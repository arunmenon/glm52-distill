#!/usr/bin/env python3
"""08_sweep.py — recipe sweep conductor (autoloop review rebuild).

Successor to 08_conductor.py for the mix/lr/epochs sweep, implementing the
review's minimal safe adaptation plan:

  finding 1  every result-affecting input is a mandatory plan field and is
             hashed into the trial id
  finding 2  a trial is SUCCESS only when the remote result.json was written
             atomically, carries status=success, the matching config digest,
             and every required metric; anything else is a FAILED attempt,
             retried up to attempt_cap
  finding 4  no promotion/ranking until every trial has a terminal state
  finding 5  live Vast credit + instance price checked BEFORE each trial:
             credit - pessimistic_next_cost must stay above the reserve
  finding 6  frozen plan digest; per-attempt atomic JSON files (fsync), the
             ledger is derived, torn writes cannot corrupt state
  finding 7  two-stage gate: hard eligibility floors, then rank; fail-closed
             on missing metrics
  finding 8  no halving; all trials run on the full corpus

Usage:
  export VAST_API_KEY=...
  python3 08_sweep.py --plan sweep_plan.yaml --instance 49077564 run
  python3 08_sweep.py --plan sweep_plan.yaml report
  touch runs/sweep/HALT      # cooperative stop between trials
"""

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml

REPO_DIR = Path(__file__).parent
SWEEP_ROOT = REPO_DIR / "runs" / "sweep"
HALT = SWEEP_ROOT / "HALT"
SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=20"]
VAST_API = "https://console.vast.ai/api/v0"
HEARTBEAT_STALE_S = 900     # remote heartbeat older than this => hung => kill
POLL_S = 60


def atomic_write(path: Path, obj: dict):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha12(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def vast_get(path: str) -> dict:
    key = os.environ.get("VAST_API_KEY") or sys.exit("VAST_API_KEY not set")
    r = requests.get(f"{VAST_API}{path}", timeout=30,
                     headers={"Authorization": f"Bearer {key}"})
    r.raise_for_status()   # fail CLOSED: any API error aborts the sweep
    return r.json()


def vast_credit() -> float:
    return float(vast_get("/users/current/").get("credit") or 0.0)


def vast_instance(iid: int) -> dict:
    d = vast_get(f"/instances/{iid}/")
    return d.get("instances") or d


class Box:
    def __init__(self, instance_id: int):
        self.iid = instance_id
        inst = vast_instance(instance_id)
        if inst.get("actual_status") != "running":
            sys.exit(f"instance {instance_id} not running: "
                     f"{inst.get('actual_status')}")
        self.ip = inst["public_ipaddr"]
        self.port = inst["ports"]["22/tcp"][0]["HostPort"]
        self.dph = float(inst.get("dph_total") or 1.0)

    def ssh(self, cmd: str, timeout: int = 120):
        return subprocess.run(
            ["ssh", *SSH_OPTS, "-p", str(self.port), f"root@{self.ip}", cmd],
            capture_output=True, text=True, timeout=timeout)

    def scp_to(self, local: Path, remote: str):
        r = subprocess.run(
            ["scp", *SSH_OPTS, "-P", str(self.port), str(local),
             f"root@{self.ip}:{remote}"], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"scp_to failed: {r.stderr[-300:]}")

    def scp_from(self, remote: str, local: Path) -> bool:
        r = subprocess.run(
            ["scp", *SSH_OPTS, "-P", str(self.port),
             f"root@{self.ip}:{remote}", str(local)],
            capture_output=True, text=True)
        return r.returncode == 0


TRIAL_SCRIPT = r"""#!/bin/bash
# generated per-trial; runs as its own process group (setsid from conductor)
set -u
T=/root/sweep/{tid}
mkdir -p $T
exec > $T/trial.log 2>&1
touch $T/heartbeat
( while true; do touch $T/heartbeat; sleep 60; done ) &
HB=$!
fail() {{ echo "$1"; python3 - <<PYEOF
import json, os
tmp = "$T/result.tmp"
json.dump({{"status": "failed", "reason": "$1",
           "config_digest": "{tid}"}}, open(tmp, "w"))
os.replace(tmp, "$T/result.json")
PYEOF
kill $HB 2>/dev/null; exit 1; }}

source /venv/main/bin/activate
cd /root/repo
git rev-parse HEAD | grep -q ^{code_rev} || fail wrong_code_rev
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && fail gpu_busy
FREE_GB=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
[ "$FREE_GB" -lt {min_free_gb} ] && fail disk_low

export HF_TOKEN={hf_token}
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/venv/main/bin/python -m accelerate.commands.launch \
  --config_file configs/ds_zero3_1gpu_optoffload.yaml 04_train_kd.py \
  --model {model_path} --data {data_path} --out $T/ckpt \
  --alpha {alpha} --lr {lr} --epochs {epochs} \
  --micro-bsz {micro_bsz} --grad-accum {grad_accum} \
  --max-seq-len {max_seq_len} --seed {seed} {mix_args} \
  || fail train_failed
[ -f $T/ckpt/final/model.safetensors ] || fail no_final_model
# vLLM needs bare text-model tensor names (journal: language_model prefix)
/venv/main/bin/python - <<PYEOF || fail rename_failed
from safetensors.torch import load_file, save_file
p = "$T/ckpt/final/model.safetensors"
sd = load_file(p)
save_file({{k.replace("model.language_model.", "model."): v
           for k, v in sd.items()}}, p, metadata={{"format": "pt"}})
PYEOF

TP={tp} TASKS="ifeval,gsm8k_cot" OUTDIR=$T/evals MODEL_LEN=16384 \
  bash 07_eval_benchmarks.sh $T/ckpt/final || fail eval_failed

/venv/main/bin/python - <<PYEOF || fail parse_failed
import glob, json, os
out = {{}}
res = glob.glob("$T/evals/*/**/results_*.json", recursive=True)
assert res, "no lm-eval results"
d = json.load(open(sorted(res)[-1]))["results"]
g = d.get("gsm8k_cot", {{}})
out["gsm8k_strict"] = g.get("exact_match,strict-match")
out["gsm8k_flexible"] = g.get("exact_match,flexible-extract")
ts = glob.glob("$T/evals/*_ifeval_thinkstripped.json")
assert ts, "no thinkstripped ifeval"
t = json.load(open(ts[-1]))
out["ifeval_prompt_strict_ts"] = t["prompt_level_strict_acc"]
out["ifeval_inst_strict_ts"] = t["inst_level_strict_acc"]
assert all(v is not None for v in out.values()), f"missing metric: {{out}}"
tmp = "$T/result.tmp"
json.dump({{"status": "success", "config_digest": "{tid}",
           "metrics": out}}, open(tmp, "w"))
os.replace(tmp, "$T/result.json")
PYEOF
rm -rf $T/ckpt   # finding 11: no checkpoint accumulation across trials
kill $HB 2>/dev/null
echo TRIAL_DONE
"""


def load_plan(path: Path) -> dict:
    plan = yaml.safe_load(path.read_text())
    for section in ("fixed", "axes", "gate", "budget"):
        if section not in plan:
            sys.exit(f"plan missing section: {section}")
    return plan


def expand_trials(plan: dict, code_rev: str, data_digest: str) -> list[dict]:
    axes = plan["axes"]
    keys = sorted(axes)
    trials = []
    for combo in itertools.product(*(axes[k] for k in keys)):
        cfg = dict(plan["fixed"])
        cfg.update(dict(zip(keys, combo)))
        cfg["code_rev"] = code_rev
        cfg["data_digest"] = data_digest
        trials.append({"config": cfg, "trial_id": sha12(cfg)})
    return trials


def attempt_path(run_dir: Path, tid: str) -> Path:
    return run_dir / f"attempt_{tid}.json"


def read_attempt(run_dir: Path, tid: str) -> dict:
    p = attempt_path(run_dir, tid)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {"state": "corrupt"}


def guard_money(plan: dict, box: Box):
    credit = vast_credit()
    next_cost = min(plan["budget"]["per_trial_cap_usd"],
                    box.dph * plan["budget"]["trial_wall_cap_s"] / 3600)
    if credit - next_cost < plan["budget"]["reserve_usd"]:
        sys.exit(f"MONEY GUARD: credit ${credit:.2f} - next ${next_cost:.2f} "
                 f"< reserve ${plan['budget']['reserve_usd']:.2f}")


def kill_stale(box: Box, tid: str):
    # kill the whole process group of any previous attempt (finding 2/5);
    # marker never matches this ssh because the command lives in a file
    box.ssh(f"pkill -9 -g $(pgrep -f 'sweeptrial_{tid}_marker' | head -1) "
            f"2>/dev/null; pkill -9 -f 'sweeptrial_{tid}_marker' 2>/dev/null; "
            f"true")


def run_trial(plan: dict, box: Box, trial: dict, run_dir: Path) -> bool:
    tid = trial["trial_id"]
    cfg = trial["config"]
    apath = attempt_path(run_dir, tid)
    prev = read_attempt(run_dir, tid)
    attempts = prev.get("attempts", 0)
    if prev.get("state") == "success":
        return True
    if attempts >= plan["budget"]["attempt_cap"]:
        return False
    kill_stale(box, tid)

    guard_money(plan, box)
    atomic_write(apath, {"state": "running", "trial_id": tid, "config": cfg,
                         "attempts": attempts + 1, "started": time.time()})

    mix_args = ""
    if cfg["mix_ratio"] > 0:
        mix_args = (f"--mix-data {cfg['mix_data']} "
                    f"--mix-ratio {cfg['mix_ratio']}")
    script = TRIAL_SCRIPT.format(
        tid=tid, code_rev=cfg["code_rev"][:12],
        min_free_gb=plan["budget"]["min_free_disk_gb"],
        hf_token=os.environ.get("HF_TOKEN", ""),
        mix_args=mix_args, **{k: cfg[k] for k in
                              ("model_path", "data_path", "alpha", "lr",
                               "epochs", "micro_bsz", "grad_accum",
                               "max_seq_len", "seed", "tp")})
    local = run_dir / f"trial_{tid}.sh"
    local.write_text(script)
    box.scp_to(local, f"/root/trial_{tid}.sh")
    box.ssh(f"setsid bash /root/trial_{tid}.sh sweeptrial_{tid}_marker "
            f">/dev/null 2>&1 < /dev/null &")

    deadline = time.time() + plan["budget"]["trial_wall_cap_s"]
    result_local = run_dir / f"result_{tid}.json"
    while time.time() < deadline:
        time.sleep(POLL_S)
        if HALT.exists():
            kill_stale(box, tid)
            atomic_write(apath, {**read_attempt(run_dir, tid),
                                 "state": "halted"})
            sys.exit("HALT observed; trial killed and marked halted")
        hb = box.ssh(f"stat -c %Y /root/sweep/{tid}/heartbeat 2>/dev/null; "
                     f"date +%s")
        parts = hb.stdout.split()
        if len(parts) == 2 and int(parts[1]) - int(parts[0]) > HEARTBEAT_STALE_S:
            break   # hung
        if box.scp_from(f"/root/sweep/{tid}/result.json", result_local):
            try:
                res = json.loads(result_local.read_text())
            except json.JSONDecodeError:
                continue   # torn remote write never happens post-rename; retry
            required = plan["gate"]["eligibility"]["required_metrics"]
            ok = (res.get("status") == "success"
                  and res.get("config_digest") == tid
                  and all(res.get("metrics", {}).get(m) is not None
                          for m in required))
            atomic_write(apath, {
                "state": "success" if ok else "failed",
                "trial_id": tid, "config": cfg, "attempts": attempts + 1,
                "result": res, "finished": time.time()})
            return ok
    kill_stale(box, tid)
    atomic_write(apath, {"state": "failed", "trial_id": tid, "config": cfg,
                         "attempts": attempts + 1, "reason": "wall_or_hang"})
    return False


def gate_and_rank(plan: dict, trials: list[dict], run_dir: Path) -> dict:
    elig = plan["gate"]["eligibility"]
    rows, incomplete = [], []
    for t in trials:
        a = read_attempt(run_dir, t["trial_id"])
        if a.get("state") == "success":
            rows.append({"trial_id": t["trial_id"], "config": t["config"],
                         "metrics": a["result"]["metrics"]})
        elif a.get("attempts", 0) < plan["budget"]["attempt_cap"]:
            incomplete.append(t["trial_id"])
    if incomplete:   # finding 4: no ranking over a partial sweep
        return {"state": "incomplete", "pending": incomplete}
    eligible = [r for r in rows
                if r["metrics"]["ifeval_prompt_strict_ts"]
                >= elig["ifeval_prompt_strict_ts_min"]
                and r["metrics"]["gsm8k_strict"] >= elig["gsm8k_strict_min"]]
    ranked = sorted(eligible,
                    key=lambda r: -r["metrics"][plan["gate"]["ranking"]])
    return {"state": "complete", "n_trials": len(trials),
            "n_succeeded": len(rows), "n_eligible": len(eligible),
            "ranked": ranked}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="sweep_plan.yaml")
    ap.add_argument("--instance", type=int)
    ap.add_argument("cmd", choices=["run", "report", "plan"])
    args = ap.parse_args()

    plan = load_plan(REPO_DIR / args.plan)
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)

    code_rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR,
                              capture_output=True, text=True).stdout.strip()

    if args.cmd == "plan":
        trials = expand_trials(plan, code_rev, "PREVIEW")
        for t in trials:
            print(t["trial_id"], {k: t["config"][k] for k in
                                  ("mix_ratio", "lr", "epochs")})
        return

    if args.cmd == "run" and not args.instance:
        sys.exit("--instance required for run")

    box = Box(args.instance) if args.cmd == "run" else None
    if box:
        dig = box.ssh("sha256sum /root/store/packed/mt_qwen35_9b/train/"
                      "data-00000-of-00001.arrow | cut -c1-12")
        data_digest = dig.stdout.strip() or sys.exit("no data digest from box")
    else:
        data_digest = "REPORT"

    # freeze the plan (finding 6): digest covers everything but data/code,
    # which live inside each trial id
    plan_digest = sha12(plan)
    run_dir = SWEEP_ROOT / plan_digest
    run_dir.mkdir(parents=True, exist_ok=True)
    frozen = run_dir / "plan_frozen.json"
    if frozen.exists():
        if json.loads(frozen.read_text())["plan"] != plan:
            sys.exit("plan file changed under an existing run dir; "
                     "start a new sweep instead of editing this one")
    else:
        atomic_write(frozen, {"plan": plan, "plan_digest": plan_digest,
                              "code_rev": code_rev})

    if args.cmd == "report":
        trials = expand_trials(plan, code_rev, data_digest)
        print(json.dumps({"note": "report uses current code_rev for ids; "
                          "run ids may differ if code moved"}, indent=1))
        print(json.dumps(gate_and_rank(plan, trials, run_dir), indent=1))
        return

    trials = expand_trials(plan, code_rev, data_digest)
    print(f"sweep {plan_digest}: {len(trials)} trials, "
          f"box ${box.dph:.2f}/hr, credit ${vast_credit():.2f}")
    for i, t in enumerate(trials, 1):
        if HALT.exists():
            sys.exit("HALT present; stopping between trials")
        cfg = t["config"]
        print(f"[{i}/{len(trials)}] {t['trial_id']} "
              f"mix={cfg['mix_ratio']} lr={cfg['lr']} ep={cfg['epochs']}")
        ok = run_trial(plan, box, t, run_dir)
        print("  ->", "SUCCESS" if ok else "FAILED/exhausted")
    report = gate_and_rank(plan, trials, run_dir)
    atomic_write(run_dir / "sweep_report.json", report)
    print(json.dumps(report, indent=1)[:2000])


if __name__ == "__main__":
    main()
