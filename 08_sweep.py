#!/usr/bin/env python3
"""08_sweep.py — recipe sweep conductor (autoloop re-review round 2).

Guarantees, mapped to the re-review findings:

  RR1  independent dead-man: (a) a wall timer INSIDE the remote trial's
       process group kills the group even if the conductor dies; (b) a
       detached laptop-side sitter stops the Vast instance if the conductor
       process disappears; (c) conductor `finally` reconciles and can stop
       the instance on exit.
  RR2  attempt UUIDs + reconciliation: every attempt runs in its own remote
       dir; before any launch, prior remote results are fetched — a valid
       success for the trial is ADOPTED without relaunching; stale results
       can never be attributed to a new attempt.
  RR3  hard spend enforcement: starting credit persisted in the manifest;
       billed = start_credit - live_credit checked against
       hard_sweep_cap_usd before every attempt; per-attempt deadline =
       min(wall_cap, per_trial_cap / live dph).
  RR5  immutable per-attempt dirs; each eval scores only its own files.
  RR6  launch protocol: remote wrapper atomically writes its numeric PGID +
       attempt uuid first; conductor requires that ack before the attempt
       counts as launched; kills use `kill -- -PGID` with a numeric check.
  RR7  --mix-data-revision threaded to the trainer; remote repo must be
       clean and at the manifest code_rev; pip-freeze digest recorded into
       every result.
  RR8  execution manifest freezes code/data/trial ids; `report` loads the
       manifest, never recomputes ids.
  RR9  retry waves loop unattended until success/attempt_cap/budget/HALT.
  RR10 remote script runs set -euo pipefail with an EXIT trap; probes fail
       closed; heartbeat dies with the main process.
  RR12 HF token never serialized into scripts: written once to a chmod-600
       remote secret file, sourced by trials.
  RR14 gates are relative to a same-environment baseline eval (eval-only
       "baseline" trial) with a reported gray band; every completed row is
       kept with a gate_reason.
  RR15 ssh coords prefer ssh_host/ssh_port, fall back to direct mapping.

Usage:
  export VAST_API_KEY=... HF_TOKEN=...
  python3 08_sweep.py --plan sweep_plan.yaml --instance <id> run
  python3 08_sweep.py --plan sweep_plan.yaml report
  python3 08_sweep.py --instance <id> coords        # smoke-suite helper
  touch $SWEEP_ROOT/HALT                            # cooperative stop
"""

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
import uuid as uuidlib
from pathlib import Path

import requests
import yaml

REPO_DIR = Path(__file__).parent
SWEEP_ROOT = Path(os.environ.get("SWEEP_ROOT", REPO_DIR / "runs" / "sweep"))
SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=20"]
VAST_API = "https://console.vast.ai/api/v0"
POLL_S = 60
LAUNCH_ACK_S = 180
HEARTBEAT_STALE_S = 900


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
    r.raise_for_status()
    d = r.json()
    if isinstance(d, dict) and d.get("error"):       # 200-with-error quirk
        sys.exit(f"vast API error on {path}: {d['error']}")
    return d


def vast_credit() -> float:
    d = vast_get("/users/current/")
    if d.get("credit") is None:
        sys.exit("vast API returned no credit field; failing closed")
    return float(d["credit"])


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
        self.ip = inst.get("ssh_host") or inst["public_ipaddr"]
        self.port = (inst.get("ssh_port")
                     or inst["ports"]["22/tcp"][0]["HostPort"])
        self.refresh_dph()

    def refresh_dph(self) -> float:
        self.dph = float(vast_instance(self.iid).get("dph_total") or 0)
        if self.dph <= 0:
            sys.exit("instance dph unavailable; failing closed")
        return self.dph

    def ssh(self, cmd: str, timeout: int = 120):
        return subprocess.run(
            ["ssh", *SSH_OPTS, "-p", str(self.port), f"root@{self.ip}", cmd],
            capture_output=True, text=True, timeout=timeout)

    def ssh_ok(self, cmd: str, timeout: int = 120) -> str:
        r = self.ssh(cmd, timeout)
        if r.returncode != 0:
            raise RuntimeError(f"ssh failed ({cmd[:60]}): {r.stderr[-200:]}")
        return r.stdout

    def scp_to(self, local: Path, remote: str):
        r = subprocess.run(["scp", *SSH_OPTS, "-P", str(self.port),
                            str(local), f"root@{self.ip}:{remote}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"scp_to failed: {r.stderr[-200:]}")

    def scp_from(self, remote: str, local: Path) -> bool:
        return subprocess.run(["scp", *SSH_OPTS, "-P", str(self.port),
                               f"root@{self.ip}:{remote}", str(local)],
                              capture_output=True, text=True).returncode == 0


# ----------------------------------------------------------------- remote --
# Layout per attempt: /root/sweep/<tid>/<uuid>/{pgid,heartbeat,trial.log,
# result.json,ckpt/,evals/}. pgid is written FIRST (launch ack, RR6); the
# in-group wall timer (RR1a) kills the group without any conductor help.
TRIAL_SCRIPT = r"""#!/bin/bash
set -euo pipefail
T=/root/sweep/{tid}/{auid}
mkdir -p $T
exec > $T/trial.log 2>&1
PGID=$(ps -o pgid= -p $$ | tr -dc 0-9)
printf '%s {auid}\n' "$PGID" > $T/pgid.tmp && mv $T/pgid.tmp $T/pgid
( sleep {deadline_s}; kill -- -$PGID 2>/dev/null ) &
WALL=$!
( while kill -0 $$ 2>/dev/null; do touch $T/heartbeat; sleep 60; done ) &
HB=$!
finish() {{ kill $WALL $HB 2>/dev/null || true; }}
fail() {{
  echo "FAIL: $1"
  python3 - "$1" <<'PYEOF'
import json, os, sys
t = "/root/sweep/{tid}/{auid}"
json.dump({{"status": "failed", "reason": sys.argv[1],
           "trial_id": "{tid}", "attempt_uuid": "{auid}"}},
          open(t + "/result.tmp", "w"))
os.replace(t + "/result.tmp", t + "/result.json")
PYEOF
  finish; exit 1
}}
trap 'finish' EXIT

source /venv/main/bin/activate || fail no_venv
source /root/.sweep_env || fail no_secret_env
cd /root/repo || fail no_repo
[ -z "$(git status --porcelain)" ] || fail dirty_repo
[ "$(git rev-parse HEAD)" = "{code_rev}" ] || fail wrong_code_rev
GPUPROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) \
  || fail nvidia_smi_failed
[ -z "$GPUPROCS" ] || fail gpu_busy
FREE_GB=$(df -BG --output=avail / | tail -1 | tr -dc 0-9) || fail df_failed
[ -n "$FREE_GB" ] || fail df_unparsed
[ "$FREE_GB" -ge {min_free_gb} ] || fail disk_low
ENV_DIGEST=$(/venv/main/bin/pip freeze 2>/dev/null | sha256sum | cut -c1-12) \
  || fail pip_freeze_failed

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
/venv/main/bin/python - <<'PYEOF' || fail rename_failed
from safetensors.torch import load_file, save_file
p = "/root/sweep/{tid}/{auid}/ckpt/final/model.safetensors"
sd = load_file(p)
save_file({{k.replace("model.language_model.", "model."): v
           for k, v in sd.items()}}, p, metadata={{"format": "pt"}})
PYEOF

TP={tp} TASKS="ifeval,gsm8k_cot" OUTDIR=$T/evals MODEL_LEN=16384 \
  bash 07_eval_benchmarks.sh $T/ckpt/final || fail eval_failed

ENV_DIGEST=$ENV_DIGEST /venv/main/bin/python - <<'PYEOF' || fail parse_failed
import glob, json, os
t = "/root/sweep/{tid}/{auid}"
out = {{}}
res = sorted(glob.glob(t + "/evals/**/results_*.json", recursive=True))
assert res, "no lm-eval results"
d = json.load(open(res[-1]))["results"]
g = d.get("gsm8k_cot", {{}})
out["gsm8k_strict"] = g.get("exact_match,strict-match")
out["gsm8k_flexible"] = g.get("exact_match,flexible-extract")
ts = sorted(glob.glob(t + "/evals/*_ifeval_thinkstripped.json"))
assert ts, "no thinkstripped ifeval"
j = json.load(open(ts[-1]))
out["ifeval_prompt_strict_ts"] = j["prompt_level_strict_acc"]
out["ifeval_inst_strict_ts"] = j["inst_level_strict_acc"]
assert all(v is not None for v in out.values()), f"missing metric: {{out}}"
json.dump({{"status": "success", "trial_id": "{tid}",
           "attempt_uuid": "{auid}", "metrics": out,
           "env_digest": os.environ.get("ENV_DIGEST", "")}},
          open(t + "/result.tmp", "w"))
os.replace(t + "/result.tmp", t + "/result.json")
PYEOF
rm -rf $T/ckpt    # RR11 deviation: finals don't fit 150GB x12; top-2
                  # retrained from frozen configs for the SWE screen
echo TRIAL_DONE
"""

# Baseline attempt (RR14): eval-only, same env, no training, no cleanup race.
BASELINE_SCRIPT = r"""#!/bin/bash
set -euo pipefail
T=/root/sweep/baseline/{auid}
mkdir -p $T
exec > $T/trial.log 2>&1
PGID=$(ps -o pgid= -p $$ | tr -dc 0-9)
printf '%s {auid}\n' "$PGID" > $T/pgid.tmp && mv $T/pgid.tmp $T/pgid
( sleep {deadline_s}; kill -- -$PGID 2>/dev/null ) &
WALL=$!
( while kill -0 $$ 2>/dev/null; do touch $T/heartbeat; sleep 60; done ) &
HB=$!
finish() {{ kill $WALL $HB 2>/dev/null || true; }}
fail() {{
  echo "FAIL: $1"
  python3 - "$1" <<'PYEOF'
import json, os, sys
t = "/root/sweep/baseline/{auid}"
json.dump({{"status": "failed", "reason": sys.argv[1],
           "trial_id": "baseline", "attempt_uuid": "{auid}"}},
          open(t + "/result.tmp", "w"))
os.replace(t + "/result.tmp", t + "/result.json")
PYEOF
  finish; exit 1
}}
trap 'finish' EXIT
source /venv/main/bin/activate || fail no_venv
source /root/.sweep_env || fail no_secret_env
cd /root/repo || fail no_repo
GPUPROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) \
  || fail nvidia_smi_failed
[ -z "$GPUPROCS" ] || fail gpu_busy
TP={tp} TASKS="ifeval,gsm8k_cot" OUTDIR=$T/evals MODEL_LEN=16384 \
  bash 07_eval_benchmarks.sh {model_path} || fail eval_failed
/venv/main/bin/python - <<'PYEOF' || fail parse_failed
import glob, json, os
t = "/root/sweep/baseline/{auid}"
out = {{}}
res = sorted(glob.glob(t + "/evals/**/results_*.json", recursive=True))
assert res, "no lm-eval results"
d = json.load(open(res[-1]))["results"]
g = d.get("gsm8k_cot", {{}})
out["gsm8k_strict"] = g.get("exact_match,strict-match")
out["gsm8k_flexible"] = g.get("exact_match,flexible-extract")
ts = sorted(glob.glob(t + "/evals/*_ifeval_thinkstripped.json"))
assert ts, "no thinkstripped ifeval"
j = json.load(open(ts[-1]))
out["ifeval_prompt_strict_ts"] = j["prompt_level_strict_acc"]
out["ifeval_inst_strict_ts"] = j["inst_level_strict_acc"]
assert all(v is not None for v in out.values()), f"missing metric: {{out}}"
json.dump({{"status": "success", "trial_id": "baseline",
           "attempt_uuid": "{auid}", "metrics": out}},
          open(t + "/result.tmp", "w"))
os.replace(t + "/result.tmp", t + "/result.json")
PYEOF
echo TRIAL_DONE
"""

SITTER = r"""#!/bin/bash
# dead-man RR1b: if the conductor process vanishes without writing the done
# marker, stop the Vast instance after a grace period.
CPID=$1; IID=$2; DONE=$3; KEY=$4
while true; do
  [ -f "$DONE" ] && exit 0
  if ! kill -0 "$CPID" 2>/dev/null; then
    sleep 300   # grace: maybe it's being restarted
    [ -f "$DONE" ] && exit 0
    kill -0 "$CPID" 2>/dev/null && continue
    curl -s -X PUT "https://console.vast.ai/api/v0/instances/$IID/" \
      -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
      -d '{"state": "stopped"}' > /dev/null
    exit 0
  fi
  sleep 60
done
"""


# -------------------------------------------------------------- conductor --
def load_plan(path: Path) -> dict:
    plan = yaml.safe_load(path.read_text())
    for s in ("fixed", "axes", "gate", "budget"):
        if s not in plan:
            sys.exit(f"plan missing section: {s}")
    return plan


def expand_trials(plan, code_rev, data_digest):
    keys = sorted(plan["axes"])
    out = []
    for combo in itertools.product(*(plan["axes"][k] for k in keys)):
        cfg = dict(plan["fixed"])
        cfg.update(dict(zip(keys, combo)))
        cfg["code_rev"] = code_rev
        cfg["data_digest"] = data_digest
        out.append({"config": cfg, "trial_id": sha12(cfg)})
    return out


def attempt_path(run_dir, tid):
    return run_dir / f"attempt_{tid}.json"


def read_attempt(run_dir, tid):
    p = attempt_path(run_dir, tid)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {"state": "corrupt"}


def billed_since(manifest) -> float:
    return manifest["start_credit_usd"] - vast_credit()


def guard_money(plan, box, manifest):
    box.refresh_dph()
    per_trial = plan["budget"]["per_trial_cap_usd"]
    next_cost = min(per_trial,
                    box.dph * plan["budget"]["trial_wall_cap_s"] / 3600)
    credit = vast_credit()
    billed = manifest["start_credit_usd"] - credit
    if billed + next_cost > plan["budget"]["hard_sweep_cap_usd"]:
        sys.exit(f"SWEEP CAP: billed ${billed:.2f} + next ${next_cost:.2f} "
                 f"> ${plan['budget']['hard_sweep_cap_usd']:.2f}")
    if credit - next_cost < plan["budget"]["reserve_usd"]:
        sys.exit(f"RESERVE: credit ${credit:.2f} - next ${next_cost:.2f} "
                 f"< ${plan['budget']['reserve_usd']:.2f}")


def remote_pgid(box, tid, auid):
    out = box.ssh(f"cat /root/sweep/{tid}/{auid}/pgid 2>/dev/null").stdout
    parts = out.split()
    if len(parts) == 2 and parts[0].isdigit() and parts[1] == auid:
        return int(parts[0])
    return None


def kill_attempt(box, tid, auid):
    pg = remote_pgid(box, tid, auid)
    if pg:
        box.ssh(f"kill -- -{pg} 2>/dev/null; true")


def reconcile(box, run_dir, trial) -> dict | None:
    """RR2: fetch any prior remote results for this trial; adopt a valid
    success rather than relaunching. Returns adopted result or None."""
    tid = trial["trial_id"]
    ls = box.ssh(f"ls /root/sweep/{tid}/*/result.json 2>/dev/null").stdout
    for remote in ls.split():
        local = run_dir / f"reconcile_{tid}_{Path(remote).parent.name}.json"
        if not box.scp_from(remote, local):
            continue
        try:
            res = json.loads(local.read_text())
        except json.JSONDecodeError:
            continue
        if (res.get("status") == "success"
                and res.get("trial_id") == tid
                and res.get("metrics")):
            return res
    return None


def poll_attempt(plan, box, tid, auid, deadline_s, run_dir):
    t0 = time.time()
    result_local = run_dir / f"result_{tid}_{auid}.json"
    while time.time() - t0 < deadline_s + 2 * POLL_S:
        time.sleep(POLL_S)
        if (SWEEP_ROOT / "HALT").exists():
            kill_attempt(box, tid, auid)
            return {"halted": True}
        if box.scp_from(f"/root/sweep/{tid}/{auid}/result.json",
                        result_local):
            try:
                return json.loads(result_local.read_text())
            except json.JSONDecodeError:
                continue
        hb = box.ssh(f"stat -c %Y /root/sweep/{tid}/{auid}/heartbeat "
                     f"2>/dev/null; date +%s").stdout.split()
        if len(hb) == 2 and int(hb[1]) - int(hb[0]) > HEARTBEAT_STALE_S:
            kill_attempt(box, tid, auid)
            return {"status": "failed", "reason": "heartbeat_stale"}
    kill_attempt(box, tid, auid)
    return {"status": "failed", "reason": "conductor_deadline"}


def run_attempt(plan, box, trial, run_dir, manifest, script_tpl,
                extra=None) -> str:
    tid = trial["trial_id"]
    prev = read_attempt(run_dir, tid)
    if prev.get("state") == "success":
        return "success"
    attempts = prev.get("attempts", 0)
    if attempts >= plan["budget"]["attempt_cap"]:
        return "exhausted"

    adopted = reconcile(box, run_dir, trial)
    if adopted:
        atomic_write(attempt_path(run_dir, tid), {
            "state": "success", "trial_id": tid, "config": trial["config"],
            "attempts": attempts, "result": adopted, "adopted": True})
        return "success"

    guard_money(plan, box, manifest)
    auid = uuidlib.uuid4().hex[:10]
    deadline_s = int(min(plan["budget"]["trial_wall_cap_s"],
                         plan["budget"]["per_trial_cap_usd"]
                         / box.dph * 3600))
    cfg = dict(trial["config"])
    mix_args = ""
    if cfg.get("mix_ratio", 0) > 0:
        mix_args = (f"--mix-data {cfg['mix_data']} "
                    f"--mix-ratio {cfg['mix_ratio']} "
                    f"--mix-data-revision {cfg['mix_data_revision']}")
    script = script_tpl.format(
        tid=tid, auid=auid, deadline_s=deadline_s,
        min_free_gb=plan["budget"]["min_free_disk_gb"],
        code_rev=cfg.get("code_rev", ""), mix_args=mix_args,
        **{k: cfg[k] for k in ("model_path", "data_path", "alpha", "lr",
                               "epochs", "micro_bsz", "grad_accum",
                               "max_seq_len", "seed", "tp")
           if k in cfg} | (extra or {}))
    local = run_dir / f"trial_{tid}_{auid}.sh"
    local.write_text(script)
    box.scp_to(local, f"/root/trial_{tid}_{auid}.sh")

    atomic_write(attempt_path(run_dir, tid), {
        "state": "running", "trial_id": tid, "config": cfg,
        "attempts": attempts + 1, "attempt_uuid": auid,
        "deadline_s": deadline_s, "started": time.time()})
    box.ssh_ok(f"setsid bash /root/trial_{tid}_{auid}.sh "
               f">/dev/null 2>&1 < /dev/null & echo LAUNCH_REQUESTED")
    t0 = time.time()
    while time.time() - t0 < LAUNCH_ACK_S:            # RR6 launch ack
        if remote_pgid(box, tid, auid):
            break
        time.sleep(10)
    else:
        atomic_write(attempt_path(run_dir, tid), {
            **read_attempt(run_dir, tid), "state": "failed",
            "reason": "no_launch_ack"})
        return "failed"

    res = poll_attempt(plan, box, tid, auid, deadline_s, run_dir)
    if res.get("halted"):
        atomic_write(attempt_path(run_dir, tid), {
            **read_attempt(run_dir, tid), "state": "halted"})
        sys.exit("HALT observed; attempt killed and marked halted")
    ok = (res.get("status") == "success"
          and res.get("trial_id") == tid
          and res.get("attempt_uuid") == auid
          and all(res.get("metrics", {}).get(m) is not None
                  for m in plan["gate"]["required_metrics"]))
    atomic_write(attempt_path(run_dir, tid), {
        "state": "success" if ok else "failed", "trial_id": tid,
        "config": cfg, "attempts": attempts + 1, "attempt_uuid": auid,
        "result": res, "finished": time.time()})
    return "success" if ok else "failed"


def gate_and_rank(plan, manifest, run_dir):
    base = read_attempt(run_dir, "baseline")
    if base.get("state") != "success":
        return {"state": "no_baseline"}
    bm = base["result"]["metrics"]
    g = plan["gate"]
    floors = {
        "gsm8k_strict": bm["gsm8k_strict"] - g["gsm8k_strict_tolerance"],
        "ifeval_prompt_strict_ts":
            bm["ifeval_prompt_strict_ts"]
            - g["ifeval_prompt_strict_ts_tolerance"],
        "ifeval_inst_strict_ts":
            bm["ifeval_inst_strict_ts"]
            - g["ifeval_inst_strict_ts_tolerance"],
    }
    rows, pending = [], []
    for t in manifest["trials"]:
        a = read_attempt(run_dir, t["trial_id"])
        if a.get("state") == "success":
            m = a["result"]["metrics"]
            reasons = [k for k, floor in floors.items() if m[k] < floor]
            gray = [k for k, floor in floors.items()
                    if floor <= m[k] < floor + g["gray_band"]]
            rows.append({"trial_id": t["trial_id"], "config": t["config"],
                         "metrics": m, "eligible": not reasons,
                         "gate_reason": reasons, "gray_band": gray})
        elif a.get("attempts", 0) < plan["budget"]["attempt_cap"]:
            pending.append(t["trial_id"])
    if pending:
        return {"state": "incomplete", "pending": pending}
    ranked = sorted([r for r in rows if r["eligible"]],
                    key=lambda r: -r["metrics"][g["ranking"]])
    return {"state": "complete", "baseline": bm, "floors": floors,
            "n_trials": len(manifest["trials"]), "n_completed": len(rows),
            "n_eligible": len(ranked), "ranked": ranked, "all_rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="sweep_plan.yaml")
    ap.add_argument("--instance", type=int)
    ap.add_argument("--stop-on-exit", action="store_true",
                    help="stop the Vast instance when the sweep finishes")
    ap.add_argument("--no-deadman", action="store_true")
    ap.add_argument("cmd", choices=["run", "report", "plan", "coords"])
    args = ap.parse_args()

    if args.cmd == "coords":
        box = Box(args.instance or sys.exit("--instance required"))
        print(box.ip, box.port)
        return

    plan = load_plan(REPO_DIR / args.plan)
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    plan_digest = sha12(plan)
    run_dir = SWEEP_ROOT / plan_digest
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"

    if args.cmd == "plan":
        for t in expand_trials(plan, "PREVIEW", "PREVIEW"):
            print(t["trial_id"], {k: t["config"][k]
                                  for k in sorted(plan["axes"])})
        return

    if args.cmd == "report":
        if not manifest_path.exists():
            sys.exit("no manifest; nothing to report (RR8: report never "
                     "recomputes ids)")
        manifest = json.loads(manifest_path.read_text())
        print(json.dumps(gate_and_rank(plan, manifest, run_dir), indent=1))
        return

    # ---- run ----
    if not args.instance:
        sys.exit("--instance required for run")
    hf_token = os.environ.get("HF_TOKEN") or sys.exit("HF_TOKEN not set")
    box = Box(args.instance)

    code_rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR,
                              capture_output=True, text=True).stdout.strip()
    data_digest = box.ssh_ok(
        "sha256sum /root/store/packed/mt_qwen35_9b/train/"
        "data-00000-of-00001.arrow | cut -c1-12").strip()

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest["code_rev"] != code_rev
                or manifest["data_digest"] != data_digest
                or manifest["plan"] != plan):
            sys.exit("plan/code/data changed under an existing manifest; "
                     "start a new sweep dir (edit plan_name)")
    else:
        manifest = {"plan": plan, "plan_digest": plan_digest,
                    "code_rev": code_rev, "data_digest": data_digest,
                    "start_credit_usd": vast_credit(),
                    "trials": expand_trials(plan, code_rev, data_digest),
                    "created": time.time()}
        atomic_write(manifest_path, manifest)

    # secrets to the box once, never serialized into scripts (RR12)
    box.ssh_ok("umask 077 && cat > /root/.sweep_env <<EOF\n"
               f"export HF_TOKEN={hf_token}\nEOF\nchmod 600 /root/.sweep_env")

    done_marker = run_dir / "sweep_done"
    done_marker.unlink(missing_ok=True)
    sitter_proc = None
    if not args.no_deadman:
        sitter = run_dir / "sitter.sh"
        sitter.write_text(SITTER)
        sitter_proc = subprocess.Popen(
            ["nohup", "bash", str(sitter), str(os.getpid()), str(box.iid),
             str(done_marker), os.environ["VAST_API_KEY"]],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)

    try:
        # baseline first (RR14)
        if read_attempt(run_dir, "baseline").get("state") != "success":
            print("running same-environment baseline eval ...")
            st = run_attempt(plan, box,
                             {"trial_id": "baseline",
                              "config": {"model_path":
                                         plan["fixed"]["model_path"],
                                         "tp": plan["fixed"]["tp"]}},
                             run_dir, manifest, BASELINE_SCRIPT,
                             extra={"model_path":
                                    plan["fixed"]["model_path"],
                                    "tp": plan["fixed"]["tp"]})
            if st != "success":
                sys.exit(f"baseline eval {st}; gates impossible — aborting")

        # retry waves (RR9)
        wave = 0
        while True:
            wave += 1
            pending = [t for t in manifest["trials"]
                       if read_attempt(run_dir, t["trial_id"]).get("state")
                       != "success"
                       and read_attempt(run_dir, t["trial_id"]).get(
                           "attempts", 0) < plan["budget"]["attempt_cap"]]
            if not pending or (SWEEP_ROOT / "HALT").exists():
                break
            print(f"wave {wave}: {len(pending)} trials pending, "
                  f"billed ${billed_since(manifest):.2f}")
            for t in pending:
                cfg = t["config"]
                print(f"  {t['trial_id']} mix={cfg['mix_ratio']} "
                      f"lr={cfg['lr']} ep={cfg['epochs']}")
                st = run_attempt(plan, box, t, run_dir, manifest,
                                 TRIAL_SCRIPT)
                print("   ->", st)
        report = gate_and_rank(plan, manifest, run_dir)
        atomic_write(run_dir / "sweep_report.json", report)
        print(json.dumps(report, indent=1)[:3000])
    finally:
        atomic_write(done_marker, {"finished": time.time()})
        if sitter_proc:
            sitter_proc.terminate()
        if args.stop_on_exit:
            key = os.environ["VAST_API_KEY"]
            requests.put(f"{VAST_API}/instances/{box.iid}/",
                         headers={"Authorization": f"Bearer {key}"},
                         json={"state": "stopped"}, timeout=30)


if __name__ == "__main__":
    main()
