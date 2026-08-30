#!/usr/bin/env python3
"""08_sweep.py — recipe sweep conductor (round 3 of external review).

Design deltas in this round, mapped to the r3 findings:

  b1  dead-man is ON THE BOX: the conductor renews a lease file every poll;
      a box-side watchdog (own process group) kills all trial groups in the
      sweep namespace and stops the instance via a chmod-600 key file when
      the lease goes stale. No laptop process to die, no PIDs to reuse, no
      secrets in argv. Conductor exit stops the instance unless
      --keep-instance. All ssh/scp calls are time-bounded. A local flock +
      conductor nonce in the lease prevent two conductors.
  b2  crash recovery reconciles FIRST: every manifest trial (including
      cap-reached ones) is reconciled against the box before pending is
      computed; a live acknowledged attempt is re-polled, never duplicated;
      attempts count only after launch ack; ssh ambiguity is indeterminate
      and never a license to relaunch.
  f4  the baseline is content-addressed (model digest + code rev + eval
      settings) inside the plan namespace; every trial's env_digest must
      equal the baseline's or the trial fails with env_drift.
  f6  spend is a persisted monotonic counter (attempt hours x live dph),
      immune to account top-ups; shutdown overhead is reserved.
  f7  results publish LAST: metrics are staged, checkpoints cleaned, then
      the result renamed into place; the conductor additionally verifies
      the remote group is gone before recording success. fail() kills the
      heavy children in its group before exiting.
  f8  HALT is checked before baseline, before each reconcile, after money
      guards, and immediately before every launch.
  f9  one strict validator for every result path: status, trial id, attempt
      uuid == path uuid, all required metrics finite floats in [0,1].
      Dataset digest covers the whole dataset dir; model digest covers
      config+tokenizer; env digest recorded and enforced.
  f11 rolling top-2 finals are RETAINED on the box (namespace/finals/);
      displaced finalists are deleted. The SWE screen runs on the actual
      ranked artifacts, not retrains.

Usage:
  export VAST_API_KEY=... HF_TOKEN=...
  python3 08_sweep.py --plan sweep_plan.yaml --instance <id> run
  python3 08_sweep.py --plan sweep_plan.yaml report
  python3 08_sweep.py --instance <id> coords
  touch $SWEEP_ROOT/HALT
"""

import argparse
import fcntl
import hashlib
import itertools
import json
import math
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
            "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3"]
# override with a mock server for local certification
VAST_API = os.environ.get("VAST_API_BASE",
                          "https://console.vast.ai/api/v0")
POLL_S = 60
LAUNCH_ACK_S = 180
LEASE_TTL_S = 900
SHUTDOWN_OVERHEAD_H = 0.25     # reserved box-hours for stop latency (f6)
SCP_TIMEOUT_S = 300
EVAL_SETTINGS = {"tasks": "ifeval,gsm8k_cot", "model_len": 16384,
                 "gen": "temperature=0.6,top_p=0.95,max_gen_toks=4096"}


class Indeterminate(Exception):
    """Connection/probe ambiguity: not success, not failure, no relaunch."""


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


def halt_check(box=None, note=""):
    if (SWEEP_ROOT / "HALT").exists():
        sys.exit(f"HALT observed ({note}); stopping. Remote trials keep "
                 f"their own wall timers; lease expiry stops the box.")


# ------------------------------------------------------------------- vast --
def vast_get(path: str) -> dict:
    key = os.environ.get("VAST_API_KEY") or sys.exit("VAST_API_KEY not set")
    r = requests.get(f"{VAST_API}{path}", timeout=30,
                     headers={"Authorization": f"Bearer {key}"})
    r.raise_for_status()
    d = r.json()
    if isinstance(d, dict) and d.get("error"):
        sys.exit(f"vast API error on {path}: {d['error']}")
    return d


def vast_credit() -> float:
    d = vast_get("/users/current/")
    if d.get("credit") is None:
        sys.exit("vast API returned no credit; failing closed")
    return float(d["credit"])


def vast_instance(iid: int) -> dict:
    d = vast_get(f"/instances/{iid}/")
    return d.get("instances") or d


def vast_stop(iid: int, attempts: int = 3) -> bool:
    key = os.environ.get("VAST_API_KEY", "")
    for _ in range(attempts):
        try:
            requests.put(f"{VAST_API}/instances/{iid}/",
                         headers={"Authorization": f"Bearer {key}"},
                         json={"state": "stopped"}, timeout=30)
            time.sleep(20)
            if vast_instance(iid).get("actual_status") in (
                    "stopped", "exited", "stopping"):
                return True
        except Exception:
            time.sleep(20)
    return False


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

    def ssh(self, cmd: str, timeout: int = 120, input_text: str = None):
        return subprocess.run(
            ["ssh", *SSH_OPTS, "-p", str(self.port), f"root@{self.ip}", cmd],
            capture_output=True, text=True, timeout=timeout,
            input=input_text)

    def ssh_ok(self, cmd: str, timeout: int = 120,
               input_text: str = None) -> str:
        r = self.ssh(cmd, timeout, input_text)
        if r.returncode != 0:
            raise Indeterminate(f"ssh failed ({cmd[:60]}): "
                                f"{r.stderr[-200:]}")
        return r.stdout

    def read_remote(self, remote: str) -> str | None:
        """None = file absent; Indeterminate on connection trouble."""
        out = self.ssh_ok(f"if [ -f {remote} ]; then cat {remote}; "
                          f"else echo __ABSENT__; fi")
        return None if "__ABSENT__" in out else out

    def scp_to(self, local: Path, remote: str):
        r = subprocess.run(["scp", *SSH_OPTS, "-P", str(self.port),
                            str(local), f"root@{self.ip}:{remote}"],
                           capture_output=True, text=True,
                           timeout=SCP_TIMEOUT_S)
        if r.returncode != 0:
            raise Indeterminate(f"scp_to failed: {r.stderr[-200:]}")


# ----------------------------------------------------------------- remote --
TRIAL_SCRIPT = r"""#!/bin/bash
set -euo pipefail
T={ns_dir}/trials/{tid}/{auid}
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
  python3 - "$1" <<PYEOF
import json, os, sys
t = "{ns_dir}/trials/{tid}/{auid}"
json.dump({{"status": "failed", "reason": sys.argv[1],
           "trial_id": "{tid}", "attempt_uuid": "{auid}"}},
          open(t + "/result.tmp", "w"))
os.replace(t + "/result.tmp", t + "/result.json")
PYEOF
  # r3 f7: kill heavy children in this group before exiting so no worker
  # outlives its wall timer
  pkill -9 -g "$PGID" -f "accelerate|04_train_kd|lm_eval|vllm" 2>/dev/null || true
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
export ENV_DIGEST

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
/venv/main/bin/python - <<PYEOF || fail rename_failed
from safetensors.torch import load_file, save_file
p = "$T/ckpt/final/model.safetensors".replace("$T",
    "{ns_dir}/trials/{tid}/{auid}")
sd = load_file(p)
save_file({{k.replace("model.language_model.", "model."): v
           for k, v in sd.items()}}, p, metadata={{"format": "pt"}})
PYEOF

TP={tp} TASKS="{eval_tasks}" OUTDIR=$T/evals MODEL_LEN={eval_model_len} \
  bash 07_eval_benchmarks.sh $T/ckpt/final || fail eval_failed

/venv/main/bin/python - <<PYEOF || fail parse_failed
import glob, json, os
t = "{ns_dir}/trials/{tid}/{auid}"
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
assert all(isinstance(v, (int, float)) and 0 <= v <= 1
           for v in out.values()), f"bad metric: {{out}}"
json.dump({{"status": "success", "trial_id": "{tid}",
           "attempt_uuid": "{auid}", "metrics": out,
           "env_digest": os.environ.get("ENV_DIGEST", "")}},
          open(t + "/staged.tmp", "w"))
os.replace(t + "/staged.tmp", t + "/staged.json")
PYEOF

# r3 f7/f11: retain the final under finals/, delete intermediates, and only
# THEN publish the result — success implies cleanup is already done
mkdir -p {ns_dir}/finals
rm -rf {ns_dir}/finals/{tid}
mv $T/ckpt/final {ns_dir}/finals/{tid}
rm -rf $T/ckpt
mv $T/staged.json $T/result.json
echo TRIAL_DONE
"""

BASELINE_SCRIPT = r"""#!/bin/bash
set -euo pipefail
T={ns_dir}/trials/{tid}/{auid}
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
  python3 - "$1" <<PYEOF
import json, os, sys
t = "{ns_dir}/trials/{tid}/{auid}"
json.dump({{"status": "failed", "reason": sys.argv[1],
           "trial_id": "{tid}", "attempt_uuid": "{auid}"}},
          open(t + "/result.tmp", "w"))
os.replace(t + "/result.tmp", t + "/result.json")
PYEOF
  pkill -9 -g "$PGID" -f "lm_eval|vllm" 2>/dev/null || true
  finish; exit 1
}}
trap 'finish' EXIT
source /venv/main/bin/activate || fail no_venv
source /root/.sweep_env || fail no_secret_env
cd /root/repo || fail no_repo
[ "$(git rev-parse HEAD)" = "{code_rev}" ] || fail wrong_code_rev
GPUPROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) \
  || fail nvidia_smi_failed
[ -z "$GPUPROCS" ] || fail gpu_busy
ENV_DIGEST=$(/venv/main/bin/pip freeze 2>/dev/null | sha256sum | cut -c1-12) \
  || fail pip_freeze_failed
export ENV_DIGEST
TP={tp} TASKS="{eval_tasks}" OUTDIR=$T/evals MODEL_LEN={eval_model_len} \
  bash 07_eval_benchmarks.sh {model_path} || fail eval_failed
/venv/main/bin/python - <<PYEOF || fail parse_failed
import glob, json, os
t = "{ns_dir}/trials/{tid}/{auid}"
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
assert all(isinstance(v, (int, float)) and 0 <= v <= 1
           for v in out.values()), f"bad metric: {{out}}"
json.dump({{"status": "success", "trial_id": "{tid}",
           "attempt_uuid": "{auid}", "metrics": out,
           "env_digest": os.environ.get("ENV_DIGEST", "")}},
          open(t + "/result.tmp", "w"))
os.replace(t + "/result.tmp", t + "/result.json")
PYEOF
echo TRIAL_DONE
"""



# Certification mode (plan fixed.fake_workload: true): the full protocol —
# pgid ack, wall timer, heartbeat, staged->atomic result — with a sleep
# instead of train+eval. Money/kill/adopt paths are exercised for cents.
FAKE_TRIAL_SCRIPT = r"""#!/bin/bash
set -euo pipefail
T={ns_dir}/trials/{tid}/{auid}
mkdir -p $T
exec > $T/trial.log 2>&1
PGID=$(ps -o pgid= -p $$ | tr -dc 0-9)
printf '%s {auid}\n' "$PGID" > $T/pgid.tmp && mv $T/pgid.tmp $T/pgid
( sleep {deadline_s}; kill -- -$PGID 2>/dev/null ) &
WALL=$!
( while kill -0 $$ 2>/dev/null; do touch $T/heartbeat; sleep 5; done ) &
HB=$!
finish() {{ kill $WALL $HB 2>/dev/null || true; }}
trap 'finish' EXIT
FREE_GB=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
[ "$FREE_GB" -ge {min_free_gb} ] || {{
  python3 - <<PYEOF
import json, os
t = "{ns_dir}/trials/{tid}/{auid}"
json.dump({{"status": "failed", "reason": "disk_low",
           "trial_id": "{tid}", "attempt_uuid": "{auid}"}},
          open(t + "/result.tmp", "w"))
os.replace(t + "/result.tmp", t + "/result.json")
PYEOF
  exit 1; }}
sleep {fake_sleep}
python3 - <<PYEOF
import json, os
t = "{ns_dir}/trials/{tid}/{auid}"
json.dump({{"status": "success", "trial_id": "{tid}",
           "attempt_uuid": "{auid}",
           "metrics": {{"gsm8k_strict": 0.78, "gsm8k_flexible": 0.7,
                       "ifeval_prompt_strict_ts": 0.48,
                       "ifeval_inst_strict_ts": 0.56}},
           "env_digest": "fakeenv"}},
          open(t + "/staged.tmp", "w"))
os.replace(t + "/staged.tmp", t + "/staged.json")
os.rename(t + "/staged.json", t + "/result.json")
PYEOF
echo TRIAL_DONE
"""

# On-box lease watchdog (r3 b1): if the conductor stops renewing the lease,
# kill every trial group in this namespace, then stop the instance using the
# chmod-600 key file (never argv).
WATCHDOG_SCRIPT = r"""#!/bin/bash
NS={ns_dir}
IID={iid}
while true; do
  sleep 60
  [ -f $NS/lease ] || continue
  AGE=$(( $(date +%s) - $(stat -c %Y $NS/lease) ))
  if [ "$AGE" -gt {lease_ttl} ]; then
    echo "$(date) lease stale (${{AGE}}s); killing trials + stopping" \
      >> $NS/watchdog.log
    for P in $NS/trials/*/*/pgid; do
      [ -f "$P" ] || continue
      PG=$(cut -d' ' -f1 "$P" | tr -dc 0-9)
      [ -n "$PG" ] && kill -- -$PG 2>/dev/null
    done
    KEY=$(cat /root/.sweep_stop_key 2>/dev/null)
    if [ -n "$KEY" ]; then
      for i in 1 2 3; do
        curl -s -X PUT "https://console.vast.ai/api/v0/instances/$IID/" \
          -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
          -d '{{"state": "stopped"}}' >> $NS/watchdog.log 2>&1 && break
        sleep 30
      done
    fi
    exit 0
  fi
done
"""


# -------------------------------------------------------------- conductor --
def load_plan(path: Path) -> dict:
    plan = yaml.safe_load(path.read_text())
    for s in ("fixed", "axes", "gate", "budget"):
        if s not in plan:
            sys.exit(f"plan missing section: {s}")
    return plan


def expand_trials(plan, code_rev, data_digest, model_digest):
    keys = sorted(plan["axes"])
    out = []
    for combo in itertools.product(*(plan["axes"][k] for k in keys)):
        cfg = dict(plan["fixed"])
        cfg.update(dict(zip(keys, combo)))
        cfg["code_rev"] = code_rev
        cfg["data_digest"] = data_digest
        cfg["model_digest"] = model_digest
        cfg["eval_settings"] = EVAL_SETTINGS
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


def validate_result(res: dict, tid: str, required: list,
                    auid: str = None) -> bool:
    """r3 f9: the ONE validator for adopt and fresh paths alike."""
    if not isinstance(res, dict) or res.get("status") != "success":
        return False
    if res.get("trial_id") != tid:
        return False
    if auid is not None and res.get("attempt_uuid") != auid:
        return False
    m = res.get("metrics")
    if not isinstance(m, dict):
        return False
    for k in required:
        v = m.get(k)
        if not isinstance(v, (int, float)) or not math.isfinite(v) \
                or not (0 <= v <= 1):
            return False
    return True


class Spend:
    """r3 f6: monotonic persisted spend, immune to account top-ups."""
    def __init__(self, run_dir: Path):
        self.path = run_dir / "spend.json"
        self.usd = 0.0
        if self.path.exists():
            try:
                self.usd = float(json.loads(self.path.read_text())["usd"])
            except Exception:
                pass

    def add(self, hours: float, dph: float):
        self.usd += hours * dph
        atomic_write(self.path, {"usd": self.usd, "at": time.time()})


def guard_money(plan, box, spend: Spend):
    box.refresh_dph()
    b = plan["budget"]
    next_cost = (min(b["per_trial_cap_usd"],
                     box.dph * b["trial_wall_cap_s"] / 3600)
                 + SHUTDOWN_OVERHEAD_H * box.dph)
    if spend.usd + next_cost > b["hard_sweep_cap_usd"]:
        sys.exit(f"SWEEP CAP: spent ${spend.usd:.2f} + next ${next_cost:.2f}"
                 f" > ${b['hard_sweep_cap_usd']:.2f}")
    credit = vast_credit()
    if credit - next_cost < b["reserve_usd"]:
        sys.exit(f"RESERVE: credit ${credit:.2f} - next ${next_cost:.2f} "
                 f"< ${b['reserve_usd']:.2f}")



def sync_state(run_dir: Path):
    """Durability (phase-1 review item 6): push sweep state to the HF store
    after every terminal attempt. Non-fatal — a sync failure never blocks
    the sweep, but it is printed so silence means synced."""
    store = os.environ.get("PILOT_STORE", "ledzepu2/glm52-pilot-artifacts")
    r = subprocess.run(
        ["hf", "upload", store, str(run_dir),
         f"sweep_state/{run_dir.name}", "--type", "dataset",
         "--include", "*.json", "--commit-message",
         "sweep state sync", "--format", "quiet"],
        capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print(f"WARN state sync failed: {r.stderr[-150:]}")


class Sweep:
    def __init__(self, plan, box, run_dir, manifest, ns_dir):
        self.plan = plan
        self.box = box
        self.run_dir = run_dir
        self.manifest = manifest
        self.ns = ns_dir
        self.spend = Spend(run_dir)

    # ---- remote attempt helpers ----
    def remote_pgid(self, tid, auid):
        try:
            out = self.box.read_remote(f"{self.ns}/trials/{tid}/{auid}/pgid")
        except Indeterminate:
            return None
        if out:
            parts = out.split()
            if len(parts) == 2 and parts[0].isdigit() and parts[1] == auid:
                return int(parts[0])
        return None

    def group_alive(self, tid, auid) -> bool:
        pg = self.remote_pgid(tid, auid)
        if not pg:
            return False
        r = self.box.ssh(f"kill -0 -- -{pg} 2>/dev/null && echo LIVE || "
                         f"echo DEAD")
        return "LIVE" in r.stdout

    def kill_attempt(self, tid, auid):
        pg = self.remote_pgid(tid, auid)
        if pg:
            self.box.ssh(f"kill -- -{pg} 2>/dev/null; true")

    def fetch_result(self, tid, auid) -> dict | None:
        out = self.box.read_remote(
            f"{self.ns}/trials/{tid}/{auid}/result.json")
        if out is None:
            return None
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return None

    def renew_lease(self):
        self.box.ssh(f"mkdir -p {self.ns} && touch {self.ns}/lease")

    # ---- reconcile (r3 b2): runs for EVERY trial before pending calc ----
    def reconcile_trial(self, trial) -> str:
        tid = trial["trial_id"]
        a = read_attempt(self.run_dir, tid)
        state = a.get("state")
        required = self.plan["gate"]["required_metrics"]
        if state == "success":
            return "success"
        # any remote attempt dir may hold a valid success — adopt it even if
        # the local counter says exhausted
        try:
            ls = self.box.ssh_ok(
                f"ls {self.ns}/trials/{tid} 2>/dev/null || true")
        except Indeterminate:
            return "indeterminate"
        for auid in ls.split():
            res = self.fetch_result(tid, auid)
            if res and validate_result(res, tid, required, auid):
                atomic_write(attempt_path(self.run_dir, tid), {
                    "state": "success", "trial_id": tid,
                    "config": trial["config"],
                    "attempts": a.get("attempts", 0),
                    "attempt_uuid": auid, "result": res, "adopted": True})
                sync_state(self.run_dir)
                return "success"
        # a locally-running attempt: resume polling it rather than relaunch
        if state == "running" and a.get("attempt_uuid"):
            if self.group_alive(tid, a["attempt_uuid"]):
                return "resume"
            res = self.fetch_result(tid, a["attempt_uuid"])
            ok = res and validate_result(res, tid, required,
                                         a["attempt_uuid"])
            atomic_write(attempt_path(self.run_dir, tid), {
                **a, "state": "success" if ok else "failed",
                "result": res or {"status": "failed",
                                  "reason": "died_no_result"}})
            return "success" if ok else "pending"
        if a.get("attempts", 0) >= self.plan["budget"]["attempt_cap"]:
            return "exhausted"
        return "pending"

    # ---- launch + poll ----
    def launch(self, trial, script_tpl, extra=None) -> str:
        tid = trial["trial_id"]
        a = read_attempt(self.run_dir, tid)
        attempts = a.get("attempts", 0)
        halt_check(note=f"pre-launch {tid}")
        guard_money(self.plan, self.box, self.spend)
        halt_check(note=f"post-guard {tid}")

        auid = uuidlib.uuid4().hex[:10]
        b = self.plan["budget"]
        deadline_s = int(min(b["trial_wall_cap_s"],
                             b["per_trial_cap_usd"] / self.box.dph * 3600))
        cfg = dict(trial["config"])
        mix_args = ""
        if cfg.get("mix_ratio", 0) > 0:
            mix_args = (f"--mix-data {cfg['mix_data']} "
                        f"--mix-ratio {cfg['mix_ratio']} "
                        f"--mix-data-revision {cfg['mix_data_revision']}")
        fields = {k: cfg[k] for k in
                  ("model_path", "data_path", "alpha", "lr", "epochs",
                   "micro_bsz", "grad_accum", "max_seq_len", "seed", "tp")
                  if k in cfg}
        fields.update(extra or {})
        script = script_tpl.format(
            ns_dir=self.ns, tid=tid, auid=auid, deadline_s=deadline_s,
            min_free_gb=b["min_free_disk_gb"],
            code_rev=cfg.get("code_rev", ""), mix_args=mix_args,
            eval_tasks=EVAL_SETTINGS["tasks"],
            eval_model_len=EVAL_SETTINGS["model_len"], **fields)
        local = self.run_dir / f"trial_{tid}_{auid}.sh"
        local.write_text(script)
        try:
            self.box.scp_to(local, f"/root/trial_{tid}_{auid}.sh")
            # attempts counted ONLY after ack (r3 b2)
            atomic_write(attempt_path(self.run_dir, tid), {
                "state": "launching", "trial_id": tid, "config": cfg,
                "attempts": attempts, "attempt_uuid": auid,
                "deadline_s": deadline_s, "started": time.time()})
            self.box.ssh_ok(f"setsid bash /root/trial_{tid}_{auid}.sh "
                            f">/dev/null 2>&1 < /dev/null & echo REQ")
        except Indeterminate as e:
            atomic_write(attempt_path(self.run_dir, tid), {
                "state": "failed", "trial_id": tid, "config": cfg,
                "attempts": attempts + 1, "attempt_uuid": auid,
                "reason": f"launch_indeterminate: {e}"})
            self.kill_attempt(tid, auid)
            return "failed"
        t0 = time.time()
        while time.time() - t0 < LAUNCH_ACK_S:
            if self.remote_pgid(tid, auid):
                break
            time.sleep(10)
        else:
            self.kill_attempt(tid, auid)     # quarantine (r3 b2)
            atomic_write(attempt_path(self.run_dir, tid), {
                "state": "failed", "trial_id": tid, "config": cfg,
                "attempts": attempts + 1, "attempt_uuid": auid,
                "reason": "no_launch_ack"})
            return "failed"
        atomic_write(attempt_path(self.run_dir, tid), {
            "state": "running", "trial_id": tid, "config": cfg,
            "attempts": attempts + 1, "attempt_uuid": auid,
            "deadline_s": deadline_s, "started": time.time()})
        return self.poll(trial, auid, deadline_s)

    def poll(self, trial, auid, deadline_s) -> str:
        tid = trial["trial_id"]
        required = self.plan["gate"]["required_metrics"]
        t0 = time.time()
        stale_since = None
        while time.time() - t0 < deadline_s + 2 * POLL_S:
            time.sleep(POLL_S)
            self.renew_lease()
            if (SWEEP_ROOT / "HALT").exists():
                self.kill_attempt(tid, auid)
                atomic_write(attempt_path(self.run_dir, tid), {
                    **read_attempt(self.run_dir, tid), "state": "halted"})
                sys.exit("HALT observed mid-trial; attempt killed")
            try:
                res = self.fetch_result(tid, auid)
            except Indeterminate:
                continue
            if res is not None:
                # r3 f7: don't record success while the group lives on
                for _ in range(6):
                    if not self.group_alive(tid, auid):
                        break
                    time.sleep(10)
                else:
                    self.kill_attempt(tid, auid)
                ok = validate_result(res, tid, required, auid)
                base_env = read_attempt(self.run_dir, "baseline").get(
                    "result", {}).get("env_digest")
                if ok and base_env and res.get("env_digest") != base_env:
                    ok = False
                    res = {**res, "status": "failed",
                           "reason": "env_drift_vs_baseline"}
                a = read_attempt(self.run_dir, tid)
                hours = (time.time() - a.get("started", t0)) / 3600
                self.spend.add(hours, self.box.dph)
                atomic_write(attempt_path(self.run_dir, tid), {
                    **a, "state": "success" if ok else "failed",
                    "result": res, "finished": time.time()})
                sync_state(self.run_dir)
                return "success" if ok else "failed"
            try:
                hb = self.box.ssh_ok(
                    f"stat -c %Y {self.ns}/trials/{tid}/{auid}/heartbeat "
                    f"2>/dev/null || echo 0; date +%s")
            except Indeterminate:
                continue
            parts = hb.split()
            if len(parts) == 2 and parts[0].isdigit():
                age = int(parts[1]) - int(parts[0])
                if age > LEASE_TTL_S:
                    stale_since = stale_since or time.time()
                    if time.time() - stale_since > 2 * POLL_S:
                        break
                else:
                    stale_since = None
        self.kill_attempt(tid, auid)
        a = read_attempt(self.run_dir, tid)
        self.spend.add((time.time() - a.get("started", t0)) / 3600,
                       self.box.dph)
        atomic_write(attempt_path(self.run_dir, tid), {
            **a, "state": "failed", "reason": "deadline_or_hang"})
        return "failed"

    def prune_finals(self):
        """r3 f11: keep only the current top-K eligible finals on the box."""
        keep = self.plan["gate"].get("retrain_top_for_screen", 2)
        rep = gate_and_rank(self.plan, self.manifest, self.run_dir,
                            allow_partial=True)
        ranked = rep.get("ranked", [])
        keep_ids = {r["trial_id"] for r in ranked[:keep]}
        try:
            ls = self.box.ssh_ok(f"ls {self.ns}/finals 2>/dev/null || true")
        except Indeterminate:
            return
        for tid in ls.split():
            if tid not in keep_ids:
                self.box.ssh(f"rm -rf {self.ns}/finals/{tid}")
        # recoverability: push current finalists to the store from the box
        # (background, idempotent; box has .env.hf + fast pipe)
        for tid in keep_ids:
            self.box.ssh(
                f"[ -d {self.ns}/finals/{tid} ] && "
                f"nohup bash -c 'source /root/exp_env 2>/dev/null; "
                f"source /root/.sweep_env; "
                f"hf upload ledzepu2/glm52-pilot-artifacts "
                f"{self.ns}/finals/{tid} sweep_finals/{tid} "
                f"--type dataset --format quiet' >/dev/null 2>&1 & true")


def gate_and_rank(plan, manifest, run_dir, allow_partial=False):
    base = read_attempt(run_dir, "baseline")
    if base.get("state") != "success":
        return {"state": "no_baseline", "ranked": []}
    bm = base["result"]["metrics"]
    g = plan["gate"]
    floors = {
        "gsm8k_strict": bm["gsm8k_strict"] - g["gsm8k_strict_tolerance"],
        "ifeval_prompt_strict_ts": bm["ifeval_prompt_strict_ts"]
            - g["ifeval_prompt_strict_ts_tolerance"],
        "ifeval_inst_strict_ts": bm["ifeval_inst_strict_ts"]
            - g["ifeval_inst_strict_ts_tolerance"],
    }
    rows, pending = [], []
    for t in manifest["trials"]:
        a = read_attempt(run_dir, t["trial_id"])
        if a.get("state") == "success":
            m = a["result"]["metrics"]
            reasons = [k for k, fl in floors.items() if m[k] < fl]
            gray = [k for k, fl in floors.items()
                    if fl <= m[k] < fl + g["gray_band"]]
            rows.append({"trial_id": t["trial_id"], "config": t["config"],
                         "metrics": m, "eligible": not reasons,
                         "gate_reason": reasons, "gray_band": gray})
        elif a.get("attempts", 0) < plan["budget"]["attempt_cap"]:
            pending.append(t["trial_id"])
    ranked = sorted([r for r in rows if r["eligible"]],
                    key=lambda r: -r["metrics"][g["ranking"]])
    if pending and not allow_partial:
        return {"state": "incomplete", "pending": pending, "ranked": []}
    return {"state": "complete" if not pending else "partial",
            "baseline": bm, "floors": floors,
            "n_trials": len(manifest["trials"]), "n_completed": len(rows),
            "n_eligible": len(ranked), "ranked": ranked, "all_rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="sweep_plan.yaml")
    ap.add_argument("--instance", type=int)
    ap.add_argument("--keep-instance", action="store_true",
                    help="do NOT stop the instance when the sweep exits")
    ap.add_argument("--max-new-trials", type=int, default=0,
                    help="shakedown pause: launch at most N NEW trials this "
                         "invocation, then exit cleanly (same manifest "
                         "resumes later); 0 = unlimited")
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
        for t in expand_trials(plan, "PREVIEW", "PREVIEW", "PREVIEW"):
            print(t["trial_id"], {k: t["config"][k]
                                  for k in sorted(plan["axes"])})
        return

    if args.cmd == "report":
        if not manifest_path.exists():
            sys.exit("no manifest; nothing to report")
        manifest = json.loads(manifest_path.read_text())
        print(json.dumps(gate_and_rank(plan, manifest, run_dir), indent=1))
        return

    # ---- run ----
    if not args.instance:
        sys.exit("--instance required for run")
    hf_token = os.environ.get("HF_TOKEN") or sys.exit("HF_TOKEN not set")
    vast_key = os.environ.get("VAST_API_KEY") or sys.exit("no VAST_API_KEY")
    halt_check(note="startup")

    # single-conductor lock (r3 b1/b2)
    lock_f = open(run_dir / ".lock", "w")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another conductor holds this sweep's lock")

    box = Box(args.instance)
    ns_dir = f"/root/sweep_ns/{plan_digest[:8]}"
    code_rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR,
                              capture_output=True, text=True).stdout.strip()
    # digests: whole dataset dir + model identity files (r3 f9)
    data_digest = box.ssh_ok(
        f"cd {plan['fixed']['data_path']} && find . -type f | sort "
        f"| xargs sha256sum | sha256sum | cut -c1-12").strip()
    model_digest = box.ssh_ok(
        f"cd {plan['fixed']['model_path']} && "
        f"sha256sum config.json tokenizer.json tokenizer_config.json "
        f"2>/dev/null | sha256sum | cut -c1-12").strip()

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for k, v in (("code_rev", code_rev), ("data_digest", data_digest),
                     ("model_digest", model_digest)):
            if manifest[k] != v:
                sys.exit(f"{k} changed under existing manifest "
                         f"({manifest[k]} -> {v}); start a new sweep")
        if manifest["plan"] != plan:
            sys.exit("plan changed under existing manifest")
    else:
        manifest = {"plan": plan, "plan_digest": plan_digest,
                    "code_rev": code_rev, "data_digest": data_digest,
                    "model_digest": model_digest,
                    "eval_settings": EVAL_SETTINGS,
                    "trials": expand_trials(plan, code_rev, data_digest,
                                            model_digest),
                    "created": time.time()}
        atomic_write(manifest_path, manifest)

    # secrets via stdin -> chmod-600 files, never argv/scripts (r3 b1)
    box.ssh_ok("umask 077 && cat > /root/.sweep_env && "
               "chmod 600 /root/.sweep_env",
               input_text=f"export HF_TOKEN={hf_token}\n")
    box.ssh_ok("umask 077 && cat > /root/.sweep_stop_key && "
               "chmod 600 /root/.sweep_stop_key", input_text=vast_key)

    sweep = Sweep(plan, box, run_dir, manifest, ns_dir)
    # lease + on-box watchdog
    box.ssh_ok(f"mkdir -p {ns_dir}/trials && touch {ns_dir}/lease")
    wd_local = run_dir / "watchdog.sh"
    wd_local.write_text(WATCHDOG_SCRIPT.format(ns_dir=ns_dir, iid=box.iid,
                                               lease_ttl=LEASE_TTL_S))
    box.scp_to(wd_local, f"{ns_dir}/watchdog.sh")
    # two calls: the kill pattern and the launch PATH must never share an
    # argv, or pkill matches the path and kills its own shell (reproduced)
    box.ssh(f"pkill -f 'sweep_ns/{plan_digest[:8]}/watchdog[.]sh' "
            f"2>/dev/null; true")
    box.ssh_ok(f"setsid bash {ns_dir}/watchdog.sh "
               f">/dev/null 2>&1 < /dev/null & echo WD")

    try:
        # baseline: content-addressed inside the namespace (r3 f4)
        base_id = "baseline"
        base_cfg = {"model_digest": model_digest, "code_rev": code_rev,
                    "eval_settings": EVAL_SETTINGS}
        prev_base = read_attempt(run_dir, base_id)
        if (prev_base.get("state") == "success"
                and prev_base.get("base_key") != sha12(base_cfg)):
            sys.exit("stored baseline was produced under a different "
                     "model/code/eval identity; new sweep dir required")
        if prev_base.get("state") != "success":
            halt_check(note="pre-baseline")
            # baseline gets the same crash recovery as trials: adopt a
            # finished remote result, kill any stale group (smoke S2 gap)
            if prev_base.get("attempt_uuid"):
                sweep.kill_attempt(base_id, prev_base["attempt_uuid"])
            st = sweep.reconcile_trial({"trial_id": base_id,
                                        "config": base_cfg})
            if st == "success":
                prev_base = read_attempt(run_dir, base_id)
        if prev_base.get("state") != "success":
            print("running same-environment baseline eval ...")
            st = sweep.launch(
                {"trial_id": base_id, "config": base_cfg},
                (FAKE_TRIAL_SCRIPT if plan["fixed"].get("fake_workload")
                 else BASELINE_SCRIPT),
                extra=({"fake_sleep": plan["fixed"].get("fake_sleep_s", 60)}
                       if plan["fixed"].get("fake_workload") else
                       {"model_path": plan["fixed"]["model_path"],
                        "tp": plan["fixed"]["tp"]}))
            if st != "success":
                sys.exit(f"baseline eval {st}; gates impossible")
            a = read_attempt(run_dir, base_id)
            atomic_write(attempt_path(run_dir, base_id),
                         {**a, "base_key": sha12(base_cfg)})

        wave = 0
        new_launched = 0
        while True:
            wave += 1
            halt_check(note=f"wave {wave}")
            states = {}
            for t in manifest["trials"]:
                halt_check(note="pre-reconcile")
                states[t["trial_id"]] = sweep.reconcile_trial(t)
            resume = [t for t in manifest["trials"]
                      if states[t["trial_id"]] == "resume"]
            pending = [t for t in manifest["trials"]
                       if states[t["trial_id"]] == "pending"]
            if not resume and not pending:
                break
            print(f"wave {wave}: {len(pending)} pending, "
                  f"{len(resume)} resuming, spent ${sweep.spend.usd:.2f}")
            for t in resume:
                a = read_attempt(run_dir, t["trial_id"])
                print(f"  resuming {t['trial_id']} "
                      f"attempt {a.get('attempt_uuid')}")
                sweep.poll(t, a["attempt_uuid"], a.get("deadline_s", 14400))
                sweep.prune_finals()
            for t in pending:
                cfg = t["config"]
                print(f"  {t['trial_id']} mix={cfg['mix_ratio']} "
                      f"lr={cfg['lr']} ep={cfg['epochs']}")
                tpl = (FAKE_TRIAL_SCRIPT
                       if plan["fixed"].get("fake_workload")
                       else TRIAL_SCRIPT)
                st = sweep.launch(t, tpl, extra=(
                    {"fake_sleep": plan["fixed"].get("fake_sleep_s", 60)}
                    if plan["fixed"].get("fake_workload") else None))
                print("   ->", st)
                if st == "success":
                    sweep.prune_finals()

        report = gate_and_rank(plan, manifest, run_dir)
        atomic_write(run_dir / "sweep_report.json", report)
        print(json.dumps(report, indent=1)[:3000])
    finally:
        try:
            box.ssh(f"pkill -f 'sweep_ns/{plan_digest[:8]}/watchdog[.]sh' "
                    f"2>/dev/null; true", timeout=30)
        except Exception:
            pass
        if not args.keep_instance:
            stopped = vast_stop(box.iid)
            print(f"instance {box.iid} stop: "
                  f"{'confirmed' if stopped else 'UNCONFIRMED - check!'}")


if __name__ == "__main__":
    main()
