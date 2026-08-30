#!/usr/bin/env bash
# 08_sweep_smoke.sh — fault-injection certification for 08_sweep.py.
# Rewritten per re-review finding 4: isolated run root, exact-state
# assertions, no tautologies, nonzero exit on any failure.
#
#   VAST_API_KEY=... HF_TOKEN=... bash 08_sweep_smoke.sh <instance_id>
#
# Portability: uses python for timeouts (macOS has no coreutils timeout).
set -uo pipefail
IID="${1:?usage: 08_sweep_smoke.sh <instance_id>}"
cd "$(dirname "$0")"

export SWEEP_ROOT="$(mktemp -d)/sweeproot"
mkdir -p "$SWEEP_ROOT"
echo "isolated SWEEP_ROOT=$SWEEP_ROOT"
read -r BOX_IP BOX_PORT < <(python3 08_sweep.py --instance "$IID" coords)
SSH="ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes -p $BOX_PORT root@$BOX_IP"
pass=0; fail=0
check() { if eval "$2"; then echo "PASS  $1"; pass=$((pass+1));
          else echo "FAIL  $1"; fail=$((fail+1)); fi; }
run_to() {  # run_to <seconds> <cmd...> — hard timeout, returns cmd rc or 124
  python3 - "$@" <<'PY'
import subprocess, sys
try:
    r = subprocess.run(sys.argv[2:], timeout=int(sys.argv[1]))
    sys.exit(r.returncode)
except subprocess.TimeoutExpired:
    sys.exit(124)
PY
}

# tiny plan: 1 trial, small caps; epochs 0-ish is invalid so use real epochs
# but scenarios never let training finish
python3 - <<'PY'
import yaml
p = yaml.safe_load(open("sweep_plan.yaml"))
p["plan_name"] = "smoke"
p["axes"] = {"mix_ratio": [0.0], "lr": [1.0e-5], "epochs": [1]}
p["budget"]["trial_wall_cap_s"] = 600
p["budget"]["per_trial_cap_usd"] = 0.40
p["budget"]["hard_sweep_cap_usd"] = 2.00
yaml.safe_dump(p, open("sweep_smoke_plan.yaml", "w"))
PY

remote_groups() { $SSH "pgrep -f 'trial_.*_.*\.sh' 2>/dev/null | wc -l"; }

echo "== S2: bad balance credentials fail closed, before any spend =="
BEFORE=$(find "$SWEEP_ROOT" -name 'attempt_*.json' 2>/dev/null | wc -l)
if VAST_API_KEY=broken_key run_to 60 python3 08_sweep.py \
     --plan sweep_smoke_plan.yaml --instance "$IID" --no-deadman run; then
  check "S2 bad key must not exit 0" false
else
  check "S2 bad key exits nonzero" true
fi
AFTER=$(find "$SWEEP_ROOT" -name 'attempt_*.json' 2>/dev/null | wc -l)
check "S2 no attempt files created" "[ \"$BEFORE\" -eq \"$AFTER\" ]"
check "S2 no remote trial groups" "[ \"\$(remote_groups)\" -eq 0 ]"

echo "== S4: disk gate trips remotely before training =="
python3 - <<'PY'
import yaml
p = yaml.safe_load(open("sweep_smoke_plan.yaml"))
p["plan_name"] = "smoke_disk"
p["budget"]["min_free_disk_gb"] = 99999
yaml.safe_dump(p, open("sweep_disk_plan.yaml", "w"))
PY
run_to 900 python3 08_sweep.py --plan sweep_disk_plan.yaml \
  --instance "$IID" --no-deadman run || true
DR=$(ls -d "$SWEEP_ROOT"/*/ 2>/dev/null | tail -1)
check "S4 exact reason=disk_low recorded" \
  "grep -l disk_low \"$DR\"attempt_*.json 2>/dev/null | grep -q ."
check "S4 accelerate never launched" \
  "! \$SSH 'grep -rl accelerate.commands.launch /root/sweep/*/*/trial.log 2>/dev/null | xargs -r grep -l 04_train_kd 2>/dev/null | grep -q .'"
check "S4 no remote trial groups" "[ \"\$(remote_groups)\" -eq 0 ]"

echo "== S1: conductor kill mid-trial; remote wall timer + reconcile =="
python3 08_sweep.py --plan sweep_smoke_plan.yaml --instance "$IID" \
  --no-deadman run > "$SWEEP_ROOT/s1.log" 2>&1 &
CPID=$!
AUID=""
for i in $(seq 1 60); do
  A=$(find "$SWEEP_ROOT" -name 'attempt_*.json' ! -name '*baseline*' \
      -exec grep -l '"state": "running"' {} + 2>/dev/null | head -1)
  if [ -n "$A" ]; then AUID=$(python3 -c "import json,sys;print(json.load(open('$A')).get('attempt_uuid',''))"); fi
  [ -n "$AUID" ] && break
  sleep 10
done
check "S1 attempt reached running with uuid" "[ -n \"$AUID\" ]"
kill -9 "$CPID" 2>/dev/null
sleep 5
check "S1 conductor dead" "! kill -0 $CPID 2>/dev/null"
# the smoke wall cap is 600s (or $-derived deadline, smaller): wait it out
echo "  waiting for in-group wall timer (<= 700s) ..."
DEADLINE_OK=false
for i in $(seq 1 70); do
  if [ "$(remote_groups)" -eq 0 ]; then DEADLINE_OK=true; break; fi
  sleep 10
done
check "S1 remote wall timer killed the process group" "$DEADLINE_OK"
# restart: must reconcile (adopt or exactly one NEW uuid), never duplicate
run_to 120 python3 08_sweep.py --plan sweep_smoke_plan.yaml \
  --instance "$IID" --no-deadman run > "$SWEEP_ROOT/s1b.log" 2>&1 || true
N_UUIDS=$($SSH "ls -d /root/sweep/*/ 2>/dev/null | grep -v baseline | head -1 | xargs -r -I{} ls {} | wc -l")
check "S1 restart created at most one new attempt dir" "[ \"$N_UUIDS\" -le 2 ]"

echo "== S3: torn local attempt file can never read as success =="
F=$(find "$SWEEP_ROOT" -name 'attempt_*.json' | head -1)
if [ -n "$F" ]; then
  printf '{"state": "succ' > "$F"
  OUT=$(run_to 60 python3 08_sweep.py --plan sweep_smoke_plan.yaml report 2>&1)
  echo "$OUT" | grep -q '"state": "complete"' \
    && check "S3 torn file must not yield complete" false \
    || check "S3 torn file blocks completion" true
else
  check "S3 attempt file existed to corrupt" false
fi

echo "== S5: HALT mid-trial kills remote group and exits promptly =="
rm -f "$SWEEP_ROOT"/*/attempt_*.json 2>/dev/null
$SSH "rm -rf /root/sweep" 2>/dev/null
python3 08_sweep.py --plan sweep_smoke_plan.yaml --instance "$IID" \
  --no-deadman run > "$SWEEP_ROOT/s5.log" 2>&1 &
CPID=$!
LIVE=false
for i in $(seq 1 60); do
  if [ "$(remote_groups)" -gt 0 ]; then LIVE=true; break; fi
  sleep 10
done
check "S5 a trial group became live" "$LIVE"
touch "$SWEEP_ROOT/HALT"
HALT_OK=false
for i in $(seq 1 12); do
  kill -0 "$CPID" 2>/dev/null || { HALT_OK=true; break; }
  sleep 15
done
check "S5 conductor exited within POLL_S+margin of HALT" "$HALT_OK"
sleep 10
check "S5 remote group killed on HALT" "[ \"\$(remote_groups)\" -eq 0 ]"
check "S5 halted state recorded" \
  "grep -rl '\"state\": \"halted\"' \"$SWEEP_ROOT\"/*/attempt_*.json 2>/dev/null | grep -q ."
rm -f "$SWEEP_ROOT/HALT"

echo
if [ "$fail" -ne 0 ]; then
  echo "SMOKE FAILED: $fail scenario check(s)"
  exit 1
fi
echo "SMOKE PASSED: $pass checks"
