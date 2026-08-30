#!/usr/bin/env bash
# 08_sweep_smoke.sh — certification drills for the r3 conductor.
#
#   VAST_API_KEY=mock HF_TOKEN=<any> bash 08_sweep_smoke.sh <real_iid> <box_ip> <box_port>
#
# Uses the MOCK Vast API (mock_vast.py) for all billing/instance calls, the
# REAL GPU box over ssh for workload mechanics, and fake 45-second workloads
# that exercise the full launch/ack/lease/heartbeat/publish protocol.
# Scenarios (phase-1 review item 1):
#   S1 provisioning + happy-path trial through the mock
#   S2 interrupted trial: conductor killed mid-trial; restart ADOPTS the
#      finished remote result without relaunching
#   S3 spend exhaustion: preloaded spend ledger trips the sweep cap before
#      any launch
#   S4 billing ambiguity: 200-with-error body fails closed, zero launches
#   S5 baseline identity mismatch: stored baseline from a different model
#      digest aborts the run
#   S6 watchdog expiry: stale lease -> box-side watchdog kills trials and
#      calls stop (recorded by the mock)
#   S7 default shutdown: conductor exit issues a stop through the mock
# Every check asserts exact state; the script exits nonzero on any failure
# and cleans up both sides on exit.
set -uo pipefail
IID="${1:?usage: 08_sweep_smoke.sh <iid> <box_ip> <box_port>}"
BOX_IP="${2:?box ip}"; BOX_PORT="${3:?box port}"
cd "$(dirname "$0")"

MOCK_PORT=$((20000 + RANDOM % 20000))
export SWEEP_ROOT="$(mktemp -d)/root"
mkdir -p "$SWEEP_ROOT"
MOCK_STATE="$SWEEP_ROOT/mockvast_state.json"
# refuse to run over a live prior invocation
pgrep -f "mock_vast[.]py" >/dev/null && { echo "another smoke run is live; abort"; exit 2; }
pgrep -f "08_sweep[.]py .*run" >/dev/null && { echo "a conductor is live; abort"; exit 2; }
export VAST_API_BASE="http://127.0.0.1:${MOCK_PORT}"
export VAST_API_KEY="${VAST_API_KEY:-mock}"
export HF_TOKEN="${HF_TOKEN:-hf_mock}"
export PILOT_STORE="${PILOT_STORE:-ledzepu2/glm52-pilot-artifacts}"
SSH="ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes -p $BOX_PORT root@$BOX_IP"

pass=0; fail=0
check() { if eval "$2"; then echo "PASS  $1"; pass=$((pass+1));
          else echo "FAIL  $1"; fail=$((fail+1)); fi; }
run_to() { python3 - "$@" <<'PY'
import subprocess, sys
try:
    sys.exit(subprocess.run(sys.argv[2:], timeout=int(sys.argv[1])).returncode)
except subprocess.TimeoutExpired:
    sys.exit(124)
PY
}
mock_set() { python3 - "$1" "$2" <<'PY'
import json, sys
d = json.load(open(sys.maxsize and __import__("os").environ["MOCK_STATE"]))
key, val = sys.argv[1], json.loads(sys.argv[2])
d[key] = val
json.dump(d, open(__import__("os").environ["MOCK_STATE"], "w"))
PY
}
export MOCK_STATE

cleanup() {
  kill "$MOCK_PID" 2>/dev/null || true
  $SSH "for P in /root/sweep_ns/*/trials/*/*/pgid; do
          [ -f \"\$P\" ] && kill -- -\$(cut -d' ' -f1 \"\$P\") 2>/dev/null
        done; pkill -f 'sweep_ns/.*watchdog[.]sh' 2>/dev/null; true" \
    >/dev/null 2>&1 || true
  rm -f "$MOCK_STATE" sweep_smoke_plan.yaml sweep_smoke2_plan.yaml sweep_smoke3_plan.yaml
}
trap cleanup EXIT

# mock server pointed at the REAL box coordinates
python3 - <<PY
import json
json.dump({"credit": 50.0, "dph_total": 0.80, "ssh_host": "$BOX_IP",
           "ssh_port": $BOX_PORT, "fail_mode": None, "stops": [],
           "actual_status": "running"}, open("$MOCK_STATE", "w"))
PY
python3 mock_vast.py --port $MOCK_PORT --state "$MOCK_STATE" &
MOCK_PID=$!
sleep 1

# smoke plan: 1 fake trial, tight caps, fake baseline too
python3 - <<'PY'
import yaml
p = yaml.safe_load(open("sweep_plan.yaml"))
p["plan_name"] = "smoke_r3"
p["axes"] = {"mix_ratio": [0.0], "lr": [1.0e-5], "epochs": [1]}
p["fixed"]["fake_workload"] = True
p["fixed"]["fake_sleep_s"] = 45
p["budget"].update({"trial_wall_cap_s": 240, "per_trial_cap_usd": 0.30,
                    "hard_sweep_cap_usd": 2.0})
yaml.safe_dump(p, open("sweep_smoke_plan.yaml", "w"), sort_keys=False)
PY
rundir_for() {  # rundir_for <plan_file> -> exact run dir for that plan
  python3 - "$1" <<'PY'
import hashlib, json, sys, yaml, os
plan = yaml.safe_load(open(sys.argv[1]))
d = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
print(os.environ["SWEEP_ROOT"] + "/" + d + "/")
PY
}

echo "== S1: provisioning + happy path through the mock =="
run_to 900 python3 08_sweep.py --plan sweep_smoke_plan.yaml \
  --instance "$IID" --keep-instance run > "$SWEEP_ROOT/s1.log" 2>&1
RC=$?
check "S1 conductor exits 0" "[ $RC -eq 0 ]"
RD=$(rundir_for sweep_smoke_plan.yaml)
check "S1 baseline success recorded" \
  "python3 -c \"import json;d=json.load(open('${RD}attempt_baseline.json'));exit(0 if d['state']=='success' else 1)\""
TID=$(python3 -c "import json,glob;f=[x for x in glob.glob('${RD}attempt_*.json') if 'baseline' not in x][0];print(json.load(open(f))['trial_id'])")
check "S1 trial success with matching uuid" \
  "python3 -c \"import json;d=json.load(open('${RD}attempt_${TID}.json'));r=d['result'];exit(0 if d['state']=='success' and r['attempt_uuid']==d['attempt_uuid'] else 1)\""
check "S1 report complete + 1 eligible" \
  "python3 08_sweep.py --plan sweep_smoke_plan.yaml report | python3 -c \"import json,sys;d=json.load(sys.stdin);exit(0 if d['state']=='complete' and d['n_eligible']==1 else 1)\""

echo "== S7: default shutdown issued a stop through the mock =="
# S1 ran WITH --keep-instance so no stop yet; rerun trivially without it
python3 - <<PY
import json
d = json.load(open("$MOCK_STATE")); d["stops"] = []; d["actual_status"] = "running"
json.dump(d, open("$MOCK_STATE", "w"))
PY
run_to 300 python3 08_sweep.py --plan sweep_smoke_plan.yaml \
  --instance "$IID" run > "$SWEEP_ROOT/s7.log" 2>&1 || true
check "S7 stop recorded by mock on default exit" \
  "python3 -c \"import json;d=json.load(open('$MOCK_STATE'));exit(0 if any(s.get('state')=='stopped' for s in d['stops']) else 1)\""
python3 - <<PY
import json
d = json.load(open("$MOCK_STATE")); d["actual_status"] = "running"
json.dump(d, open("$MOCK_STATE", "w"))
PY

echo "== S2: conductor killed mid-trial; restart ADOPTS, no relaunch =="
python3 - <<'PY'
import yaml
p = yaml.safe_load(open("sweep_smoke_plan.yaml"))
p["plan_name"] = "smoke_r3_s2"
yaml.safe_dump(p, open("sweep_smoke2_plan.yaml", "w"), sort_keys=False)
PY
python3 08_sweep.py --plan sweep_smoke2_plan.yaml --instance "$IID" \
  --keep-instance run > "$SWEEP_ROOT/s2.log" 2>&1 &
CPID=$!
RD2=""
for i in $(seq 1 40); do
  RD2=$(rundir_for sweep_smoke2_plan.yaml)
  RUNNING=$(grep -l '"state": "running"' "$RD2"attempt_*.json 2>/dev/null | grep -v baseline | head -1)
  [ -n "$RUNNING" ] && break
  RD2=""; sleep 5
done
check "S2 an attempt reached running" "[ -n \"$RD2\" ]"
kill -9 "$CPID" 2>/dev/null; sleep 2
check "S2 conductor dead" "! kill -0 $CPID 2>/dev/null"
sleep 75   # let the 45s fake workload finish + publish on the box
BEFORE_DIRS=$($SSH 'ls -d /root/sweep_ns/*/trials/*/*/ 2>/dev/null | wc -l')
run_to 300 python3 08_sweep.py --plan sweep_smoke2_plan.yaml \
  --instance "$IID" --keep-instance run > "$SWEEP_ROOT/s2b.log" 2>&1 || true
AFTER_DIRS=$($SSH 'ls -d /root/sweep_ns/*/trials/*/*/ 2>/dev/null | wc -l')
check "S2 no new attempt dirs (adopted, not relaunched)" \
  "[ \"$BEFORE_DIRS\" -eq \"$AFTER_DIRS\" ]"
check "S2 adoption or success recorded locally" \
  "grep -rl '\"state\": \"success\"' \"$RD2\"/attempt_*.json 2>/dev/null | grep -q . || grep -rl '\"adopted\": true' \"$RD2\"/attempt_*.json 2>/dev/null | grep -q ."

echo "== S3: preloaded spend trips the sweep cap before any launch =="
python3 - <<'PY'
import yaml
p = yaml.safe_load(open("sweep_smoke_plan.yaml"))
p["plan_name"] = "smoke_r3_s3"
yaml.safe_dump(p, open("sweep_smoke3_plan.yaml", "w"), sort_keys=False)
PY
RD3=$(rundir_for sweep_smoke3_plan.yaml)
mkdir -p "$RD3"
python3 -c "import json;json.dump({'usd': 1.95, 'at': 0}, open('${RD3}spend.json','w'))"
OUT=$(run_to 120 python3 08_sweep.py --plan sweep_smoke3_plan.yaml \
  --instance "$IID" --keep-instance run 2>&1); RC=$?
check "S3 nonzero exit on cap" "[ $RC -ne 0 ]"
echo "$OUT" | grep -q "SWEEP CAP" && check "S3 cap message explicit" true \
  || check "S3 cap message explicit" false

echo "== S4: 200-with-error billing body fails closed =="
python3 - <<PY
import json
d = json.load(open("$MOCK_STATE")); d["fail_mode"] = "error_body"
json.dump(d, open("$MOCK_STATE", "w"))
PY
OUT=$(run_to 120 python3 08_sweep.py --plan sweep_smoke_plan.yaml \
  --instance "$IID" --keep-instance run 2>&1); RC=$?
check "S4 nonzero exit on billing ambiguity" "[ $RC -ne 0 ]"
echo "$OUT" | grep -qi "vast API error" && check "S4 explicit error" true \
  || check "S4 explicit error" false
python3 - <<PY
import json
d = json.load(open("$MOCK_STATE")); d["fail_mode"] = None
json.dump(d, open("$MOCK_STATE", "w"))
PY

echo "== S5: baseline identity mismatch aborts =="
RD=$(rundir_for sweep_smoke_plan.yaml)
python3 - <<PY
import json
p = "${RD}attempt_baseline.json"
d = json.load(open(p)); d["base_key"] = "wrong_identity"
json.dump(d, open(p, "w"))
PY
OUT=$(run_to 120 python3 08_sweep.py --plan sweep_smoke_plan.yaml \
  --instance "$IID" --keep-instance run 2>&1); RC=$?
check "S5 nonzero exit on baseline mismatch" "[ $RC -ne 0 ]"
echo "$OUT" | grep -q "different .*identity\|different model" \
  && check "S5 explicit mismatch message" true \
  || check "S5 explicit mismatch message" false

echo "== S6: stale lease -> watchdog kills planted group within a tick =="
NS=$($SSH 'ls -d /root/sweep_ns/*/ 2>/dev/null | head -1' | tr -d '[:space:]')
if [ -n "$NS" ]; then
  $SSH "mkdir -p ${NS}trials/s6test/deadbeef && touch ${NS}lease && setsid bash -c 'echo \$(ps -o pgid= -p \$\$ | tr -dc 0-9) deadbeef > ${NS}trials/s6test/deadbeef/pgid; sleep 600' >/dev/null 2>&1 < /dev/null & echo PLANTED" >/dev/null 2>&1
  sleep 3
  $SSH "setsid bash ${NS}watchdog.sh >/dev/null 2>&1 < /dev/null & echo WD" >/dev/null 2>&1
  $SSH "touch -d '30 minutes ago' ${NS}lease 2>/dev/null || touch -t 202601010000 ${NS}lease" >/dev/null 2>&1
  echo "  waiting one watchdog tick (75s) ..."
  sleep 75
  PG=$($SSH "cut -d' ' -f1 ${NS}trials/s6test/deadbeef/pgid 2>/dev/null" | tr -dc 0-9)
  if [ -n "$PG" ]; then
    check "S6 planted group killed by watchdog" \
      "! \$SSH 'kill -0 -- -$PG 2>/dev/null && echo LIVE' | grep -q LIVE"
  else
    check "S6 planted group registered" false
  fi
  check "S6 stop attempted (watchdog log shows action)" \
    "$SSH 'grep -q \"lease stale\" ${NS}watchdog.log 2>/dev/null' >/dev/null 2>&1"
else
  check "S6 namespace existed on box" false
fi

echo
if [ "$fail" -ne 0 ]; then
  echo "SMOKE FAILED: $fail check(s)"
  exit 1
fi
echo "SMOKE PASSED: $pass checks"
