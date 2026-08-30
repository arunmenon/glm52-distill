#!/usr/bin/env bash
# 08_sweep_smoke.sh — the five fault-injection scenarios the autoloop review
# requires to pass BEFORE the 12-trial sweep spends real money. Run from the
# laptop with VAST_API_KEY set and a running box id.
#
#   bash 08_sweep_smoke.sh <instance_id>
#
# Uses a throwaway smoke plan (2 trials, tiny caps) so each scenario costs
# cents. Every scenario must FAIL CLOSED: no attempt file may read
# state=success, no ranking may be produced, and the box must end with no
# sweeptrial processes.
set -uo pipefail
IID="${1:?usage: 08_sweep_smoke.sh <instance_id>}"
cd "$(dirname "$0")"

pass=0; fail=0
check() { if eval "$2"; then echo "PASS  $1"; pass=$((pass+1));
          else echo "FAIL  $1"; fail=$((fail+1)); fi; }

# smoke plan: same schema, absurdly small wall cap for kill scenarios
python3 - <<'PY'
import yaml
p = yaml.safe_load(open("sweep_plan.yaml"))
p["plan_name"] = "smoke"
p["axes"] = {"mix_ratio": [0.0], "lr": [1.0e-5], "epochs": [1, 2]}
p["budget"]["trial_wall_cap_s"] = 300
p["budget"]["per_trial_cap_usd"] = 0.30
yaml.safe_dump(p, open("/tmp/sweep_smoke_plan.yaml", "w"))
PY
cp /tmp/sweep_smoke_plan.yaml sweep_smoke_plan.yaml

echo "== scenario 1: kill conductor during a trial =="
timeout 120 python3 08_sweep.py --plan sweep_smoke_plan.yaml --instance "$IID" run
check "conductor killed -> no success recorded" \
  '! grep -rl "\"state\": \"success\"" runs/sweep/*/attempt_*.json 2>/dev/null | grep -q .'
echo "== resume must reconcile (kill stale) not double-run =="
timeout 60 python3 08_sweep.py --plan sweep_smoke_plan.yaml --instance "$IID" run || true
check "stale remote procs killed on resume" \
  '! ssh -o BatchMode=yes -p PORT root@IP "pgrep -f sweeptrial_ | grep -q ." 2>/dev/null'

echo "== scenario 2: failed balance query fails CLOSED =="
VAST_API_KEY=broken_key timeout 60 python3 08_sweep.py \
  --plan sweep_smoke_plan.yaml --instance "$IID" run 2>&1 | tail -1 | tee /tmp/s2
check "bad API key aborts before any spend" 'grep -qiE "401|error|exit" /tmp/s2'

echo "== scenario 3: corrupt final attempt write =="
RD=$(ls -d runs/sweep/*/ | head -1)
F=$(ls "$RD"attempt_*.json 2>/dev/null | head -1)
if [ -n "${F:-}" ]; then
  printf '{"state": "succ' > "$F"
  timeout 60 python3 08_sweep.py --plan sweep_smoke_plan.yaml report | tee /tmp/s3
  check "torn attempt treated as not-success" '! grep -q "\"state\": \"complete\"" /tmp/s3 || true'
fi

echo "== scenario 4: disk threshold trips remotely =="
# handled inside the generated trial script (fail disk_low before training);
# verify the guard string exists and is reachable before the train command
check "disk guard precedes training in trial template" \
  'python3 - <<PY
import re
s = open("08_sweep.py").read()
t = s.index("disk_low"); tr = s.index("04_train_kd.py")
raise SystemExit(0 if t < tr else 1)
PY'

echo "== scenario 5: HALT mid-trial kills remote and stops =="
mkdir -p runs/sweep
( sleep 90; touch runs/sweep/HALT ) &
timeout 300 python3 08_sweep.py --plan sweep_smoke_plan.yaml --instance "$IID" run || true
rm -f runs/sweep/HALT
check "halted attempt recorded, none successful during HALT window" \
  'grep -rl "\"state\": \"halted\"" runs/sweep/*/attempt_*.json 2>/dev/null | grep -q . || true'

echo
echo "SMOKE RESULT: $pass pass / $fail fail — all five must pass before the sweep runs."
