#!/bin/bash
# env_vm_launch.sh — self-healing generation stack for the expansion run.
# Runs ON the env VM (not piped over ssh: pgrep patterns must not see the
# ssh command line — that self-match killed workers 4x in the WALK era).
#
# Prereqs on the VM (see ops/env_vm_provision.md):
#   /root/exp/            repo scripts + runs/expansion/bugfix/<tasks file>
#   /root/venv/           python env with minisweagent, pandas, hf
#   /root/exp/.env.hf     exports PILOT_STORE=ledzepu2/glm52-pilot-artifacts
#                         and HF_TOKEN=<fine-grained WRITE token>
#   docker working (DOCKER_API_VERSION=1.43 in /etc/environment on Vast KVM)
#
# Usage:
#   export TEACHER_BASE_URL=... TEACHER_API_KEY=...
#   bash env_vm_launch.sh <tasks_file.json> [n_workers]
#
# Self-healing layers (all proven in the WALK regen run):
#   1. per-task sealed ledgers  -> restarts never redo finished tasks
#   2. nohup+disown detached    -> survives ssh loss / harness reaping
#   3. watchdog                 -> restarts dead/hung workers (75 min log
#                                  age), but HOLDS when the teacher endpoint
#                                  is down instead of churning (the churn
#                                  once poisoned 23 ledgers)
#   4. sync-on-seal             -> every new ledger pushed to HF within
#                                  ~4 min; a reclaimed host loses only
#                                  in-flight rollouts (the unsynced WALK VM
#                                  lost 44 ledgers / $66.73)
set -euo pipefail
cd /root/exp
TASKS_FILE="${1:?usage: env_vm_launch.sh <tasks_file.json> [n_workers]}"
NW="${2:-3}"
: "${TEACHER_BASE_URL:?}" "${TEACHER_API_KEY:?}"
source /root/exp/.env.hf
RUN_DIR=runs/expansion/bugfix
HF_PREFIX=expansion/runs
TOTAL=$(python3 -c "import json;print(len(json.load(open('$TASKS_FILE'))))")

# hard gate: never spend without a CLEAN decontam verdict on this exact file
/root/venv/bin/python trajectory_decontam.py "$TASKS_FILE" \
    --output /root/exp/gate_expansion.json >/dev/null
python3 -c "import json,sys; sys.exit(0 if json.load(open('/root/exp/gate_expansion.json'))['verdict']=='CLEAN' else 1)" \
    || { echo GATE_NOT_CLEAN; exit 1; }

cat > exp_worker.py <<'PYEOF'
# argv: <part_number> <inert_marker>  (marker is matched by pgrep; the
# worker never inspects it — see ssh-pgrep-self-match history)
import importlib.util
import sys
from pathlib import Path
spec = importlib.util.spec_from_file_location(
    "xb", "/root/exp/09e_expansion_bugfix.py")
xb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xb)
xb.G.TASKS_FILE = Path(f"/root/exp/exp_part{sys.argv[1]}.json")
xb.G.run()
PYEOF

# split remaining (unsealed) tasks round-robin per tier across workers
/root/venv/bin/python - "$TASKS_FILE" "$NW" <<'PYEOF'
import json
import pathlib
import sys
tasks = json.load(open(sys.argv[1]))
n_workers = int(sys.argv[2])
sealed = {p.parent.name
          for p in pathlib.Path("runs/expansion/bugfix").glob("*/result.json")}
todo = [t for t in tasks if t["instance_id"] not in sealed]
parts = [[] for _ in range(n_workers)]
for tier in ("easy", "medium", "hard"):
    for i, t in enumerate([t for t in todo if t["tier"] == tier]):
        parts[i % n_workers].append(t)
for i, p in enumerate(parts):
    json.dump(p, open(f"exp_part{i+1}.json", "w"), indent=1)
    print(f"part{i+1}: {len(p)} tasks")
PYEOF

cat > sync_on_seal.sh <<SEOF
#!/bin/bash
source /root/exp/.env.hf
LAST=-1
while true; do
  N=\$(ls $RUN_DIR/*/result.json 2>/dev/null | wc -l)
  if [ "\$N" != "\$LAST" ]; then
    /root/venv/bin/hf upload "\$PILOT_STORE" $RUN_DIR $HF_PREFIX \
        --type dataset --commit-message "sync-on-seal: \$N ledgers" \
        --format quiet >/dev/null 2>&1 \
      && LAST=\$N && echo "[sync] \$N ledgers \$(date '+%H:%M')" >> sync.log
  fi
  [ "\$N" -ge $TOTAL ] && exit 0
  sleep 240
done
SEOF
chmod +x sync_on_seal.sh

cat > watchdog.sh <<WEOF
#!/bin/bash
teacher_ok() {
  curl -s --max-time 15 -o /dev/null -w "%{http_code}" "$TEACHER_BASE_URL" \
    -H "Authorization: Bearer $TEACHER_API_KEY" \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen/qwen3.8-27b","messages":[{"role":"user","content":"x"}],"max_tokens":1}' \
    | grep -qE "200"
}
while true; do
  sleep 150
  N=\$(ls $RUN_DIR/*/result.json 2>/dev/null | wc -l)
  [ "\$N" -ge $TOTAL ] && exit 0
  for W in \$(seq 1 $NW); do
    REMAIN=\$(/root/venv/bin/python - \$W <<'RPY'
import json, pathlib, sys
part = json.load(open(f"/root/exp/exp_part{sys.argv[1]}.json"))
sealed = {p.parent.name for p in
          pathlib.Path("/root/exp/runs/expansion/bugfix").glob("*/result.json")}
print(sum(1 for t in part if t["instance_id"] not in sealed))
RPY
)
    [ "\$REMAIN" -eq 0 ] && continue
    ALIVE=\$(pgrep -c -f "xpart\${W}_marker" || true)
    AGE=\$(( \$(date +%s) - \$(stat -c %Y exp_w\${W}.log 2>/dev/null || echo 0) ))
    if [ "\$ALIVE" -eq 0 ] || [ "\$AGE" -gt 4500 ]; then
      if ! teacher_ok; then
        touch TEACHER_DOWN
        echo "[watchdog] \$(date) teacher down - holding" >> watchdog.log
        continue
      fi
      rm -f TEACHER_DOWN
      echo "[watchdog] \$(date) worker \$W alive=\$ALIVE age=\$AGE restart" >> watchdog.log
      pkill -9 -f "xpart\${W}_marker" 2>/dev/null
      docker ps -q --filter name=minisweagent | xargs -r docker rm -f
      sleep 5
      nohup /root/venv/bin/python -u exp_worker.py \$W xpart\${W}_marker \
          >> exp_w\${W}.log 2>&1 &
    fi
  done
done
WEOF
chmod +x watchdog.sh

for W in $(seq 1 "$NW"); do
  nohup /root/venv/bin/python -u exp_worker.py "$W" "xpart${W}_marker" \
      >> "exp_w${W}.log" 2>&1 &
done
nohup bash sync_on_seal.sh >/dev/null 2>&1 &
nohup bash watchdog.sh >/dev/null 2>&1 &
disown -a
sleep 10
echo "workers: $(pgrep -c -f xpart || true) | sync: $(pgrep -c -f sync_on_seal || true) | watchdog: $(pgrep -c -f 'watchdog[.]sh' || true)"
echo LAUNCH_DONE
