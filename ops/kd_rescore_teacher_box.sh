#!/bin/bash
# kd_rescore_teacher_box.sh — teacher-forced top-20 logprob rescore for the
# expansion run's verified trajectories. Run ON the teacher box AFTER
# generation halts (frees the GPU from serving, reuses the same box+model).
#
# Usage:
#   export ENV_VM_SSH_PORT=<port> ENV_VM_IP=<ip>
#   bash kd_rescore_teacher_box.sh
set -euo pipefail
: "${ENV_VM_SSH_PORT:?}" "${ENV_VM_IP:?}"
exec > /root/kd_rescore.out 2>&1
echo "=== KD rescore start $(date) ==="

pkill -f "serve_watchdog" 2>/dev/null || true
S=$(pgrep -f "vllm serve" || true); [ -n "$S" ] && kill $S
sleep 20
P=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); [ -n "$P" ] && kill -9 $P || true
sleep 10
echo "gpu free: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

VSSH="ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes -p $ENV_VM_SSH_PORT root@$ENV_VM_IP"
mkdir -p /workspace/traj /workspace/shards
$VSSH '/root/venv/bin/python -c "
import json, glob
for f in glob.glob(\"/root/exp/runs/expansion/bugfix/*/result.json\"):
    d = json.load(open(f))
    if d[\"task_verified\"]: print(d[\"instance_id\"])"' > /root/verified_ids.txt
echo "verified tasks: $(wc -l < /root/verified_ids.txt)"
while read -r IID; do
  mkdir -p "/workspace/traj/$IID"
  scp -o BatchMode=yes -P "$ENV_VM_SSH_PORT" \
      "root@$ENV_VM_IP:/root/exp/runs/expansion/bugfix/$IID/*" \
      "/workspace/traj/$IID/" >/dev/null 2>&1
done < /root/verified_ids.txt
scp -o BatchMode=yes -P "$ENV_VM_SSH_PORT" \
    "root@$ENV_VM_IP:/root/exp/09c_rescore_logprobs.py" \
    "root@$ENV_VM_IP:/root/exp/.env.hf" /workspace/ >/dev/null

source /venv/main/bin/activate
cd /workspace
RESCORE_GPU_UTIL=0.85 python3 09c_rescore_logprobs.py \
    --traj-dir traj --out-dir shards --only-verified 2>&1 | tail -40
echo "shards: $(ls /workspace/shards/*.npz 2>/dev/null | wc -l)"
source /workspace/.env.hf
pip install -q "huggingface_hub[cli]" 2>/dev/null
hf upload "$PILOT_STORE" /workspace/shards expansion/kd_shards \
    --type dataset --commit-message "expansion KD shards: top-20 logprobs" \
    --format quiet 2>&1 | tail -1
echo "=== KD rescore done $(date) ==="
