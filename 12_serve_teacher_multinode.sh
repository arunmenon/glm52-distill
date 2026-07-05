#!/usr/bin/env bash
# 12_serve_teacher_multinode.sh — GLM-5.2 FP8 across VM-A + VM-B (16 GPUs, 2 nodes).
# Run on VM-A AFTER 11_ray_cluster.sh has both nodes up (ray status shows 16 GPUs).
#
# Layout: TP=8 inside each VM (PCIe-local), PP=2 across VMs (activations only
# cross the network — the layout that tolerates Ethernet).
#
# Both VMs must have the model weights at the SAME local path ($MODEL) —
# download on each VM via 00_setup.sh; do NOT try to serve from GCS directly.
set -euo pipefail
cd "$(dirname "$0")"
source "$HOME/venvs/serve/bin/activate"
source common.sh

MODEL="${MODEL:-/models/glm-5.2-fp8}"
LOG="${LOG:-teacher_serve.log}"

ray status | grep -q "16.0/16.0 GPU\|GPU: 16" || \
  echo "WARNING: ray status does not show 16 GPUs — check 11_ray_cluster.sh on both VMs"

launch_vllm "$MODEL" glm-5.2 "$LOG" \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 128

# weight load on 2 nodes can take 30-60 min
wait_ready_and_smoke "http://localhost:8000/v1" glm-5.2 240 30 || {
  echo "check $LOG on VM-A and 'ray status'" >&2; exit 1; }
