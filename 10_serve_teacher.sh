#!/usr/bin/env bash
# 10_serve_teacher.sh — GLM-5.2 FP8 across all 16 GPUs (single node).
#
# NOTES
#  - --return-tokens-as-token-ids is REQUIRED by 02_generate.py (exact id capture);
#    launch_vllm in common.sh sets it (plus seed + prefix caching) for every server.
#  - No --reasoning-parser: thinking traces stay in the token stream on purpose,
#    so they are part of the distillation signal.
#  - If your vLLM build has stable expert parallelism for GLM-5.2, benchmark
#    "-tp 16 --enable-expert-parallel" against "-tp 8 -pp 2" for 10 minutes and
#    keep the faster.
set -euo pipefail
cd "$(dirname "$0")"
source "$HOME/venvs/serve/bin/activate"
source common.sh

MODEL="${MODEL:-/models/glm-5.2-fp8}"
LOG="${LOG:-teacher_serve.log}"

launch_vllm "$MODEL" glm-5.2 "$LOG" \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 128

# up to ~2h for 750GB weight load on slow disk
wait_ready_and_smoke "http://localhost:8000/v1" glm-5.2 240 30
