#!/usr/bin/env bash
# 00_setup.sh — environments, model downloads, sanity checks. Run once on Day 0.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/models}"
TEACHER_REPO="${TEACHER_REPO:-zai-org/GLM-5.2-FP8}"     # verify exact repo id on HF before running
STUDENT_A_REPO="${STUDENT_A_REPO:-zai-org/GLM-4.5-Air}"
STUDENT_B_REPO="${STUDENT_B_REPO:-Qwen/Qwen3-8B}"

echo "== [1/5] Disk check (need >= 2.5 TB free on ${MODELS_DIR} volume) =="
df -h "$(dirname "$MODELS_DIR")" || true
FREE_GB=$(df --output=avail -BG "$(dirname "$MODELS_DIR")" | tail -1 | tr -dc '0-9')
if [ "${FREE_GB}" -lt 2500 ]; then
  echo "WARNING: only ${FREE_GB} GB free; pipeline needs ~2.5 TB." >&2
fi

echo "== [2/5] GPU check =="
nvidia-smi --query-gpu=index,name,memory.total --format=csv
N_GPU=$(nvidia-smi -L | wc -l)
EXPECTED_GPUS="${EXPECTED_GPUS:-$N_GPU}"   # set EXPECTED_GPUS=8 (VM-A/B) or 2 (VM-C) to enforce
[ "$N_GPU" -eq "$EXPECTED_GPUS" ] || { echo "ERROR: expected $EXPECTED_GPUS GPUs, found $N_GPU" >&2; exit 1; }

echo "== [3/5] Serving env =="
python3 -m venv "$HOME/venvs/serve"
"$HOME/venvs/serve/bin/pip" install -q --upgrade pip
"$HOME/venvs/serve/bin/pip" install -q "vllm>=0.10" "huggingface_hub[hf_transfer]" aiohttp pyarrow
"$HOME/venvs/serve/bin/pip" freeze > env_serve.lock.txt

echo "== [4/5] Training env =="
python3 -m venv "$HOME/venvs/train"
"$HOME/venvs/train/bin/pip" install -q --upgrade pip
"$HOME/venvs/train/bin/pip" install -q "torch>=2.7" "transformers>=4.55" \
  "trl>=0.19" peft accelerate deepspeed datasets wandb pyarrow \
  datasketch lm-eval "vllm>=0.10"
"$HOME/venvs/train/bin/pip" install -q flash-attn --no-build-isolation || \
  echo "flash-attn build failed; 04_train_kd.py will fall back to sdpa"
"$HOME/venvs/train/bin/pip" freeze > env_train.lock.txt

echo "== [5/5] Model downloads =="
export HF_HUB_ENABLE_HF_TRANSFER=1
HF="$HOME/venvs/serve/bin/hf"
if [ "${SKIP_TEACHER:-0}" = "1" ]; then
  # VM-C mode: tokenizer/config only (needed for the gate check), not 750GB of weights
  $HF download "$TEACHER_REPO" --local-dir "$MODELS_DIR/glm-5.2-fp8" \
    --include "*.json" --include "*.txt" --include "tokenizer*" --include "*.py"
else
  $HF download "$TEACHER_REPO" --local-dir "$MODELS_DIR/glm-5.2-fp8"
fi
$HF download "$STUDENT_A_REPO" --local-dir "$MODELS_DIR/glm-4.5-air" ${SKIP_TEACHER:+--include "*.json" --include "*.txt" --include "tokenizer*" --include "*.py"}
$HF download "$STUDENT_B_REPO" --local-dir "$MODELS_DIR/qwen3-8b"

echo "== Done. Next: python 00_check_tokenizer.py --teacher $MODELS_DIR/glm-5.2-fp8 --student-a $MODELS_DIR/glm-4.5-air =="
