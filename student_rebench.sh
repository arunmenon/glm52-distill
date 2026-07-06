#!/usr/bin/env bash
# student_rebench.sh — staged, budget-gated re-run to produce the FIRST
# trustworthy v0 delta (journal Discovery #6 fixes). Each stage is cheap and
# gates the next, so a wrong assumption costs ~$1, not ~$30.
#
# Stage 0 (CPU, ~free): re-pack the 1,140-row corpus with the thinking-format
#   packer. The loud </think>-strip guard fires here or passes — no GPU spent.
# Stage 1 (~$5): benchmark BASE Qwen3-8B with the thinking-aware config +
#   IFEval re-score. If base does NOT recover toward its published range
#   (~GSM8K flexible ~0.8, IFEval strict ~0.4+), the eval fix is still
#   incomplete — STOP, do not retrain (retraining would only make a 2nd artifact).
# Stage 2 (~$20): retrain v0 on the re-packed data (more steps than the
#   68-step pilot), benchmark, push. Only runs if GATE=pass is set.
#
#   Stage 0+1:  bash student_rebench.sh
#   Then, if base recovered:  GATE=pass bash student_rebench.sh
set -euo pipefail
cd "$(dirname "$0")"
source common.sh
source /workspace/.pilot_env
source /workspace/venv/bin/activate 2>/dev/null || true
export WANDB_MODE=disabled
STATE_DIR=pilot_student/.state
QWEN=/workspace/models/qwen3-8b
TEACHER_TOK=/workspace/models/glm-5.2-tok
AIR_TOK=/workspace/models/glm-4.5-air-tok
BENCH_EPOCHS="${BENCH_EPOCHS:-3}"

if ! done_p model; then
  hf_pull Qwen/Qwen3-8B "$QWEN"
  hf_pull zai-org/GLM-5.2-FP8 "$TEACHER_TOK" "tokenizer*"
  hf_pull zai-org/GLM-4.5-Air "$AIR_TOK" "tokenizer*"
  mark model
fi
if ! done_p pulled; then
  store_pull "corpus/*"; store_pull "data/*"; store_pull "gate_result.json"
  mark pulled
fi

# ---- Stage 0: re-pack with the thinking-format packer (CPU) ----
rm -rf pilot_student/packed_v2
python 03_pack_dataset.py --corpus corpus --heldout data/prompts_heldout.jsonl \
  --teacher "$TEACHER_TOK" --student-a "$AIR_TOK" --student-b "$QWEN" \
  --out pilot_student/packed_v2 --gate gate_result.json --max-len 6144 \
  --exclude-report pilot_student/contamination_report.json
# eyeball one packed sample's think placement
python - <<'PY'
from datasets import load_from_disk
d = load_from_disk("pilot_student/packed_v2/student_b")["train"]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/workspace/models/qwen3-8b", trust_remote_code=True)
ex = d[0]
txt = tok.decode([i for i, l in zip(ex["input_ids"], ex["labels"]) if l != -100])
print("=== decoded assistant span (first 400 chars) ===")
print(txt[:400])
assert "<think>" in txt and "</think>" in txt, "think tags missing from packed sample!"
print("=== think tags present: OK ===")
PY
echo ">>> Stage 0 OK: re-pack + guard passed."

# ---- Stage 1: BASE anchor ----
if ! done_p base_bench; then
  TP=4 TASKS="ifeval,gsm8k_cot" OUTDIR=pilot_student/evals_v2/base \
    bash 07_eval_benchmarks.sh "$QWEN"
  store_push pilot_student/evals_v2/base evals_v2/base
  mark base_bench
  echo ">>> Stage 1 done. INSPECT base scores above."
  echo ">>> If base recovered (GSM8K flexible ~0.8, IFEval strict ~0.4+):"
  echo ">>>   GATE=pass bash student_rebench.sh"
  echo ">>> If not, the eval fix is incomplete — do NOT retrain."
fi

# ---- Stage 2: retrain + benchmark v0 (gated) ----
if [ "${GATE:-}" != "pass" ]; then
  echo "GATE not set to pass — stopping after the base anchor (by design)."
  exit 0
fi
if ! done_p v0_train; then
  accelerate launch --config_file configs/ds_zero3_4gpu.yaml 04_train_kd.py \
    --model "$QWEN" --data pilot_student/packed_v2/student_b \
    --out pilot_student/ckpts/qwen3_v2 --alpha 0.0 --grad-accum 8 \
    --epochs "$BENCH_EPOCHS"
  mark v0_train
fi
if ! done_p v0_bench; then
  TP=4 TASKS="ifeval,gsm8k_cot" OUTDIR=pilot_student/evals_v2/v0 \
    bash 07_eval_benchmarks.sh pilot_student/ckpts/qwen3_v2/final
  store_push pilot_student/evals_v2 evals_v2
  store_push pilot_student/ckpts/qwen3_v2/final qwen3_glm_distill_v0_1
  mark v0_bench
fi
echo "REBENCH COMPLETE — base vs v0 in pilot_student/evals_v2/"
