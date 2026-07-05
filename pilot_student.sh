#!/usr/bin/env bash
# pilot_student.sh — LEG 2: the Student-A path on real weights, consuming the
# corpus the teacher leg produced. Defaults to 4 GPUs; sized for 4x RTX PRO 6000
# (384 GB, same GPU as the GCP fleet) or 4x H100.
#
#   1. pull leg-1 artifacts (corpus/data/gate) from $PILOT_STORE
#   2. pack with remap at --max-len 6144 (fits 80 GB GPUs; also exercises the
#      README's OOM-rollback path)
#   3. GLM-4.5-Air LoRA KD smoke: ZeRO-3 over 4 GPUs, ~1 epoch
#   4. merge adapter (needs ~450 GB CPU RAM — checked before starting)
#   5. reload merged Air in vLLM (tp=4) and generate held-out answers
#   6. push adapter + evals (NOT the 212 GB merged model — it is reproducible)
#
# EXTENDED=1 (~$300 budget) additionally:
#   7. train Qwen3-8B (full FT SFT) on the real corpus -> "qwen3-glm-distill-v0"
#   8. GSM8K + IFEval on base vs distilled (no teacher needed) -> benchmark delta
#
# Run AFTER pilot_teacher.sh. Same PILOT_STORE/HF_TOKEN as leg 1 (or one shared
# network volume with leg-1 outputs at pilot_teacher/, then PILOT_STORE stays unset
# and the pull phase is a no-op — paths below resolve either way).
set -euo pipefail
cd "$(dirname "$0")"
source common.sh

STATE_DIR=pilot_student/.state
AIR=/workspace/models/glm-4.5-air
TEACHER_TOK=/workspace/models/glm-5.2-tok
QWEN_TOK=/workspace/models/qwen3-8b-tok
export WANDB_MODE=disabled

mkdir -p pilot_student/evals
ensure_pod_env

if ! done_p model; then
  hf_pull zai-org/GLM-4.5-Air "$AIR"
  hf_pull zai-org/GLM-5.2-FP8 "$TEACHER_TOK" "tokenizer*"
  hf_pull Qwen/Qwen3-8B "$QWEN_TOK" "tokenizer*" "chat_template*" "*config.json" "vocab.json" "merges.txt"
  mark model
fi

if ! done_p pulled; then
  store_pull "corpus/*"
  store_pull "data/*"
  store_pull "evals/*"
  store_pull "gate_result.json"
  # shared-volume fallback: leg 1 left everything under pilot_teacher/
  [ -d corpus ] || { ln -s pilot_teacher/corpus corpus; ln -s pilot_teacher/data data;
                     ln -s pilot_teacher/gate_result.json gate_result.json; }
  mark pulled
fi

if ! done_p audit; then
  # Retroactive benchmark-contamination scan (pilot prompt set predates the
  # decontamination step); contaminated rows are excluded at pack time.
  python benchmark_audit.py --corpus corpus --out pilot_student/contamination_report.json
  mark audit
fi

if ! done_p packed; then
  python 03_pack_dataset.py --corpus corpus --heldout data/prompts_heldout.jsonl \
    --teacher "$TEACHER_TOK" --student-a "$AIR" --student-b "$QWEN_TOK" \
    --out pilot_student/packed --gate gate_result.json --max-len 6144 \
    --exclude-report pilot_student/contamination_report.json
  store_push pilot_student/contamination_report.json contamination_report.json
  mark packed
fi

if ! done_p train; then
  FREE_RAM_GB=$(free -g | awk '/^Mem:/{print $7}')
  [ "$FREE_RAM_GB" -ge 450 ] || echo "WARNING: ${FREE_RAM_GB} GB free RAM; the merge step needs ~450 GB"
  accelerate launch --config_file configs/ds_zero3_4gpu.yaml 04_train_kd.py \
    --model "$AIR" --data pilot_student/packed/student_a --out pilot_student/ckpts/air_kd \
    --alpha 0.5 --temperature 2.0 --lora --lr 1e-4 --epochs 1 --grad-accum 8
  mark train
fi

if ! done_p merge; then
  python merge_lora.py --base "$AIR" --adapter pilot_student/ckpts/air_kd/adapter \
    --out pilot_student/ckpts/air_kd/final
  mark merge
fi

if ! done_p student_gens; then
  python 05_eval_gen.py --model pilot_student/ckpts/air_kd/final \
    --heldout data/prompts_heldout.jsonl --out pilot_student/evals/air_kd.jsonl --tp 4
  mark student_gens
fi

# ---- EXTENDED=1: mini-distill Qwen3-8B on the real corpus + benchmark delta ----
if [ "${EXTENDED:-0}" = "1" ]; then
  QWEN=/workspace/models/qwen3-8b
  if ! done_p qwen_train; then
    hf_pull Qwen/Qwen3-8B "$QWEN"
    accelerate launch --config_file configs/ds_zero3_4gpu.yaml 04_train_kd.py \
      --model "$QWEN" --data pilot_student/packed/student_b \
      --out pilot_student/ckpts/qwen3_sft --alpha 0.0 --grad-accum 8
    mark qwen_train
  fi
  if ! done_p bench; then
    uv pip install -q -c /tmp/pip_constraints.txt lm-eval
    TP=4 TASKS="ifeval,gsm8k_cot" OUTDIR=pilot_student/evals/benchmarks \
      bash 07_eval_benchmarks.sh "$QWEN" pilot_student/ckpts/qwen3_sft/final
    mark bench
  fi
fi

if ! done_p pushed; then
  store_push pilot_student/ckpts/air_kd/adapter adapter_air_kd
  store_push pilot_student/evals evals_h100
  [ -d pilot_student/ckpts/qwen3_sft/final ] && store_push pilot_student/ckpts/qwen3_sft/final qwen3_glm_distill_v0
  mark pushed
fi

echo "================ PILOT COMPLETE — terminate this pod now ================"
echo "-- held-out generations from the merged KD checkpoint:"
head -c 600 pilot_student/evals/air_kd.jsonl; echo
