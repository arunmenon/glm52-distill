#!/usr/bin/env bash
# run_experiment.sh — ON-POD idempotent recipe for ONE trial (or the anchor).
# Reuses 03/04/merge_lora/07/rescore_ifeval unchanged. mark/done_p make every
# step resumable; writes trials/<id>/result.json for the conductor to merge.
#
#   bash run_experiment.sh <trial_config.json>
# config.json: {experiment_id, alpha, lr, lora_rank, epochs, rung, mode}
#   mode = "anchor" (eval base only) | "train" (pack->train->merge->eval)
set -euo pipefail
cd "$(dirname "$0")"
CFG="${1:?trial config json}"
source common.sh 2>/dev/null || true
source /workspace/.pilot_env 2>/dev/null || true
source /workspace/venv/bin/activate 2>/dev/null || true
export WANDB_MODE=disabled

jqf() { python3 -c "import json,sys; print(json.load(open('$CFG')).get('$1',''))"; }
EID=$(jqf experiment_id); MODE=$(jqf mode); ALPHA=$(jqf alpha); LR=$(jqf lr)
RANK=$(jqf lora_rank); EPOCHS=$(jqf epochs); RUNG=$(jqf rung)
PROXY=$(jqf proxy_rows)
STUDENT=/workspace/models/qwen3-8b
TDIR="trials/$EID"; STATE_DIR="$TDIR/.state"
mkdir -p "$TDIR"; BENCH="$TDIR/bench"
mark(){ mkdir -p "$STATE_DIR"; touch "$STATE_DIR/$1"; }; done_p(){ [ -f "$STATE_DIR/$1" ]; }

TASKS="${TASKS:-gsm8k_cot,ifeval}"

if [ "$MODE" = "anchor" ]; then
  # base-model anchor (Discovery #6/#7 gate): eval the UNTRAINED student
  if ! done_p bench; then
    TP=4 TASKS="$TASKS" OUTDIR="$BENCH" bash 07_eval_benchmarks.sh "$STUDENT"
    mark bench
  fi
  python objective.py --bench-dir "$BENCH" --out "$TDIR/result.json"
  echo "ANCHOR-DONE $EID"
  exit 0
fi

# ---- train mode ----
# pack ONCE per corpus×student (cached across trials)
PACK=/workspace/glm52-distill/packed_shared/student_b
if [ ! -d "$PACK" ]; then
  python 03_pack_dataset.py --corpus corpus --heldout "$(jqf heldout || echo data/prompts_heldout.jsonl)" \
    --teacher /workspace/models/glm-5.2-tok --student-a "$STUDENT" --student-b "$STUDENT" \
    --out /workspace/glm52-distill/packed_shared --gate gate_result.json --max-len 6144 \
    --exclude-report contamination_report.json 2>/dev/null || \
  python 03_pack_dataset.py --corpus corpus --heldout data/prompts_heldout.jsonl \
    --teacher "$STUDENT" --student-a "$STUDENT" --student-b "$STUDENT" \
    --out /workspace/glm52-distill/packed_shared --gate gate_result.json --max-len 6144
fi

LIMIT=0; [ "$RUNG" = "0" ] && LIMIT="${PROXY:-800}"
if ! done_p trained; then
  accelerate launch --config_file configs/ds_zero3_4gpu.yaml 04_train_kd.py \
    --model "$STUDENT" --data "$PACK" --out "$TDIR/ckpt" \
    --alpha "$ALPHA" --lr "$LR" --epochs "$EPOCHS" --grad-accum 8 \
    ${LIMIT:+--max-train-samples $LIMIT} \
    $([ "$ALPHA" != "0.0" ] && echo "--lora --lora-rank $RANK")
  mark trained
fi

# non-LoRA (alpha 0) saves final directly; LoRA needs merge
FINAL="$TDIR/ckpt/final"
if [ ! -d "$FINAL" ] && [ -d "$TDIR/ckpt/adapter" ]; then
  python merge_lora.py --base "$STUDENT" --adapter "$TDIR/ckpt/adapter" --out "$FINAL"
fi

if ! done_p bench; then
  TP=4 TASKS="$TASKS" OUTDIR="$BENCH" bash 07_eval_benchmarks.sh "$FINAL"
  mark bench
fi
ANCHOR_ARG=""
[ -f anchors/qwen.json ] && ANCHOR_ARG="--anchor anchors/qwen.json"
python objective.py --bench-dir "$BENCH" $ANCHOR_ARG --out "$TDIR/result.json"
echo "TRIAL-DONE $EID  $(cat "$TDIR/result.json" | python3 -c 'import json,sys;print("score",json.load(sys.stdin)["score"])')"
