#!/usr/bin/env bash
# run_all.sh — full pipeline with phase gates. Edit paths at the top, then run.
# Safe to re-run: generation is shard-resumable; completed phases are marked in .state/.
set -euo pipefail

MODELS=/models
BEST_OF="${BEST_OF:-1}"   # set BEST_OF=4 for rejection sampling (~4x gen cost; use ~100k prompts)
PILOT="${PILOT:-0}"       # PILOT=1 -> 500-prompt end-to-end dry run (~2-3h) BEFORE the real thing
N_TOTAL=$([ "$PILOT" = "1" ] && echo 500 || echo 150000)
N_HELDOUT=$([ "$PILOT" = "1" ] && echo 50 || echo 1000)
export WANDB_MODE="${WANDB_MODE:-online}"   # set WANDB_MODE=offline if no API key
TEACHER=$MODELS/glm-5.2-fp8
STUDENT_A=$MODELS/glm-4.5-air
STUDENT_B=$MODELS/qwen3-8b
mkdir -p .state evals

mark()   { touch ".state/$1"; }
done_p() { [ -f ".state/$1" ]; }

# Refuse to mix pilot and full-run state: rerunning without PILOT=1 after a
# pilot would otherwise skip phases and silently reuse the 500-prompt corpus.
if [ -f .state/mode ]; then
  if [ "$(cat .state/mode)" != "$PILOT" ]; then
    echo "ERROR: existing .state/ is from a PILOT=$(cat .state/mode) run." >&2
    echo "Clear it and the outputs first: rm -rf .state data corpus packed ckpts evals gate_result.json" >&2
    exit 1
  fi
else
  echo "$PILOT" > .state/mode
fi

# ---------- Phase 0 ----------
if ! done_p setup; then
  bash 00_setup.sh
  mark setup
fi

source "$HOME/venvs/train/bin/activate"
if ! done_p gate; then
  python 00_check_tokenizer.py --teacher "$TEACHER" --student-a "$STUDENT_A"
  mark gate
fi
LOGIT_KD_OK=$(python3 -c 'import json;print(1 if json.load(open("gate_result.json"))["logit_kd_ok"] else 0)')
echo "LOGIT_KD_OK=$LOGIT_KD_OK"

if ! done_p prompts; then
  python 01_build_prompts.py --out data/ --n-total "$N_TOTAL" --n-heldout "$N_HELDOUT"
  mark prompts
fi

# ---------- Phase 1: teacher up, generate ----------
if ! done_p corpus; then
  bash 10_serve_teacher.sh
  source "$HOME/venvs/serve/bin/activate"
  python 02_generate.py --prompts data/prompts.jsonl --out corpus/ \
    --base-url http://localhost:8000/v1 --model glm-5.2 --concurrency 112 \
    --best-of "$BEST_OF"
  pkill -f "vllm serve" || true
  sleep 20
  source "$HOME/venvs/train/bin/activate"
  mark corpus
fi

# ---------- Phase 2: pack + train ----------
if ! done_p packed; then
  python 03_pack_dataset.py --corpus corpus/ --heldout data/prompts_heldout.jsonl \
    --teacher "$TEACHER" --student-a "$STUDENT_A" --student-b "$STUDENT_B" \
    --out packed/ --gate gate_result.json
  mark packed
fi

if ! done_p train_qwen_sft; then
  accelerate launch --config_file configs/ds_zero3.yaml 04_train_kd.py \
    --model "$STUDENT_B" --data packed/student_b --out ckpts/qwen3_sft --alpha 0.0
  mark train_qwen_sft
fi

if ! done_p train_air_sft; then
  accelerate launch --config_file configs/ds_zero3.yaml 04_train_kd.py \
    --model "$STUDENT_A" --data packed/student_a --out ckpts/air_sft \
    --alpha 0.0 --lora --lr 1e-4
  python merge_lora.py --base "$STUDENT_A" --adapter ckpts/air_sft/adapter --out ckpts/air_sft/final
  mark train_air_sft
fi

if [ "$LOGIT_KD_OK" = "1" ] && ! done_p train_air_kd; then
  accelerate launch --config_file configs/ds_zero3.yaml 04_train_kd.py \
    --model "$STUDENT_A" --data packed/student_a --out ckpts/air_kd \
    --alpha 0.5 --temperature 2.0 --lora --lr 1e-4
  python merge_lora.py --base "$STUDENT_A" --adapter ckpts/air_kd/adapter --out ckpts/air_kd/final
  mark train_air_kd
fi

# ---------- Phase 3: eval ----------
if ! done_p student_gens; then
  python 05_eval_gen.py --model ckpts/qwen3_sft/final --heldout data/prompts_heldout.jsonl --out evals/qwen3_sft.jsonl --tp 8
  python 05_eval_gen.py --model ckpts/air_sft/final  --heldout data/prompts_heldout.jsonl --out evals/air_sft.jsonl  --tp 8
  [ "$LOGIT_KD_OK" = "1" ] && python 05_eval_gen.py --model ckpts/air_kd/final --heldout data/prompts_heldout.jsonl --out evals/air_kd.jsonl --tp 8
  mark student_gens
fi

if ! done_p judged; then
  bash 10_serve_teacher.sh
  python 05_eval_gen.py --served --base-url http://localhost:8000/v1 --model glm-5.2 \
    --heldout data/prompts_heldout.jsonl --out evals/teacher_ref.jsonl
  python 06_judge.py --ref evals/teacher_ref.jsonl --cand evals/qwen3_sft.jsonl --out evals/judge_qwen3_sft.json
  python 06_judge.py --ref evals/teacher_ref.jsonl --cand evals/air_sft.jsonl  --out evals/judge_air_sft.json
  [ "$LOGIT_KD_OK" = "1" ] && python 06_judge.py --ref evals/teacher_ref.jsonl --cand evals/air_kd.jsonl --out evals/judge_air_kd.json
  pkill -f "vllm serve" || true
  sleep 20
  mark judged
fi

if ! done_p benchmarks; then
  CKPTS="ckpts/qwen3_sft/final ckpts/air_sft/final"
  [ "$LOGIT_KD_OK" = "1" ] && CKPTS="$CKPTS ckpts/air_kd/final"
  bash 07_eval_benchmarks.sh $CKPTS
  mark benchmarks
fi

echo "================ PIPELINE COMPLETE ================"
echo "Judge summaries:";   ls evals/judge_*.json
echo "Benchmark results:"; ls evals/benchmarks/ || true
