#!/usr/bin/env bash
# pilot_runpod.sh — single-GPU end-to-end rehearsal of the whole pipeline.
#
# Uses Qwen3-8B as BOTH the teacher stand-in and the student: the tokenizer
# gate passes trivially, so the token-exact logit-KD path (the riskiest code)
# is exercised end to end, including best-of-N judging, packing, ZeRO-3 LoRA
# training, adapter merge, vLLM reload of the merged checkpoint, and judging.
#
# Everything lands under pilot/ with its own state marks; idempotent — rerun
# after a failure and completed phases are skipped. `rm -rf pilot` for a clean run.
set -euo pipefail
cd "$(dirname "$0")"
source common.sh

STATE_DIR=pilot/.state
PORT=8000
BASE="http://localhost:$PORT/v1"
MODEL_DIR=/workspace/models/qwen3-8b
SERVED_NAME=teacher
export WANDB_MODE=disabled

mkdir -p pilot/evals
ensure_pod_env
mark deps   # kept for state compatibility; ensure_pod_env is idempotent

serve_standin() {
  launch_vllm "$MODEL_DIR" "$SERVED_NAME" pilot/serve.log \
    --max-model-len 16384 --gpu-memory-utilization 0.9
  wait_ready_and_smoke "$BASE" "$SERVED_NAME" 60 10
}

if ! done_p model; then
  hf_pull Qwen/Qwen3-8B "$MODEL_DIR"
  mark model
fi

if ! done_p gate; then
  python 00_check_tokenizer.py --teacher "$MODEL_DIR" --student-a "$MODEL_DIR" \
    --out pilot/gate_result.json
  mark gate
fi

if ! done_p prompts; then
  python 01_build_prompts.py --out pilot/data --n-total 200 --n-heldout 24
  mark prompts
fi

if ! done_p corpus; then
  serve_standin
  python 02_generate.py --prompts pilot/data/prompts.jsonl --out pilot/corpus \
    --base-url "$BASE" --model "$SERVED_NAME" --concurrency 32 \
    --shard-size 100 --best-of 2
  kill_vllm
  mark corpus
fi

if ! done_p packed; then
  python 03_pack_dataset.py --corpus pilot/corpus --heldout pilot/data/prompts_heldout.jsonl \
    --teacher "$MODEL_DIR" --student-a "$MODEL_DIR" --student-b "$MODEL_DIR" \
    --out pilot/packed --gate pilot/gate_result.json
  mark packed
fi

if ! done_p train; then
  accelerate launch --config_file configs/ds_zero3_1gpu.yaml 04_train_kd.py \
    --model "$MODEL_DIR" --data pilot/packed/student_a --out pilot/ckpts/kd \
    --alpha 0.5 --temperature 2.0 --lora --lr 1e-4 --epochs 1
  mark train
fi

if ! done_p merge; then
  python merge_lora.py --base "$MODEL_DIR" --adapter pilot/ckpts/kd/adapter \
    --out pilot/ckpts/kd/final
  mark merge
fi

if ! done_p student_gens; then
  python 05_eval_gen.py --model pilot/ckpts/kd/final \
    --heldout pilot/data/prompts_heldout.jsonl --out pilot/evals/kd.jsonl --tp 1
  mark student_gens
fi

if ! done_p judged; then
  serve_standin
  python 05_eval_gen.py --served --base-url "$BASE" --model "$SERVED_NAME" \
    --heldout pilot/data/prompts_heldout.jsonl --out pilot/evals/teacher_ref.jsonl
  python 06_judge.py --ref pilot/evals/teacher_ref.jsonl --cand pilot/evals/kd.jsonl \
    --out pilot/evals/judge_kd.json --base-url "$BASE" --judge "$SERVED_NAME"
  kill_vllm
  mark judged
fi

echo "================ PILOT COMPLETE ================"
echo "-- corpus manifest:"; cat pilot/corpus/MANIFEST.json
echo "-- judge summary:";   cat pilot/evals/judge_kd.json
