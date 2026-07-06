#!/usr/bin/env bash
# 07_eval_benchmarks.sh — lm-eval-harness benchmarks for each final checkpoint.
# Usage: bash 07_eval_benchmarks.sh ckpts/qwen3_sft/final ckpts/air_sft/final ckpts/air_kd/final
# Overridable: TP (tensor parallel, default 8), TASKS, OUTDIR — the pilot legs
# call this with TP=4 TASKS="ifeval,gsm8k_cot".
set -euo pipefail
# GCP VMs have the train venv; pilot pods have /workspace/venv already active.
[ -f "$HOME/venvs/train/bin/activate" ] && source "$HOME/venvs/train/bin/activate"

TASKS="${TASKS:-ifeval,gsm8k_cot,mmlu_pro}"   # add leaderboard tasks as needed
TP="${TP:-8}"
OUTDIR="${OUTDIR:-evals/benchmarks}"
# Thinking-model settings (journal Discovery #6): these students emit a long
# <think> block BEFORE the answer. Defaults (max_gen_toks 256, greedy) truncate
# mid-thought and the answer-extraction regex never sees the final answer,
# scoring a capable model near zero. Give room to think + sampling params that
# match the model's reasoning mode; the harness extracts the post-</think> span.
MAX_GEN_TOKS="${MAX_GEN_TOKS:-4096}"
GEN_KWARGS="${GEN_KWARGS:-temperature=0.6,top_p=0.95,max_gen_toks=${MAX_GEN_TOKS}}"
MODEL_LEN="${MODEL_LEN:-16384}"   # prompt + long thinking must fit
mkdir -p "$OUTDIR"

for MODEL in "$@"; do
  NAME=$(echo "$MODEL" | tr '/' '_')
  echo "== Evaluating $MODEL (gen: $GEN_KWARGS) =="
  lm_eval --model vllm \
    --model_args "pretrained=${MODEL},tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=${MODEL_LEN},seed=42,trust_remote_code=True" \
    --tasks "$TASKS" \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --gen_kwargs "$GEN_KWARGS" \
    --batch_size auto \
    --seed 42 \
    --output_path "${OUTDIR}/${NAME}" \
    --log_samples

  # IFEval scores the RAW response against instructions; a think block fails
  # them regardless of answer quality. Re-score on think-stripped responses
  # (journal Discovery #6). Non-fatal: the raw lm-eval score is still written.
  if echo "$TASKS" | grep -q ifeval; then
    echo "-- IFEval think-stripped re-score for $NAME --"
    python3 rescore_ifeval.py --samples "${OUTDIR}/${NAME}"/**/samples_ifeval_*.jsonl \
      "${OUTDIR}/${NAME}"/samples_ifeval_*.jsonl 2>&1 \
      | tee "${OUTDIR}/${NAME}_ifeval_thinkstripped.json" || \
      echo "WARN: IFEval re-score failed; use raw score with the think-block caveat"
  fi
done

echo "== Summary =="
OUTDIR="$OUTDIR" python3 - <<'EOF'
import glob, json, os
outdir = os.environ["OUTDIR"]
for path in sorted(glob.glob(f"{outdir}/*/results*.json")):
    d = json.load(open(path))
    name = os.path.relpath(path, outdir).split(os.sep)[0]
    for task, m in d.get("results", {}).items():
        keys = [k for k in m if "acc" in k or "exact_match" in k or "strict" in k]
        vals = {k: round(m[k], 4) for k in keys if isinstance(m[k], (int, float))}
        print(f"{name:40s} {task:15s} {vals}")
EOF

# NOTE: LiveCodeBench and AIME'25 use their own official harnesses:
#   LiveCodeBench: https://github.com/LiveCodeBench/LiveCodeBench  (run with --model vllm, seed 42)
#   AIME 2025:     evaluate pass@1 with 8 samples/question, temperature 0.6 via 05_eval_gen.py-style generation
# Run them last; they are the slowest.
