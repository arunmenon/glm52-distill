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
mkdir -p "$OUTDIR"

for MODEL in "$@"; do
  NAME=$(echo "$MODEL" | tr '/' '_')
  echo "== Evaluating $MODEL =="
  lm_eval --model vllm \
    --model_args "pretrained=${MODEL},tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=8192,seed=42,trust_remote_code=True" \
    --tasks "$TASKS" \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --batch_size auto \
    --seed 42 \
    --output_path "${OUTDIR}/${NAME}" \
    --log_samples
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
