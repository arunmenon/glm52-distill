#!/usr/bin/env bash
# run_vm_c.sh — VM-C (2 GPUs): the driver node. Builds prompts, runs the tokenizer
# gate, drives corpus generation against the teacher on VM-A, runs Qwen3 held-out
# generation (fits on 2 GPUs), and drives judging in Phase 3.
# Requires: export GCS_BUCKET=gs://...  export TEACHER_URL=http://<vm-a-ip>:8000/v1
# Model downloads needed here: tokenizers only is enough for the gate, but
# 00_setup.sh downloading qwen3-8b lets this VM also do Qwen evals.
set -euo pipefail
: "${GCS_BUCKET:?export GCS_BUCKET=gs://my-bucket/glm52-distill}"
: "${TEACHER_URL:?export TEACHER_URL=http://<vm-a internal ip>:8000/v1}"
BEST_OF="${BEST_OF:-4}"
PILOT="${PILOT:-0}"
N_TOTAL=$([ "$PILOT" = "1" ] && echo 500 || echo 100000)   # 100k default: BoN-4 budget
N_HELDOUT=$([ "$PILOT" = "1" ] && echo 50 || echo 1000)
mkdir -p .state gate
mark(){ touch ".state/$1"; }; done_p(){ [ -f ".state/$1" ]; }

# Refuse to mix pilot and full-run state: rerunning without PILOT=1 after a
# pilot would otherwise skip phases and push the 500-prompt corpus as real.
if [ -f .state/mode ]; then
  if [ "$(cat .state/mode)" != "$PILOT" ]; then
    echo "ERROR: existing .state/ is from a PILOT=$(cat .state/mode) run." >&2
    echo "Clear local state/outputs AND the GCS copies first:" >&2
    echo "  rm -rf .state data corpus evals gate" >&2
    echo "  gcloud storage rm -r \$GCS_BUCKET/data \$GCS_BUCKET/corpus \$GCS_BUCKET/gate" >&2
    exit 1
  fi
else
  echo "$PILOT" > .state/mode
fi
source "$HOME/venvs/train/bin/activate"

# ---- Phase 0: gate + prompts ----
if ! done_p gate; then
  python 00_check_tokenizer.py --teacher /models/glm-5.2-fp8 --student-a /models/glm-4.5-air \
    --out gate/gate_result.json
  bash 20_sync.sh push gate gate
  mark gate
fi
if ! done_p prompts; then
  python 01_build_prompts.py --out data/ --n-total "$N_TOTAL" --n-heldout "$N_HELDOUT"
  bash 20_sync.sh push data data
  mark prompts
fi

# ---- Phase 1: drive generation against VM-A's teacher ----
if ! done_p corpus; then
  # Resume any shards a previous (preempted) VM already pushed.
  bash 20_sync.sh pull corpus corpus/ || true
  source "$HOME/venvs/serve/bin/activate"
  # Generation runs for days; push completed shards every 10 min so a
  # preemption loses at most one in-flight shard, not the whole phase.
  ( while true; do sleep 600; bash 20_sync.sh push corpus corpus || true; done ) &
  SYNC_PID=$!
  python 02_generate.py --prompts data/prompts.jsonl --out corpus/ \
    --base-url "$TEACHER_URL" --model glm-5.2 --concurrency 112 --best-of "$BEST_OF"
  kill "$SYNC_PID" 2>/dev/null || true
  bash 20_sync.sh push corpus corpus
  source "$HOME/venvs/train/bin/activate"
  mark corpus
  echo ">>> Corpus done + pushed. Tell VM-A to tear down the teacher and start training."
fi

# ---- Phase 3a: Qwen3 held-out generation on 2 GPUs ----
if ! done_p qwen_gens; then
  bash 20_sync.sh pull ckpts/qwen3_sft_final ckpts/qwen3_sft/final
  python 05_eval_gen.py --model ckpts/qwen3_sft/final \
    --heldout data/prompts_heldout.jsonl --out evals/qwen3_sft.jsonl --tp 2
  bash 20_sync.sh push evals evals
  mark qwen_gens
fi

# ---- Phase 3b: judging (teacher must be re-hosted on VM-A+B first) ----
if ! done_p judged; then
  echo ">>> Re-host the teacher (11_ray_cluster.sh + 12_serve_teacher_multinode.sh on A+B), then press enter."
  read -rp "..."
  bash 20_sync.sh pull evals evals/
  python 05_eval_gen.py --served --base-url "$TEACHER_URL" --model glm-5.2 \
    --heldout data/prompts_heldout.jsonl --out evals/teacher_ref.jsonl
  for cand in evals/qwen3_sft.jsonl evals/air_sft.jsonl evals/air_kd.jsonl; do
    [ -f "$cand" ] || continue
    name=$(basename "$cand" .jsonl)
    python 06_judge.py --ref evals/teacher_ref.jsonl --cand "$cand" \
      --out "evals/judge_${name}.json" --base-url "$TEACHER_URL" --judge glm-5.2
  done
  bash 20_sync.sh push evals evals
  mark judged
fi
echo "VM-C done. Judge summaries in evals/judge_*.json"
