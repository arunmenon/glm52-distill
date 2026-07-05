#!/usr/bin/env bash
# pilot_teacher.sh — LEG 1: real GLM-5.2-FP8 on whatever node fits it (~750 GB
# weights + KV headroom). Defaults to TP=8 (8x H200 SXM, 1,128 GB); override
# for other shapes, e.g. 6-7x B200: TP=2 PP=3 bash pilot_teacher.sh
#
#   1. real GLM-5.2-FP8 serves under vLLM + token-id smoke
#   2. measured generation throughput  -> real corpus sizing for the GCP run
#   3. real tokenizer gate             -> expect kd_mode=remap
#   4. ~500-prompt corpus, best-of-4, top-20 logprobs
#   5. teacher held-out reference + judge directionality check (truncated
#      answers must LOSE to full answers — validates GLM-5.2 verdict parsing)
#   6. pack with remap -> prints the real unmappable-token drop rate
#   7. push corpus/data/gate/evals to $PILOT_STORE for the H100 leg
#
# Cost control: phases are state-marked and resumable; terminate the pod the
# moment "PILOT COMPLETE" prints. Set PILOT_STORE (HF dataset repo) + HF_TOKEN
# unless both legs share one network volume.
set -euo pipefail
cd "$(dirname "$0")"
source common.sh

STATE_DIR=pilot_teacher/.state
# ---- GLM-5.2 launch profile (audit fix: defaults in-repo, not operator memory) ----
export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"   # 2h warmup tax otherwise
N_GPU=$(nvidia-smi -L 2>/dev/null | wc -l)
case "${TP:-auto}${PP:-}" in
  auto*)  # shape-appropriate parallelism when not set explicitly
    case "$N_GPU" in
      8) TP=8; PP=1 ;;
      6|7) TP=2; PP=3 ;;
      4) TP=4; PP=1 ;;
      *) TP="$N_GPU"; PP=1 ;;
    esac ;;
esac
# GLM-5.2 needs the cu130 env (ensure_glm_env); default to it when present
[ -x /opt/venv/bin/vllm ] && VENV_DIR="${VENV_DIR:-/opt/venv}" && PIN_ENV="${PIN_ENV:-0}"
PORT=8000
BASE="http://localhost:$PORT/v1"
TEACHER="${TEACHER:-/workspace/models/glm-5.2-fp8}"
TEACHER_REPO="${TEACHER_REPO:-zai-org/GLM-5.2-FP8}"
SERVED_NAME=glm-5.2   # kept stable; 02_generate/06_judge address the server by this name
AIR_TOK=/workspace/models/glm-4.5-air-tok
QWEN_TOK=/workspace/models/qwen3-8b-tok
# EXTENDED=1 (~$300 budget): generate a 4k-prompt production-grade corpus instead
# of the 500-prompt diagnostic one. Generation is shard-resumable, so terminate
# at T-minus-30-min and every completed shard is kept — corpus size self-adjusts
# to the throughput the hardware actually delivers.
N_TOTAL="${N_TOTAL:-$([ "${EXTENDED:-0}" = "1" ] && echo 4000 || echo 500)}"
N_HELDOUT="${N_HELDOUT:-50}"
export WANDB_MODE=disabled

mkdir -p pilot_teacher/evals
ensure_pod_env

if ! done_p model; then
  hf_pull "$TEACHER_REPO" "$TEACHER"
  hf_pull zai-org/GLM-4.5-Air "$AIR_TOK" "tokenizer*" "chat_template*" "*config.json"
  hf_pull Qwen/Qwen3-8B "$QWEN_TOK" "tokenizer*" "chat_template*" "*config.json" "vocab.json" "merges.txt"
  mark model
fi

if ! done_p gate; then
  python 00_check_tokenizer.py --teacher "$TEACHER" --student-a "$AIR_TOK" \
    --out pilot_teacher/gate_result.json
  mark gate
fi

if ! done_p prompts; then
  python 01_build_prompts.py --out pilot_teacher/data --n-total "$N_TOTAL" --n-heldout "$N_HELDOUT"
  mark prompts
fi


# ---- CUTOVER=1: scripted early stop (audit fix: no manual kill+touch ritual) ----
# Gracefully stops generation, verifies the manifest, marks the corpus phase,
# then falls through to refs -> pack -> push like any normal completion.
if [ "${CUTOVER:-0}" = "1" ] && ! done_p corpus; then
  pkill -f "02_generate[.]py" 2>/dev/null || true
  sleep 5
  python - <<'PYEOF'
import json, glob, sys
m = json.load(open("pilot_teacher/corpus/MANIFEST.json"))
shards = set(glob.glob("pilot_teacher/corpus/shard_*.parquet"))
missing = [k for k in m if k.endswith(".parquet")
           and f"pilot_teacher/corpus/{k}" not in shards]
if missing:
    sys.exit(f"manifest lists missing shards: {missing}")
print(f"manifest consistent: {sum(1 for k in m if k.endswith('.parquet'))} sealed shards")
PYEOF
  mark corpus
  echo "CUTOVER: generation stopped, corpus marked; continuing to refs/pack/push"
fi

if ! done_p corpus; then
  # Re-enterable: reuse an already-live-and-HEALTHY server (curl, not pgrep —
  # a wedged process must not suppress the relaunch). Grace for a booting one.
  curl -sf "$BASE/models" > /dev/null 2>&1 || pgrep -f "vllm[ ]serve" > /dev/null 2>&1 || {
    launch_vllm "$TEACHER" glm-5.2 pilot_teacher/serve.log \
      --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP" \
      --max-model-len 32768 --gpu-memory-utilization 0.92 --max-num-seqs 128 \
      ${VLLM_EXTRA_ARGS:-}
  }
  wait_ready_and_smoke "$BASE" glm-5.2 240 30   # allow up to 2h

  GEN_T0=$SECONDS
  python 02_generate.py --prompts pilot_teacher/data/prompts.jsonl --out pilot_teacher/corpus \
    --base-url "$BASE" --model glm-5.2 --concurrency 96 --shard-size 250 --best-of 4
  GEN_ELAPSED=$((SECONDS - GEN_T0)) python - <<'PY'
import glob, os
import pyarrow.parquet as pq
kept = sum(
    sum(len(r) for r in pq.read_table(s, columns=["response_token_ids"])
        .column("response_token_ids").to_pylist())
    for s in glob.glob("pilot_teacher/corpus/shard_*.parquet"))
el = int(os.environ["GEN_ELAPSED"])
print(f"THROUGHPUT: kept {kept} tokens in {el}s -> {kept/max(el,1):.0f} tok/s kept-side; "
      f"~{4*kept/max(el,1):.0f} tok/s decode-side (best-of-4). "
      f"Size the GCP corpus with this number, not the README guess.")
PY

  mark corpus
fi

# Separate phase so an early generation stop (cost control) can still complete
# the run: re-launches the server if it is not already up.
if ! done_p refs; then
  curl -sf "$BASE/models" > /dev/null 2>&1 || {
    launch_vllm "$TEACHER" glm-5.2 pilot_teacher/serve.log \
      --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP" \
      --max-model-len 32768 --gpu-memory-utilization 0.92 --max-num-seqs 128 \
      ${VLLM_EXTRA_ARGS:-}
    wait_ready_and_smoke "$BASE" glm-5.2 240 30
  }
  # Teacher reference answers + judge directionality check while the server is up:
  # truncate each reference to 30% and judge truncated-vs-full. Expect ~all "loss"
  # for the truncated side; ~all "tie" would mean verdict parsing is broken.
  python 05_eval_gen.py --served --base-url "$BASE" --model glm-5.2 \
    --heldout pilot_teacher/data/prompts_heldout.jsonl --out pilot_teacher/evals/teacher_ref.jsonl
  python - <<'PY'
import json
rows = [json.loads(l) for l in open("pilot_teacher/evals/teacher_ref.jsonl")]
with open("pilot_teacher/evals/teacher_truncated.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps({**r, "answer": r["answer"][: max(1, len(r["answer"]) // 3)]},
                           ensure_ascii=False) + "\n")
PY
  python 06_judge.py --ref pilot_teacher/evals/teacher_ref.jsonl \
    --cand pilot_teacher/evals/teacher_truncated.jsonl \
    --out pilot_teacher/evals/judge_directionality.json --base-url "$BASE" --judge glm-5.2
  kill_vllm
  mark refs
fi

if ! done_p packed; then
  python 03_pack_dataset.py --corpus pilot_teacher/corpus \
    --heldout pilot_teacher/data/prompts_heldout.jsonl \
    --teacher "$TEACHER" --student-a "$AIR_TOK" --student-b "$QWEN_TOK" \
    --out pilot_teacher/packed --gate pilot_teacher/gate_result.json
  mark packed
fi

if ! done_p pushed; then
  store_push pilot_teacher/corpus corpus
  store_push pilot_teacher/data data
  store_push pilot_teacher/evals evals
  store_push pilot_teacher/gate_result.json gate_result.json
  mark pushed
fi

echo "================ PILOT COMPLETE — terminate this pod now ================"
echo "-- gate:";  cat pilot_teacher/gate_result.json
echo "-- judge directionality (truncated side should be ~all loss):"
cat pilot_teacher/evals/judge_directionality.json
