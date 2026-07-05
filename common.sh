#!/usr/bin/env bash
# common.sh — shared helpers sourced by the serve scripts and pilot legs.
# Callers activate their own python env first (vllm/python must be on PATH).

# ---- phase-state helpers (caller sets STATE_DIR) ----
mark()   { mkdir -p "$STATE_DIR"; touch "$STATE_DIR/$1"; }
done_p() { [ -f "$STATE_DIR/$1" ]; }

# ---- vLLM serving ----
# launch_vllm <model_dir> <served_name> <log_file> [extra vllm args...]
# Flags every server in this pipeline needs: token-id logprobs (02_generate),
# no reasoning parser (thinking stays in the distillation signal), fixed seed.
launch_vllm() {
  local model="$1" name="$2" log="$3"; shift 3
  # VLLM_BIN: point at an alternate vllm (e.g. venv2 with transformers 5.x for
  # glm_moe_dsa) without disturbing the pinned training env.
  nohup "${VLLM_BIN:-vllm}" serve "$model" \
    --served-model-name "$name" \
    --enable-prefix-caching \
    --return-tokens-as-token-ids \
    --seed 42 \
    --port "${PORT:-8000}" \
    "$@" > "$log" 2>&1 &
  echo "vLLM starting (pid $!, log $log)"
}

# wait_ready_and_smoke <base_url> <served_name> [tries] [sleep_s]
# Polls /models, then asserts token-id logprob mode with a 1-shot completion.
wait_ready_and_smoke() {
  local base="$1" model="$2" tries="${3:-240}" pause="${4:-30}"
  for i in $(seq 1 "$tries"); do
    if curl -sf "$base/models" > /dev/null 2>&1; then
      echo "READY after ~$((i*pause))s"
      curl -s "$base/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_tokens\":64,\"logprobs\":true,\"top_logprobs\":5,\"seed\":42}" \
        | python3 -c 'import sys,json;d=json.load(sys.stdin);c=d["choices"][0];assert c["logprobs"]["content"][0]["token"].startswith("token_id:"),"token-id mode NOT active";print("SMOKE TEST OK")'
      return 0
    fi
    sleep "$pause"
  done
  echo "ERROR: server at $base not ready after $((tries*pause))s" >&2
  return 1
}

kill_vllm() { pkill -f "vllm serve" || true; sleep "${1:-10}"; }

# ---- RunPod pod environment (idempotent; pinned CUDA-12.8 stack that is
# verified on Blackwell pods: torch 2.8.0+cu128 / vllm 0.11.0 / transformers 4.x) ----
ensure_pod_env() {
  export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1
  mkdir -p /workspace/hf
  local venv="${VENV_DIR:-/workspace/venv}"
  printf 'torch==2.8.0\ntransformers==4.57.6\nnumpy<2.3\nvllm==0.11.0\n' > /tmp/pip_constraints.txt
  # PIN_ENV=0: the venv is managed manually (e.g. cu13 stack for GLM-5.2) —
  # never clobber it with the pinned cu128 stack, just verify and activate.
  if [ "${PIN_ENV:-1}" != "0" ] && \
     ! "$venv/bin/python" -c "import vllm, torch, datasets, peft, accelerate, deepspeed, datasketch" 2>/dev/null; then
    if [ -x "$venv/bin/python" ]; then
      # NEVER install into a venv this function did not create — a custom
      # stack (e.g. the GLM-5.2 cu130 env) failing the training-deps probe
      # must not be clobbered. Operator decides: PIN_ENV=0 or rebuild.
      echo "ERROR: $venv exists but fails the training-stack probe." >&2
      echo "Set PIN_ENV=0 to use it as-is, or remove it to rebuild pinned." >&2
      return 1
    fi
    python3 -m venv "$venv"
    "$venv/bin/pip" install -q -U pip uv
    "$venv/bin/uv" pip install --python "$venv/bin/python" --torch-backend=cu128 \
      -c /tmp/pip_constraints.txt \
      torch==2.8.0 vllm==0.11.0 "transformers<5" "numpy<2.3" hf_transfer \
      datasets peft accelerate deepspeed datasketch pyarrow aiohttp
  fi
  "$venv/bin/python" -c "import vllm, torch; print('env OK: vllm', vllm.__version__, '| torch', torch.__version__)"
  # shellcheck disable=SC1091
  source "$venv/bin/activate"
}


# ---- GLM-5.2 serving environment (CUDA-13 driver hosts ONLY) ----
# The battle-tested cu130 stack: vLLM 0.24.0 (native glm_moe_dsa) + pinned
# datasets/fsspec pair. Build on LOCAL disk (container disk), never on
# network volumes (partial-copy corruption observed 2026-07-05).
ensure_glm_env() {
  export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1
  local venv="${VENV_DIR:-/opt/venv}"
  if ! "$venv/bin/python" -c "import vllm" 2>/dev/null; then
    rm -rf "$venv"
    python3 -m venv "$venv"
    "$venv/bin/pip" install -q -U pip uv
    export UV_LINK_MODE=copy
    "$venv/bin/uv" pip install -q --python "$venv/bin/python" --torch-backend=auto \
      "vllm==0.24.0" hf_transfer "datasets==5.0.0" datasketch pyarrow aiohttp
  fi
  "$venv/bin/python" -c "import vllm, torch, datasets; print('glm env OK: vllm', vllm.__version__, '| torch', torch.__version__)"
  # shellcheck disable=SC1091
  source "$venv/bin/activate"
}

# hf_pull <repo_id> <local_dir> [include_pattern ...]
# One --include per call (multi-include is flaky on some hf CLI versions).
hf_pull() {
  local repo="$1" dst="$2"; shift 2
  if [ "$#" -eq 0 ]; then
    hf download "$repo" --local-dir "$dst"
  else
    local pat
    for pat in "$@"; do
      hf download "$repo" --local-dir "$dst" --include "$pat"
    done
  fi
}

# ---- artifact store: cross-pod hand-off via a private HF dataset repo ----
# export PILOT_STORE=<user>/glm52-pilot-artifacts   (needs HF_TOKEN with write)
# Unset PILOT_STORE = both legs share one network volume; nothing to move.
store_push() {  # store_push <local_path> <remote_subpath>
  [ -n "${PILOT_STORE:-}" ] || { echo "[store] PILOT_STORE unset; leaving $1 on this volume"; return 0; }
  hf repo create "$PILOT_STORE" --repo-type dataset --private 2>/dev/null || true
  hf upload "$PILOT_STORE" "$1" "$2" --repo-type dataset
  echo "[store] pushed $1 -> $PILOT_STORE/$2"
}
store_pull() {  # store_pull <remote_subpath_pattern> ; lands at ./<subpath>
  [ -n "${PILOT_STORE:-}" ] || { echo "[store] PILOT_STORE unset; assuming $1 already on this volume"; return 0; }
  hf download "$PILOT_STORE" --repo-type dataset --local-dir . --include "$1"
  echo "[store] pulled $1"
}
