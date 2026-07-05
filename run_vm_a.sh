#!/usr/bin/env bash
# run_vm_a.sh — VM-A (8 GPUs): Ray head + teacher stage 1 -> Qwen3-8B training -> lm-eval.
# Requires: export GCS_BUCKET=gs://...   Run 00_setup.sh here first.
set -euo pipefail
: "${GCS_BUCKET:?export GCS_BUCKET=gs://my-bucket/glm52-distill}"
mkdir -p .state
mark(){ touch ".state/$1"; }; done_p(){ [ -f ".state/$1" ]; }

# ---- Phase 1: serve (VM-C drives generation; this VM just hosts) ----
if ! done_p served; then
  bash 11_ray_cluster.sh head
  echo ">>> Now run on VM-B:  HEAD_IP=$(hostname -I | awk '{print $1}') bash 11_ray_cluster.sh worker"
  read -rp "Press enter once VM-B has joined (ray status shows 16 GPUs)..."
  bash 12_serve_teacher_multinode.sh
  echo ">>> Teacher is up at http://$(hostname -I | awk '{print $1}'):8000/v1"
  echo ">>> Now run run_vm_c.sh on VM-C. Press enter here AFTER VM-C finishes generation."
  read -rp "..."
  pkill -f "vllm serve" || true; ray stop || true; sleep 20
  mark served
fi

# ---- Phase 2: train Qwen3-8B (student B) ----
source "$HOME/venvs/train/bin/activate"
if ! done_p pulled_data; then
  bash 20_sync.sh pull corpus corpus/
  bash 20_sync.sh pull data data/
  bash 20_sync.sh pull gate gate/ || true
  python 03_pack_dataset.py --corpus corpus/ --heldout data/prompts_heldout.jsonl \
    --teacher /models/glm-5.2-fp8 --student-a /models/glm-4.5-air \
    --student-b /models/qwen3-8b --out packed/ --gate gate/gate_result.json
  bash 20_sync.sh push packed/student_a packed/student_a   # VM-B pulls this
  mark pulled_data
fi

if ! done_p train_qwen; then
  accelerate launch --config_file configs/ds_zero3_8gpu.yaml 04_train_kd.py \
    --model /models/qwen3-8b --data packed/student_b --out ckpts/qwen3_sft \
    --alpha 0.0 --grad-accum 16
  bash 20_sync.sh push ckpts/qwen3_sft/final ckpts/qwen3_sft_final
  mark train_qwen
fi

# ---- Phase 3: benchmarks for all checkpoints (pull Air ckpts from VM-B) ----
if ! done_p bench; then
  bash 20_sync.sh pull ckpts/air_sft_final ckpts/air_sft/final || true
  bash 20_sync.sh pull ckpts/air_kd_final  ckpts/air_kd/final  || true
  CK="ckpts/qwen3_sft/final"
  [ -d ckpts/air_sft/final ] && CK="$CK ckpts/air_sft/final"
  [ -d ckpts/air_kd/final ]  && CK="$CK ckpts/air_kd/final"
  bash 07_eval_benchmarks.sh $CK
  bash 20_sync.sh push evals evals
  mark bench
fi

# ---- Phase 3b: re-host teacher (with VM-B) as judge; VM-C drives judging ----
if ! done_p rehost; then
  bash 11_ray_cluster.sh head
  echo ">>> Now on VM-B: HEAD_IP=$(hostname -I | awk '{print $1}') bash run_vm_b.sh  (re-joins the cluster)"
  read -rp "Press enter once VM-B has re-joined (ray status shows 16 GPUs)..."
  bash 12_serve_teacher_multinode.sh
  echo ">>> Teacher re-hosted. Tell VM-C to run its judging phase."
  read -rp "Press enter AFTER VM-C finishes judging..."
  pkill -f "vllm serve" || true; ray stop || true
  mark rehost
fi
echo "VM-A done."
