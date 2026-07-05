#!/usr/bin/env bash
# run_vm_b.sh — VM-B (8 GPUs): Ray worker for stage 2 of the teacher -> GLM-4.5-Air
# LoRA training (SFT and KD variants) -> push checkpoints to GCS.
# Requires: export GCS_BUCKET=gs://...   Run 00_setup.sh here first.
set -euo pipefail
: "${GCS_BUCKET:?export GCS_BUCKET=gs://my-bucket/glm52-distill}"
mkdir -p .state
mark(){ touch ".state/$1"; }; done_p(){ [ -f ".state/$1" ]; }

# ---- Phase 1: join the Ray cluster (interactive; VM-A tells you when) ----
if ! done_p served; then
  : "${HEAD_IP:?export HEAD_IP=<vm-a internal ip>}"
  HEAD_IP="$HEAD_IP" bash 11_ray_cluster.sh worker
  echo ">>> Worker joined. Wait until VM-A prints 'pushed packed/student_a'"
  echo ">>> (that happens in VM-A's Phase 2, after packing), then press enter."
  read -rp "..."
  ray stop || true
  mark served
fi

# ---- Phase 2: train GLM-4.5-Air (student A) ----
source "$HOME/venvs/train/bin/activate"
if ! done_p pulled_data; then
  # VM-A packs and pushes this partway through its own Phase 2; poll until it lands.
  until bash 20_sync.sh pull packed/student_a packed/student_a; do
    echo "packed/student_a not in GCS yet (VM-A still packing?); retrying in 60s..."
    sleep 60
  done
  bash 20_sync.sh pull gate gate/ || true
  mark pulled_data
fi
LOGIT_KD_OK=$(python3 -c 'import json;print(1 if json.load(open("gate/gate_result.json"))["logit_kd_ok"] else 0)' 2>/dev/null || echo 0)

if ! done_p train_air_sft; then
  accelerate launch --config_file configs/ds_zero3_8gpu.yaml 04_train_kd.py \
    --model /models/glm-4.5-air --data packed/student_a --out ckpts/air_sft \
    --alpha 0.0 --lora --lr 1e-4 --grad-accum 16
  python merge_lora.py --base /models/glm-4.5-air --adapter ckpts/air_sft/adapter --out ckpts/air_sft/final
  bash 20_sync.sh push ckpts/air_sft/final ckpts/air_sft_final
  mark train_air_sft
fi

if [ "$LOGIT_KD_OK" = "1" ] && ! done_p train_air_kd; then
  accelerate launch --config_file configs/ds_zero3_8gpu.yaml 04_train_kd.py \
    --model /models/glm-4.5-air --data packed/student_a --out ckpts/air_kd \
    --alpha 0.5 --temperature 2.0 --lora --lr 1e-4 --grad-accum 16
  python merge_lora.py --base /models/glm-4.5-air --adapter ckpts/air_kd/adapter --out ckpts/air_kd/final
  bash 20_sync.sh push ckpts/air_kd/final ckpts/air_kd_final
  mark train_air_kd
fi

# ---- Phase 3: held-out generation for Air checkpoints (8 GPUs available here) ----
if ! done_p air_gens; then
  bash 20_sync.sh pull data data/
  python 05_eval_gen.py --model ckpts/air_sft/final --heldout data/prompts_heldout.jsonl --out evals/air_sft.jsonl --tp 8
  [ "$LOGIT_KD_OK" = "1" ] && python 05_eval_gen.py --model ckpts/air_kd/final --heldout data/prompts_heldout.jsonl --out evals/air_kd.jsonl --tp 8
  bash 20_sync.sh push evals evals
  mark air_gens
fi

# ---- Phase 3b: re-join Ray so VM-A can re-host the teacher for judging ----
if ! done_p rehost; then
  : "${HEAD_IP:?export HEAD_IP=<vm-a internal ip> (needed again for the re-host)}"
  HEAD_IP="$HEAD_IP" bash 11_ray_cluster.sh worker
  echo ">>> Re-joined. Press enter AFTER VM-C finishes judging."
  read -rp "..."
  ray stop || true
  mark rehost
fi
echo "VM-B done."
