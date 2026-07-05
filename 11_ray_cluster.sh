#!/usr/bin/env bash
# 11_ray_cluster.sh — bring up the 2-node Ray cluster for multi-node teacher serving.
#
#   On VM-A (head):    bash 11_ray_cluster.sh head
#   On VM-B (worker):  HEAD_IP=<vm-a-internal-ip> bash 11_ray_cluster.sh worker
#
# GCP prerequisites (do these once):
#   - Both VMs in the SAME zone, ideally a compact placement policy.
#   - gVNIC enabled + Tier_1 networking on both 8-GPU VMs (up to 400 Gbps
#     internal on g4 8-GPU shapes). PP=2 ships activations only, so even
#     50-100 Gbps effective is fine for batch generation.
#   - Firewall: allow tcp:6379 (ray) and tcp:10000-19999 (ray workers) and
#     NCCL's dynamic TCP range between the two VMs' internal IPs.
set -euo pipefail
source "$HOME/venvs/serve/bin/activate"

# NCCL over GCP internal network (no InfiniBand on G4):
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens,eth0}"
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ens,eth0}"

ROLE="${1:?head|worker}"
if [ "$ROLE" = "head" ]; then
  ray stop || true
  ray start --head --port=6379 --num-gpus=8
  echo "Ray head up. Internal IP: $(hostname -I | awk '{print $1}')"
  echo "Now on VM-B: HEAD_IP=$(hostname -I | awk '{print $1}') bash 11_ray_cluster.sh worker"
elif [ "$ROLE" = "worker" ]; then
  : "${HEAD_IP:?Set HEAD_IP=<vm-a internal ip>}"
  ray stop || true
  ray start --address="$HEAD_IP:6379" --num-gpus=8
  echo "Ray worker joined $HEAD_IP."
else
  echo "usage: 11_ray_cluster.sh head|worker" >&2; exit 1
fi
