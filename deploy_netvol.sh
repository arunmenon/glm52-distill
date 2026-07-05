#!/usr/bin/env bash
# deploy_netvol.sh — deploy a teacher pod the durable way (audit fix #4):
# network volume FIRST, then a pod mounting it, with the CUDA-13 constraint
# GLM-5.2 requires. Weights downloaded onto the volume survive pod churn.
#
#   RUNPOD_API_KEY=... bash deploy_netvol.sh [DC] [GPU_TYPE] [GPU_COUNT] [VOL_GB]
# Defaults: US-CA-2, NVIDIA B200, 6, 1200.
# Prints VOLUME_ID and POD_ID; idempotent on the volume (reuses by name).
set -euo pipefail

KEY="${RUNPOD_API_KEY:-$(cat ~/.runpod/api_key 2>/dev/null | tr -d '[:space:]')}"
[ -n "$KEY" ] || { echo "need RUNPOD_API_KEY or ~/.runpod/api_key" >&2; exit 1; }
DC="${1:-US-CA-2}"
GPU="${2:-NVIDIA B200}"
CNT="${3:-6}"
VOL_GB="${4:-1200}"
VOL_NAME="glm52-weights-${DC}"

GQL() { curl -s "https://api.runpod.io/graphql?api_key=$KEY" -H 'Content-Type: application/json' -d "$1"; }

VOL_ID=$(GQL '{"query":"query { myself { networkVolumes { id name dataCenterId } } }"}' \
  | python3 -c "import sys,json
for v in json.load(sys.stdin)['data']['myself']['networkVolumes']:
    if v['name']=='$VOL_NAME' and v['dataCenterId']=='$DC': print(v['id']); break")
if [ -z "$VOL_ID" ]; then
  VOL_ID=$(GQL "{\"query\":\"mutation { createNetworkVolume(input:{name:\\\"$VOL_NAME\\\", size:$VOL_GB, dataCenterId:\\\"$DC\\\"}) { id } }\"}" \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"]["createNetworkVolume"]["id"])')
  echo "created volume $VOL_ID ($VOL_NAME, ${VOL_GB}GB, $DC)"
else
  echo "reusing volume $VOL_ID ($VOL_NAME)"
fi

RES=$(GQL "{\"query\":\"mutation { podFindAndDeployOnDemand(input:{cloudType: SECURE, gpuCount: $CNT, gpuTypeId: \\\"$GPU\\\", dataCenterId: \\\"$DC\\\", networkVolumeId: \\\"$VOL_ID\\\", volumeMountPath: \\\"/workspace\\\", containerDiskInGb: 100, volumeInGb: 0, name: \\\"glm52-teacher\\\", imageName: \\\"runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404\\\", ports: \\\"22/tcp\\\", startSsh: true, allowedCudaVersions: [\\\"13.0\\\"]}) { id costPerHr } }\"}")
echo "$RES"
echo "$RES" | grep -q '"id"' || { echo "DEPLOY FAILED (no CUDA-13 $GPU x$CNT in $DC right now?)" >&2; exit 1; }
echo "VOLUME_ID=$VOL_ID"
