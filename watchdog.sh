#!/usr/bin/env bash
# watchdog.sh — ON-POD spend guard (audit fix #8: laptop loops get reaped;
# the pod must be able to stop itself).
#
#   nohup bash watchdog.sh <MAX_MINUTES> [POD_ID] >> watchdog.log 2>&1 &
#
# After MAX_MINUTES of wall clock: gracefully stops generation, gives the
# per-seal pushes 3 minutes to flush, then — if POD_ID and RUNPOD_API_KEY are
# available — stops the pod via the RunPod API. Without credentials it still
# halts the GPU burn by killing the serve process.
set -u
MAX_MIN="${1:?usage: watchdog.sh <MAX_MINUTES> [POD_ID]}"
POD_ID="${2:-${RUNPOD_POD_ID:-}}"   # RUNPOD_POD_ID is injected by RunPod
KEY="${RUNPOD_API_KEY:-}"

echo "watchdog armed: ${MAX_MIN} min, pod=${POD_ID:-unknown}"
sleep $((MAX_MIN * 60))

echo "watchdog: time budget exhausted — stopping generation"
pkill -f "02_generate[.]py" 2>/dev/null
sleep 180   # let per-seal pushes flush
pkill -f "vllm serve" 2>/dev/null

if [ -n "$POD_ID" ] && [ -n "$KEY" ]; then
  echo "watchdog: stopping pod $POD_ID"
  curl -s "https://api.runpod.io/graphql?api_key=$KEY" \
    -H 'Content-Type: application/json' \
    -d "{\"query\":\"mutation { podStop(input:{podId:\\\"$POD_ID\\\"}) { id desiredStatus } }\"}"
else
  echo "watchdog: no API credentials — GPU processes killed, pod left running"
fi
