#!/usr/bin/env bash
# 20_sync.sh — GCS artifact sync for local-disk-only VMs.
#
#   bash 20_sync.sh push <local_dir> <remote_subpath>
#   bash 20_sync.sh pull <remote_subpath> <local_dir>
#
# Set GCS_BUCKET once, e.g.: export GCS_BUCKET=gs://my-bucket/glm52-distill
#
# WHY THIS EXISTS: the VMs have local disk only. Local SSD contents are LOST on
# VM stop/preemption. Every phase output (prompts, corpus shards, checkpoints,
# evals) must be pushed to GCS immediately after it is produced, and pulled by
# whichever VM consumes it next. run_vm_*.sh scripts call this automatically.
set -euo pipefail

: "${GCS_BUCKET:?Set GCS_BUCKET, e.g. export GCS_BUCKET=gs://my-bucket/glm52-distill}"

CMD="${1:?push|pull}"
case "$CMD" in
  push)
    SRC="${2:?local dir}"; DST="${3:?remote subpath}"
    gcloud storage rsync -r "$SRC" "$GCS_BUCKET/$DST"
    echo "[sync] pushed $SRC -> $GCS_BUCKET/$DST"
    ;;
  pull)
    SRC="${2:?remote subpath}"; DST="${3:?local dir}"
    mkdir -p "$DST"
    gcloud storage rsync -r "$GCS_BUCKET/$SRC" "$DST"
    echo "[sync] pulled $GCS_BUCKET/$SRC -> $DST"
    ;;
  *) echo "usage: 20_sync.sh push|pull <src> <dst>" >&2; exit 1 ;;
esac
