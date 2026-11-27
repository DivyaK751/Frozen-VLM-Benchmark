#!/bin/bash
# Runs inside the Vertex AI job. Stages the data this job needs onto the local disk, then runs the
# command it was given (training or evaluation).
#
#   DATA=none|pilot|stage1|stage2|both  bash scripts/cloud/vertex_entry.sh <command ...>
#
# The bucket is mounted by Vertex at /gcs/<bucket> (Cloud Storage FUSE). Reading single images from it is far
# too slow, so the zips are copied to local disk and extracted there (DATA_PIPELINE.md section 5).
# For train.py the data_root flag is added automatically.
set -euo pipefail

MOUNT=${BUCKET_MOUNT:-/gcs/frozen-vlm-benchmark}
D=${DATA_DIR:-/tmp/data}
DATA=${DATA:-none}
mkdir -p "$D/processed" "$D/raw/pretrain" "$D/raw/instruct" "$D/images"

log() { echo "[entry $(date +%H:%M:%S)] $*"; }

# Unzip $1 into $2. If the zip had an extra top-level folder, move its contents up so that $3 exists.
extract() {
  local zip=$1 dest=$2 probe=$3
  mkdir -p "$dest"
  log "extracting $(basename "$zip") -> $dest"
  unzip -q -o "$zip" -d "$dest"
  if [ ! -f "$dest/$probe" ]; then
    local sub
    sub=$(find "$dest" -mindepth 1 -maxdepth 1 -type d | head -1)
    if [ -n "$sub" ] && [ -f "$sub/$probe" ]; then
      log "zip had a top-level folder ($(basename "$sub")); moving contents up"
      find "$sub" -mindepth 1 -maxdepth 1 -exec mv -t "$dest" {} +
      rmdir "$sub"
    fi
  fi
  [ -f "$dest/$probe" ] || { log "ERROR: $dest/$probe not found after extraction"; exit 1; }
  rm -f "$zip"                                           # free the disk
}

log "staging data: $DATA"
case "$DATA" in none) ;; *) cp "$MOUNT/processed/image_checks/bad_images.json" "$D/processed/bad_images.json" ;; esac

if [ "$DATA" = pilot ]; then
  # data/pilot uploaded once from the desktop: bash scripts/cloud/launch_vertex.sh upload-pilot
  cp -r "$MOUNT/pilot_data" "$D/pilot"
fi
if [ "$DATA" = stage1 ] || [ "$DATA" = both ]; then
  cp "$MOUNT/raw/pretrain/blip_laion_cc_sbu_558k.json" "$D/raw/pretrain/"
  cp "$MOUNT/raw/pretrain/images.zip" /tmp/pretrain_images.zip
  extract /tmp/pretrain_images.zip "$D/images/pretrain" "00217/002170933.jpg"
fi
if [ "$DATA" = stage2 ] || [ "$DATA" = both ]; then
  cp "$MOUNT/raw/instruct/llava_instruct_150k.json" "$D/raw/instruct/"
  cp "$MOUNT/raw/coco/train2017.zip" /tmp/train2017.zip
  extract /tmp/train2017.zip "$D/images/coco/train2017" "000000369358.jpg"
fi
log "data ready"; df -h /tmp | tail -1

# train.py needs to know where the data is; other commands do not
case "$*" in *scripts/train.py*) set -- "$@" --data_root "$D" ;; esac

log "running: $*"
"$@"
log "finished"
