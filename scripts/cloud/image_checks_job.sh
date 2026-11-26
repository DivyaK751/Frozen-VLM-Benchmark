#!/bin/bash
# Runs ON the cloud VM as its startup script (see launch_image_checks.sh). It needs no
# internet: code, Python wheels and data all come from the bucket over Private Google Access.
#
#   1. bad-image scan of both training zips  -> bad_images.json (read by the Dataset)
#   2. POPE near-duplicate check (pHash)     -> pope_near_dups/
#
# Results, the log and a DONE or FAILED marker go to $OUT; then the VM powers off.
set -euo pipefail

BUCKET=gs://frozen-vlm-benchmark
JOB=$BUCKET/jobs/image_checks
OUT=$BUCKET/processed/image_checks
WORK=/mnt/work
LOG=$WORK/job.log

[ -f /var/run/image_checks_started ] && exit 0   # startup scripts rerun on every boot
touch /var/run/image_checks_started

mkdir -p $WORK && cd $WORK
exec > >(tee -a $LOG) 2>&1
finish() {
  status=$1
  gcloud storage cp $LOG $OUT/job.log || true
  echo "$status $(date -u +%FT%TZ)" | gcloud storage cp - $OUT/$status || true
  poweroff
}
trap 'echo "FAILED at line $LINENO"; finish FAILED' ERR

echo "== $(date -u) start on $(nproc) CPUs"
gcloud storage cp $JOB/code.tgz "$JOB/wheels/*.whl" .
mkdir -p repo pylib && tar -xzf code.tgz -C repo
for w in *.whl; do python3 -m zipfile -e "$w" pylib/; done
export PYTHONPATH=$WORK/pylib

echo "== $(date -u) copying data"
mkdir -p data/raw/pope/data
gcloud storage cp $BUCKET/raw/pretrain/images.zip $BUCKET/raw/coco/train2017.zip data/
gcloud storage cp "$JOB/pope/*.parquet" data/raw/pope/data/
ls -la data data/raw/pope/data

cd repo
W=$(nproc)
echo "== $(date -u) bad-image scan"
python3 scripts/check_images.py --zip ../data/images.zip --zip ../data/train2017.zip \
  --out ../results/bad_images.json --workers $W
gcloud storage cp ../results/bad_images.json $OUT/bad_images.json

echo "== $(date -u) POPE near-duplicate check"
python3 scripts/check_pope_near_dups.py --zip ../data/images.zip --zip ../data/train2017.zip \
  --pope-dir ../data/raw/pope --out-dir ../results/pope_near_dups --workers $W
gcloud storage cp -r ../results/pope_near_dups $OUT/

echo "== $(date -u) done"
finish DONE
