#!/bin/bash
# Launch the full-data image checks (bad-image scan + POPE near-duplicates) on a cloud VM.
#
#   bash scripts/cloud/launch_image_checks.sh            # upload inputs, create the VM
#   bash scripts/cloud/launch_image_checks.sh status     # DONE / FAILED / still running
#   bash scripts/cloud/launch_image_checks.sh fetch      # download results to data/processed/
#   bash scripts/cloud/launch_image_checks.sh teardown   # delete the VM (run after fetch)
#
# The VM has no public IP (Columbia blocks them) and no internet: it reads the bucket
# through Private Google Access, and gets code and Python packages from the bucket too.
# Cost: e2-standard-16 + 400 GB disk for about 45 minutes, under $1. The VM powers itself
# off when done; teardown deletes it so the disk stops costing money.
set -euo pipefail
cd "$(dirname "$0")/../.."

PROJECT=encouraging-art-515701-r8
ZONE=us-central1-a
REGION=us-central1
VM=image-checks
BUCKET=gs://frozen-vlm-benchmark
JOB=$BUCKET/jobs/image_checks
OUT=$BUCKET/processed/image_checks
G="gcloud --project $PROJECT --quiet"

case "${1:-launch}" in
launch)
  # Python packages for Debian 12 (Python 3.11), same versions as the local .venv.
  WHEELS=$(mktemp -d)
  .venv/bin/pip download --quiet --only-binary=:all: --platform manylinux2014_x86_64 --platform manylinux_2_28_x86_64 \
    --python-version 3.11 --no-deps -d "$WHEELS" numpy==2.0.2 pillow==11.3.0 pyarrow==21.0.0
  tar -czf "$WHEELS/code.tgz" src scripts
  $G storage cp "$WHEELS"/*.whl $JOB/wheels/
  $G storage cp "$WHEELS/code.tgz" $JOB/code.tgz
  $G storage cp data/raw/pope/data/*.parquet $JOB/pope/
  $G storage rm "$OUT/DONE" "$OUT/FAILED" 2>/dev/null || true

  # Lets a VM without a public IP reach Cloud Storage. Free; changes only this setting.
  $G compute networks subnets update default --region $REGION --enable-private-ip-google-access

  $G compute instances create $VM --zone $ZONE \
    --machine-type e2-standard-16 --no-address --scopes cloud-platform \
    --image-family debian-12 --image-project debian-cloud \
    --boot-disk-size 400GB --boot-disk-type pd-balanced \
    --metadata-from-file startup-script=scripts/cloud/image_checks_job.sh
  echo "Started. Check with: bash $0 status   (expect about 45 minutes)"
  ;;
status)
  $G storage ls $OUT/ 2>/dev/null | grep -E '/(DONE|FAILED)$' && exit 0
  echo "still running; last log lines:"
  $G compute instances get-serial-port-output $VM --zone $ZONE 2>/dev/null | grep startup-script | tail -5
  ;;
fetch)
  mkdir -p data/processed
  $G storage cp $OUT/bad_images.json data/processed/bad_images.json
  $G storage cp -r $OUT/pope_near_dups data/processed/
  $G storage cp $OUT/job.log data/processed/image_checks_job.log
  echo "Fetched to data/processed/ (bad_images.json is what the Dataset reads)"
  ;;
teardown)
  $G compute instances delete $VM --zone $ZONE
  ;;
*)
  echo "usage: $0 [launch|status|fetch|teardown]"; exit 1 ;;
esac
