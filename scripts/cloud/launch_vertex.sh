#!/bin/bash
# Run training / evaluation on a Vertex AI GPU from your desktop.
#
#   bash scripts/cloud/launch_vertex.sh setup                      # once: enable APIs, create the image repository
#   bash scripts/cloud/launch_vertex.sh build                      # once, and again when requirements.txt changes
#   bash scripts/cloud/launch_vertex.sh upload-pilot               # once: copy data/pilot to the bucket
#   bash scripts/cloud/launch_vertex.sh run NAME DATA -- COMMAND   # submit a job (uploads the latest code first)
#   bash scripts/cloud/launch_vertex.sh jobs                       # list recent jobs and their state
#   bash scripts/cloud/launch_vertex.sh logs JOB_ID                # follow a job's log
#   bash scripts/cloud/launch_vertex.sh cancel JOB_ID              # stop a job (stops the billing)
#
# DATA is what the job copies to its local disk first: none | pilot | stage1 | stage2 | both
# COMMAND is run from the repo root inside the job. Examples:
#   run smoke none -- nvidia-smi
#   run overfit pilot -- python scripts/train.py --config configs/overfit_stage2.json --output_dir /gcs/frozen-vlm-benchmark/runs/overfit
#   run e1 stage1 -- python scripts/train.py --config configs/e1_stage1.json --output_dir /gcs/frozen-vlm-benchmark/runs/e1_stage1
#   run eval_e3 none -- python scripts/evaluate.py --run_dir /gcs/frozen-vlm-benchmark/runs/e3_seed0 --name e3_seed0 --out_dir /gcs/frozen-vlm-benchmark/eval
#
# Settings (environment variables):
#   MACHINE=a2-highgpu-1g ACCEL=NVIDIA_TESLA_A100 COUNT=1   the GPU (for a cheaper test: MACHINE=g2-standard-8 ACCEL=NVIDIA_L4)
#   SPOT=1        use Spot VMs (about 60-70% cheaper, but can be stopped at any time; training resumes from its last
#                 checkpoint when you submit the same job again, so only use it for runs that checkpoint)
#   DISK_GB=400   boot disk (the zips plus the extracted images need about 100 GB; Stage 1 + 2 together about 200 GB)
#   DRY_RUN=1     print what would be submitted, change nothing
set -euo pipefail
cd "$(dirname "$0")/../.."

PROJECT=${PROJECT:-build-a-small-vlm}
REGION=${REGION:-us-central1}
BUCKET=${BUCKET:-build-a-small-vlm-data}
REPO=vlm
IMAGE=$REGION-docker.pkg.dev/$PROJECT/$REPO/train:latest
MACHINE=${MACHINE:-a2-highgpu-1g}
ACCEL=${ACCEL:-NVIDIA_TESLA_A100}
COUNT=${COUNT:-1}
DISK_GB=${DISK_GB:-400}
SPOT=${SPOT:-0}
DRY_RUN=${DRY_RUN:-0}
G="gcloud --project $PROJECT --quiet"

# Re-quote an argument that your shell already unquoted, so it reaches the job as one argument again.
quote_arg() {
  case "$1" in
    *[!A-Za-z0-9_./:=,@%+-]*)
      case "$1" in *\'*) echo "arguments containing a single quote are not supported" >&2; exit 1;; esac
      printf "'%s'" "$1" ;;
    *) printf '%s' "$1" ;;
  esac
}

doit() { if [ "$DRY_RUN" = 1 ]; then echo "DRY RUN: $*"; else "$@"; fi; }

case "${1:-help}" in
setup)
  doit $G services enable aiplatform.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com storage.googleapis.com
  doit $G artifacts repositories create $REPO --repository-format=docker --location=$REGION \
    --description="training images" || echo "(repository may already exist)"
  ;;
build)
  # Built by Cloud Build in the cloud, so Docker is not needed on your machine.
  doit $G builds submit --region=$REGION --config=scripts/cloud/cloudbuild.yaml --substitutions=_IMAGE="$IMAGE" .
  ;;
upload-pilot)
  [ -d data/pilot ] || { echo "data/pilot not found (run scripts/download_pilot.py and scripts/make_subsets.py first)"; exit 1; }
  doit $G storage rsync -r data/pilot gs://$BUCKET/pilot_data
  ;;
run)
  NAME=${2:?usage: run NAME DATA -- COMMAND}; DATA=${3:?usage: run NAME DATA -- COMMAND}
  [ "${4:-}" = "--" ] || { echo "usage: run NAME DATA -- COMMAND"; exit 1; }
  shift 4
  [ $# -gt 0 ] || { echo "no command given"; exit 1; }
  case "$DATA" in none|pilot|stage1|stage2|both) ;; *) echo "DATA must be none|pilot|stage1|stage2|both"; exit 1;; esac
  CMD=""
  for a in "$@"; do CMD="$CMD $(quote_arg "$a")"; done
  CMD=${CMD# }
  case "$CMD" in *\"*|*\\*) echo "the command must not contain double quotes or backslashes"; exit 1;; esac

  # ship the current code (the image only holds the dependencies)
  TMP=$(mktemp -d); tar -czf "$TMP/code.tgz" src scripts configs
  doit $G storage cp "$TMP/code.tgz" gs://$BUCKET/jobs/train/code.tgz

  STRATEGY=""
  [ "$SPOT" = 1 ] && STRATEGY=$'scheduling:\n  strategy: SPOT\n  restartJobOnWorkerRestart: true'
  cat > "$TMP/job.yaml" <<YAML
workerPoolSpecs:
- machineSpec:
    machineType: $MACHINE
    acceleratorType: $ACCEL
    acceleratorCount: $COUNT
  replicaCount: 1
  diskSpec:
    bootDiskType: pd-ssd
    bootDiskSizeGb: $DISK_GB
  containerSpec:
    imageUri: $IMAGE
    command: ["bash", "-c"]
    args: ["tar xzf /gcs/$BUCKET/jobs/train/code.tgz -C /app && DATA=$DATA BUCKET_MOUNT=/gcs/$BUCKET bash /app/scripts/cloud/vertex_entry.sh $CMD"]
$STRATEGY
YAML
  if [ "$DRY_RUN" = 1 ]; then echo "--- job.yaml ---"; cat "$TMP/job.yaml"; fi
  doit $G ai custom-jobs create --region=$REGION --display-name="$NAME" --config="$TMP/job.yaml"
  echo "Submitted. Follow it with:  bash $0 jobs   and   bash $0 logs JOB_ID"
  ;;
jobs)
  doit $G ai custom-jobs list --region=$REGION --limit=10 --sort-by=~createTime \
    --format="table(name.basename(),displayName,state,createTime.date())"
  ;;
logs)   doit $G ai custom-jobs stream-logs "${2:?usage: logs JOB_ID}" --region=$REGION ;;
cancel) doit $G ai custom-jobs cancel "${2:?usage: cancel JOB_ID}" --region=$REGION ;;
*)
  sed -n 2,28p "$0"
  ;;
esac
