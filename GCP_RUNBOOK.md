# Running on Google Cloud (Vertex AI, one A100)

Your laptop or desktop is for editing code and running the tests. Everything that needs a GPU (the overfit test on real
models, the pilot, E1, E2, E3, E0, the ablation and all evaluation) runs as a **Vertex AI custom job** that you submit from
your desktop with one command. The job starts a GPU machine, runs your command, writes results to the bucket, and **stops
(and stops billing) when the command ends**.

> **Status:** these scripts were written and dry-run tested, but not run against your project. Start with the 5-minute
> smoke job in step 4. If anything fails, send the error text and we will fix the script.

## 0. Do this today: request GPU quota (this is the critical path)

New projects often have **zero** Vertex GPU quota, and quota requests can take **several business days**. Without it, no
job below will start (the error is `RESOURCE_EXHAUSTED ... custom_model_training_nvidia_a100_gpus`).

1. Open the Cloud Console, project `encouraging-art-515701-r8`: **IAM & Admin → Quotas & System Limits**.
2. Filter by service **Vertex AI API** and search for **A100**. Request **1** for each of these in **us-central1**:
   - *Custom model training Nvidia A100 GPUs per region* (on-demand)
   - the *preemptible / Spot* version of it (optional, for cheaper runs later)
3. Also request **1 L4** (*Custom model training Nvidia L4 GPUs per region*) in us-central1. It is a cheaper GPU that can run the
   overfit test and a short pilot if the A100 approval is slow.
4. Justification text that works: *"University course project (Foundation Model Engineering). Fine-tuning a 0.5B vision-language
   model, about 40 GPU-hours in total on one GPU, deadline Dec 8."*
5. Check which billing account the project uses and that the $400 pooled credits are on it. If a request shows "not eligible",
   ask your instructor or TA whether the course has a Google Cloud education contact.

If the A100 is not approved by about **Dec 1**, tell me: there is a fallback (a Compute Engine GPU VM, using the same no-public-IP
pattern as your earlier image-check job) and a reduced-scope plan (see section 7).

## 1. One-time setup (on your desktop, about 15 minutes)

You need the [gcloud CLI](https://cloud.google.com/sdk/docs/install) and a shell (macOS, Linux, or WSL on Windows).

```bash
gcloud auth login
gcloud config set project encouraging-art-515701-r8
git checkout training-loop            # the branch with the code from this handoff
bash scripts/cloud/launch_vertex.sh setup     # enables the APIs, creates the image repository
bash scripts/cloud/launch_vertex.sh build     # builds the training image in the cloud (3-5 min, no Docker needed)
bash scripts/cloud/launch_vertex.sh upload-pilot    # uploads data/pilot (about 3 GB) so cloud jobs can use it
```

Notes:
- `build` only needs repeating when `requirements.txt` changes. Code changes need no rebuild: every `run` ships your latest
  `src/`, `scripts/` and `configs/` to the job.
- The training data (47 GB) is already in `gs://frozen-vlm-benchmark/raw/`. Jobs copy what they need to the machine's local disk.
- If a job later fails with `Permission denied` on `/gcs/...`, grant the service account named in the error the role
  **Storage Object Admin** on the bucket (Console → Cloud Storage → bucket → Permissions).
- Set a **budget alert**: Console → Billing → Budgets & alerts → create a budget (for example $300) with alerts at 50%, 75% and 90%.

## 2. What runs where

| Task | Where | Cost |
|---|---|---|
| Unit tests (`tests/*.py`), editing, reading logs | your machine | free |
| Quick "does it run" check of `scripts/train.py` on the overfit data | your machine, optional: `--device cpu --dtype float32 --max_steps 20 --micro_batch_size 2 --grad_accum 1` (slow, a smoke test only). On a Mac you can also try `--device mps`. | free |
| The convincing overfit test (loss near 0, shuffled images raise the loss) | cloud GPU | under $1 |
| Pilot (10K examples, measures speed, memory, cost) | cloud A100 | about $2-4 |
| E1, E2, E3, E0, ablation, evaluation | cloud A100 | the bulk of the budget |

## 3. How to submit a job

```bash
bash scripts/cloud/launch_vertex.sh run NAME DATA -- COMMAND
```

- `NAME`: a label that shows in the job list.
- `DATA`: what the job copies to local disk first: `none` (evaluation, smoke test), `pilot`, `stage1` (captions, 27 GB zip),
  `stage2` (COCO, 19 GB zip) or `both`.
- `COMMAND`: runs from the repo root inside the job. For training, `--data_root` is added for you. Send outputs to
  `/gcs/frozen-vlm-benchmark/...`; that path is the bucket, so results survive the job.
- Do not put double quotes or backslashes in the command.

Then watch it:
```bash
bash scripts/cloud/launch_vertex.sh jobs              # list jobs and their state (PENDING, RUNNING, SUCCEEDED, FAILED)
bash scripts/cloud/launch_vertex.sh logs JOB_ID       # follow the log (JOB_ID is the number in the first column)
bash scripts/cloud/launch_vertex.sh cancel JOB_ID     # stop a job
```
A job that stays PENDING for many minutes usually means no GPU is available right now or the quota is missing. The job page in
the console (Vertex AI → Training → Custom jobs) shows the reason.

## 4. The run order

**Step 4.0, smoke test (5 min, a few cents).** Proves the image, the GPU and the bucket all work.
```bash
bash scripts/cloud/launch_vertex.sh run smoke none -- nvidia-smi
```
Then, the same way, check that the code can see the GPU:
```bash
bash scripts/cloud/launch_vertex.sh run smoke2 none -- python -c 'import torch; print(torch.cuda.get_device_name(0))'
```
(Use single quotes for the Python code, as above.)

**Step 4.1, overfit test on a GPU (about 10 min).** Must pass before any real run.
```bash
bash scripts/cloud/launch_vertex.sh run overfit pilot -- python scripts/train.py --config configs/overfit_stage2.json --output_dir /gcs/frozen-vlm-benchmark/runs/overfit
```
In the log, look for loss near 0 and the line `image shuffle check ... OK: the model uses the image`.

**Step 4.2, pilot (about 20-30 min including loading the models).** Measures what the real runs will cost.
```bash
bash scripts/cloud/launch_vertex.sh run pilot1 pilot -- python scripts/train.py --config configs/pilot_stage1.json --output_dir /gcs/frozen-vlm-benchmark/runs/pilot_stage1
```
Read `tokens_per_s` and `peak_mem_gb` from the log. Send them to me and we set the micro-batch size, decide on gradient checkpointing,
and compute the real GPU-hours for E1 (then update the cost table in the abstract). Then do one pilot **eval** to measure
evaluation time:
```bash
bash scripts/cloud/launch_vertex.sh run pilot_eval none -- python scripts/evaluate.py --run_dir /gcs/frozen-vlm-benchmark/runs/pilot_stage1 --name pilot --modes image gray no_image --max_samples 64 --out_dir /gcs/frozen-vlm-benchmark/eval
```

**Step 4.3, the real runs.** Copy the exact commands from `TRAINING.md` section 3, adding `run NAME DATA --` in front and
`--output_dir /gcs/frozen-vlm-benchmark/runs/<name>` at the end. For example E1:
```bash
bash scripts/cloud/launch_vertex.sh run e1 stage1 -- python scripts/train.py --config configs/e1_stage1.json --output_dir /gcs/frozen-vlm-benchmark/runs/e1_stage1
```
Stage 2 jobs use `DATA=stage2` and `--init_projector /gcs/frozen-vlm-benchmark/runs/e1_stage1/final/projector.pt`. Evaluation uses
`DATA=none` and `--run_dir /gcs/frozen-vlm-benchmark/runs/<name>` (see `EVAL.md`).

**Every job downloads Qwen and SigLIP from Hugging Face at start**, which takes a couple of minutes.

## 5. Cost control

- Cost is the job's running time x the machine price. The abstract assumed about $4.43 per hour for `a2-highgpu-1g` including the
  Vertex fee; check the current price on the [pricing page](https://cloud.google.com/vertex-ai/pricing) before the big runs.
- Staging data costs GPU time too: Stage 1 copies and extracts 27 GB before training starts (plan for 10-20 minutes). Run `E1` as one
  job, and do not submit many small Stage 1 jobs.
- After every session run `bash scripts/cloud/launch_vertex.sh jobs` and make sure nothing is RUNNING that should not be.
- **Spot VMs** (`SPOT=1 bash scripts/cloud/launch_vertex.sh run ...`) are much cheaper but can be stopped at any time. Training checkpoints
  every 200 steps or 15 minutes into `output_dir`, so re-submitting the same command resumes. But a restart repeats the data staging, so
  use Spot only for the long runs once the pipeline is proven, and keep the first E1 on-demand.
- Log every run in a cost sheet: the job, wall hours (`final/summary.json` has `wall_hours_this_session` and
  `est_cost_usd_this_session`) and the dollars.

## 6. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `RESOURCE_EXHAUSTED ... quota` | No GPU quota yet (section 0). Try `MACHINE=g2-standard-8 ACCEL=NVIDIA_L4` for small tests. |
| Job stays PENDING | No capacity right now, or quota. Cancel and retry, or try later. |
| `Permission denied` on `/gcs/...` | Give the service account in the error Storage Object Admin on the bucket. |
| `CUDA out of memory` | Lower `--micro_batch_size` (raise `--grad_accum` to keep the effective batch), add `--gradient_checkpointing true`. |
| `FileNotFoundError` for an image | Wrong `DATA` for the stage (Stage 1 needs `stage1`, Stage 2 needs `stage2`), or the zip extracted to a different folder; the entry script prints the layout it expects. |
| Hugging Face download errors | Re-submit; if it persists, the job has no internet (tell me) and we pre-stage the models in the bucket. |
| Job FAILED right after starting | Open the job's log. The first lines show whether it is the code unpacking, the data staging or the command. |
| Spot job restarted | Expected. It resumes from the last checkpoint; check `train_log.jsonl` for the step it continued from. |

## 7. If the quota is late: reduced scope

Following the abstract's own fallback order (section 10), cut in this order and say so in the report:
1. drop the rank-4 ablation run (keep rank 64 if only one is affordable);
2. train Stage 1 on a 200K-caption subset (`--subset_size 200000`);
3. pool to 81 tokens (`--pool_grid 9`; this needs the token-count constants in `src/config.py` and `src/data/formatting.py` to change too, so ask first);
4. as a last resort, a single seed for E2 and E3 (report it as a limitation, since the abstract promised two).

E0 and the E2 vs E3 comparison on both benchmarks are the last things to cut.
