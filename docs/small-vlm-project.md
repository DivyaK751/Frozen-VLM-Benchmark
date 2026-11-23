# Small VLM Project: Master Context

Paste this file at the start of a new chat, together with `docs/NOTES.md`. Last updated: Nov 22, 2026 (Task 1 complete, Task 2 next).

## 1. Project in brief
- **Course:** COMS 4995, Foundation Model Engineering, Fall 2026. Group project, Team 14, Idea #13 (Building a small vision-language model).
- **Title:** Building a Small Vision-Language Model: How Far Can a Frozen Encoder and a Trained Connector Go?
- **Members:** Divyasri Kadekar (dk3480), Lakshya Khandelwal (lk3068), Ritwik Sharan (rs4764), Vinod Walia (vw2317).
- **Question:** With a frozen SigLIP 2 So400m encoder and a 0.5B Qwen2.5 LLM, how much does adding LoRA improve ScienceQA-IMG, TextVQA and POPE compared with training only the MLP projector, and on which object types does each model hallucinate?
- **Hypotheses:**
  - H1: Projector + LoRA beats projector-only on all three benchmarks, with the largest gain on ScienceQA. *(Confounded by answer format, see section 7.)*
  - H2: Projector-only hallucinates more (lower POPE F1, higher "yes" rate), with the largest gap on the adversarial split.
  - H3: Blanking the image lowers TextVQA and POPE far more than ScienceQA, and the drop is smaller for projector-only (more reliance on language priors).

## 2. Architecture
Image (384 px) -> frozen SigLIP 2 So400m (`google/siglip2-so400m-patch14-384`) -> **pooling (27x27 -> 14x14, 196 tokens)** -> 2-layer MLP projector with GELU, 1152 -> 896 (trained) -> Qwen2.5-0.5B-Instruct, which reads the visual tokens then the text prompt and generates the answer.

**Verified on Colab (Colab's own package versions, not the pinned ones):**
- Vision encoder output: `[1, 729, 1152]` (27x27 grid). Vision tower is 428.2M params.
- LLM: 494.0M params, hidden size 896.
- Neither model is gated (no Hugging Face token needed).
- **To do:** re-run this shape check inside the pinned venv before the first real run (optional download of ~5 GB into the Hugging Face cache). Task 2 will need this anyway.

**Decided (Nov 22): pooling = `adaptive_avg_pool2d` to 14x14 = 196 visual tokens per image, applied before the projector.**
- Reason: keeps every patch. `avg_pool2d(kernel=2)` would give 13x13 = 169 tokens but drops the last row and column (~7% of patches, always the bottom/right edge), a systematic crop that could confound TextVQA and POPE. Cost: ~16% more tokens than 169.
- Caveat for the report: adaptive windows are uneven (2 or 3 patches wide, overlapping), so some output tokens average more patches than others.
- Pooling first means the projector runs on 196 tokens instead of 729 (cheaper).
- E1 and E3 inherit this choice; changing it later means retraining E1.
- Output size is a function argument, so the 81- and 729-token variants (Track B) are a config change.

**Connector interface (done, in `src/small_vlm/connector.py`):**
- `pool_visual_tokens(tokens, grid_size=27, output_size=14)`: `[B, grid_size**2, C] -> [B, output_size**2, C]`. Asserts the token count equals `grid_size**2`.
- `VisualProjector(in_dim=1152, out_dim=896)`: `fc1` (in_dim -> out_dim), `act` (GELU), `fc2` (out_dim -> out_dim). `[B, N, in_dim] -> [B, N, out_dim]`. **1,836,800 parameters.** Layer names `fc1`/`act`/`fc2` give readable `state_dict` keys for the E1 checkpoint.
- Tests in `tests/test_connector.py` (5 pass): pooling shape, constant preserved, identity pooling (`output_size=27`) preserves values and positions, connector output shape `[2, 196, 896]`, parameter count 1,836,800.

## 3. Repository and environment

### Repo and branches
- **Shared repo (owned by Divya):** `Frozen-VLM-Benchmark` (private, I am a collaborator).
- **My branch:** `no-lora` (the without-LoRA half). Never commit directly to `main`; merge through a pull request.
- **Old personal repo `small-vlm`:** kept as a backup until everything is confirmed in the shared repo. Do not edit it. Do not delete it yet.
- **Local path (Windows, Git Bash):** `C:\Columbia\Fall_Sem_26\Foundation Model Engineering\Frozen-VLM-Benchmark`

### Layout
```
Frozen-VLM-Benchmark/
  README.md             # overview, setup, workflow (draft on no-lora, pending team agreement)
  pyproject.toml        # package metadata, requires Python >=3.11,<3.12, src layout
  requirements.txt      # exact pinned versions (pip freeze, 74 lines)
  runs.csv              # one row per kept run: GPU, hours, peak memory, dollars, commit
  src/small_vlm/        # the importable package (connector.py done; model.py, data.py next)
  configs/  scripts/  tests/  eval/  docs/
  .gitignore            # excludes .venv, data, checkpoints, weights, credentials
```
Imports look like `from small_vlm.connector import pool_visual_tokens, VisualProjector`. **Paths in the task list below use this layout.**

### Environment (identical on every machine)
- **Python 3.11** (laptop: 3.11.8). Same minor version on laptop, VM and Vertex container. Record the exact patch version in `docs/NOTES.md`.
- **Virtual environment:** `venv` in `.venv/` inside the repo (not conda). `venv` reuses an installed Python, so install Python 3.11 first.
- **Editable install:** `pip install -e .` puts a pointer to `src/small_vlm` in site-packages, so `import small_vlm` works from any folder and edits take effect immediately. Verified: it resolves to the repo's `src/small_vlm/__init__.py`.
- **Pinned versions** (from `requirements.txt`): torch 2.15.0 (CPU build on the laptop), torchvision 0.29.2, transformers 5.24.0 (new major version, differs from Colab), accelerate 1.16.0, peft 0.22.0, datasets 5.2.0, numpy 2.4.6, pandas 3.1.1, scipy 1.17.1, pillow 13.0.0, pytest 9.2.0, google-cloud-storage 3.17.1, plus tqdm, pyyaml and transitive dependencies.
- **Verified:** imports of torch, transformers, peft, datasets, accelerate succeed with these versions; `pytest` runs (`python -m pytest tests/test_connector.py` passes on Python 3.11.8).

### Setup for a teammate (Windows, Git Bash)
```bash
git clone <repo-link>
cd Frozen-VLM-Benchmark
git checkout -b <your-branch>        # or git checkout no-lora to join mine
py -3.11 -m venv .venv
source .venv/Scripts/activate        # prompt should show (.venv)
python --version                     # expect 3.11.x
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -c "import small_vlm; print(small_vlm.__file__)"
pytest
```
**Linux (VM):** `python3.11 -m venv .venv && source .venv/bin/activate`, then the same pip commands.

### Known risks
- `requirements.txt` was frozen on Windows. A Windows-only package could fail on Linux: if `pip install -r` errors on one, remove that line and note it here.
- On a Linux GPU VM, check `torch.cuda.is_available()` on the cheap L4 VM before any A100 spend. The torch version stays the same; only the build (CPU vs. CUDA) differs.
- When adding a package: install it in the venv, then regenerate with `pip freeze --exclude-editable > requirements.txt`, and tell the team.
- transformers 5.24.0 is a new major version and differs from the Colab check. Read the docs for the pinned version before relying on remembered APIs (especially for SigLIP 2 loading and chat templates).

## 4. Training plan
| Run | What trains | Data | Notes |
|---|---|---|---|
| E1 (Stage 1, alignment) | Projector only | LLaVA-Pretrain, 558K pairs, 1 epoch | Encoder and LLM frozen. Starting LR ~1e-3, batch ~256 (confirm on pilot). |
| E2 | Projector only | LLaVA-Instruct-150K | Starts from E1. **My half.** |
| E3 | Projector + LoRA (rank 16, attention + MLP layers of the LLM) | LLaVA-Instruct-150K | Starts from the same E1. Starting LRs: ~2e-4 LoRA, ~2e-5 projector. **Teammates' half.** |

E2 and E3 must share the same E1 checkpoint, data order, seed, step count, chat template, loss masking and evaluation code.

**Data and licenses**
- `liuhaotian/LLaVA-Pretrain` (558K pairs)
- `liuhaotian/LLaVA-Instruct-150K` (COCO images)
- Evaluation: ScienceQA (image subset, test), TextVQA (val), POPE (random, popular, adversarial).

## 5. Team split and my scope
The team divided the project in two halves:
1. **Without LoRA, end to end** = **mine**: data pipeline, connector, training loop, E1, E2, the evaluation harness, baselines, and analysis for E2.
2. **With LoRA, end to end** = teammates: E3 and its analysis.

Agreements to make with the LoRA half (not yet done):
- Branch names and merging: pull requests into `main`, nobody commits to `main` directly.
- I produce E1 first and push it to Cloud Storage with a short README (pooling choice, LR, steps, commit hash), because E3 depends on it.
- One shared codebase. LoRA is a config flag, not a separate implementation.
- One shared evaluation harness. Per-sample outputs saved so paired tests work across E2 and E3.
- Same Python 3.11 and the same `requirements.txt` for everyone.
- README ownership and final wording: my draft lives on `no-lora`; agree with Divya before merging so `main` has no conflicting edits. The course requires setup steps, the command for each table and figure, and a note on which results need a GPU.
- Contribution statement in the final report must match what each person actually did.
- **Tell the LoRA half the pooling decision (196 tokens, adaptive) and the connector interface, since E3 inherits both.**

**Track B (personal, after the course requirement is covered):** run E3 myself, the text-only and blank-image baselines, E4, 81/729-token variants, shuffled-image control, video-frame experiments. Use my own credits and be honest about attribution (team work vs. my own).

**Portfolio angle (image/video generation roles):** a VLM is on the understanding side. Bridges: checking whether generated images match prompts (POPE-style hallucination testing), recaptioning and data curation, video-frame questions, rigorous evaluation. Pair with the diffusion projects already in progress.

## 6. Infrastructure state (done)
- **GCP project created**, billing linked, **$100 credit applied** (a second coupon may raise this to $200).
- **Budget alert:** target $100 for now, multiple thresholds. **To do: raise the target to $200 when the second coupon is applied.** Alerts only send email; they do not stop spending.
- **APIs enabled:** Agent Platform API (the new name for Vertex AI), Compute Engine.
- **Quota (us-central1):**
  - Custom model training preemptible Nvidia A100 GPUs (Agent Platform): **8** (spot A100 40 GB, the main one).
  - Compute Engine NVIDIA L4 GPUs: 1, preemptible L4: 1 (dev VM).
  - Custom model training L4 and L4 preemptible (Agent Platform): requested, **approved**.
  - Optional later: request 1 for Custom model training Nvidia A100 GPUs (on-demand) as a backup for preempted runs.
  - Ignore all "serving", "restricted image training" and "committed" rows.
- **Bucket:** `small_vlm`, us-central1 (Iowa), Standard, not public, soft delete on.
- **Region rule: everything stays in us-central1** (Vertex jobs, VMs in zones like us-central1-a/b/c/f, any other storage). Cross-region traffic costs money; downloading to a laptop costs egress.

## 7. Evaluation design
**Metrics**
- ScienceQA-IMG: multiple-choice accuracy.
- TextVQA (val): VQA accuracy.
- POPE (random, popular, adversarial): F1, accuracy, "yes" rate, plus a per-object-type breakdown (parse the object from the question).

**Baselines:** text-only Qwen (no image), blank-image input, E2 vs. E3, published LLaVA-1.5-7B and TinyLLaVA scores (not re-run), POPE chance (50%).

**Professor's feedback:** LLaVA-Instruct-150K has no OCR or multiple-choice science data, so TextVQA may sit near the floor and ScienceQA may mostly reflect answer format. Report format-failure rates and treat a near-floor result as legitimate.

**My response in the design**
- For ScienceQA and TextVQA, report three numbers: valid-answer rate, accuracy among valid answers, overall accuracy.
- Add a format-free ScienceQA score (pick the option letter with the highest logit).
- Rewrite H1 as format-confounded and add a prediction for the format-free score.
- Pre-commit that a near-floor TextVQA result is valid, using the text-only baseline as reference.
- Optional E4: add a small slice of ScienceQA-train/TextVQA-train to Stage 2 to test the professor's explanation.

**Statistics:** single seed, so use bootstrap confidence intervals and McNemar's test for E2 vs. E3. Save per-sample outputs for everything. Hand-inspect about 50 outputs and categorize failures (format, OCR miss, hallucinated object, language-prior answer).

**What the benchmarks miss:** POPE checks only yes/no object existence (not attributes, counts, relations, or free-form descriptions); ScienceQA can be partly solved from text priors.

## 8. Cost plan
- **Prices (check the current pricing page):** abstract used ~$4.43/hr for an on-demand A100 on Vertex (includes management fee); the course sheet lists ~$3.67 on-demand and ~$1.80 spot for a2-highgpu-1g; L4 ~$0.71/hr on-demand. Spot with checkpoints can roughly halve cost.
- **Estimates (unmeasured until the 10K pilot):** E1 6-9 GPU-h, E2 5-8, E3 6-8, evaluation of ~4 models 8-12, dev and pilots 10-15 (L4).
  - Track A (E1 + E2 + pilots): ~$60-85 on-demand, ~$30-45 spot.
  - Full core plan: ~$125-175 on-demand, ~$65-95 spot.
- **Rules:** debug on Colab or the L4 VM, use the A100 only for pilots and real runs; spot with checkpoints to Cloud Storage at least every 30 min (test resume first); stop idle VMs (a forgotten A100 costs ~$88/day); keep a 20% reserve; log every kept run in `runs.csv` (GPU, wall-clock, peak memory, dollars, commit).

## 9. Course deadlines
| Date | Item |
|---|---|
| Week of Nov 18 | Optional TA checkpoint (15 min, ungraded). That week has passed; skip it unless the TA offers another slot. |
| Dec 8 | Slides due (frozen after) |
| Dec 9 | Presentation in class (10 min + 3 min questions; every member speaks) |
| Dec 18 | Final report (6-8 pages, NeurIPS template) + frozen GitHub repo with README. Also: contribution statement, AI-use disclosure, confidential peer evaluation. |

No late days on any project deliverable. Running out of credits does not extend deadlines.

## 10. My calendar (one-person-friendly)
| Dates | Goal |
|---|---|
| Nov 22-25 | Tasks 1-4: connector (**Task 1 done Nov 22**), model assembly, data code, training loop, shape tests |
| Nov 26-27 | Overfit 100 examples, 10K pilot (real GPU-hours), launch E1 on spot |
| Nov 27-30 | E1 trains; build the evaluation harness meanwhile; push E1 checkpoint for the LoRA half |
| Dec 1-3 | E2 from E1; validate the harness on a pilot checkpoint |
| Dec 3-6 | Score all models, blank-image run, POPE breakdowns, manual check of ~50 outputs |
| Dec 7 | Plots, tables, slides (due Dec 8) |
| Dec 9 | Presentation |
| Dec 10-18 | Report, repo freeze, README, disclosures, peer evaluation |

Draft Method and Experimental Setup while E1 trains.

## 11. How we work together
- **I write all the code myself; Claude guides and reviews.** Each task comes with a goal, hints and a test that must pass. I search docs first and ask only when stuck.
- **Purpose first:** at the start of each task, Claude explains the purpose in plain English with a diagram before any implementation. No jumping ahead; one step at a time, and I confirm before moving on.
- **One task per message.** Paste only the one function and the exact error, not whole files.
- **Start a new chat per phase** and paste this file plus `docs/NOTES.md`.
- **NOTES.md is updated at the end of each task** (batched, not after every step): Claude provides ready-to-paste entries and the updated files at that point. Decisions and surprises still get recorded then, with dates.
- Expected values in tests (such as parameter counts) are derived from a formula or from the code, not from memory. Lesson from Task 1: a hand-computed count was wrong, the test caught it.

## 12. Task list
**Setup (done Nov 21-22):** GCP, bucket, shared repo and branch, venv, pinned requirements, editable install.

1. **Pooling and projector** (`src/small_vlm/connector.py`) **DONE Nov 22**
   - `pool_visual_tokens` (adaptive pooling to 14x14, 196 tokens) and `VisualProjector` (1152 -> 896 -> 896, GELU, 1,836,800 parameters).
   - 5 tests pass in `tests/test_connector.py`.
   - To confirm: committed and pushed on `no-lora`.
2. **Model assembly** (`src/small_vlm/model.py`) **<- current task**
   - Goal: frozen SigLIP 2 encoder (bf16, `no_grad`) -> `pool_visual_tokens` -> `VisualProjector` -> splice visual tokens into the Qwen text embeddings at the image placeholder -> Qwen forward.
   - Unit test: the number of visual tokens spliced in equals the number of image placeholders times 196; output shapes are right on CPU with tiny dummy inputs.
   - **Decisions to settle at the start of Task 2 (record in NOTES.md):**
     - Image placeholder mechanism: a new special token (for example `<image>`, which means resizing the Qwen embeddings) or an existing token. Check the pinned transformers docs.
     - How one placeholder expands into 196 embedding slots (and how attention mask and labels stay aligned; visual positions get label -100 in Task 3).
     - dtype of the projector and LLM (bf16 vs. fp32 master weights for the projector).
     - How to load only the SigLIP 2 vision tower in the pinned transformers version (the full checkpoint also contains a text tower).
   - **Gotchas to keep in mind:**
     - Encoder: `torch.no_grad()` and `requires_grad_(False)`, `eval()` mode.
     - LLM is frozen in E1 and E2 but **must not be wrapped in `no_grad`**: gradients still have to flow back through it to reach the projector. Freeze with `requires_grad_(False)` only.
     - Only the projector's parameters should have `requires_grad=True`. Add a test that counts trainable parameters (expect 1,836,800).
     - Check the gradient-checkpointing plus frozen-embeddings behaviour in the docs before Task 4.
     - Task 2 needs the real models for a shape check; use the optional ~5 GB download, or test with tiny random-weight configs first.
3. **Data** (`src/small_vlm/data.py`): dataset, Qwen chat template, image placeholder, **loss masking on assistant tokens only**; decode a batch and inspect by eye.
4. **Training loop** (`scripts/train.py`): gradient checkpointing on the LLM, bf16, checkpoint to Cloud Storage every <=30 min with working resume, logging of loss, throughput, peak memory.
5. **Validate:** overfit 100 examples to near-zero loss, then run the 10K pilot and record real GPU-hours and memory.
6. **E1 on Vertex (spot)**, push the checkpoint plus README to `small_vlm`, then **E2** from the same checkpoint.
7. **Evaluation harness** (lmms-eval wrapper or own): text-only baseline first, then the VLM; per-sample outputs; format-failure rates; POPE breakdowns; blank-image run; latency, throughput and peak memory.
8. **Analysis and deliverables.**

## 13. Prompt to start the next chat
> I'm working on a COMS 4995 group project (small VLM, Team 14). Attached are `small-vlm-project.md` and `docs/NOTES.md`. Treat them as the source of truth. I'm writing the code myself and want guidance, not full solutions. Task 1 is done (pooling = adaptive 14x14, 196 tokens; `pool_visual_tokens` and `VisualProjector` in `src/small_vlm/connector.py`; 5 tests pass). We are on Task 2: model assembly in `src/small_vlm/model.py`. First explain the purpose of Task 2 in plain English with a diagram, and wait for me to confirm before the goal, hints and test. Keep replies short, one step at a time, and give me the updated NOTES.md entries at the end of the task.
