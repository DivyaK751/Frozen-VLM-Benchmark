# Small VLM Project: Master Context

Paste this file at the start of a new chat, together with `docs/NOTES.md`. Last updated: Nov 25, 2026 (Tasks 1-2 complete, Task 3 next, plan shrunk).

## 0. Working rule: finish first, polish later
Get the whole required path working before adding anything: data -> train -> E1 -> E2 -> evaluation -> report. For every task, do the minimum that makes its test pass. No extra smoke scripts, optional tests, "break-it" variants, or side checks unless a real bug shows up. Anything cut lives in section 13 (Deferred) and is added back only after the full pipeline works.

## 1. Project in brief
- **Course:** COMS 4995, Foundation Model Engineering, Fall 2026. Group project, Team 14, Idea #13 (Building a small vision-language model).
- **Title:** Building a Small Vision-Language Model: How Far Can a Frozen Encoder and a Trained Connector Go?
- **Members:** Divyasri Kadekar (dk3480), Lakshya Khandelwal (lk3068), Ritwik Sharan (rs4764), Vinod Walia (vw2317).
- **Question:** With a frozen SigLIP 2 So400m encoder and a 0.5B Qwen2.5 LLM, how much does adding LoRA improve ScienceQA-IMG, TextVQA and POPE compared with training only the MLP projector, and on which object types does each model hallucinate?
- **Hypotheses:**
  - H1: Projector + LoRA beats projector-only on all three benchmarks, with the largest gain on ScienceQA. *(Confounded by answer format, see section 7.)*
  - H2: Projector-only hallucinates more (lower POPE F1, higher "yes" rate), with the largest gap on the adversarial split.
  - H3: Blanking the image lowers TextVQA and POPE far more than ScienceQA, and the drop is smaller for projector-only (more reliance on language priors).

## 2. Architecture (built and verified)
Image (384 px) -> frozen SigLIP 2 So400m (`google/siglip2-so400m-patch14-384`) -> pooling (27x27 -> 14x14 = 196 tokens) -> 2-layer MLP projector with GELU, 1152 -> 896 (trained) -> Qwen2.5-0.5B-Instruct, which reads the visual tokens then the text prompt.

**Pooling (decided Nov 22):** `adaptive_avg_pool2d` to 14x14, applied before the projector. Keeps every patch (`avg_pool2d(kernel=2)` gives 13x13 but drops the last row and column). Caveat for the report: adaptive windows are uneven (2 or 3 patches wide, overlapping). E1 and E3 inherit this. Output size is a function argument, so other token counts are a config change.

**Connector (`src/small_vlm/connector.py`, Task 1):** `pool_visual_tokens(tokens, grid_size=27, output_size=14)`; `VisualProjector(in_dim=1152, out_dim=896)` with layers `fc1`/`act`/`fc2`, 1,836,800 parameters. 5 tests pass.

**Model (`src/small_vlm/model.py`, Task 2):**
- `SmallVLM(encoder, llm, projector, image_token_id, grid_size=27, pool_size=14)`; `tokens_per_image = pool_size**2`.
- `forward(input_ids, pixel_values, attention_mask=None, labels=None)` returns the HF causal-LM output (loss, logits). One image per row. Each row must contain exactly `tokens_per_image` ids equal to `image_token_id`, else `ValueError` (checked before the encoder runs). The model overwrites those embeddings with the projected image vectors and calls the LLM with `inputs_embeds=`.
- `encode_images(pixel_values)`: `[B,3,H,W] -> [B, tokens_per_image, llm_hidden]`.
- `build_small_vlm(*, pool_size=14, dtype=torch.float32)` returns `(model, tokenizer)` and loads the real checkpoints. Raises `RuntimeError` on missing SigLIP vision keys.
- Freezing: encoder `requires_grad_(False)`, `eval()`, `no_grad` only around its call; `SmallVLM.train()` keeps the encoder in eval. LLM: `requires_grad_(False)` only (no `no_grad`, so gradients flow through it), not forced to eval. Only the projector trains: 1,836,800 parameters.
- Tests in `tests/test_model.py` cover shapes, splice order (two images), per-row count check, mask/label passthrough, frozen modules, dtype flow, and a tiny random SigLIP + Qwen2 forward/backward.

**Verified on the pinned versions (transformers 5.24.0):**
- Vision class is `SiglipVisionModel` (checkpoint `model_type: "siglip"`), `forward(pixel_values)`; `last_hidden_state` is `[B, 729, 1152]` (after `post_layernorm`, no CLS). `pooler_output` is computed but unused. Output matches the full `SiglipModel`.
- Image processor (`AutoImageProcessor` of the checkpoint): resizes to 384x384 by **squashing** (aspect ratio not preserved), no center crop, mean = std = 0.5, range [-1, 1].
- Qwen: 494.0M params, hidden 896, tied embeddings (one freeze covers the output head). `len(tokenizer)` = 151,665, embedding rows = 151,936. Image placeholder = Qwen's existing image-pad token, id 151655 (single token), no resize.
- `from_pretrained(..., dtype=...)` works in 5.24.0.
- dtypes on GPU: encoder and LLM bf16, projector fp32. Flow: encode (bf16) -> pool (bf16) -> cast fp32 -> projector -> cast to LLM embedding dtype -> splice. CPU tests are fp32. The bf16 path on real models is first exercised in the Task 5 pilot.

## 3. Repository and environment

### Repo and branches
- **Shared repo (owned by Divya):** `Frozen-VLM-Benchmark` (private, I am a collaborator).
- **My branch:** `no-lora`. Never commit directly to `main`; merge through a pull request.
- **Old personal repo `small-vlm`:** backup only. Do not edit or delete until everything is confirmed in the shared repo.
- **Local path (Windows, Git Bash):** `C:\Columbia\Fall_Sem_26\Foundation Model Engineering\Frozen-VLM-Benchmark`

### Layout
```
Frozen-VLM-Benchmark/
  README.md  pyproject.toml  requirements.txt  runs.csv  .gitignore
  src/small_vlm/   # connector.py, model.py done; data.py next
  configs/  scripts/  tests/  eval/  docs/
```
Imports: `from small_vlm.connector import pool_visual_tokens, VisualProjector`.

### Environment (identical on every machine)
- Python 3.11 (laptop 3.11.8), `venv` in `.venv/`, `pip install -r requirements.txt`, then `pip install -e .`.
- Pinned: torch 2.15.0 (CPU build on laptop), torchvision 0.29.2, transformers 5.24.0, accelerate 1.16.0, peft 0.22.0, datasets 5.2.0, numpy 2.4.6, pandas 3.1.1, scipy 1.17.1, pillow 13.0.0, pytest 9.2.0, google-cloud-storage 3.17.1, plus tqdm, pyyaml and transitive dependencies.
- Adding a package: install it, run `pip freeze --exclude-editable > requirements.txt`, tell the team.
- Linux VM: `python3.11 -m venv .venv && source .venv/bin/activate`, then the same pip commands. If `pip install -r` fails on a Windows-only package, remove that line and note it. Check `torch.cuda.is_available()` on the L4 VM before any A100 spend.

## 4. Training plan
| Run | What trains | Data | Notes |
|---|---|---|---|
| E1 (alignment) | Projector only | LLaVA-Pretrain (558K pairs available) | Encoder and LLM frozen. Starting LR ~1e-3, batch ~256 (confirm on pilot). **Size decided after the pilot** (see below) |
| E2 | Projector only | LLaVA-Instruct-150K | Starts from E1. **My half** |
| E3 | Projector + LoRA (rank 16, LLM attention + MLP) | LLaVA-Instruct-150K | Starts from the same E1. LRs ~2e-4 LoRA, ~2e-5 projector. **Teammates' half** |

**E1 size (shrink):** if the full 558K x 1 epoch does not fit comfortably after the pilot, use a fixed seeded random subset (for example ~200K pairs), 1 epoch. Hard date: E1 launches by Nov 29. If not, cut E1 further. Tell the LoRA half whatever is chosen, since E3 starts from this checkpoint.

E2 and E3 must share the same E1 checkpoint, data order, seed, step count, chat template, loss masking and evaluation code.

Evaluation data: ScienceQA (image subset, test), TextVQA (val), POPE (random, popular, adversarial).

## 5. Team split and my scope
1. **Without LoRA, end to end** = mine: data pipeline, connector, training loop, E1, E2, evaluation harness, baselines, analysis for E2.
2. **With LoRA, end to end** = teammates: E3 and its analysis.

Agreements needed with the LoRA half (only these for now):
- Pull requests into `main`, nobody commits to `main` directly.
- Tell them: pooling (196 tokens, adaptive), the `SmallVLM`/connector interfaces, the squash preprocessing, and `build_small_vlm` (E3 reuses it).
- I push E1 to Cloud Storage with a short README (pooling, LR, steps, subset, commit hash), because E3 depends on it.
- One shared codebase (LoRA is a config flag they add) and one shared evaluation harness with per-sample outputs saved.
- README wording, contribution statement: handle after results exist (Dec 10+). The contribution statement must match what each person did.

Track B (personal extras) is deferred, see section 13.

## 6. Infrastructure state (done)
- GCP project, billing, $100 credit (a second coupon may raise it to $200; then raise the budget alert target). Alerts only send email and do not stop spending.
- APIs enabled: Agent Platform (Vertex AI), Compute Engine.
- Quota (us-central1): 8 preemptible A100 40 GB for custom model training (Agent Platform, the main one); 1 L4 and 1 preemptible L4 on Compute Engine (dev VM); custom-training L4 and preemptible L4 approved.
- Bucket `small_vlm`, us-central1, Standard, not public.
- **Region rule: everything stays in us-central1.**

## 7. Evaluation design
**Metrics:** ScienceQA-IMG multiple-choice accuracy; TextVQA (val) VQA accuracy; POPE (random, popular, adversarial) F1, accuracy and "yes" rate, with a per-object-type breakdown (parse the object from the question).

**Baselines:** text-only Qwen (no image), blank-image input (needed for H3), E2 vs E3, published LLaVA-1.5-7B and TinyLLaVA scores (cited, not re-run), POPE chance (50%).

**Professor's feedback:** LLaVA-Instruct-150K has no OCR or multiple-choice science data, so TextVQA may sit near the floor and ScienceQA may mostly reflect answer format.
- For ScienceQA and TextVQA, report three numbers: valid-answer rate, accuracy among valid answers, overall accuracy.
- H1 is stated as format-confounded.
- A near-floor TextVQA result is valid, with the text-only baseline as reference.

**Statistics:** single seed. Bootstrap confidence intervals, McNemar's test for E2 vs E3 (per-sample outputs saved for every model). Hand-inspect ~30 outputs and categorize failures (format, OCR miss, hallucinated object, language-prior answer).

**What the benchmarks miss:** POPE checks only yes/no object existence; ScienceQA can be partly solved from text priors.

**Harness decision (Nov 25):** own minimal harness (no lmms-eval), to avoid dependency friction with the pinned versions.

## 8. Cost plan
- Prices (check the current page): ~$3.67/hr on-demand and ~$1.80/hr spot for a2-highgpu-1g (course sheet); the abstract used ~$4.43/hr on-demand on Vertex; L4 ~$0.71/hr.
- Estimates (unmeasured until the 10K pilot): E1 full 6-9 GPU-h (scales roughly with the number of samples, so a subset is proportionally cheaper), E2 5-8, E3 6-8 (teammates), evaluation ~4 models 8-12, dev and pilots 10-15 (L4).
- Rules: debug on Colab or the L4 VM; A100 only for the pilot and real runs; stop idle VMs (a forgotten A100 costs ~$88/day); keep a 20% reserve; log every kept run in `runs.csv`.
- Checkpoints are small because only the projector trains: 1,836,800 fp32 params ~ 7.3 MB, plus two AdamW states, about 22 MB total (derived). So checkpoint often (every ~15 min) to Cloud Storage; resume is simple.

## 9. Course deadlines
| Date | Item |
|---|---|
| Dec 8 | Slides due (frozen after) |
| Dec 9 | Presentation in class (10 min + 3 min questions; every member speaks) |
| Dec 18 | Final report (6-8 pages, NeurIPS template) + frozen GitHub repo with README. Also: contribution statement, AI-use disclosure, confidential peer evaluation |

No late days on any project deliverable. Running out of credits does not extend deadlines.

## 10. My calendar (revised Nov 25)
| Dates | Goal |
|---|---|
| Nov 25 | Task 2 done. Commit and push |
| Nov 26 | Task 3: data |
| Nov 27 | Task 4: training loop |
| Nov 28 | Task 5: overfit 100, 10K pilot, decide E1 size, **launch E1 (hard date Nov 29)** |
| Nov 28-30 | E1 trains. Build the evaluation harness meanwhile (text-only baseline first). Push the E1 checkpoint for the LoRA half. Draft Method and Experimental Setup |
| Dec 1-3 | E2 from E1. If E1 finishes early, start E2 early |
| Dec 3-6 | Score all models, blank-image run, POPE breakdown, hand-check ~30 outputs |
| Dec 7 | Plots, tables, slides |
| Dec 8 / 9 | Slides due / presentation |
| Dec 10-18 | Report, README, repo freeze, disclosures, peer evaluation |

## 11. How we work together
- I write all the code myself; Claude guides and reviews. Each task has a goal, hints and a test that must pass. I search docs first and ask only when stuck.
- Purpose first: at the start of each task, Claude explains the purpose in plain English with a diagram, and I confirm before moving on.
- One task per message. Paste only the one function and the exact error.
- New chat per phase: paste this file plus `docs/NOTES.md`.
- NOTES.md is updated at the end of each task (short, batched). Claude provides ready-to-paste entries.
- Expected values in tests come from a formula or from the code, not from memory.
- Shrink rule (section 0) applies to every task.

## 12. Task list
**Setup (done Nov 21-22).**

1. **Pooling and projector** (`connector.py`) **DONE Nov 22.**
2. **Model assembly** (`model.py`, `tests/test_model.py`) **DONE Nov 25.**
3. **Data** (`src/small_vlm/data.py`) **<- next**
   - Dataset for LLaVA-Pretrain and LLaVA-Instruct (check the real JSON format first: single-turn captions vs multi-turn conversations, image path format).
   - Preprocess images with the checkpoint's image processor (384x384 squash, mean = std = 0.5).
   - Qwen chat template; one image marker expanded to exactly 196 image-pad ids per image.
   - Loss masking: label -100 on image slots, system/user text and padding; labels only on assistant tokens (all assistant turns for multi-turn).
   - Collate with padding and an attention mask.
   - Tests (keep to these): exactly 196 placeholders per image; labels are -100 on image and prompt positions and equal `input_ids` on assistant tokens; decode one batch and inspect by eye.
   - Develop against a handful of local samples. The full data download happens in Task 5, onto the VM or bucket in us-central1.
4. **Training loop** (`scripts/train.py`)
   - bf16 encoder and LLM, fp32 projector, AdamW on the projector only, warmup + cosine LR, a fixed seed, deterministic data order from the seed.
   - Log loss, tokens/s and peak memory.
   - Checkpoint (projector + optimizer + step) every ~15 min to local disk and `gs://small_vlm`; resume by loading it and skipping to the step.
   - Gradient checkpointing is off by default; turn it on only if the pilot runs out of memory (then read the pinned `GradientCheckpointingLayer` source first).
   - One config file per run. Do not implement LoRA (the LoRA half adds it as a config key).
5. **Validate:** overfit 100 examples to near-zero loss (L4 VM or Colab); run the 10K pilot on a spot A100. This is also the first bf16 real-model run and the first `torch.cuda.is_available()` check. Record GPU-hours, memory, tokens/s in `runs.csv`, then decide the E1 size and launch.
6. **E1** on Vertex (spot). Push the checkpoint plus short README to `small_vlm`. Then **E2** from the same checkpoint.
7. **Evaluation harness** (own, minimal): text-only baseline first, then the VLM; per-sample outputs (JSONL); valid-answer rates; POPE per-object breakdown; blank-image run.
8. **Analysis and deliverables:** bootstrap CIs, McNemar E2 vs E3, ~30 outputs by hand, plots and tables, slides (Dec 8), report (Dec 18), README with the command for each table and figure, repo freeze, contribution statement, AI-use disclosure, peer evaluation.

## 13. Deferred (add back only after the full pipeline works)
- Track B: run E3 myself, E4 (small ScienceQA/TextVQA-train slice in Stage 2), 81/729-token variants, shuffled-image control, video-frame experiments, portfolio write-up.
- Format-free ScienceQA score (pick the option letter with the highest logit) and a matching H1 prediction. First thing to add if there is time.
- Latency and throughput benchmarking in the evaluation (training logs already give tokens/s).
- lmms-eval wrapper.
- Maintaining `scripts/smoke_test_siglip.py` (not committed; the results it produced are recorded in NOTES.md).
- Comparing bf16 vs fp32 pooling.
- Extra connector tests, request for on-demand A100 quota, archiving the old `small-vlm` repo (after Dec 18).
- README "Reproducing results" table (fill in during Dec 10-18).

## 14. Prompt to start the next chat
> I'm working on a COMS 4995 group project (small VLM, Team 14). Attached are `small-vlm-project.md` and `docs/NOTES.md`. Treat them as the source of truth. I'm writing the code myself and want guidance, not full solutions. Tasks 1 and 2 are done and pushed. We are on Task 3: data in `src/small_vlm/data.py`. The plan is shrunk: do the minimum that passes each task's tests, no extra checks. First explain the purpose of Task 3 in plain English with a diagram, and wait for me to confirm before the goal, hints and test. Keep replies short, one step at a time, and give me the updated NOTES.md entries at the end of the task.
