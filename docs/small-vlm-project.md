# Small VLM Project: Master Context

Paste this file at the start of a new chat, together with `docs/NOTES.md`. Last updated: Nov 19, 2026.

## 1. Project in brief
- **Course:** COMS 4995, Foundation Model Engineering, Fall 2026. Group project, Team 14, Idea #13 (Building a small vision-language model).
- **Title:** Building a Small Vision-Language Model: How Far Can a Frozen Encoder and a Trained Connector Go?
- **Members:** Divyasri Kadekar (dk3480), Lakshya Khandelwal (lk3068), Ritwik Sharan (rs4764), Vinod Walia (vw2317).
- **Question:** With a frozen SigLIP 2 So400m encoder and a 0.5B Qwen2.5 LLM, how much does adding LoRA improve ScienceQA-IMG, TextVQA and POPE compared with training only the MLP projector, and on which object types does each model hallucinate?
- **Hypotheses:**
  - H1: Projector + LoRA beats projector-only on all three benchmarks, with the largest gain on ScienceQA. *(Confounded by answer format, see section 6.)*
  - H2: Projector-only hallucinates more (lower POPE F1, higher "yes" rate), with the largest gap on the adversarial split.
  - H3: Blanking the image lowers TextVQA and POPE far more than ScienceQA, and the drop is smaller for projector-only (more reliance on language priors).

## 2. Architecture (from the abstract)
Image (384 px) -> frozen SigLIP 2 So400m (`google/siglip2-so400m-patch14-384`) -> pooling -> 2-layer MLP projector with GELU, 1152 -> 896 (trained) -> Qwen2.5-0.5B-Instruct, which reads the visual tokens then the text prompt and generates the answer.

**Verified on Colab:**
- Vision encoder output: `[1, 729, 1152]` (27x27 grid). Vision tower is 428.2M params.
- LLM: 494.0M params, hidden size 896.
- Neither model is gated (no Hugging Face token needed).

**Open decision: pooling.** 27 is odd, so 2x2 pooling is not clean.
- `avg_pool2d(kernel=2)` -> 13x13 = 169 tokens (drops the last row and column).
- `adaptive_avg_pool2d` to 14x14 -> 196 tokens (keeps everything).
- Pick one, record the reason in `docs/NOTES.md`, and write a shape test.

## 3. Training plan
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

## 4. Team split and my scope
The team divided the project in two halves:
1. **Without LoRA, end to end** = **mine**: data pipeline, connector, training loop, E1, E2, the evaluation harness, baselines, and analysis for E2.
2. **With LoRA, end to end** = teammates: E3 and its analysis.

Agreements to make with the LoRA half (not yet done):
- I produce E1 first and push it to Cloud Storage with a short README (pooling choice, LR, steps, commit hash), because E3 depends on it.
- One shared codebase. LoRA is a config flag, not a separate implementation.
- One shared evaluation harness. Per-sample outputs saved so paired tests work across E2 and E3.
- Contribution statement in the final report must match what each person actually did.

**Track B (personal, after the course requirement is covered):** run E3 myself, the text-only and blank-image baselines, E4, 81/729-token variants, shuffled-image control, video-frame experiments. Use my own credits and be honest about attribution (team work vs. my own).

**Portfolio angle (image/video generation roles):** a VLM is on the understanding side. Bridges: checking whether generated images match prompts (POPE-style hallucination testing), recaptioning and data curation, video-frame questions, rigorous evaluation. Pair with the diffusion projects already in progress.

## 5. Infrastructure state (done)
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
- **GitHub repo:** `small-vlm` (private), cloned locally. Structure: `src/`, `configs/`, `scripts/`, `tests/`, `eval/`, `docs/`, `runs.csv`. `.gitignore` excludes data, checkpoints, weights, credentials.

## 6. Evaluation design
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

## 7. Cost plan
- **Prices (check the current pricing page):** abstract used ~$4.43/hr for an on-demand A100 on Vertex (includes management fee); the course sheet lists ~$3.67 on-demand and ~$1.80 spot for a2-highgpu-1g; L4 ~$0.71/hr on-demand. Spot with checkpoints can roughly halve cost.
- **Estimates (unmeasured until the 10K pilot):** E1 6-9 GPU-h, E2 5-8, E3 6-8, evaluation of ~4 models 8-12, dev and pilots 10-15 (L4).
  - Track A (E1 + E2 + pilots): ~$60-85 on-demand, ~$30-45 spot.
  - Full core plan: ~$125-175 on-demand, ~$65-95 spot.
- **Rules:** debug on Colab or the L4 VM, use the A100 only for pilots and real runs; spot with checkpoints to Cloud Storage at least every 30 min (test resume first); stop idle VMs (a forgotten A100 costs ~$88/day); keep a 20% reserve; log every kept run in `runs.csv` (GPU, wall-clock, peak memory, dollars, commit).

## 8. Course deadlines
| Date | Item |
|---|---|
| Week of Nov 18 | Optional TA checkpoint (15 min, ungraded). Bring a one-page status note: what runs end to end, first results, credits spent, main open problem. |
| Dec 8 | Slides due (frozen after) |
| Dec 9 | Presentation in class (10 min + 3 min questions; every member speaks) |
| Dec 18 | Final report (6-8 pages, NeurIPS template) + frozen GitHub repo with README. Also: contribution statement, AI-use disclosure, confidential peer evaluation. |

No late days on any project deliverable. Running out of credits does not extend deadlines.

## 9. My calendar (one-person-friendly)
| Dates | Goal |
|---|---|
| Nov 19-20 | Connector, model assembly, data code, shape tests, overfit 100 examples, 10K pilot |
| Nov 21-24 | Launch E1 on spot; build the evaluation harness while it trains |
| Nov 25-29 | E2 (E3 by the LoRA half); validate the harness on a pilot checkpoint |
| Nov 30-Dec 3 | Score all models, blank-image run, POPE breakdowns, manual check of ~50 outputs |
| Dec 4-7 | Plots, tables, slides |
| Dec 9 | Presentation |
| Dec 10-18 | Report, repo freeze, README, disclosures, peer evaluation |

Draft Method and Experimental Setup while E1 trains.

## 10. How we work together
- **I write all the code myself; Claude guides and reviews.** Each task comes with a goal, hints and a test that must pass. I search docs first and ask only when stuck.
- **One task per message.** Paste only the one function and the exact error, not whole files.
- **Start a new chat per phase** and paste this file plus `docs/NOTES.md`.
- **Keep `docs/NOTES.md` updated after every decision or run.**

## 11. Task list
1. **Pooling and projector** (`src/connector.py`) **<- current task**
   - Pooling function: `[B, 729, 1152]` -> reshape to 27x27 grid -> pool -> flatten to `[B, N, 1152]`. Hints: `reshape`, `permute`, `F.avg_pool2d`, `F.adaptive_avg_pool2d`.
   - Projector `nn.Module`: Linear 1152->896, GELU, Linear 896->896.
   - Test in `tests/test_connector.py`: `torch.randn(2, 729, 1152)` through both gives `[2, N, 896]` for the chosen N. Run with `pytest` on laptop CPU.
2. **Model assembly** (`src/model.py`): frozen encoder (bf16, `no_grad`), connector, Qwen LLM; splice visual tokens into the text embeddings at the image placeholder; unit test that the number of visual tokens equals the number of placeholders.
3. **Data** (`src/data.py`): dataset, Qwen chat template, image placeholder, **loss masking on assistant tokens only**; decode a batch and inspect by eye.
4. **Training loop** (`scripts/train.py`): gradient checkpointing on the LLM, bf16, checkpoint to Cloud Storage every <=30 min with working resume, logging of loss, throughput, peak memory.
5. **Validate:** overfit 100 examples to near-zero loss, then run the 10K pilot and record real GPU-hours and memory.
6. **E1 on Vertex (spot)**, push the checkpoint plus README to `small_vlm`, then **E2** from the same checkpoint.
7. **Evaluation harness** (lmms-eval wrapper or own): text-only baseline first, then the VLM; per-sample outputs; format-failure rates; POPE breakdowns; blank-image run; latency, throughput and peak memory.
8. **Analysis and deliverables.**

## 12. Prompt to start the next chat
> I'm working on a COMS 4995 group project (small VLM, Team 14). Attached are `small-vlm-project.md` and `docs/NOTES.md`. I'm writing the code myself and want guidance, not full solutions. We are on Task 1 (pooling and projector). Keep replies short and one task at a time.
