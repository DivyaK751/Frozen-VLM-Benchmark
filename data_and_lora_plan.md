# Data Pipeline and LoRA: Implementation Plan

**Owners:** Divya and Lakshya

**Parts covered:** Part 1 (data pipeline) and Part 2 (LoRA)

**Used by:** Ritwik and Vinod (projector, eval pipeline)

**How we work:** we build both parts on our own, with no waiting on the other team. We set the formats ourselves (see [Image-token format](#image-token-format-our-decision)), and Ritwik and Vinod build on what we hand over. When a design choice is unclear, follow [abstract.md](abstract.md).

---

## Part 1: Data pipeline

**Goal:** turn raw datasets into batches the model can train on, in a fixed format that the projector and eval code can build on.

### Steps

1. **Download and store the data.**
   - Stage 1: LLaVA-Pretrain, which is 558K captions plus about 27 GB of images.
   - Stage 2: LLaVA-Instruct-150K conversations plus COCO train2017 images (about 18 GB).
   - Store everything in a Google Cloud Storage bucket so Vertex jobs can read it.
2. **Remove POPE overlap.** Take the COCO image IDs from POPE's 500 images and drop any Stage 2 conversation that uses one of them. Record how many images and conversations were removed for the report.
3. **Convert to Qwen's chat format.** LLaVA data looks like `{"human": "<image>\nWhat is...", "gpt": "..."}`, and it needs to become Qwen's `<|im_start|>user ... <|im_end|>` template.
   - Stage 1 becomes: user = image + "Describe the image", assistant = caption.
   - Stage 2 keeps its multi-turn conversations as they are.
4. **Expand the image placeholder.** Replace each `<image>` with **169 copies of a special image token**. The model later swaps these positions for the projector's output. We fix this format ourselves (see [Image-token format](#image-token-format-our-decision)).
5. **Build loss masks.** Training should only score the assistant's words, so every system, user and image token gets label `-100`. A mistake here is the most common silent bug.
6. **Preprocess images** with the SigLIP processor: resize to 384 px and normalize.
7. **Write the Dataset and collator.** They handle tokenization, truncation to a maximum length (about 2048 tokens), padding and batching. Each batch contains:
   - `pixel_values`
   - `input_ids`
   - `attention_mask`
   - `labels`
8. **Make the text-only variant for E0.** Use the same conversations with the image tokens removed.
9. **Run sanity checks.**
   - Decode a few samples back to text and read them.
   - Check that the masks cover the right tokens.
   - Build a 10K pilot subset and a 100-example subset for overfitting tests.

**Handoff:** a function like `get_dataloader(stage, split, subset_size, text_only=False)` that the training loop and eval code can call.

### Time estimate (steps 1–7)

| Step | Hands-on time | Notes |
|---|---|---|
| 1. Download and store in Google Cloud Storage | 2–3 h | Plus **3–8 h of background waiting** for about 45 GB of images. The fastest route is to download onto a cloud VM and copy straight to the bucket, not through a laptop. |
| 2. Remove POPE overlap | 1–2 h | Simple ID matching once the files are in place. |
| 3. Convert to Qwen chat format | 2–3 h | Mostly reading the LLaVA JSON and Qwen's chat template. |
| 4. Expand the image token | 1–2 h | The format is fixed below, so this is mostly implementation. |
| 5. Loss masks | 2–4 h | Easy to write, but slow to get right, and checking it carefully is worth the time. |
| 6. Image preprocessing | 1 h | The SigLIP processor does most of the work. |
| 7. Dataset and collator | 3–5 h | Padding, truncation and batching, plus testing on real batches. |
| Debugging buffer | 2–3 h | Something always breaks, such as corrupt images, overly long conversations or tokenizer edge cases. |
| **Total** | **14–23 h** | About 3–5 working days for one person; about 2–3 days with two people working in parallel. |

**Ways to save time:**
- Start the downloads on day 1 and write the code while they run.
- Build everything against the 10K pilot subset first. It's quick to download and lets you test steps 3–7 before the full data arrives.
- Split the work: one person takes steps 1–2 while the other takes steps 3–5.

### Task split: Divya and Lakshya

The work splits into two tracks that run in parallel:

- **Divya** owns the **data and images** track: downloading, filtering, image loading, and the Dataset class that serves batches.
- **Lakshya** owns the **text and tokens** track: chat formatting, image-token expansion and loss masks.

The two tracks meet at one function, which Lakshya writes and Divya's Dataset calls:

```python
format_example(conversation, stage, text_only=False) -> {"input_ids": [...], "labels": [...]}
```

#### Divya: data and images (about 9–13 h)

| # | Task | Covers step | Done when | Time |
|---|---|---|---|---|
| D1 | Download the **10K pilot subset** first (JSON + images) and share it with Lakshya | 1 | Lakshya can load the pilot JSON | 0.5 h |
| D2 | Start the **full downloads** to a cloud VM and copy them to the Cloud Storage bucket: LLaVA-Pretrain (558K) and LLaVA-Instruct-150K + COCO train2017 | 1 | All files are in the bucket and file counts match | 2–3 h (+ background waiting) |
| D3 | **Remove POPE overlap**: get POPE's 500 COCO image IDs, drop matching Stage 2 conversations, and save the filtered JSON | 2 | Filtered JSON saved; counts of removed images and conversations written down | 1–2 h |
| D4 | **Image preprocessing**: load each image with the SigLIP processor (384 px, normalize), and skip or log corrupt and missing images | 6 | Returns `pixel_values` of shape `[3, 384, 384]`; a list of bad images is saved | 1 h |
| D5 | **Make subsets**: the 10K pilot and the 100-example overfit set, for both stages | 9 | The subsets exist as JSON files with fixed seeds | 0.5–1 h |
| D6 | **Dataset class**: reads the JSON, loads the image (D4), calls Lakshya's `format_example`, and returns one example | 7 | `dataset[i]` returns `pixel_values`, `input_ids` and `labels` | 2–3 h |
| D7 | **Collator and `get_dataloader`**: pad `input_ids` with the pad token, pad `labels` with `-100`, build `attention_mask`, and stack the images | 7 | A batch of 8 loads with the right shapes | 1–2 h |
| D8 | **Data sanity checks**: example counts per stage, image check on 20 random samples, and a time-to-load-one-batch measurement | 9 | Short notes added to this file | 1 h |

**D1 results (done 2026-11-23).** Run with `python scripts/download_pilot.py` (seed 0). The output is in `data/pilot/`, which git ignores.

| | Source size | Pilot examples | Images | Missing / corrupt | Size |
|---|---|---|---|---|---|
| Stage 1 | LLaVA-Pretrain: 558,128 | 10,000 | 10,000 | 0 / 0 | 498 MB |
| Stage 2 | LLaVA-Instruct-150K: 157,712 | 10,000 | 9,627 unique | 0 / 0 | 1.58 GB |

- Lakshya only needs `stage1_pilot_10k.json` (2.9 MB) and `stage2_pilot_10k.json` (13.8 MB). Running the script again with the same seed recreates the same pilot.
- Hugging Face rate-limits users who aren't logged in (about 3,000 requests per 5 minutes). Run `hf auth login` (inside `.venv`) before D2.
- POPE overlap was checked in D3: the pilot has none (see below).

**D3 results (done 2026-11-24).** Run with `python scripts/check_pope_overlap.py`. The report is saved to `data/processed/pope_overlap.json`.

| | Conversations | Unique images | Overlapping with POPE | Removed |
|---|---|---|---|---|
| POPE (`lmms-lab/POPE`, 9,000 questions) | — | 500 | — | — |
| LLaVA-Instruct-150K (full) | 157,712 | 81,479 | **0** | 0 |
| Stage 2 pilot (10K) | 10,000 | 9,627 | **0** | 0 |

- **No filtering is needed.** Use the original `llava_instruct_150k.json` and the pilot as they are. LLaVA-Instruct is built from COCO train2014, and POPE uses val2014; the two never share images.
- Checks run: COCO IDs parse as plain integers on both sides, both fall in COCO's ID range, and a known POPE ID (310196) is found.
- **Still open:** the near-duplicate (perceptual-hash) check between POPE and the LLaVA-Pretrain images (LAION/CC/SBU). It needs all 558K images, so run it as a cloud job later.

**D4 results (done 2026-11-24).** The code is in `src/data/images.py`:

| Function | What it does |
|---|---|
| `load_image(path)` | Opens a file path or file object and returns `(RGB image, None)`, or `(None, error)` for a missing, truncated or invalid file |
| `preprocess(images)` | Runs the SigLIP processor and returns `pixel_values` `[N, 3, 384, 384]`, float32, in [-1, 1] |
| `blank_image()` | Gray (128, 128, 128) 384×384 image for the blank-image test; about 0 after normalization |

- **Decisions:**
  - Stretch images to 384×384 (SigLIP's default), with no padding to square.
  - Convert non-RGB images with `.convert("RGB")`, as LLaVA does.
  - Use the slow PIL processor (`use_fast=False`) explicitly, so train and eval always match.
- **SigLIP processor settings:** bilinear resize, rescale 1/255, mean and std 0.5. `do_convert_rgb` is unset, which is why `load_image` converts to RGB itself.
- **Pilot check:** all 19,627 images load, with 0 bad. The originals were 19,551 RGB and 76 grayscale, and all 76 grayscale images convert correctly.
- **Speed:** 3.1 ms per image on a laptop. That's about 30 CPU-minutes per Stage 1 epoch, so it won't slow the GPU down with 4–8 data-loading workers.
- **Bad-image log:** `scripts/check_images.py` scans a folder or a zip directly, without extracting, and writes the bad files and their errors to JSON. Tested on a zip with a truncated JPEG and a non-image file: both were caught. **To do:** run it on the full `images.zip` and `train2017.zip` in the cloud before training, and have the D6 Dataset skip the files it lists.

**D2 results (done 2026-11-24).** The full training data is in `gs://frozen-vlm-benchmark/raw/` (project `encouraging-art-515701-r8`, region `us-central1`). The zips are stored unextracted.

| Bucket path | Size | Contents | Check |
|---|---|---|---|
| `raw/pretrain/blip_laion_cc_sbu_558k.json` | 180.5 MB | 558,128 examples | — |
| `raw/pretrain/images.zip` | 27.36 GB | 558,128 images | 0 of 558,128 JSON images missing |
| `raw/instruct/llava_instruct_150k.json` | 228.9 MB | 157,712 examples, 81,479 unique images | — |
| `raw/coco/train2017.zip` | 19.34 GB | 118,287 images | md5 `cced6f7f71b7629ddf16f17bbcfab6b2` matches COCO; 0 of 81,479 JSON images missing |

How it was done (to repeat later, e.g. for eval data):
- The download ran on a temporary `e2-standard-4` VM in `us-central1-a`, using `hf download` for the Hugging Face files and `wget` for COCO, then `gcloud storage cp` to the bucket.
- **The Columbia organization blocks public IPs on VMs** (`constraints/compute.vmExternalIpAccess`). Workaround: create the VM with `--no-address`, give it internet through a Cloud NAT (`nat-router` / `nat-config`), and SSH in with `--tunnel-through-iap` (firewall rule `allow-iap-ssh` for `35.235.240.0/20`).
- Create the VM with `--scopes=cloud-platform`, or it can't write to the bucket.
- The VM, NAT, router and firewall rule were deleted afterwards. Total cost was about $2–3; storage is about $0.90 per month.
- Training jobs should copy the zips to local disk and extract them there, rather than reading individual files from the bucket.

#### Lakshya: text and tokens (about 7–13 h)

| # | Task | Covers step | Done when | Time |
|---|---|---|---|---|
| L1 | **Lock the image-token format**: load the Qwen tokenizer, confirm that `<\|vision_start\|>`, `<\|image_pad\|>` and `<\|vision_end\|>` exist, and record their IDs in the format section below | 4 | The token IDs are filled in below | 0.5 h |
| L2 | **Put the format in code**: one `config.py` with `NUM_IMAGE_TOKENS = 169`, the token IDs, the system prompt and `MAX_LEN = 2048`, so the format is set in a single place | 4 | Both tracks import the constants from `config.py` | 0.5 h |
| L3 | **Chat formatting**: convert LLaVA conversations to Qwen's chat template. Stage 1 is "Describe the image" plus the caption; Stage 2 is multi-turn | 3 | 5 pilot examples printed as text look correct | 2–3 h |
| L4 | **Image-token expansion**: replace each `<image>` with 169 image tokens | 4 | Every tokenized example contains exactly 169 image-token IDs | 1–2 h |
| L5 | **Loss masks**: set labels to `-100` for system, user and image tokens and keep labels only on assistant replies | 5 | Decoding only the unmasked labels shows assistant text and nothing else | 2–4 h |
| L6 | **Truncation**: cut to a maximum length of 2048 tokens without breaking the image tokens; log how many examples get cut | 7 | The share of truncated examples is recorded | 0.5–1 h |
| L7 | **Text-only variant (E0)**: `text_only=True` drops the image tokens and keeps everything else the same | 8 | No image tokens appear in text-only output | 0.5–1 h |
| L8 | **Text sanity checks**: decode 10 random examples per stage with their masks and read them | 9 | Short notes added to this file | 1 h |

#### Schedule

| Day | Divya | Lakshya |
|---|---|---|
| 1 | D1 (pilot download), start D2 (full downloads run in background) | L1 (lock the token format), L2 (`config.py`) |
| 2 | D3 (POPE overlap), D4 (image preprocessing), D5 (subsets) | L3 (chat formatting), L4 (token expansion) |
| 3 | D6 (Dataset class, using a stub `format_example` until L5 is done) | L5 (loss masks), L6 (truncation) |
| 4 | D7 (collator and `get_dataloader`), D8 | L7 (text-only), L8 |
| 5 | **Together:** connect the tracks, load the 10K pilot end to end, and hand `get_dataloader` to Ritwik and Vinod | |

**Final check (both):** one batch from the pilot loader, decoded back to text, shows the right image-token count, correct masks and the matching image.

#### Image-token format (our decision)

We set this format. Ritwik and Vinod build their code against it.

- **Image tokens:** `<|vision_start|>` + 169 × `<|image_pad|>` + `<|vision_end|>`. These reuse tokens already in the Qwen2.5 tokenizer, so no new tokens are added and the embeddings don't need resizing.
- **Token IDs:** *fill in after L1*: `vision_start = ___`, `image_pad = ___`, `vision_end = ___`
- **Tokens per image: 169.** SigLIP gives a 27×27 grid; 2×2 average pooling with stride 2 gives 13×13 = 169 (abstract.md §6). This is the single constant `NUM_IMAGE_TOKENS` in `config.py`, so changing it later is a one-line edit.
- **Position:** the image comes at the start of the first user turn, before the question:
  ```
  <|im_start|>system\n{Qwen default system prompt}<|im_end|>
  <|im_start|>user\n<|vision_start|><|image_pad|>×169<|vision_end|>\n{question}<|im_end|>
  <|im_start|>assistant\n{answer}<|im_end|>
  ```
- **Batch contents:**
  - `pixel_values` `[B, 3, 384, 384]`: raw images from the SigLIP processor. The model runs SigLIP itself; we don't precompute features.
  - `input_ids`, `attention_mask` `[B, L]`: maximum length 2048.
  - `labels` `[B, L]`: -100 everywhere except assistant replies.
- **How the model uses it:** embed `input_ids`, then replace the positions where `input_ids == image_pad_id` with the projector output `[B, 169, 896]`.
- **Modes of `format_example`:**
  - `inference=True`: no answer; the prompt ends at `<|im_start|>assistant\n`. Used for eval.
  - `text_only=True`: no image tokens. Used for E0 and the no-image test.

---

## Part 2: LoRA

**Goal:** adapt Qwen cheaply on top of the projector, so that E2 and E3 differ only in whether LoRA is switched on.

### Steps

1. **Wrap Qwen with PEFT.**
   - Settings: rank 16, alpha 32, dropout 0.05.
   - Target all 7 linear layers in each block: `q, k, v, o` (attention) and `gate, up, down` (MLP).
   - That gives about 8.8M trainable parameters, compared with about 1.8M for the projector.
2. **Combine LoRA with the projector.**
   - To test without waiting for the projector team, build a small stand-in model from the abstract.md §6 spec: frozen SigLIP → 2×2 pooling → 2-layer MLP (1152 → 896, GELU, randomly initialized) → Qwen, using the image-token replacement described in the format section.
   - For the real E3 runs, swap the stand-in projector for the Stage 1 projector checkpoint. The shapes are the same, so nothing else changes.
   - Trainable parameters are the projector plus LoRA. SigLIP and the base Qwen weights stay frozen.
   - Print the trainable parameter count to confirm this.
3. **Use one config flag.** Something like `use_lora: true/false`, so that E2 and E3 share all other code and settings.
4. **Set up the optimizer.**
   - Use separate learning rates for LoRA (around 2e-4) and the projector (around 2e-5 in Stage 2).
   - Keep the same scheduler, steps and seed for E2 and E3.
5. **Handle memory.** Use bf16 and gradient checkpointing. Gradient checkpointing has a small known quirk when used with PEFT, so test it on the pilot run.
6. **Save and load correctly.**
   - Save the LoRA adapter and the projector separately.
   - For eval, merge the adapter into Qwen (`merge_and_unload`) and give Ritwik and Vinod one model they can load.
7. **Run sanity checks.**
   - Before training, the LoRA model's output should exactly match base Qwen, because LoRA starts at zero.
   - Train on the 100-example subset and confirm the loss drops toward 0.
8. **Launch the runs:**
   - E3 with seeds 0 and 1
   - E0, a text-only LoRA run
   - the rank ablation at ranks 4 and 64
   - E2 vs. E3 loss-curve plots

---

## Order of work

| When | Data pipeline | LoRA |
|---|---|---|
| Week 1 | Steps 1–7; 10K pilot loader ready | Steps 1–3, using the stand-in model and dummy data |
| Week 2 | Steps 8–9; POPE overlap counts | Steps 4–7 on the pilot data |
| Week 3 | Support the full runs | Full E3, E0 and rank-ablation runs |

**Handoff to Ritwik and Vinod:** once the pilot loader works, share `get_dataloader`, `format_example`, `config.py` and the [Image-token format](#image-token-format-our-decision) section. They build the projector and eval code on these.

**The one input we need from them:** the full E3 runs start from the Stage 1 projector checkpoint. Until then, all LoRA work runs on the stand-in projector, so nothing is blocked.
