# Data Pipeline: Handoff to Projector and Eval

**From:** Divya and Lakshya. **For:** Ritwik and Vinod.

The data pipeline (Part 1, tasks D1–D8 and L1–L8) is finished and tested. This page covers what you need to build the projector, the training loop and the eval code on top of it. The full design notes and test results are in [data_and_lora_plan.md](data_and_lora_plan.md).

---

## 1. Quick start

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download_pilot.py     # 10K pilot per stage, about 2 GB, into data/pilot/
.venv/bin/python scripts/make_subsets.py       # 100-example overfit sets
.venv/bin/python tests/test_loader.py          # expect: ALL CHECKS PASSED
```

```python
from src.data.loader import get_dataloader

dl = get_dataloader(stage=2, split="pilot", batch_size=8)
batch = next(iter(dl))
```

`data/` is git-ignored, so each machine downloads its own copy.

## 2. What a batch contains

| Key | Shape / type | Contents |
|---|---|---|
| `pixel_values` | float `[B, 3, 384, 384]` | Raw images after the SigLIP processor, in [-1, 1]. **SigLIP is not run for you**: the model runs the frozen encoder itself. Absent when `text_only=True`. |
| `input_ids` | long `[B, L]` | Qwen chat tokens, padded on the **right** with `151643` (`<\|endoftext\|>`). `L` is rounded up to a multiple of 8. |
| `attention_mask` | long `[B, L]` | 1 = real token, 0 = padding. |
| `labels` | long `[B, L]` | `-100` everywhere except assistant answers and their closing `<\|im_end\|>`. Can go straight into a causal-LM loss. |

Each example looks like this, with one image per example in the first user turn:

```
<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n
<|im_start|>user\n<|vision_start|><|image_pad|>×169<|vision_end|>\n{question}<|im_end|>\n
<|im_start|>assistant\n{answer}<|im_end|>\n
```

## 3. What the model has to do with the image tokens

The text contains **169 placeholder tokens** (`<|image_pad|>`, ID `151655`) for each image. Your forward pass should:

1. Run SigLIP on `pixel_values` to get a 27×27 grid of patch features.
2. Apply 2×2 average pooling with stride 2 to get 13×13 = **169** features, then run the projector into Qwen's hidden size.
3. Embed `input_ids` as usual, then **overwrite the embeddings at positions where `input_ids == IMAGE_PAD_ID`** with the 169 projector outputs, in order.

```python
mask = input_ids == IMAGE_PAD_ID                                # [B, L], exactly 169 True per row
embeds = llm.get_input_embeddings()(input_ids)
embeds[mask] = projected.reshape(-1, projected.shape[-1]).to(embeds.dtype)  # projected: [B, 169, D]
```

The image tokens always have label `-100`, so they never count towards the loss. No new tokens are added to the tokenizer, so the embedding matrix doesn't need resizing.

All of these constants live in [src/config.py](src/config.py). Import them from there rather than hard-coding them:

| Constant | Value |
|---|---|
| `LLM_NAME` | `Qwen/Qwen2.5-0.5B-Instruct` |
| `NUM_IMAGE_TOKENS` | 169 |
| `VISION_START_ID` / `IMAGE_PAD_ID` / `VISION_END_ID` | 151652 / 151655 / 151653 |
| `IM_START_ID` / `IM_END_ID` | 151644 / 151645 (`IM_END_ID` is the EOS for generation) |
| `PAD_TOKEN_ID` | 151643 |
| `MAX_LEN` | 2048 |
| `IGNORE_INDEX` | -100 |

The image encoder is `google/siglip2-so400m-patch14-384` ([src/data/images.py](src/data/images.py)).

## 4. `get_dataloader` options

```python
get_dataloader(stage, split="pilot", subset_size=None, text_only=False, batch_size=8,
               shuffle=True, seed=0, num_workers=4, data_root="data")
```

| Argument | Values | Use |
|---|---|---|
| `stage` | `1`, `2` | 1 = alignment on LLaVA-Pretrain captions; 2 = instruction tuning on LLaVA-Instruct-150K |
| `split` | `"overfit"` (100), `"pilot"` (10K), `"full"` | Start with `overfit` to debug, use `pilot` to measure speed, memory and cost, and `full` for real runs |
| `subset_size` | int or `None` | A random N examples, fixed by `seed` |
| `text_only` | bool | No images and no image tokens, for E0 and the no-image test |
| `seed` | int | The same seed gives the same subset and batch order |

### Dataset sizes

| Split | Stage 1 | Stage 2 |
|---|---|---|
| overfit | 100 | 100 (240 Q/A turns) |
| pilot | 10,000 | 10,000 (9,627 images, 23,035 Q/A turns) |
| full | 558,128 | 157,712 (81,479 images, 361,411 Q/A turns) |

### Sequence lengths

| | Min | Median | Max |
|---|---|---|---|
| Stage 1 (full) | 209 | 218 | 268 |
| Stage 2 (full) | 213 | 363 | 1,124 |

No example reaches `MAX_LEN = 2048`, so nothing is truncated. About 6% of Stage 1 tokens are trained on, since it's mostly image tokens plus a short caption. About 43% of Stage 2 tokens are trained on.

## 5. Running on the full data (Vertex / cloud)

The full data is in `gs://frozen-vlm-benchmark/raw/`, stored as zips (about 47 GB in total, not extracted):

| Bucket path | Extract to (under `data_root`) |
|---|---|
| `raw/pretrain/blip_laion_cc_sbu_558k.json` | `raw/pretrain/` |
| `raw/pretrain/images.zip` | `images/pretrain/` |
| `raw/instruct/llava_instruct_150k.json` | `raw/instruct/` |
| `raw/coco/train2017.zip` | `images/coco/train2017/` |

- **Copy the zips to the job's local disk and extract them there.** Reading single files from the bucket is far too slow.
- Image paths in the JSONs are relative, for example `00217/002170933.jpg` or `000000369358.jpg`. After extraction, check that `images/pretrain/00217/002170933.jpg` exists. If the zip added an extra top-level folder, move the files up one level.
- **Bad images:** the full zips have already been scanned, and **none are bad**. Copy `gs://frozen-vlm-benchmark/processed/image_checks/bad_images.json` to `data/processed/bad_images.json`, which the loader reads automatically (it skips any listed image). If an image still fails at runtime, the loader warns and swaps in another example instead of crashing.
- Use 4–8 data-loading workers. Loading takes about 3–5 ms per example on a laptop CPU, so it won't hold the GPU back.

## 6. Notes for the eval pipeline

- **Prompts for generation:** use `format_example(conversation, stage, inference=True)` from [src/data/formatting.py](src/data/formatting.py). It drops the answer and ends the prompt at `<|im_start|>assistant\n`, which is exactly the training format.
- **Padding:** batched generation needs **left** padding. The training collator pads on the right, so the eval code should pad on its own.
- **Stopping:** stop generation at `IM_END_ID` (151645). The model is trained to emit it after every answer.
- **System prompt:** we use Qwen2.5's default system prompt, the same one Qwen's chat template and lmms-eval insert when no system message is given. Keep it unchanged so eval matches training.
- **Image preprocessing:** use `load_image` and `preprocess` from [src/data/images.py](src/data/images.py), so eval images get the same resize (stretched to 384×384), RGB conversion and normalization.
- **Blank-image test:** `blank_image()` in the same file returns a gray image that normalizes to about 0.
- **POPE contamination: none.** LLaVA-Instruct (COCO train2014) shares **no image IDs** with POPE (COCO val2014); the report is in `data/processed/pope_overlap.json`. A perceptual-hash check of all 558K Pretrain and 118K COCO images found **no near-duplicates** of POPE images either. All 207 close candidates were checked by eye; see `data/processed/pope_near_dups/` and the D3 notes in [data_and_lora_plan.md](data_and_lora_plan.md).

## 7. Checks you can run

| Command | What it checks |
|---|---|
| `python tests/test_loader.py` | Batch shapes, 169 image tokens per example, padding and image tokens never trained on, `text_only`, seeding, truncation |
| `python tests/test_formatting.py` | Output matches Qwen's own `apply_chat_template` token for token on all 20,000 pilot examples |
| `python scripts/sanity_check.py` | Counts, loader speed, and `data/processed/sanity/samples.html`, which shows each image next to its decoded text and the trained span. All 20 samples were checked by hand: every image matches its text. |
| `python scripts/check_formatting.py` | Writes decoded examples, with trained tokens in `[[ ]]`, to a text file you can read |

**Before any real run, do the overfit test.** Train on `split="overfit"` for a few hundred steps, and the loss should fall close to 0. Then shuffle the images between examples: the loss should rise clearly. If it doesn't, the model isn't using the image, which usually means the image-token replacement in step 3 of section 3 is wrong.

## 8. Still open

Nothing on the data side. The two former items were done on 2026-11-26:
- **Bad-image scan:** 0 bad images across all 676,415 training images.
- **POPE near-duplicate check:** 0 near-duplicates.

The cloud job is `scripts/cloud/launch_image_checks.sh`.

## 9. File map

| File | What it does |
|---|---|
| [src/config.py](src/config.py) | Shared constants: token IDs, image-token count, prompts, max length |
| [src/data/formatting.py](src/data/formatting.py) | `format_example`: LLaVA conversation → Qwen tokens + labels (training, `text_only`, `inference`) |
| [src/data/images.py](src/data/images.py) | `load_image`, `preprocess` (SigLIP processor), `blank_image` |
| [src/data/dataset.py](src/data/dataset.py) | `VLMDataset`: one example → `pixel_values`, `input_ids`, `labels`; skips bad images |
| [src/data/loader.py](src/data/loader.py) | `Collator` and `get_dataloader`, which you call from training |
| [scripts/download_pilot.py](scripts/download_pilot.py) | Downloads the 10K pilot per stage |
| [scripts/make_subsets.py](scripts/make_subsets.py) | Builds the 100-example overfit sets |
| [scripts/check_images.py](scripts/check_images.py) | Scans a folder or zip for unreadable images |
| [scripts/check_pope_overlap.py](scripts/check_pope_overlap.py) | POPE / training-data overlap report (exact COCO IDs) |
| [scripts/check_pope_near_dups.py](scripts/check_pope_near_dups.py), [scripts/review_pope_near_dups.py](scripts/review_pope_near_dups.py) | POPE near-duplicate check (pHash) and its by-eye review page |
| [scripts/cloud/](scripts/cloud/) | Cloud job that runs the bad-image scan and the near-duplicate check on the full zips |
| [scripts/sanity_check.py](scripts/sanity_check.py), [scripts/check_formatting.py](scripts/check_formatting.py) | Human-readable checks |
| [tests/](tests/) | `test_loader.py`, `test_formatting.py` |
