# Data Pipeline and LoRA: Implementation Plan

**Owners:** Divya and Lakshya

**Parts covered:** Part 1 (data pipeline) and Part 2 (LoRA)

**Depends on us:** Ritwik and Vinod (projector, eval pipeline)

---

## Part 1: Data pipeline

**Goal:** turn raw datasets into batches the model can train on, and give the projector and eval teams a clear agreed format for those batches.

### Steps

1. **Download and store the data.**
   - Stage 1: LLaVA-Pretrain, which is 558K captions plus about 27 GB of images.
   - Stage 2: LLaVA-Instruct-150K conversations plus COCO train2017 images (about 18 GB).
   - Store everything in a Google Cloud Storage bucket so Vertex jobs can read it.
2. **Remove POPE overlap.** Take the COCO image IDs from POPE's 500 images and drop any Stage 2 conversation that uses one of them. Record how many images and conversations were removed for the report.
3. **Convert to Qwen's chat format.** LLaVA data looks like `{"human": "<image>\nWhat is...", "gpt": "..."}`, and it needs to become Qwen's `<|im_start|>user ... <|im_end|>` template.
   - Stage 1 becomes: user = image + "Describe the image", assistant = caption.
   - Stage 2 keeps its multi-turn conversations as they are.
4. **Expand the image placeholder.** Replace each `<image>` with **169 copies of a special image token**. The model later swaps these positions for the projector's output. Agree on this format with Ritwik and Vinod early.
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
| 4. Expand the image token | 2–4 h | Includes agreeing on the format with the projector team. Can take longer if you go back and forth. |
| 5. Loss masks | 2–4 h | Easy to write, but slow to get right, and checking it carefully is worth the time. |
| 6. Image preprocessing | 1 h | The SigLIP processor does most of the work. |
| 7. Dataset and collator | 3–5 h | Padding, truncation and batching, plus testing on real batches. |
| Debugging buffer | 2–3 h | Something always breaks, such as corrupt images, overly long conversations or tokenizer edge cases. |
| **Total** | **15–25 h** | About 3–5 working days for one person; about 2–3 days with two people working in parallel. |

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

#### Lakshya: text and tokens (about 8–14 h)

| # | Task | Covers step | Done when | Time |
|---|---|---|---|---|
| L1 | **Agree on the image-token format** with Ritwik and Vinod: the token name (e.g. `<image>`), the fact that it is repeated 169 times, and where it goes in the prompt | 4 | The format is written below in this file and all three tracks have signed off | 1–2 h |
| L2 | **Add the image token** to the Qwen tokenizer as a special token and record its ID | 4 | The tokenizer encodes `<image>` as a single ID | 0.5 h |
| L3 | **Chat formatting**: convert LLaVA conversations to Qwen's chat template. Stage 1 is "Describe the image" plus the caption; Stage 2 is multi-turn | 3 | 5 pilot examples printed as text look correct | 2–3 h |
| L4 | **Image-token expansion**: replace each `<image>` with 169 image tokens | 4 | Every tokenized example contains exactly 169 image-token IDs | 1–2 h |
| L5 | **Loss masks**: set labels to `-100` for system, user and image tokens and keep labels only on assistant replies | 5 | Decoding only the unmasked labels shows assistant text and nothing else | 2–4 h |
| L6 | **Truncation**: cut to a maximum length of 2048 tokens without breaking the image tokens; log how many examples get cut | 7 | The share of truncated examples is recorded | 0.5–1 h |
| L7 | **Text-only variant (E0)**: `text_only=True` drops the image tokens and keeps everything else the same | 8 | No image tokens appear in text-only output | 0.5–1 h |
| L8 | **Text sanity checks**: decode 10 random examples per stage with their masks and read them | 9 | Short notes added to this file | 1 h |

#### Schedule

| Day | Divya | Lakshya |
|---|---|---|
| 1 | D1 (pilot download), start D2 (full downloads run in background) | L1 (meet the projector team), L2 |
| 2 | D3 (POPE overlap), D4 (image preprocessing), D5 (subsets) | L3 (chat formatting), L4 (token expansion) |
| 3 | D6 (Dataset class, using a stub `format_example` until L5 is done) | L5 (loss masks), L6 (truncation) |
| 4 | D7 (collator and `get_dataloader`), D8 | L7 (text-only), L8 |
| 5 | **Together:** connect the tracks, load the 10K pilot end to end, and hand `get_dataloader` to Ritwik and Vinod | |

**Final check (both):** one batch from the pilot loader, decoded back to text, shows the right image-token count, correct masks and the matching image.

#### Agreed image-token format

*To be filled in after L1.*

- Token string:
- Token ID:
- Tokens per image: 169
- Position in the prompt:

---

## Part 2: LoRA

**Goal:** adapt Qwen cheaply on top of the projector, so that E2 and E3 differ only in whether LoRA is switched on.

### Steps

1. **Wrap Qwen with PEFT.**
   - Settings: rank 16, alpha 32, dropout 0.05.
   - Target all 7 linear layers in each block: `q, k, v, o` (attention) and `gate, up, down` (MLP).
   - That gives about 8.8M trainable parameters, compared with about 1.8M for the projector.
2. **Combine LoRA with the projector.**
   - Load the Stage 1 projector checkpoint.
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
| Week 1 | Steps 1–7; 10K pilot loader ready (this unblocks the projector team) | Steps 1–3, tested on dummy data |
| Week 2 | Steps 8–9; POPE overlap counts | Steps 4–7 on the pilot data |
| Week 3 | Support the full runs | Full E3, E0 and rank-ablation runs |

**Biggest dependency:** step 4 of the data pipeline, the image-token format. It decides how the projector's output is inserted into Qwen's input, and the data pipeline, LoRA and projector all depend on it. Agree on it with Ritwik and Vinod before writing steps 4, 5 and 7.
