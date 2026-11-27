# Model and training loop

Covers the model (`src/model/`), the trainer (`src/train/`), the run configs (`configs/`) and `scripts/train.py`.
It builds on the data pipeline in [DATA_PIPELINE.md](DATA_PIPELINE.md) and follows [abstract.md](abstract.md).
The evaluation harness is in [EVAL.md](EVAL.md).

## 1. What is in it

| File | What it does |
|---|---|
| `src/model/projector.py` | `VisualProjector`: Linear 1152→896, GELU, Linear 896→896 (1,836,800 parameters) |
| `src/model/vlm.py` | `SmallVLM`: frozen SigLIP → `pool_patches` (169 tokens) → projector → Qwen. Overwrites the 169 `<\|image_pad\|>` embeddings. `add_lora()` turns E2 into E3. `build_model()` loads the real checkpoints. |
| `src/train/config.py` | `TrainConfig`: every setting for one run |
| `src/train/trainer.py` | The loop: accumulation, optimizer, LR schedule, checkpoint/resume, logging, overfit diagnostic |
| `scripts/train.py` | Command line: `--config configs/X.json` plus any override |
| `configs/` | `overfit_stage{1,2}`, `pilot_stage1`, `e1_stage1`, `e2_stage2_projector`, `e3_stage2_lora`, `e0_text_only_lora` |
| `tests/test_model.py`, `test_train.py`, `test_build_model.py` | CPU tests on tiny random models, no downloads |

## 2. Design decisions

- **E2 vs E3 differ only in `lora`.** Same data order, steps, batch size, projector LR and seed. `tests/test_train.py` checks that `e2_stage2_projector.json` and `e3_stage2_lora.json` differ only in the LoRA fields.
- **Trained parameters:** only those with `requires_grad`: the projector, plus LoRA (rank 16, alpha 32, dropout 0.05, all 7 linear layers per block). SigLIP and Qwen's base weights are frozen. E0 (`text_only`) trains LoRA only.
- **Exact gradient accumulation.** One optimizer step is `grad_accum` micro-batches, and the loss is divided by the number of trained tokens in the *whole step* (`num_items_in_batch`). So the result equals one big batch, even though answers differ in length. Tested.
- **Learning rate:** linear warm-up (3%), then cosine to 0. Separate rates for the projector (`lr_projector`) and LoRA (`lr_lora`). AdamW, no weight decay, gradient clipping at 1.0.
- **Dtypes:** SigLIP and Qwen in bf16, projector and LoRA weights in fp32. CPU runs use `"dtype": "float32"`.
- **Data order depends only on `(seed, epoch)`.** Resuming skips ahead without loading any images.
- **Resume:** the checkpoint holds the trainable weights, optimizer, step and RNG state (about 22 MB for the projector, about 105 MB more with LoRA rank 16). It is saved every `save_every_steps` steps or `save_every_minutes` minutes, keeping the last `keep_last`. **Re-running the same command resumes.** Tested: a run killed mid-way ends with the same weights as an uninterrupted run.
- **NaN guard:** a non-finite loss stops the run before any checkpoint is written.
- **Outputs** (in `output_dir`, which can be a `/gcs/<bucket>/...` path on Vertex):
  - `train_log.jsonl`: loss, grad norm, LR, tokens/s, peak GPU memory
  - `latest.json` and `ckpt_*.pt`
  - `final/projector.pt`: the start of Stage 2, and what eval loads
  - `final/lora/`: the PEFT adapter, to merge for eval
  - `final/summary.json`: steps, final loss, wall hours, peak memory and a cost estimate for the budget log

## 3. How to run

```bash
pip install -r requirements.txt
python tests/test_model.py && python tests/test_train.py && python tests/test_build_model.py   # expect ALL CHECKS PASSED
```

**Step 1, overfit test** (needs the overfit data from `scripts/download_pilot.py` and `make_subsets.py`). It needs a GPU to be convincing: run it as the first cloud job (see [GCP_RUNBOOK.md](GCP_RUNBOOK.md) section 4). On a laptop CPU it only works as a slow "does it run" smoke test (add `--device cpu --dtype float32 --max_steps 20 --micro_batch_size 2 --grad_accum 1`):
```bash
python scripts/train.py --config configs/overfit_stage2.json
```
Expected: the loss falls close to 0, and the printed `image shuffle check` says `OK: the model uses the image`. If shuffling the images does not raise the loss, the image-token replacement is wrong. Do not start real runs until this passes. These two overfit configs use LoRA so that 100 examples can be memorised.

**Step 2, pilot** (10K examples, A100): `python scripts/train.py --config configs/pilot_stage1.json`. Read `tokens_per_s` and `peak_mem_gb` from the log. This decides the micro-batch size, whether `gradient_checkpointing` is needed, the real GPU-hours for E1, and whether E1 runs on the full 558K or a `subset_size` of 200K (abstract §10 fallback).

**Step 3, real runs** (on Vertex; the cloud job copies and extracts the zips for you, see [GCP_RUNBOOK.md](GCP_RUNBOOK.md)):
```bash
python scripts/train.py --config configs/e1_stage1.json                       # Stage 1, 1 seed
python scripts/train.py --config configs/e2_stage2_projector.json --seed 0
python scripts/train.py --config configs/e3_stage2_lora.json --seed 0
python scripts/train.py --config configs/e2_stage2_projector.json --seed 1 --output_dir outputs/e2_stage2_projector_seed1
python scripts/train.py --config configs/e3_stage2_lora.json --seed 1 --output_dir outputs/e3_stage2_lora_seed1
python scripts/train.py --config configs/e0_text_only_lora.json
python scripts/train.py --config configs/e3_stage2_lora.json --lora_rank 4 --lora_alpha 8 --output_dir outputs/a1_rank4     # ablation
python scripts/train.py --config configs/e3_stage2_lora.json --lora_rank 64 --lora_alpha 128 --output_dir outputs/a2_rank64
```
Stage 2 configs start from `outputs/e1_stage1/final/projector.pt`; change `init_projector` if it is stored elsewhere. Keep `lora_alpha = 2 × rank` in the ablation so the LoRA scale stays the same.

## 4. Settings to confirm on the pilot (starting values, not measured)

| Setting | Start value | Source / reason |
|---|---|---|
| Stage 1 effective batch, LR | 256 (32 × 8), 1e-3 | LLaVA-1.5 Stage 1 recipe |
| Stage 2 effective batch | 128 (16 × 8) | LLaVA-1.5 Stage 2 recipe |
| Stage 2 `lr_projector` | 2e-5 | from `data_and_lora_plan.md` |
| `lr_lora` | 2e-4 | from `data_and_lora_plan.md` |
| Micro-batch sizes | 32 (Stage 1), 16 (Stage 2) | guesses; raise or lower after reading `peak_mem_gb` |

**One thing to decide before E2 and E3.** 2e-5 is the rate LLaVA-1.5 uses when the whole LLM is trained. For a projector-only E2 it may be too low, so E2 would be undertrained and E3's gain would partly reflect the learning rate. The abstract requires identical hyperparameters, so the rate has to be settled before the real runs, for example with two short pilot runs of E2 at 2e-5 and 1e-4, then using the same projector LR in E2 and E3. If you change it, change it in both configs (a test enforces this).

## 5. What has and has not been verified

Verified (CPU, tiny random models, `transformers 4.57.6`, `peft 0.22.0`, `torch 2.14.1`):
- Image features land on the right placeholder positions, per sample, in row-major order; wrong placeholder counts are rejected before SigLIP runs.
- The frozen parts get no gradients, but gradients flow through the LLM to the projector.
- LoRA does not change the output at initialisation, and targets exactly the 7 linear layers in each block.
- Gradient checkpointing gives the same gradients, including for text-only E0.
- bf16 LLM with an fp32 projector trains.
- Accumulated micro-batches match one big batch; crash-and-resume gives identical weights; checkpoint pruning; NaN guard.
- The overfit test memorises the data, and shuffled images raise the loss.
- `build_model` loads a full SigLIP checkpoint into the vision-only class with no missing weights, and fails loudly if weights are missing.

**Not verified** (needs your data, GPU or the real checkpoints):
- Loading the real `google/siglip2-so400m-patch14-384` and `Qwen/Qwen2.5-0.5B-Instruct`, and the real shapes (`[B, 729, 1152]` features).
- Anything with `get_dataloader` and real images (the sandbox has no access to Hugging Face).
- Speed, memory and cost on an A100. These come from the pilot.
- `gradient_checkpointing` and bf16 on a GPU.
