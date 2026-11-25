# NOTES.md: Decision and Progress Log

> **Reminder: update this file at the end of each task (short), and after any big decision, run, or surprise.** Paste it at the start of each new chat.
> It lives in the repo as `docs/NOTES.md`. Commit it regularly.
> Last updated: Nov 25, 2026 (Task 2 complete, plan shrunk).

## Fixed setup
| Item | Value |
|---|---|
| GCP region | `us-central1` (keep everything here) |
| Bucket | `small_vlm` (Standard, not public) |
| Shared repo | `Frozen-VLM-Benchmark` (owned by Divya, private) |
| My branch | `no-lora` (never commit to `main`; use pull requests) |
| Old personal repo | `small-vlm` (backup only, do not edit, do not delete yet) |
| Package | `small_vlm`, in `src/small_vlm/`, editable install |
| Vision encoder | `google/siglip2-so400m-patch14-384`, frozen |
| LLM | `Qwen/Qwen2.5-0.5B-Instruct` |
| Encoder output | `[B, 729, 1152]` (27x27 grid) |
| Pooled output | `[B, 196, 1152]` (14x14 grid, adaptive pooling) |
| Projector output | `[B, 196, 896]` |
| LLM hidden size | 896 |
| Credits applied | $100 (second coupon may raise this to $200) |
| Budget alert target | $100 (raise to $200 with the second coupon) |
| Spot A100 quota | 8 (Agent Platform, us-central1) |

## Interfaces (stable, other code depends on these)
| Item | Signature / value |
|---|---|
| `pool_visual_tokens` | `pool_visual_tokens(tokens, grid_size=27, output_size=14)`: `[B, grid_size**2, C] -> [B, output_size**2, C]`. Asserts token count |
| `VisualProjector` | `VisualProjector(in_dim=1152, out_dim=896)`: `fc1`, `act` (GELU), `fc2`. `[B, N, in_dim] -> [B, N, out_dim]` |
| Projector parameters | 1,836,800 (= 1152*896 + 896 + 896*896 + 896) |
| `SmallVLM` | `SmallVLM(encoder, llm, projector, image_token_id, grid_size=27, pool_size=14)`. `tokens_per_image = pool_size**2` (196 by default) |
| `SmallVLM.forward` | `forward(input_ids, pixel_values, attention_mask=None, labels=None)` -> HF causal-LM output (loss, logits). One image per row; each row needs exactly `tokens_per_image` ids equal to `image_token_id`, else `ValueError` (checked before the encoder runs) |
| `SmallVLM.encode_images` | `[B,3,H,W] -> [B, tokens_per_image, llm_hidden]`, in the LLM embedding dtype |
| `build_small_vlm` | `build_small_vlm(*, pool_size=14, dtype=torch.float32) -> (model, tokenizer)`. Raises `RuntimeError` if any SigLIP vision keys are missing |
| Tests | `tests/test_connector.py`: 5. `tests/test_model.py`: fakes plus a tiny random Siglip/Qwen2 forward-backward (fill in the count from `pytest`) |
| Smoke script | `scripts/smoke_test_siglip.py`: needs the ~5 GB download, NOT run by pytest, NOT committed |

## Verified facts (transformers 5.24.0, Nov 25)
| Item | Value |
|---|---|
| Vision class | `SiglipVisionModel` (checkpoint `model_type: "siglip"`, not the Siglip2 classes). `forward(pixel_values, interpolate_pos_encoding=False)` |
| Vision output | `last_hidden_state` `[B,729,1152]` (after `post_layernorm`, no CLS). `pooler_output` is computed but unused |
| Weight loading | Vision-only load has no missing keys; output matches the full `SiglipModel` (fp32) |
| Image preprocessing | `AutoImageProcessor` of the checkpoint: resize to 384x384 (squash, aspect ratio NOT preserved; a 160x160 marker in a 640x320 image became 96x192), no center crop, mean = std = 0.5, range [-1, 1] |
| Vocabulary | `len(tokenizer)` 151,665; embedding rows 151,936 (271 unused); Qwen's existing image-pad token has id 151655 and is a single token; no resize needed |
| Qwen head | `tie_word_embeddings=True`; one `requires_grad_(False)` freezes both |
| Trainable params | 1,836,800 on the real models |
| Loading kwarg | `from_pretrained(..., dtype=...)` works in 5.24.0 |
| Checkpoint size | Projector 1,836,800 fp32 params ~ 7.3 MB; with two AdamW states ~ 22 MB (derived) |

## Environment (everyone uses the same)
| Item | Value |
|---|---|
| Python | 3.11 (laptop: 3.11.8). `pyproject.toml` requires >=3.11,<3.12 |
| Virtual env | `venv` in `.venv/` (git-ignored). Not conda |
| Install | `pip install -r requirements.txt` then `pip install -e .` |
| torch / torchvision | 2.15.0 (CPU build on laptop) / 0.29.2 |
| transformers | 5.24.0 |
| accelerate / peft / datasets | 1.16.0 / 0.22.0 / 5.2.0 |
| numpy / pandas / scipy | 2.4.6 / 3.1.1 / 1.17.1 |
| pillow / pytest | 13.0.0 / 9.2.0 |
| google-cloud-storage | 3.17.1 |

**Setup commands (Windows, Git Bash):**
```bash
git clone <repo-link> && cd Frozen-VLM-Benchmark
git checkout no-lora                 # or your own branch
py -3.11 -m venv .venv
source .venv/Scripts/activate        # Linux: python3.11 -m venv .venv ; source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -c "import small_vlm; print(small_vlm.__file__)"
pytest
```
**Adding a package:** install it in the venv, run `pip freeze --exclude-editable > requirements.txt`, commit, tell the team.

**Known risks:** `requirements.txt` was frozen on Windows, so a Windows-only package may fail on Linux (remove that line and note it here). On the Linux GPU VM, check `torch.cuda.is_available()` on the L4 first.

## Decisions
| Date | Decision | Reason |
|---|---|---|
| Nov 21 | I write all code myself, Claude guides | Learning and interview readiness |
| Nov 21 | My half = without-LoRA end to end (E1, E2, eval harness) | Team split |
| Nov 21 | Work in Divya's repo `Frozen-VLM-Benchmark` on branch `no-lora` | Team repo |
| Nov 21 | `venv` + Python 3.11 + pinned `requirements.txt` | Avoid version mismatches between machines |
| Nov 21 | `src/` layout with package `small_vlm`, `pip install -e .` | Imports work from any folder |
| Nov 22 | Pooling: `adaptive_avg_pool2d` 27x27 -> 14x14 = 196 tokens, applied before the projector | Keeps every patch. `avg_pool2d(kernel=2)` (13x13 = 169) would drop the last row and column, a systematic crop that could confound TextVQA and POPE. Pooling first means the projector runs on 196 tokens. Cost: ~16% more tokens than 169 |
| Nov 22 | Pooling output size is a function argument | Other token counts become a config change |
| Nov 22 | `VisualProjector`: Linear 1152->896, GELU, Linear 896->896, layers `fc1`/`act`/`fc2` | Readable `state_dict` keys for the E1 checkpoint. 1,836,800 parameters |
| Nov 22 | Connector code in one file, `connector.py` | Cohesive, ~30 lines |
| Nov 25 | Image placeholder = Qwen's existing image-pad token (id 151655), no embedding resize | The embedding row is overwritten and never used; the LLM is frozen so a new row could not be trained anyway |
| Nov 25 | Task 3 writes 196 placeholder ids per image; the model overwrites those embeddings. Image slots: attention mask 1, label -100 | `attention_mask` and `labels` stay aligned with `input_ids` for free |
| Nov 25 | Model enforces exactly `tokens_per_image` placeholders per row (not just in total), before running the encoder | A total-only check passes uneven rows (5 and 3); failing early avoids a wasted SigLIP forward |
| Nov 25 | dtypes on GPU: encoder and LLM bf16, projector fp32. Flow: encode (bf16) -> pool (bf16) -> cast fp32 -> projector -> cast to LLM embedding dtype -> splice. CPU tests all fp32 | fp32 master weights for the only trained module. bf16 path on real models first runs in the Task 5 pilot |
| Nov 25 | Freezing: encoder `requires_grad_(False)` + `eval()`, `no_grad` only around the encoder call; `SmallVLM.train()` re-forces the encoder to eval. LLM: `requires_grad_(False)` only, no `no_grad`, not forced to eval | Gradients must flow through the LLM to the projector. Letting the LLM follow `train()`/`eval()` keeps E3's LoRA dropout and optional gradient checkpointing working |
| Nov 25 | Encoder, LLM and projector are constructor arguments; loading lives in `build_small_vlm` | Tests build tiny random-weight models without downloads |
| Nov 25 | **Shrink plan:** do the minimum per task; cut extras into a Deferred list (see master file section 13) | Finish the whole pipeline first, add more if time permits |
| Nov 25 | **E1 size decided after the pilot;** default if the full 558K does not fit: a seeded random subset (~200K pairs), 1 epoch. E1 launches by Nov 29 | Protects the Dec 8 slides. Tell the LoRA half, since E3 starts from E1 |
| Nov 25 | Checkpoint = projector + optimizer + step (~22 MB), every ~15 min to Cloud Storage; resume by loading and skipping to the step | Only the projector trains, so checkpoints are tiny and resume is simple |
| Nov 25 | Gradient checkpointing off by default; enable only if the pilot runs out of memory | Expected to fit (0.5B LLM, short sequences), unverified until the pilot |
| Nov 25 | Own minimal evaluation harness (no lmms-eval) | Avoids dependency friction with the pinned versions |
| Nov 25 | Hand-inspect ~30 outputs (was ~50); format-free ScienceQA score and latency benchmarking deferred | Shrink |
| | Stage 1 LR / batch size: **TBD** (start ~1e-3, ~256; confirm on the pilot) | |
| | Seed: **TBD** (set in `train.py`, Task 4) | |

## Progress checklist
- [x] GCP project, billing, budget alerts
- [x] APIs enabled, GPU quota approved
- [x] Bucket `small_vlm` created
- [x] Old repo `small-vlm` with folder structure, `runs.csv` (backup)
- [x] Joined shared repo, created branch `no-lora`, copied structure, pushed
- [x] venv (Python 3.11.8), pinned `requirements.txt`, `pyproject.toml`, editable install
- [x] Draft `README.md` written on `no-lora`, pending team agreement
- [x] Task 1: pooling and projector + 5 tests (Nov 22)
- [x] Task 2: model assembly, tests, real-model check (Nov 25)
- [ ] Task 3: data, chat template, loss masking (`src/small_vlm/data.py`)
- [ ] Task 4: training loop with checkpoint/resume (`scripts/train.py`)
- [ ] Task 5: overfit 100 examples, then 10K pilot; decide E1 size
- [ ] Task 6: E1 -> push checkpoint -> E2
- [ ] Task 7: evaluation harness and baselines
- [ ] Task 8: analysis, slides (Dec 8), report (Dec 18)

## To do soon
- [ ] Commit and push Task 1-2 files and this file on `no-lora`
- [ ] Tell teammates: pooling (196 tokens, adaptive), the `SmallVLM` and connector interfaces, squash preprocessing, `build_small_vlm`, and the E1 plan (including a possible subset)
- [ ] Agree with teammates: pull requests into `main`, the shared E1 checkpoint, one shared evaluation harness with per-sample outputs
- [ ] Raise the budget target to $200 when the second coupon is applied
- [ ] Later (Dec 10+): README wording, "Reproducing results" table, contribution statement

## Run log (also record each kept run in `runs.csv`)
| Run | Date | Hardware | Hours | Cost | Result | Notes |
|---|---|---|---|---|---|---|
| | | | | | | |

## Open questions / problems
- Real GPU-hours and memory per step: unknown until the Task 5 pilot (decides the E1 size).
- Whether gradient checkpointing is needed: unknown until the pilot.

## Surprises and lessons (for the report and talk)
- Python patch versions differed between conda (3.11.18) and the system Python (3.11.8), which is why we pin the version and use one venv approach.
- Nov 22: adaptive pooling windows are uneven (2 or 3 patches wide, overlapping), so some output tokens average more patches than others. State this as a caveat in the report.
- Nov 22: the expected projector parameter count was first computed by hand wrongly (1,835,776); the code and test gave the correct 1,836,800. Derive expected numbers from a formula.
- Nov 22: a shape test alone cannot prove the permutes in pooling are right; the identity-pooling test (`output_size=27`, position-valued input) checks values and positions.
- Nov 22: a string placed above a `def` is not a docstring; put it on the first line inside the body.
- Nov 25: I expected SigLIP 2 to need Siglip2 classes. The pinned source and the checkpoint config show this fixed-resolution checkpoint uses the `siglip` classes with plain `pixel_values`. Check class names against the pinned source and the config.
- Nov 25: a fake LLM that returns its embeddings cannot test gradient flow through the LLM; the tiny random Qwen2 + SigLIP test can.
- Nov 25: a fake encoder with identical patch features cannot detect splice-order bugs; use features that vary by patch and by sample, and a batch of 2.
- Nov 25: correct shapes do not prove the weights loaded; comparing against the full `SiglipModel` output did.
- Nov 25: the image processor squashes non-square images. State it in the report.
- Nov 25: loading the full `SiglipModel` prints a bos/eos warning from its unused text config. Harmless for the vision path.
- Nov 25: the smoke-test loss varied between runs (9.53 vs 6.47) because the projector init is unseeded and the images differed; seed it in `train.py`.
