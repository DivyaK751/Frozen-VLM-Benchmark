# NOTES.md: Decision and Progress Log

> **Reminder: update this file at the end of each task, and after any big decision, run, or surprise.** Paste it at the start of each new chat.
> It lives in the repo as `docs/NOTES.md`. Commit it regularly.
> Last updated: Nov 22, 2026 (Task 1 complete).

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
| Tests | `tests/test_connector.py`, 5 tests pass |

## Environment (everyone uses the same)
| Item | Value |
|---|---|
| Python | 3.11 (laptop: 3.11.8). `pyproject.toml` requires >=3.11,<3.12 |
| Virtual env | `venv` in `.venv/` (git-ignored). Not conda |
| Install | `pip install -r requirements.txt` then `pip install -e .` |
| torch | 2.15.0 (CPU build on laptop) |
| torchvision | 0.29.2 |
| transformers | 5.24.0 (new major version; Colab check used different versions) |
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

**Known risks:** `requirements.txt` was frozen on Windows, so a Windows-only package may fail on Linux (remove that line and note it here). On the Linux GPU VM, check `torch.cuda.is_available()` on the cheap L4 first.

## Decisions
| Date | Decision | Reason |
|---|---|---|
| Nov 21 | I write all code myself, Claude guides | Learning and interview readiness |
| Nov 21 | My half = without-LoRA end to end (E1, E2, eval harness) | Team split |
| Nov 21 | Work in Divya's repo `Frozen-VLM-Benchmark` on branch `no-lora` | Team repo |
| Nov 21 | `venv` + Python 3.11 + pinned `requirements.txt` | Avoid Python/package version mismatches between machines |
| Nov 21 | `src/` layout with package `small_vlm`, `pip install -e .` | Imports work from any folder |
| Nov 22 | Schedule re-planned: Tasks 1-4 by Nov 25, E1 launch ~Nov 26-27, slides Dec 8 | Setup took longer than planned |
| Nov 22 | Pooling: `adaptive_avg_pool2d` 27x27 -> 14x14 = 196 tokens, applied before the projector | Keeps every patch. `avg_pool2d(kernel=2)` (13x13 = 169) would drop the last row and column (~7% of patches, always bottom/right), a systematic crop that could confound TextVQA and POPE. Pooling first means the projector runs on 196 tokens, not 729. Cost: ~16% more tokens than 169 |
| Nov 22 | Pooling output size is a function argument, not a constant | 81- and 729-token variants (Track B) become a config change |
| Nov 22 | `VisualProjector`: Linear 1152->896, GELU, Linear 896->896, layers named `fc1`/`act`/`fc2` | Readable `state_dict` keys for the E1 checkpoint. 1,836,800 parameters (bias on) |
| Nov 22 | Connector code (function plus class) lives in one file, `connector.py` | Cohesive, ~30 lines, one test file; split only if a second connector appears |
| Nov 22 | Working style: purpose in plain English with a diagram first; NOTES.md updated at the end of each task | My preference: no jumping ahead, batch the notes |
| | Stage 1 LR / batch size: **TBD** (start ~1e-3, ~256) | |
| | Seed: **TBD** | |
| | Image placeholder mechanism and visual-token splicing: **TBD (Task 2)** | |
| | dtype of projector and LLM (bf16 vs fp32 master weights): **TBD (Task 2)** | |

## Progress checklist
- [x] GCP project, billing, budget alerts
- [x] APIs enabled, GPU quota approved
- [x] Bucket `small_vlm` created
- [x] Old repo `small-vlm` with folder structure, `runs.csv` (backup)
- [x] Colab check: both models download, shapes verified
- [x] Joined shared repo, created branch `no-lora`, copied structure, pushed
- [x] venv (Python 3.11.8), pinned `requirements.txt`, `pyproject.toml`, editable install
- [x] Import check in venv passes with pinned versions
- [x] Draft `README.md` (repo root) written on `no-lora`, pending team agreement
- [x] Task 1: pooling and projector + 5 tests (`src/small_vlm/connector.py`, `tests/test_connector.py`) (Nov 22)
- [ ] Task 2: model assembly (`src/small_vlm/model.py`)
- [ ] Task 3: data, chat template, loss masking (`src/small_vlm/data.py`)
- [ ] Task 4: training loop with checkpoint/resume (`scripts/train.py`)
- [ ] Task 5: overfit 100 examples, then 10K pilot
- [ ] Task 6: E1 -> push checkpoint -> E2
- [ ] Task 7: evaluation harness and baselines
- [ ] Task 8: analysis, slides (Dec 8), report (Dec 18)

## To do soon
- [ ] Confirm Task 1 is committed and pushed on `no-lora`: `git add src/small_vlm/connector.py tests/test_connector.py docs/NOTES.md`, commit, `git push`
- [ ] Tell teammates the pooling decision (196 tokens, adaptive) and the connector interface, since E3 inherits both
- [ ] Agree with teammates: branch names and pull requests, shared E1 checkpoint, shared code and eval harness, same Python 3.11 and `requirements.txt`
- [ ] Agree with teammates on README ownership and final wording (draft is on the `no-lora` branch; avoid editing `main` README until agreed, to prevent merge conflicts)
- [ ] Fill in the "Reproducing results" table in `README.md` as results appear
- [ ] Re-run the SigLIP and Qwen shape check inside the venv (optional, ~5 GB cache); needed for Task 2
- [ ] Optional: add a `pytest.raises(AssertionError)` test for a wrong token count in pooling; derive the expected parameter count from a formula in the test
- [ ] Raise budget target to $200 when the second coupon is applied
- [ ] Optional: request on-demand A100 training quota (backup for preempted runs)
- [ ] Once confirmed everything is in the shared repo: decide whether to archive the old `small-vlm` repo and delete the old local folder

## Run log (also record each kept run in `runs.csv`)
| Run | Date | Hardware | Hours | Cost | Result | Notes |
|---|---|---|---|---|---|---|
| | | | | | | |

## Open questions / problems
- Task 2: how to load only the SigLIP 2 vision tower in transformers 5.24.0, and which placeholder mechanism is cleanest for splicing 196 visual tokens.

## Surprises and lessons (for the report and talk)
- Python patch versions differed between conda (3.11.18) and the system Python (3.11.8), which is why we pin the version and use one venv approach.
- Nov 22: adaptive pooling windows are uneven (2 or 3 patches wide, overlapping), so some output tokens average more patches than others. State this as a caveat in the report.
- Nov 22: the expected projector parameter count was first computed by hand wrongly (1,835,776); the code and test gave the correct 1,836,800. Derive expected numbers from a formula in the test rather than from memory.
- Nov 22: a shape test alone cannot prove the permutes in pooling are right (a wrong permute can still give the right shape). The identity-pooling test (`output_size=27`, position-valued input) is what checks that values and positions are preserved.
- Nov 22: a string placed above a `def` is not a docstring; put it on the first line inside the function or class body.
