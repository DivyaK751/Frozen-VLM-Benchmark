# NOTES.md: Decision and Progress Log

> **Reminder: update this file after every decision, run, or surprise.** Paste it at the start of each new chat.
> It lives in the repo as `docs/NOTES.md`. Commit it regularly.

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
| LLM hidden size | 896 |
| Credits applied | $100 (second coupon may raise this to $200) |
| Budget alert target | $100 (raise to $200 with the second coupon) |
| Spot A100 quota | 8 (Agent Platform, us-central1) |

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
| Nov 19 | I write all code myself, Claude guides | Learning and interview readiness |
| Nov 19 | My half = without-LoRA end to end (E1, E2, eval harness) | Team split |
| Nov 19 | Work in Divya's repo `Frozen-VLM-Benchmark` on branch `no-lora` | Team repo |
| Nov 19 | `venv` + Python 3.11 + pinned `requirements.txt` | Avoid Python/package version mismatches between machines |
| Nov 19 | `src/` layout with package `small_vlm`, `pip install -e .` | Imports work from any folder |
| | Pooling: **TBD** (13x13 = 169 tokens, or 14x14 = 196 tokens) | |
| | Stage 1 LR / batch size: **TBD** (start ~1e-3, ~256) | |
| | Seed: **TBD** | |

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
- [ ] Task 1: pooling and projector + test (`src/small_vlm/connector.py`)
- [ ] Task 2: model assembly (`src/small_vlm/model.py`)
- [ ] Task 3: data, chat template, loss masking (`src/small_vlm/data.py`)
- [ ] Task 4: training loop with checkpoint/resume (`scripts/train.py`)
- [ ] Task 5: overfit 100 examples, then 10K pilot
- [ ] Task 6: E1 -> push checkpoint -> E2
- [ ] Task 7: evaluation harness and baselines
- [ ] Task 8: analysis, slides (Dec 8), report (Dec 18)

## To do soon
- [ ] Agree with teammates: branch names and pull requests, shared E1 checkpoint, shared code and eval harness, same Python 3.11 and `requirements.txt`
- [ ] Agree with teammates on README ownership and final wording (draft is on the `no-lora` branch; avoid editing `main` README until agreed, to prevent merge conflicts)
- [ ] Fill in the "Reproducing results" table in `README.md` as results appear
- [ ] Re-run the SigLIP and Qwen shape check inside the venv (optional, ~5 GB cache)
- [ ] Raise budget target to $200 when the second coupon is applied
- [ ] Optional: request on-demand A100 training quota (backup for preempted runs)
- [ ] Optional: book the TA checkpoint (week of Nov 18)
- [ ] Once confirmed everything is in the shared repo: decide whether to archive the old `small-vlm` repo and delete the old local folder

## Run log (also record each kept run in `runs.csv`)
| Run | Date | Hardware | Hours | Cost | Result | Notes |
|---|---|---|---|---|---|---|
| | | | | | | |

## Open questions / problems
-

## Surprises and lessons (for the report and talk)
- Python patch versions differed between conda (3.11.18) and the system Python (3.11.8), which is why we pin the version and use one venv approach.
