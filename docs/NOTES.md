# NOTES.md: Decision and Progress Log

> **Reminder: update this file after every decision, run, or surprise.** Paste it at the start of each new chat.
> Put it in the repo as `docs/NOTES.md` and commit it regularly.

## Fixed setup
| Item | Value |
|---|---|
| GCP region | `us-central1` (keep everything here) |
| Bucket | `small_vlm` (Standard, not public) |
| Repo | `small-vlm` (private) |
| Vision encoder | `google/siglip2-so400m-patch14-384`, frozen |
| LLM | `Qwen/Qwen2.5-0.5B-Instruct` |
| Encoder output | `[B, 729, 1152]` (27x27 grid) |
| LLM hidden size | 896 |
| Credits applied | $100 (second coupon may raise this to $200) |
| Budget alert target | $100 (raise to $200 with the second coupon) |
| Spot A100 quota | 8 (Agent Platform, us-central1) |

## Decisions
| Date | Decision | Reason |
|---|---|---|
| Nov 19 | I write all code myself, Claude guides | Learning and interview readiness |
| Nov 19 | My half = without-LoRA end to end (E1, E2, eval harness) | Team split |
| | Pooling: **TBD** (13x13 = 169 tokens, or 14x14 = 196 tokens) | |
| | Stage 1 LR / batch size: **TBD** (start ~1e-3, ~256) | |
| | Seed: **TBD** | |

## Progress checklist
- [x] GCP project, billing, budget alerts
- [x] APIs enabled, GPU quota approved
- [x] Bucket `small_vlm` created
- [x] GitHub repo and folder structure, `runs.csv`
- [x] Colab check: both models download, shapes verified
- [ ] Task 1: pooling and projector + test (`src/connector.py`)
- [ ] Task 2: model assembly (`src/model.py`)
- [ ] Task 3: data, chat template, loss masking (`src/data.py`)
- [ ] Task 4: training loop with checkpoint/resume (`scripts/train.py`)
- [ ] Task 5: overfit 100 examples, then 10K pilot
- [ ] Task 6: E1 -> push checkpoint -> E2
- [ ] Task 7: evaluation harness and baselines
- [ ] Task 8: analysis, slides (Dec 8), report (Dec 18)

## To do soon
- [ ] Agree with the LoRA teammates: shared E1 checkpoint, shared code and eval harness
- [ ] Raise budget target to $200 when the second coupon is applied
- [ ] Optional: request on-demand A100 training quota (backup for preempted runs)
- [ ] Optional: book the TA checkpoint (week of Nov 18)

## Run log (also record each kept run in `runs.csv`)
| Run | Date | Hardware | Hours | Cost | Result | Notes |
|---|---|---|---|---|---|---|
| | | | | | | |

## Open questions / problems
-

## Surprises and lessons (for the report and talk)
-
