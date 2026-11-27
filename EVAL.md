# Evaluation harness

Covers `src/eval/` and `scripts/evaluate.py`. It evaluates a trained model on ScienceQA-IMG (test), TextVQA (val) and
POPE, in three input modes, and writes one record per question so that every number can be recomputed and given a confidence
interval. It follows [abstract.md](abstract.md) section 7.

## 1. What is in it

| File | What it does |
|---|---|
| `src/eval/tasks.py` | Loads each benchmark, builds the prompt, scores one output. Dataset formats checked against the dataset cards. |
| `src/eval/parsing.py` | Output → answer, and the exact definition of a format failure (quote this in the report). |
| `src/eval/generate.py` | `run_task` (batched greedy generation), `EvalDataset` (the 3 input modes), `load_eval_model` (rebuild a run) |
| `src/eval/metrics.py` | Accuracy, VQA accuracy, POPE F1 / precision / recall / yes-rate, format-failure rates, POPE breakdowns by object and supercategory |
| `src/eval/stats.py` | Bootstrap CIs, paired E2-vs-E3 difference, difference of gains, chance-normalised relative drop |
| `src/eval/vqa_normalize.py` | TextVQA answer normaliser, copied unchanged from lmms-eval (MIT licence) |
| `src/eval/coco_objects.py` | COCO's 80 objects in their 12 supercategories |
| `scripts/evaluate.py` | The command line |
| `tests/test_eval.py` | 22 CPU tests on tiny models and synthetic data |

## 2. How to run

```bash
# E2 / E3 / ablations: point at the training run directory (the one containing final/)
python scripts/evaluate.py --run_dir outputs/e3_stage2_lora_seed0 --name e3_seed0 --modes image gray no_image

# E0 (text-only, LoRA tuned on the same data): no images
python scripts/evaluate.py --run_dir outputs/e0_text_only_lora_seed0 --name e0

# zero-shot text-only baseline: the untouched Qwen (no --run_dir)
python scripts/evaluate.py --name zeroshot_text_only

# free-form descriptions for the hand inspection (about 50 per model)
python scripts/evaluate.py --run_dir outputs/e2_stage2_projector_seed0 --name e2_seed0 --benchmarks describe

# smoke test: 64 random questions per benchmark
python scripts/evaluate.py --run_dir ... --name smoke --max_samples 64
```
On Vertex: see [GCP_RUNBOOK.md](GCP_RUNBOOK.md) (use `DATA=none`). Output goes to `<out_dir>/<name>/<mode>/`:
`<benchmark>.jsonl` (one record per question) and `metrics.json`. A finished benchmark is skipped when you re-run (`--overwrite`
redoes it), so an interrupted evaluation continues where it stopped. The same questions are used for every model; `--max_samples`
takes a random subset fixed by `--subset_seed`.

Evaluate **each seed separately** (`--name e2_seed0`, `--name e2_seed1`, ...). `stats.py` then averages over seeds.

## 3. The three input modes

| Mode | What the model sees | Used for |
|---|---|---|
| `image` | the real image | the normal evaluation |
| `gray` | a uniform gray 384×384 image through the full pipeline, so the 169 visual tokens are still there | **main blank-image test** (H3) |
| `no_image` | the visual tokens removed, text prompt only | secondary blank-image test, and how E0 and the zero-shot baseline are evaluated |

A model without a projector (E0, zero-shot) only accepts `no_image`. The prompt goes through the same `format_example(..., inference=True)`
that was used in training, so the system prompt and chat format match.

## 4. Prompts, parsing and scoring

Prompts and metrics follow lmms-eval 0.7.4's defaults (`scienceqa_img`, `textvqa_val`, `pope`). I read lmms-eval's source and
cross-checked ours against it: the ScienceQA and POPE prompts are identical, TextVQA accuracy and POPE accuracy / precision / recall / F1
are numerically identical on clean outputs, and our ScienceQA scoring never scores below lmms-eval's.

| Benchmark | Prompt | Generation | Score |
|---|---|---|---|
| ScienceQA-IMG | `[Context: <hint>\n]<question>\nA. ..\nB. ..\nAnswer with the option's letter from the given choices directly.` | up to 16 new tokens | 1 if the parsed letter is the gold one |
| TextVQA | `<Question capitalised>\nAnswer the question using a single word or phrase.` (no OCR tokens) | up to 32 new tokens | official VQA accuracy (leave-one-out over the 10 human answers, `min(1, matches/3)`) |
| POPE | `<question>\nAnswer the question using a single word or phrase.` | up to 16 new tokens | 1 if the answer begins with the gold "yes"/"no" |

Decoding is plain greedy: Qwen's default repetition penalty and sampling settings are switched off.

**Format failures** (reported for every model and benchmark, and counted as wrong in the main score; see `src/eval/parsing.py`):
- ScienceQA: the output does not name a valid option letter. Accepted forms: `B`, `B.`, `B. text`, `(B)`, `B)`, `Answer: B`, `The answer is B`, `B` followed by a newline, and a lone lowercase `b`. Not accepted: a letter that starts a sentence ("A cat is ..."), a letter beyond the number of options.
- POPE: the output does not begin with "yes" or "no". For the confusion matrix a failure is scored as the wrong answer (the opposite of gold). `yes_rate` counts only outputs that literally began with "yes".
- TextVQA: the output is empty or longer than 10 words.

Every aggregate also gives the score on the answers that did parse (`accuracy_on_valid`, and `on_valid` for POPE), as the professor asked.
A near-floor TextVQA result is a legitimate outcome and is reported as is.

**Differences from lmms-eval** (the abstract should say these):
1. We use our own harness instead of calling lmms-eval, because our model splices image features into the embeddings and we need per-question records for bootstrap CIs. It reuses lmms-eval's prompts and metric code.
2. ScienceQA and POPE parsing is a little more lenient (a superset of what lmms-eval accepts).
3. POPE generates 16 tokens instead of 128; only the start of the answer is scored.
4. TextVQA's 32-token limit is chosen so that verbose answers are long enough to be detected as format failures.

## 5. Statistics (`src/eval/stats.py`)

```python
from src.eval import stats
load = stats.load_records
e2 = stats.Results.from_records([load("outputs/eval/e2_seed0/image/pope.jsonl"), load("outputs/eval/e2_seed1/image/pope.jsonl")], "pope")
e3 = stats.Results.from_records([load("outputs/eval/e3_seed0/image/pope.jsonl"), load("outputs/eval/e3_seed1/image/pope.jsonl")], "pope")
stats.bootstrap(e3, "f1")                        # F1 averaged over the 2 seeds, 95% CI over test items
stats.paired_difference(e2, e3, "f1")            # F1(E3) - F1(E2), same items resampled for both
stats.paired_difference(e2, e3, "yes_rate")
stats.difference_of_gains((sqa_e2, sqa_e3, "accuracy"), (tvqa_e2, tvqa_e3, "accuracy"))   # "largest gain" for H1
stats.relative_drop(e3_image, e3_gray)           # chance-normalised drop, the (CI, number of undefined resamples)
```
All of it follows the definitions fixed in abstract.md section 5: seed-averaged scores, 1,000 bootstrap resamples of the test items,
paired resampling for E2 vs E3, independent resampling for the difference of two benchmarks' gains, and
`(S_img − S_blank) / (S_img − S_chance)` for the blank-image drop (chance = mean of 1/(number of options) for ScienceQA, 0 for TextVQA, 0.5 for POPE accuracy).
Metrics: `"accuracy"` (also the VQA accuracy for TextVQA), `"f1"`, `"yes_rate"` (POPE only). Pass `split="adversarial"` to
`Results.from_records` for one POPE split. A CI "excludes zero" via `.excludes_zero`.

`metrics.json` already holds the POPE breakdowns (`breakdown_object`, `breakdown_supercategory`, and the adversarial split), with the
hallucination rate (share of absent objects the model said "yes" to) per group.

## 6. What has and has not been verified

Verified (CPU, tiny models and synthetic data, plus lmms-eval 0.7.4's source):
- Prompts and scoring agree with lmms-eval as described in section 4.
- Left-padded batched generation gives the same tokens as one prompt at a time; the image changes the output.
- The three modes build the right inputs; a trained run directory (projector + LoRA) reloads into a model that matches the trained one.
- The command line end to end (outputs, skipping finished work, `--overwrite`, text-only restrictions, describe).
- Bootstrap statistics on simulated data with known answers; the Hugging Face dataset row types and local POPE parquet loading.

**Not verified** (needs the Hub, real checkpoints or a GPU): the real dataset downloads (`lmms-lab/ScienceQA`, `lmms-lab/textvqa`,
`lmms-lab/POPE`; field names follow the dataset cards), the real tokenizer path, and speed. The first `--max_samples 64` run will show
any problem, and the pilot evaluation in the runbook measures the time per benchmark.
