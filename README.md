# Frozen-VLM-Benchmark

Building a small vision-language model (VLM): how far can a frozen image encoder and a trained connector go?

COMS 4995, Foundation Model Engineering, Fall 2026. Team 14.
Divyasri Kadekar (dk3480), Lakshya Khandelwal (lk3068), Ritwik Sharan (rs4764), Vinod Walia (vw2317).

## Overview

We follow the LLaVA recipe at small scale:

```
Image (384 px) -> frozen SigLIP 2 So400m -> pooling -> 2-layer MLP projector (1152 -> 896) -> Qwen2.5-0.5B-Instruct -> answer
```

- Vision encoder: `google/siglip2-so400m-patch14-384` (frozen)
- Language model: `Qwen/Qwen2.5-0.5B-Instruct`
- Stage 1 data: `liuhaotian/LLaVA-Pretrain` (558K image-caption pairs)
- Stage 2 data: `liuhaotian/LLaVA-Instruct-150K`
- Evaluation: ScienceQA (image subset, test), TextVQA (val), POPE (random, popular, adversarial)

**Research question:** with a frozen encoder and a 0.5B LLM, how much does adding LoRA improve ScienceQA-IMG, TextVQA and POPE compared with training only the projector, and on which object types does each model hallucinate?

## Experiments

| Run | What trains | Data |
|---|---|---|
| E1 (Stage 1, alignment) | Projector only | LLaVA-Pretrain, 1 epoch |
| E2 | Projector only (no LoRA) | LLaVA-Instruct-150K, starts from E1 |
| E3 | Projector + LoRA (rank 16) | LLaVA-Instruct-150K, starts from the same E1 |

E2 and E3 share the same E1 checkpoint, data order, seed, step count, chat template, loss masking and evaluation code. LoRA is a config option in the shared code, not a separate implementation.

## Repository layout

```
pyproject.toml        package metadata (Python >=3.11,<3.12, src layout)
requirements.txt      exact pinned package versions
runs.csv              one row per kept run: GPU, hours, peak memory, cost, git commit
src/small_vlm/        importable package (connector, model, data)
configs/              training and evaluation configs
scripts/              training and launch scripts
tests/                unit tests (shapes, placeholder counts, loss masking)
eval/                 shared evaluation harness
docs/                 project notes and context
```

## Setup

Use **Python 3.11** and the pinned versions so every machine behaves the same.

Windows (Git Bash):
```bash
git clone <repo-link>
cd Frozen-VLM-Benchmark
py -3.11 -m venv .venv
source .venv/Scripts/activate
python --version                      # expect 3.11.x
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Linux:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Check the install from any folder:
```bash
python -c "import small_vlm; print(small_vlm.__file__)"
pytest
```

`pip install -e .` is an editable install: it points Python at `src/small_vlm`, so `import small_vlm` works from anywhere and code edits take effect immediately.

**Adding a package:** install it in the venv, run `pip freeze --exclude-editable > requirements.txt`, commit, and tell the team.

**Note:** `requirements.txt` pins the CPU build of torch used on laptops. On a Linux GPU machine, check `torch.cuda.is_available()` before starting a long run.

## Git workflow

- Do not commit directly to `main`.
- Each person works on their own branch (for example `no-lora` for the without-LoRA half) and merges through a pull request.
- Never commit model weights, datasets, credentials or `.venv/`. They are in `.gitignore`.

## Compute

- Google Cloud, region `us-central1` only. Checkpoints and outputs go to the Cloud Storage bucket `small_vlm`.
- Development on a CPU, Colab or a single L4 VM. Final runs on spot A100 40 GB with checkpoints at least every 30 minutes.
- Every kept run is logged in `runs.csv` (GPU, wall-clock time, peak memory, cost, commit).

## Reproducing results

*To be filled in as results appear:* the command for each table and figure, and which steps need a GPU.

| Result | Command | Needs GPU |
|---|---|---|
| E1 alignment | TBD | Yes |
| E2 projector-only | TBD | Yes |
| E3 projector + LoRA | TBD | Yes |
| Evaluation (ScienceQA, TextVQA, POPE) | TBD | Yes |
| Tables and plots | TBD | No |

## Status

Setup complete (cloud project, bucket, environment). Implementation in progress.

## Licenses and credits

- SigLIP 2 and Qwen2.5-0.5B-Instruct: Apache 2.0
- LLaVA-Pretrain: research use, subject to LAION/CC/SBU terms
- LLaVA-Instruct-150K: CC BY 4.0, subject to OpenAI terms
- ScienceQA: CC BY-NC-SA 4.0, TextVQA: CC BY 4.0, POPE: MIT

Built with Hugging Face Transformers, PEFT and lmms-eval. Method follows LLaVA (Liu et al., 2023; 2024). AI-tool use is disclosed in the final report.
