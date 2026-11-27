"""The benchmarks: how to load them, what prompt to send, and how to score one output.

Prompts are the ones lmms-eval 0.7.4 uses by default (tasks scienceqa_img, textvqa_val, pope), so our
numbers are comparable to lmms-eval's. Differences, all deliberate and documented in parsing.py:
  - ScienceQA and POPE outputs are parsed a little more leniently than lmms-eval's exact match;
  - TextVQA uses no OCR tokens in the prompt (lmms-eval's default `ocr: false`);
  - POPE generates at most 16 tokens (lmms-eval: 128). Only the start of the answer is scored.

A Task turns a dataset row into (prompt, meta) and scores a model output against `meta`. `meta` holds
the gold answer and ids only, never the image, so it can be written to JSONL as it is.

Dataset schemas (Hugging Face, checked against the dataset cards):
  lmms-lab/ScienceQA, config ScienceQA-IMG, split test (2,017): image, question, choices[list], answer[int], hint
  lmms-lab/textvqa, split validation (5,000): image, question, answers[10 strings], question_id, ...
  lmms-lab/POPE, split test (9,000): image, question, answer["yes"|"no"], question_id, image_source,
                                    category["random"|"popular"|"adversarial"]
"""

import glob
import io
import random
import re
import statistics
from dataclasses import dataclass
from typing import Any, Callable

from src.eval.coco_objects import supercategory
from src.eval.parsing import is_textvqa_format_failure, parse_choice, parse_yes_no
from src.eval.vqa_normalize import EvalAIAnswerProcessor

SQA_POST = "\nAnswer with the option's letter from the given choices directly."
SHORT_ANSWER_POST = "\nAnswer the question using a single word or phrase."
DESCRIBE_PROMPT = "Describe the image in detail."

_POPE_OBJECT = re.compile(r"^Is there an? (.+?) in the image\??$", re.IGNORECASE)
_normalizer = EvalAIAnswerProcessor()


@dataclass(frozen=True)
class Task:
    name: str
    max_new_tokens: int
    load: Callable[..., Any]                        # (**kwargs) -> rows (indexable, len())
    make_prompt: Callable[[dict], tuple]            # row -> (prompt_text, meta)
    score: Callable[[str, dict], dict]              # (output, meta) -> fields added to the record
    scored: bool = True                             # False for "describe" (hand inspection only)


# ---------------------------------------------------------------------------------------------
# helpers


def get_pil(image):
    """PIL image from a datasets Image value (already PIL) or a raw {"bytes": ...} dict (local parquet)."""
    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(io.BytesIO(image["bytes"]))
    return image.convert("RGB")


def subset_indices(n_total, max_samples=None, seed=0):
    """All indices, or a fixed random subset of max_samples (sorted), the same for every model."""
    if max_samples is None or max_samples >= n_total:
        return list(range(n_total))
    return sorted(random.Random(seed).sample(range(n_total), max_samples))


# ---------------------------------------------------------------------------------------------
# ScienceQA (image subset, test)


def sqa_prompt(row):
    context, question, choices = row["hint"], row["question"], row["choices"]
    options = [chr(ord("A") + i) for i in range(len(choices))]
    choices_str = "\n".join(f"{o}. {c}" for o, c in zip(options, choices))
    if context:
        context = f"Context: {context}\n"
    meta = {"gold": int(row["answer"]), "n_choices": len(choices)}
    return f"{context}{question}\n{choices_str}{SQA_POST}", meta


def sqa_score(output, meta):
    idx = parse_choice(output, meta["n_choices"])
    return {"pred": None if idx is None else chr(ord("A") + idx),
            "score": float(idx == meta["gold"]),
            "format_failure": idx is None}


def load_scienceqa(**_):
    from datasets import load_dataset

    return load_dataset("lmms-lab/ScienceQA", "ScienceQA-IMG", split="test")


# ---------------------------------------------------------------------------------------------
# TextVQA (validation)


def textvqa_prompt(row):
    return f"{row['question'].capitalize()}{SHORT_ANSWER_POST}", {"answers": list(row["answers"])}


def vqa_accuracy(output, answers):
    """Official VQA accuracy: leave-one-out over the 10 human answers, min(1, matches / 3), averaged.
    Same computation as lmms-eval's textvqa_process_results."""
    pred = _normalizer(output)
    gts = [_normalizer(a) for a in answers]
    accs = []
    for i in range(len(gts)):
        others = [g for j, g in enumerate(gts) if j != i]
        accs.append(min(1.0, sum(g == pred for g in others) / 3))
    return statistics.mean(accs), pred


def textvqa_score(output, meta):
    acc, pred = vqa_accuracy(output, meta["answers"])
    return {"pred": pred, "score": acc, "format_failure": is_textvqa_format_failure(output)}


def load_textvqa(**_):
    from datasets import load_dataset

    return load_dataset("lmms-lab/textvqa", split="validation")


# ---------------------------------------------------------------------------------------------
# POPE (random, popular, adversarial)


def pope_prompt(row):
    question = row["question"].strip()
    m = _POPE_OBJECT.match(question)
    obj = m.group(1).lower() if m else None
    meta = {"gold": row["answer"].lower().strip(), "split": row["category"], "object": obj,
            "supercategory": supercategory(obj) if obj else "unknown"}
    return f"{question}{SHORT_ANSWER_POST}", meta


def pope_score(output, meta):
    pred = parse_yes_no(output)
    gold = meta["gold"]
    # A format failure is scored as the wrong answer: for the confusion matrix that means the opposite of gold.
    pred_yes = (pred == "yes") if pred else (gold == "no")
    return {"pred": pred, "score": float(pred == gold), "format_failure": pred is None,
            "pred_yes": bool(pred_yes), "said_yes": pred == "yes"}


def load_pope(pope_dir=None, **_):
    """From the Hub, or from a local copy (`hf download lmms-lab/POPE --repo-type dataset
    --include "data/*" --local-dir data/raw/pope`, which the overlap check already uses)."""
    from datasets import load_dataset

    if pope_dir:
        files = sorted(glob.glob(f"{pope_dir}/data/*.parquet"))
        if files:
            return load_dataset("parquet", data_files=files, split="train")
    return load_dataset("lmms-lab/POPE", split="test")


# ---------------------------------------------------------------------------------------------
# describe: free-form descriptions for the hand inspection (about 50 per model), not scored


def describe_prompt(row):
    return DESCRIBE_PROMPT, {"image_source": row["image_source"], "split": row["category"]}


def describe_indices(rows, n=50, seed=0):
    """One row per distinct POPE image, then a fixed random subset of n images."""
    try:
        sources = rows["image_source"]                 # Hugging Face dataset: column access, no image decoding
    except TypeError:
        sources = [r["image_source"] for r in rows]    # plain list of dicts
    first = {}
    for i, src in enumerate(sources):
        first.setdefault(src, i)
    return subset_indices_from(sorted(first.values()), n, seed)


def subset_indices_from(indices, n, seed=0):
    return sorted(random.Random(seed).sample(indices, n)) if n < len(indices) else list(indices)


TASKS = {
    "scienceqa": Task("scienceqa", 16, load_scienceqa, sqa_prompt, sqa_score),
    "textvqa": Task("textvqa", 32, load_textvqa, textvqa_prompt, textvqa_score),
    "pope": Task("pope", 16, load_pope, pope_prompt, pope_score),
    "describe": Task("describe", 200, load_pope, describe_prompt, lambda out, meta: {}, scored=False),
}
