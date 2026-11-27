"""Aggregate per-sample records into the numbers in the report.

Every function takes a list of records (dicts as written to the JSONL files by scripts/evaluate.py).
Format failures count as wrong in the main score; accuracy on the answers that did parse is reported
next to it (abstract.md section 7).
"""

from collections import defaultdict

POPE_SPLITS = ("random", "popular", "adversarial")


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def confusion_metrics(pred_yes, gold_yes):
    """Accuracy, precision, recall, F1 with "yes" as the positive class (the POPE convention)."""
    tp = sum(p and g for p, g in zip(pred_yes, gold_yes))
    fp = sum(p and not g for p, g in zip(pred_yes, gold_yes))
    fn = sum((not p) and g for p, g in zip(pred_yes, gold_yes))
    n = len(gold_yes)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": (n - fp - fn) / n if n else float("nan"), "precision": precision, "recall": recall, "f1": f1}


def aggregate_scienceqa(records):
    valid = [r for r in records if not r["format_failure"]]
    return {"n": len(records),
            "accuracy": _mean(r["score"] for r in records),               # failures count as wrong
            "format_failure_rate": 1 - len(valid) / len(records),
            "accuracy_on_valid": _mean(r["score"] for r in valid),
            "chance": _mean(1 / r["n_choices"] for r in records)}


def aggregate_textvqa(records):
    valid = [r for r in records if not r["format_failure"]]
    return {"n": len(records),
            "accuracy": _mean(r["score"] for r in records),               # VQA accuracy, failures count as scored
            "format_failure_rate": 1 - len(valid) / len(records),
            "accuracy_on_valid": _mean(r["score"] for r in valid)}


def _pope_block(records):
    gold = [r["gold"] == "yes" for r in records]
    main = confusion_metrics([r["pred_yes"] for r in records], gold)
    valid = [r for r in records if not r["format_failure"]]
    on_valid = confusion_metrics([r["pred"] == "yes" for r in valid], [r["gold"] == "yes" for r in valid]) if valid else {}
    return {"n": len(records), **main,
            "yes_rate": _mean(r["said_yes"] for r in records),
            "format_failure_rate": 1 - len(valid) / len(records),
            "on_valid": {"accuracy": on_valid.get("accuracy"), "f1": on_valid.get("f1")}}


def aggregate_pope(records):
    out = {"all": _pope_block(records)}
    for split in POPE_SPLITS:
        sub = [r for r in records if r["split"] == split]
        if sub:
            out[split] = _pope_block(sub)
    return out


def pope_breakdown(records, by="object", split=None):
    """Hallucination by object type. For each group:
       hallucination_rate = share of questions whose object is NOT in the image where the model said "yes"
       (the FPR); also accuracy and the share of "yes" answers. Sorted with the worst group first."""
    groups = defaultdict(list)
    for r in records:
        if split is None or r["split"] == split:
            groups[r[by] or "unknown"].append(r)
    out = {}
    for name, rs in groups.items():
        neg = [r for r in rs if r["gold"] == "no"]
        out[name] = {"n": len(rs), "n_absent": len(neg),
                     "hallucination_rate": _mean(r["said_yes"] for r in neg) if neg else None,
                     "accuracy": _mean(r["score"] for r in rs),
                     "yes_rate": _mean(r["said_yes"] for r in rs)}
    return dict(sorted(out.items(), key=lambda kv: -(kv[1]["hallucination_rate"] or 0)))


AGGREGATORS = {"scienceqa": aggregate_scienceqa, "textvqa": aggregate_textvqa, "pope": aggregate_pope}


def summarize(benchmark, records):
    """The metrics.json entry for one benchmark."""
    out = AGGREGATORS[benchmark](records)
    if benchmark == "pope":
        out["breakdown_supercategory"] = pope_breakdown(records, by="supercategory")
        out["breakdown_object"] = pope_breakdown(records, by="object")
        out["breakdown_supercategory_adversarial"] = pope_breakdown(records, by="supercategory", split="adversarial")
    return out
