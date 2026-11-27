"""Bootstrap statistics, following the definitions fixed in abstract.md section 5.

- Every headline number is the mean over the seeds of a condition, with a 95% percentile-bootstrap CI
  from 1,000 resamples of the TEST ITEMS (not of the seeds).
- E2 vs E3 is PAIRED: each resample picks the same items for both conditions, and the CI is on the
  difference of the two (seed-averaged) scores.
- "Largest gain": the CI of the difference between two benchmarks' gains must exclude 0. The two
  benchmarks have different items, so their resamples are independent.
- Blank-image drop is chance-normalised: (S_img - S_blank) / (S_img - S_chance), resampled jointly.

Typical use:
    e2 = Results.from_records([load_records("e2_seed0.jsonl"), load_records("e2_seed1.jsonl")], "pope")
    e3 = Results.from_records([...], "pope")
    paired_difference(e2, e3, "f1")          # -> CI(point, lo, hi) for f1(E3) - f1(E2)
"""

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np

CHUNK = 100                      # resamples processed at a time, to bound memory


@dataclass
class CI:
    point: float
    lo: float
    hi: float

    @property
    def excludes_zero(self):
        return self.lo > 0 or self.hi < 0

    def __str__(self):
        return f"{self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}]"


@dataclass
class Results:
    """Per-item results of one condition on one benchmark, for S seeds and N items."""
    ids: list                                  # item ids, the same order for every condition compared
    score: np.ndarray                          # [S, N] 0/1 (or VQA accuracy in [0, 1])
    pred_yes: Optional[np.ndarray] = None      # [S, N] bool, POPE confusion-matrix convention (failure = wrong)
    said_yes: Optional[np.ndarray] = None      # [S, N] bool, the output literally began with "yes"
    gold_yes: Optional[np.ndarray] = None      # [N] bool
    chance: Optional[np.ndarray] = None        # [N] chance-level score per item

    @classmethod
    def from_records(cls, seed_records, benchmark, split=None):
        """seed_records: one list of records per seed (as written by scripts/evaluate.py)."""
        per_seed = []
        for recs in seed_records:
            if split is not None:
                recs = [r for r in recs if r["split"] == split]
            per_seed.append(sorted(recs, key=lambda r: r["idx"]))
        ids = [r["idx"] for r in per_seed[0]]
        if any([r["idx"] for r in recs] != ids for recs in per_seed):
            raise ValueError("seeds were evaluated on different items")
        out = cls(ids=ids, score=np.array([[r["score"] for r in recs] for recs in per_seed], dtype=float))
        if benchmark == "pope":
            out.pred_yes = np.array([[r["pred_yes"] for r in recs] for recs in per_seed], dtype=bool)
            out.said_yes = np.array([[r["said_yes"] for r in recs] for recs in per_seed], dtype=bool)
            out.gold_yes = np.array([r["gold"] == "yes" for r in per_seed[0]], dtype=bool)
            out.chance = np.full(len(ids), 0.5)                              # POPE accuracy chance level
        elif benchmark == "scienceqa":
            out.chance = np.array([1 / r["n_choices"] for r in per_seed[0]])  # mean of 1/(number of options)
        elif benchmark == "textvqa":
            out.chance = np.zeros(len(ids))
        return out

    @property
    def n(self):
        return len(self.ids)


def load_records(path):
    return [json.loads(line) for line in open(path)]


# ---------------------------------------------------------------------------------------------
# metrics: each maps (Results, resample indices [B, N]) -> array [B], averaged over seeds


def _accuracy(res, idx):
    return res.score[:, idx].mean(axis=2).mean(axis=0)


def _f1(res, idx):
    p = res.pred_yes[:, idx]                              # [S, B, N]
    g = res.gold_yes[idx][None]                           # [1, B, N]
    tp, fp, fn = (p & g).sum(-1), (p & ~g).sum(-1), (~p & g).sum(-1)
    prec, rec = tp / np.maximum(tp + fp, 1), tp / np.maximum(tp + fn, 1)
    return (2 * prec * rec / np.maximum(prec + rec, 1e-12)).mean(axis=0)


def _yes_rate(res, idx):
    return res.said_yes[:, idx].mean(axis=2).mean(axis=0)


METRICS = {"accuracy": _accuracy, "f1": _f1, "yes_rate": _yes_rate}


def _check_same_items(*results):
    if any(r.ids != results[0].ids for r in results):
        raise ValueError("conditions must be evaluated on the same items, in the same order")


def _resamples(n, n_boot, seed):
    """Index matrices [<=CHUNK, n]. The same (n, n_boot, seed) always gives the same resamples, so two
    conditions evaluated with the same arguments are resampled identically (paired)."""
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_boot:
        b = min(CHUNK, n_boot - done)
        yield rng.integers(0, n, size=(b, n))
        done += b


def _samples(fn, n, n_boot, seed):
    return np.concatenate([fn(idx) for idx in _resamples(n, n_boot, seed)])


def _ci(point, samples):
    lo, hi = np.percentile(samples[np.isfinite(samples)], [2.5, 97.5])
    return CI(float(point), float(lo), float(hi))


def _point(res, metric):
    return float(METRICS[metric](res, np.arange(res.n)[None])[0])


# ---------------------------------------------------------------------------------------------
# public API


def bootstrap(res, metric, n_boot=1000, seed=0):
    """Seed-averaged value of `metric` with a 95% CI over test items."""
    s = _samples(lambda idx: METRICS[metric](res, idx), res.n, n_boot, seed)
    return _ci(_point(res, metric), s)


def difference_samples(res_a, res_b, metric, n_boot=1000, seed=0):
    """Resampled values of metric(B) - metric(A), the SAME items drawn for both (paired)."""
    _check_same_items(res_a, res_b)
    return _samples(lambda idx: METRICS[metric](res_b, idx) - METRICS[metric](res_a, idx), res_a.n, n_boot, seed)


def paired_difference(res_a, res_b, metric, n_boot=1000, seed=0):
    """The gain of B over A (for the main comparison, A = E2 and B = E3): CI of metric(B) - metric(A).
    "Significant" in the abstract's sense means CI.excludes_zero."""
    s = difference_samples(res_a, res_b, metric, n_boot, seed)
    return _ci(_point(res_b, metric) - _point(res_a, metric), s)


def difference_of_gains(gain_1, gain_2, n_boot=1000, seed=0):
    """CI of (gain on benchmark 1) - (gain on benchmark 2), for "largest gain" in H1.
    Each gain is a tuple (res_a, res_b, metric). The benchmarks have different items, so the two
    gains are resampled independently (different random streams)."""
    s1 = difference_samples(*gain_1[:2], gain_1[2], n_boot, seed)
    s2 = difference_samples(*gain_2[:2], gain_2[2], n_boot, seed + 1)
    point = (_point(gain_1[1], gain_1[2]) - _point(gain_1[0], gain_1[2])) - \
            (_point(gain_2[1], gain_2[2]) - _point(gain_2[0], gain_2[2]))
    return _ci(point, s1 - s2)


def relative_drop(res_img, res_blank, n_boot=1000, seed=0):
    """Chance-normalised drop when the image is blanked: (S_img - S_blank) / (S_img - S_chance), on the
    accuracy-type score, resampling the same items for both conditions. A value of 1 means the model
    fell to chance; 0 means blanking changed nothing. Resamples where S_img is not above chance are
    left out of the CI (the ratio is undefined there); the number left out is returned as the second value."""
    _check_same_items(res_img, res_blank)

    def fn(idx):
        s_img, s_blank = _accuracy(res_img, idx), _accuracy(res_blank, idx)
        chance = res_img.chance[idx].mean(axis=-1)
        den = s_img - chance
        return np.where(den > 0, (s_img - s_blank) / np.where(den > 0, den, 1.0), np.nan)

    s = _samples(fn, res_img.n, n_boot, seed)
    all_idx = np.arange(res_img.n)[None]
    den = _accuracy(res_img, all_idx)[0] - res_img.chance.mean()
    point = (_accuracy(res_img, all_idx)[0] - _accuracy(res_blank, all_idx)[0]) / den if den > 0 else float("nan")
    return _ci(point, s) if np.isfinite(s).any() else CI(float(point), float("nan"), float("nan")), int(np.isnan(s).sum())
