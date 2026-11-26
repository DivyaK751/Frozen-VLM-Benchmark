"""L6 + L8: text-side checks for format_example.

1. L6, truncation: format every example of a split and record the length distribution
   and how many exceed MAX_LEN (and would be cut).
2. L8, decode: 10 random pilot examples per stage, decoded with their masks, written to
   a text file for a person to read. Trained tokens are shown inside [[ ]].

Usage (repo root):
  python scripts/check_formatting.py              # full data if present, else pilot
  python scripts/check_formatting.py --split pilot
Output: data/processed/formatting/{report.json, decoded_samples.txt}
"""

import argparse
import json
import random
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.simplefilter("ignore")

from src.config import IGNORE_INDEX, IMAGE_PAD_TOKEN, MAX_LEN, NUM_IMAGE_TOKENS  # noqa: E402
from src.data.formatting import format_example, get_tokenizer  # noqa: E402
from src.data.loader import SPLITS  # noqa: E402

OUT_DIR = Path("data/processed/formatting")


def _length(args):
    conversation, stage = args
    return len(format_example(conversation, stage, max_len=10**9)["input_ids"])


def truncation_stats(path, stage, workers):
    data = json.load(open(path))
    t = time.time()
    with Pool(workers) as pool:
        lengths = pool.map(_length, [(ex["conversations"], stage) for ex in data], chunksize=512)
    s = sorted(lengths)
    over = sum(n > MAX_LEN for n in lengths)
    return {
        "file": str(path), "examples": len(data),
        "length_min": s[0], "length_median": s[len(s) // 2],
        "length_p99": s[int(len(s) * 0.99)], "length_max": s[-1],
        "truncated": over, "truncated_share": over / len(data),
        "seconds": round(time.time() - t, 1),
    }


def render(ids, labels, tok):
    """Decode with trained spans wrapped in [[ ]] and the image block shortened."""
    parts, span, trained = [], [], None
    for t, l in zip(ids + [None], labels + [None]):
        is_trained = l is not None and l != IGNORE_INDEX
        if t is None or (trained is not None and is_trained != trained):
            text = tok.decode(span)
            parts.append(f"[[{text}]]" if trained else text)
            span = []
        span.append(t)
        trained = is_trained
    return "".join(parts).replace(IMAGE_PAD_TOKEN * NUM_IMAGE_TOKENS, f"{IMAGE_PAD_TOKEN}x{NUM_IMAGE_TOKENS}")


def decoded_samples(data_root, n, seed):
    tok = get_tokenizer()
    lines = []
    for stage in (1, 2):
        data = json.load(open(Path(data_root) / SPLITS["pilot"][stage][0]))
        for i in random.Random(seed + stage).sample(range(len(data)), n):
            ex = data[i]
            out = format_example(ex["conversations"], stage)
            txt = format_example(ex["conversations"], stage, text_only=True)
            lines += [
                "=" * 100,
                f"stage {stage} | id {ex['id']} | {len(out['input_ids'])} tokens "
                f"({len(txt['input_ids'])} text-only) | "
                f"{sum(l != IGNORE_INDEX for l in out['labels'])} trained",
                "--- original LLaVA conversation ---",
                json.dumps(ex["conversations"], indent=1, ensure_ascii=False),
                "--- model input, trained tokens in [[ ]] ---",
                render(out["input_ids"], out["labels"], tok),
            ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["overfit", "pilot", "full"], default=None)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.data_root)
    split = args.split or ("full" if all((root / SPLITS["full"][s][0]).exists() for s in (1, 2)) else "pilot")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {"split": split, "max_len": MAX_LEN, "truncation": {}}
    for stage in (1, 2):
        stats = truncation_stats(root / SPLITS[split][stage][0], stage, args.workers)
        report["truncation"][f"stage{stage}"] = stats
        print(f"stage {stage} ({split}):", json.dumps(stats))

    (OUT_DIR / "decoded_samples.txt").write_text(decoded_samples(root, n=10, seed=0))
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2))
    print(f"Wrote {OUT_DIR / 'report.json'} and {OUT_DIR / 'decoded_samples.txt'}")


if __name__ == "__main__":
    main()
