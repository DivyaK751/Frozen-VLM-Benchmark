"""D8: data sanity checks.

1. Counts per split (examples, unique images, Q/A turns, answer length).
2. 20 random examples (10 per stage) rendered to an HTML page: the image next to
   the decoded model input and the part that is trained on. A person checks that
   each image matches its text (the one thing automated tests cannot check).
3. Loader speed on the pilot: steady-state examples per second.

Usage (repo root):
  python scripts/sanity_check.py
  open data/processed/sanity/samples.html
"""

import html
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `src`
warnings.simplefilter("ignore")

from src.data.dataset import VLMDataset  # noqa: E402
from src.data.formatting_stub import get_tokenizer  # noqa: E402
from src.data.loader import SPLITS, get_dataloader  # noqa: E402

IMG_TOKEN = "<|image_pad|>"
OUT_DIR = Path("data/processed/sanity")


def counts(data_root):
    rows = []
    for split in ("overfit", "pilot", "full"):
        for stage in (1, 2):
            path = Path(data_root) / SPLITS[split][stage][0]
            if not path.exists():
                continue
            data = json.load(open(path))
            answers = [t["value"] for ex in data for t in ex["conversations"] if t["from"] == "gpt"]
            rows.append({
                "split": split, "stage": stage, "examples": len(data),
                "unique_images": len({ex["image"] for ex in data}),
                "qa_turns": len(answers),
                "avg_answer_words": round(sum(len(a.split()) for a in answers) / len(answers), 1),
            })
    return rows


def render_samples(data_root, n_per_stage, seed):
    tok = get_tokenizer()
    cards = []
    for stage in (1, 2):
        json_rel, img_rel = SPLITS["pilot"][stage]
        ds = VLMDataset(Path(data_root) / json_rel, Path(data_root) / img_rel, stage)
        for i in random.Random(seed + stage).sample(range(len(ds)), n_per_stage):
            ex, item = ds.data[i], ds[i]
            ids, labels = item["input_ids"], item["labels"]
            full = tok.decode(ids).replace(IMG_TOKEN * 169, f"{IMG_TOKEN}×169")
            trained = tok.decode(ids[labels != -100])
            img_path = os.path.relpath(Path(data_root) / img_rel / ex["image"], OUT_DIR)
            cards.append(f"""
<div class="card">
  <img src="{html.escape(img_path)}">
  <div class="text">
    <div class="meta">Stage {stage} · id {html.escape(ex['id'])} · {html.escape(ex['image'])} ·
      {len(ids)} tokens, {int((labels != -100).sum())} trained</div>
    <b>Model input (decoded)</b><pre>{html.escape(full)}</pre>
    <b>Trained on (labels ≠ -100)</b><pre class="trained">{html.escape(trained)}</pre>
  </div>
</div>""")
    page = f"""<!doctype html><meta charset="utf-8"><title>D8 sanity check</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 24px; max-width: 1200px; }}
 .card {{ display: flex; gap: 16px; border-bottom: 1px solid #ddd; padding: 16px 0; }}
 img {{ width: 320px; height: 320px; object-fit: contain; background: #f4f4f4; flex: none; }}
 .text {{ flex: 1; min-width: 0; }}
 .meta {{ color: #666; font-size: 13px; margin-bottom: 8px; }}
 pre {{ white-space: pre-wrap; font-size: 12px; background: #f7f7f7; padding: 8px; }}
 .trained {{ background: #eaf6ea; }}
</style>
<h1>D8: does each image match its text?</h1>
<p>For every example, check that the trained text (green) describes the image. Seed {seed}.</p>
{''.join(cards)}"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "samples.html"
    out.write_text(page)
    return out


def speed(data_root, batch_size, num_workers, n_batches):
    results = {}
    for stage in (1, 2):
        dl = get_dataloader(stage, "pilot", batch_size=batch_size, num_workers=num_workers,
                            data_root=data_root)
        it = iter(dl)
        next(it)  # warm-up: worker start-up, tokenizer and processor loading
        t = time.time()
        for _ in range(n_batches):
            next(it)
        dt = time.time() - t
        results[f"stage{stage}"] = {
            "batch_size": batch_size, "num_workers": num_workers,
            "sec_per_batch": round(dt / n_batches, 3),
            "examples_per_sec": round(batch_size * n_batches / dt, 1),
        }
    return results


def main():
    data_root = "data"
    report = {"counts": counts(data_root)}
    print("Counts:")
    for r in report["counts"]:
        print("  ", r)

    out = render_samples(data_root, n_per_stage=10, seed=0)
    print(f"Samples page: {out}")

    report["speed"] = speed(data_root, batch_size=16, num_workers=4, n_batches=100)
    print("Speed:", json.dumps(report["speed"], indent=1))

    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2))
    print(f"Report: {OUT_DIR / 'report.json'}")


if __name__ == "__main__":
    main()
