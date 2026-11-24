"""D3: check whether any POPE test image appears in the Stage 2 training data.

POPE (lmms-lab/POPE, as loaded by lmms-eval) uses 500 COCO val2014 images.
LLaVA-Instruct-150K uses COCO images named by their train2017 file names.
COCO image IDs are unique across splits, so overlap = shared numeric IDs.

If any overlap is found, the matching conversations are removed and the
filtered JSON is written next to the original (`*_nopope.json`).

Usage:
  hf download lmms-lab/POPE --repo-type dataset --include "data/*" --local-dir data/raw/pope
  python scripts/check_pope_overlap.py
"""

import argparse
import glob
import json
from pathlib import Path

import pyarrow.parquet as pq


def pope_image_ids(pope_dir):
    """'COCO_val2014_000000310196' -> 310196"""
    ids, rows = set(), 0
    for f in sorted(glob.glob(str(pope_dir / "data" / "*.parquet"))):
        col = pq.read_table(f, columns=["image_source"]).column("image_source").to_pylist()
        rows += len(col)
        ids |= {int(s.split("_")[-1]) for s in col}
    return ids, rows


def coco_id(image_name):
    """'000000310196.jpg' -> 310196"""
    return int(Path(image_name).stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pope-dir", type=Path, default=Path("data/raw/pope"))
    ap.add_argument("--instruct", type=Path, default=Path("data/raw/instruct/llava_instruct_150k.json"))
    ap.add_argument("--pilot", type=Path, default=Path("data/pilot/stage2_pilot_10k.json"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/pope_overlap.json"))
    args = ap.parse_args()

    pope_ids, pope_rows = pope_image_ids(args.pope_dir)
    assert pope_ids, f"no POPE parquet files found in {args.pope_dir}/data"
    assert 310196 in pope_ids, "self-test failed: known POPE image ID not found"

    report = {"pope_questions": pope_rows, "pope_unique_images": len(pope_ids), "datasets": {}}

    for name, path in [("instruct_full", args.instruct), ("instruct_pilot", args.pilot)]:
        if not path.exists():
            print(f"skip {name}: {path} not found")
            continue
        data = json.load(open(path))
        keep = [ex for ex in data if coco_id(ex["image"]) not in pope_ids]
        removed = [ex for ex in data if coco_id(ex["image"]) in pope_ids]
        overlap_imgs = sorted({coco_id(ex["image"]) for ex in removed})
        report["datasets"][name] = {
            "file": str(path),
            "conversations": len(data),
            "unique_images": len({coco_id(ex["image"]) for ex in data}),
            "overlapping_images": len(overlap_imgs),
            "conversations_removed": len(removed),
            "overlapping_image_ids": overlap_imgs,
        }
        if removed:
            out = path.with_name(path.stem + "_nopope.json")
            out.write_text(json.dumps(keep))
            report["datasets"][name]["filtered_file"] = str(out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
