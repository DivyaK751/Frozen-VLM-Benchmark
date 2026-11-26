"""Second step of the POPE near-duplicate check: keep only candidates that training uses,
and render them for a person to judge.

check_pope_near_dups.py scans whole zips. COCO train2017.zip also holds most of COCO
val2014, POPE's source, so it reports POPE's own images (same COCO ID, 0 bits) even
though LLaVA-Instruct never uses them. This script:

1. drops pairs whose training image is not referenced by the training JSONs,
2. fetches just those images from the zips in the bucket (HTTP range requests via
   remotezip, so the 47 GB zips are not downloaded),
3. writes side-by-side pictures and an HTML page, closest pairs first.

Usage (repo root, gcloud logged in):
  python scripts/review_pope_near_dups.py
Output: data/processed/pope_near_dups/review/{index.html, pairs/*.jpg, used_pairs.json}
"""

import argparse
import html
import io
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from PIL import Image
from remotezip import RemoteZip

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_pope_near_dups import load_pope, save_pair  # noqa: E402

BUCKET_URL = "https://storage.googleapis.com/frozen-vlm-benchmark"
# zip name in the report -> (zip in the bucket, training JSON, how JSON names relate to zip names)
SOURCES = {
    "images.zip": ("raw/pretrain/images.zip", "data/raw/pretrain/blip_laion_cc_sbu_558k.json"),
    "train2017.zip": ("raw/coco/train2017.zip", "data/raw/instruct/llava_instruct_150k.json"),
}


def used_names(json_path):
    return {ex["image"] for ex in json.load(open(json_path))}


def is_used(zip_name, used):
    """Zip entries may carry a top folder ('train2017/x.jpg') that the JSON lacks."""
    return zip_name in used or zip_name.split("/", 1)[-1] in used or Path(zip_name).name in used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("data/processed/pope_near_dups"))
    ap.add_argument("--pope-dir", type=Path, default=Path("data/raw/pope"))
    args = ap.parse_args()

    report = json.load(open(args.dir / "report.json"))
    keep = []
    for key, (_, json_path) in SOURCES.items():
        used = used_names(json_path)
        pairs = [p for p in report["pairs"] if Path(p["source"]).name == key]
        kept = [p for p in pairs if is_used(p["train"], used)]
        print(f"{key}: {len(pairs)} candidate pairs, {len(kept)} with an image used in training "
              f"(by distance: {dict(sorted(Counter(p['distance'] for p in kept).items()))})")
        keep += kept
    keep.sort(key=lambda p: (p["distance"], p["pope"], p["train"]))

    token = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True,
                           text=True, check=True).stdout.strip()
    headers = {"Authorization": f"Bearer {token}"}
    pope = load_pope(args.pope_dir)
    out = args.dir / "review"
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    cards = []
    for key, (bucket_path, _) in SOURCES.items():
        mine = [p for p in keep if Path(p["source"]).name == key]
        if not mine:
            continue
        with RemoteZip(f"{BUCKET_URL}/{bucket_path}", headers=headers) as z:
            for p in mine:
                p["key"] = key
                p["image"] = Image.open(io.BytesIO(z.read(p["train"]))).convert("RGB")
    for i, p in enumerate(keep):
        fname = f"{i:04d}_d{p['distance']:02d}.jpg"
        save_pair(pope[p["pope"]], p.pop("image"), out / "pairs" / fname)
        p["picture"] = f"pairs/{fname}"
        cards.append(f'<figure><img src="pairs/{fname}"><figcaption>#{i} · distance {p["distance"]} · '
                     f'{html.escape(p["pope"])} (left) vs {html.escape(p["key"])}:{html.escape(p["train"])}'
                     "</figcaption></figure>")
    (out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>POPE near-duplicates</title>'
        "<style>body{font-family:system-ui;margin:24px}img{max-width:100%}figure{margin:0 0 24px}</style>"
        f"<h1>{len(keep)} candidates whose training image is used</h1>"
        "<p>Left: POPE test image. Right: training image. Closest first.</p>" + "".join(cards))
    (out / "used_pairs.json").write_text(json.dumps(keep, indent=2))
    print(f"wrote {out}/index.html ({len(keep)} pairs)")


if __name__ == "__main__":
    main()
