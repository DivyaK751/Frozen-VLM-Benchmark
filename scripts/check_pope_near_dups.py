"""D3 follow-up: near-duplicate check between POPE's test images and the training images.

D3 matched COCO IDs exactly. That cannot catch the same photo appearing under another
name, e.g. a COCO image re-posted on the web and scraped into LAION/CC/SBU
(LLaVA-Pretrain). This script compares perceptual hashes (64-bit pHash) instead:
two images whose hashes differ in few bits look the same, even after resizing or
re-compression.

1. Hash POPE's 500 unique images (from the lmms-lab/POPE parquet files).
2. Hash every training image, read directly from the zips (no extraction) or folders.
3. Report every (POPE, training) pair within --max-distance bits, with counts per
   threshold, and save side-by-side pictures of the candidates for a person to judge.

Usage:
  python scripts/check_pope_near_dups.py --zip images.zip --zip train2017.zip --workers 16
  python scripts/check_pope_near_dups.py --dir data/pilot/images      # local test on the pilot
Output: data/processed/pope_near_dups/{report.json, pairs/*.jpg, index.html}
"""

import argparse
import glob
import html
import io
import json
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
THRESHOLDS = (0, 4, 6, 8, 10, 12)

# --- pHash (same algorithm as the `imagehash` library's phash, without the dependency) ---
_N = 32
_DCT = np.sqrt(2 / _N) * np.cos(np.pi * (2 * np.arange(_N)[None, :] + 1) * np.arange(_N)[:, None] / (2 * _N))
_DCT[0] /= np.sqrt(2)
_BITS = (1 << np.arange(63, -1, -1, dtype=np.uint64))


def phash(image):
    """64-bit perceptual hash: grayscale 32x32 -> 2-D DCT -> top-left 8x8 vs its median."""
    g = np.asarray(image.convert("L").resize((_N, _N), Image.LANCZOS), dtype=np.float64)
    low = (_DCT @ g @ _DCT.T)[:8, :8]
    return int((_BITS[(low > np.median(low)).ravel()]).sum())


def hamming(a, b):
    """Bitwise distance between one uint64 hash `a` and an array of hashes `b`."""
    x = np.bitwise_xor(b, np.uint64(a))
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)


# --- hashing training images in parallel, straight from zips or folders ---
_zip = None


def _init(zip_path):
    global _zip
    _zip = zipfile.ZipFile(zip_path) if zip_path else None


def _hash_one(name):
    try:
        src = io.BytesIO(_zip.read(name)) if _zip else name
        with Image.open(src) as im:
            return name, phash(im)
    except Exception:
        return name, None  # unreadable images are check_images.py's job


def hash_source(names, zip_path, workers):
    hashes, names_ok, failed = [], [], 0
    t = time.time()
    with Pool(workers, initializer=_init, initargs=(zip_path,)) as pool:
        for i, (name, h) in enumerate(pool.imap_unordered(_hash_one, names, chunksize=256), 1):
            if h is None:
                failed += 1
            else:
                names_ok.append(name)
                hashes.append(h)
            if i % 50_000 == 0:
                print(f"  {i:,}/{len(names):,} hashed ({i / (time.time() - t):.0f}/s)", flush=True)
    return names_ok, np.array(hashes, dtype=np.uint64), failed


def list_images(zip_path=None, folder=None):
    if zip_path:
        names = zipfile.ZipFile(zip_path).namelist()
    else:
        names = [str(p) for p in Path(folder).rglob("*")]
    return sorted(n for n in names if not n.endswith("/") and Path(n).suffix.lower() in IMAGE_EXTS)


def load_pope(pope_dir):
    """Unique POPE images: {image_source: PIL image}."""
    images = {}
    for f in sorted(glob.glob(str(Path(pope_dir) / "data" / "*.parquet"))):
        table = pq.read_table(f, columns=["image_source", "image"]).to_pylist()
        for row in table:
            if row["image_source"] not in images:
                images[row["image_source"]] = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
    return images


def open_train_image(name, zip_path):
    if zip_path:
        with zipfile.ZipFile(zip_path) as z:
            return Image.open(io.BytesIO(z.read(name))).convert("RGB")
    return Image.open(name).convert("RGB")


def save_pair(a, b, path, height=256):
    a = a.resize((max(1, a.width * height // a.height), height))
    b = b.resize((max(1, b.width * height // b.height), height))
    canvas = Image.new("RGB", (a.width + b.width + 8, height), "white")
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width + 8, 0))
    canvas.save(path, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, action="append", default=[], help="zip of training images (repeatable)")
    ap.add_argument("--dir", type=Path, action="append", default=[], help="folder of training images (repeatable)")
    ap.add_argument("--pope-dir", type=Path, default=Path("data/raw/pope"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/pope_near_dups"))
    ap.add_argument("--max-distance", type=int, default=10, help="report pairs within this many bits")
    ap.add_argument("--max-pictures", type=int, default=300, help="side-by-side pictures to save")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    assert args.zip or args.dir, "give at least one --zip or --dir"

    pope = load_pope(args.pope_dir)
    assert len(pope) == 500, f"expected 500 unique POPE images, got {len(pope)}"
    pope_names = sorted(pope)
    pope_hashes = np.array([phash(pope[n]) for n in pope_names], dtype=np.uint64)
    print(f"POPE: {len(pope_names)} images hashed")

    report = {"pope_images": len(pope_names), "max_distance": args.max_distance,
              "hash": "pHash 64-bit (32x32 grayscale DCT, 8x8 low frequencies vs median)",
              "sources": {}, "pairs": []}
    sources = [(str(z), z) for z in args.zip] + [(str(d), None) for d in args.dir]
    for label, zip_path in sources:
        names = list_images(zip_path=zip_path) if zip_path else list_images(folder=label)
        print(f"{label}: {len(names):,} images", flush=True)
        t = time.time()
        names_ok, hashes, failed = hash_source(names, zip_path, args.workers)

        # 500 x N distances, one POPE image at a time (N = 558K -> 558K x 8 bytes each).
        best = np.full(len(names_ok), 65)
        found = []
        for p, ph in zip(pope_names, pope_hashes):
            d = hamming(ph, hashes)
            best = np.minimum(best, d)
            for j in np.flatnonzero(d <= args.max_distance):
                found.append({"distance": int(d[j]), "pope": p, "train": names_ok[j], "source": label})

        report["sources"][label] = {
            "images": len(names), "hashed": len(names_ok), "unreadable": failed,
            "seconds": round(time.time() - t, 1),
            # training images within d bits of ANY POPE image
            "train_images_within": {str(k): int((best <= k).sum()) for k in THRESHOLDS},
            "pope_images_within": {str(k): len({f["pope"] for f in found if f["distance"] <= k})
                                   for k in THRESHOLDS if k <= args.max_distance},
        }
        report["pairs"] += found
        print(f"  done: {json.dumps(report['sources'][label]['train_images_within'])}", flush=True)

    # Side-by-side pictures of the closest pairs, for a person to judge.
    report["pairs"].sort(key=lambda f: (f["distance"], f["pope"], f["train"]))
    pic_dir = args.out_dir / "pairs"
    pic_dir.mkdir(parents=True, exist_ok=True)
    zips = {str(z): z for z in args.zip}
    cards = []
    for i, f in enumerate(report["pairs"][: args.max_pictures]):
        fname = f"{i:04d}_d{f['distance']:02d}.jpg"
        save_pair(pope[f["pope"]], open_train_image(f["train"], zips.get(f["source"])), pic_dir / fname)
        f["picture"] = f"pairs/{fname}"
        cards.append(f'<figure><img src="pairs/{fname}"><figcaption>distance {f["distance"]} · '
                     f'{html.escape(f["pope"])} (left) vs {html.escape(f["train"])} (right)</figcaption></figure>')
    (args.out_dir / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>POPE near-duplicates</title>'
        "<style>body{font-family:system-ui;margin:24px}img{max-width:100%}figure{margin:0 0 24px}</style>"
        f"<h1>POPE vs training: {len(report['pairs'])} pairs within {args.max_distance} bits</h1>"
        "<p>Left: POPE test image. Right: training image. Judge which are the same photo.</p>"
        + "".join(cards))
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"{len(report['pairs'])} candidate pairs; wrote {args.out_dir}/report.json and index.html")


if __name__ == "__main__":
    sys.exit(main())
