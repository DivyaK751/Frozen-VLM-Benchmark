"""D4: scan images and log any that cannot be loaded (bad-image log).

Works on a folder of images or directly on a zip (no extraction needed), so it
can run on the full training zips on a cloud machine before training.

Usage:
  python scripts/check_images.py --dir data/pilot/images --out data/processed/bad_images_pilot.json
  python scripts/check_images.py --zip images.zip --zip train2017.zip --out bad_images.json --workers 8
"""

import argparse
import io
import json
import sys
import time
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `src`
from src.data.images import load_image  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

_zip = None  # one open ZipFile per worker process


def _init_zip(path):
    global _zip
    _zip = zipfile.ZipFile(path) if path else None


def _check(name):
    """Return (name, original_mode or None, error or None)."""
    try:
        source = io.BytesIO(_zip.read(name)) if _zip else name
        with Image.open(source) as im:
            mode = im.mode
        if _zip:
            source.seek(0)
    except Exception as e:
        return name, None, f"{type(e).__name__}: {e}"
    _, err = load_image(source)
    return name, mode, err


def scan(names, zip_path, workers):
    modes, bad = Counter(), {}
    with Pool(workers, initializer=_init_zip, initargs=(zip_path,)) as pool:
        for i, (name, mode, err) in enumerate(pool.imap_unordered(_check, names, chunksize=256), 1):
            if err:
                bad[name] = err
            else:
                modes[mode] += 1
            if i % 50_000 == 0:
                print(f"  {i:,}/{len(names):,} checked, {len(bad)} bad")
    return modes, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, action="append", default=[], help="folder of images (repeatable)")
    ap.add_argument("--zip", type=Path, action="append", default=[], help="zip of images (repeatable)")
    ap.add_argument("--out", type=Path, default=Path("data/processed/bad_images.json"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    assert args.dir or args.zip, "give at least one --dir or --zip"

    report = {}
    sources = [(str(d), None, sorted(str(p) for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS))
               for d in args.dir]
    for z in args.zip:
        names = [n for n in zipfile.ZipFile(z).namelist()
                 if not n.endswith("/") and Path(n).suffix.lower() in IMAGE_EXTS]
        sources.append((str(z), str(z), sorted(names)))

    for label, zip_path, names in sources:
        print(f"{label}: {len(names):,} images")
        t = time.time()
        modes, bad = scan(names, zip_path, args.workers)
        report[label] = {
            "images": len(names),
            "ok": sum(modes.values()),
            "bad": len(bad),
            "original_modes": dict(modes),
            "seconds": round(time.time() - t, 1),
            "bad_images": bad,
        }
        print(f"  done: {len(bad)} bad, modes {dict(modes)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
