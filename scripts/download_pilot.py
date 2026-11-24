"""D1: build the 10K pilot subsets for Stage 1 and Stage 2.

Steps:
  1. Download the annotation JSONs (LLaVA-Pretrain, LLaVA-Instruct-150K).
  2. Sample N examples from each with a fixed seed.
  3. Download only the images those examples use:
       - Stage 1: pulled out of the remote images.zip, without downloading all 27 GB
       - Stage 2: fetched one by one from the COCO train2017 server
  4. Check that every example has a readable image and write a summary.

Usage:
  python scripts/download_pilot.py              # full 10K pilot
  python scripts/download_pilot.py --n 20       # quick test
  python scripts/download_pilot.py --skip-images  # JSONs only
"""

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from huggingface_hub import get_token, hf_hub_download
from PIL import Image
from remotezip import RemoteZip
from tqdm import tqdm

PRETRAIN_REPO = "liuhaotian/LLaVA-Pretrain"
PRETRAIN_JSON = "blip_laion_cc_sbu_558k.json"
PRETRAIN_ZIP_URL = f"https://huggingface.co/datasets/{PRETRAIN_REPO}/resolve/main/images.zip"

INSTRUCT_REPO = "liuhaotian/LLaVA-Instruct-150K"
INSTRUCT_JSON = "llava_instruct_150k.json"
COCO_URL = "http://images.cocodataset.org/train2017/{}"


def download_json(repo, filename, raw_dir):
    path = hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", local_dir=raw_dir)
    with open(path) as f:
        return json.load(f)


def sample(data, n, seed):
    rng = random.Random(seed)
    return rng.sample(data, min(n, len(data)))


def hf_headers():
    """Use a Hugging Face token if available (HF_TOKEN or `hf auth login`): higher rate limits."""
    token = get_token()
    return {"Authorization": f"Bearer {token}"} if token else {}


def with_retries(fn, tries=8, base_delay=5):
    """Retry on errors such as HTTP 429 (rate limited), waiting longer each time."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(base_delay * 2 ** attempt)


def download_stage1_images(examples, out_dir, workers):
    """Extract only the needed files from the remote zip using HTTP range requests.

    Hugging Face rate-limits these requests, so keep `workers` low (default 4).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = [ex["image"] for ex in examples if not (out_dir / ex["image"]).exists()]
    if not wanted:
        return []
    headers = hf_headers()
    if not headers:
        print("  note: no Hugging Face token found; downloads may be rate-limited (run `hf auth login` to fix)")

    # Paths inside the zip may carry a prefix (e.g. "images/"); match on the suffix.
    with with_retries(lambda: RemoteZip(PRETRAIN_ZIP_URL, headers=headers)) as z:
        names = [i.filename for i in z.infolist() if not i.is_dir()]
    by_rel = {}
    for name in names:
        for rel in (name, name.split("/", 1)[-1]):
            by_rel.setdefault(rel, name)

    missing = [w for w in wanted if w not in by_rel]
    todo = [w for w in wanted if w in by_rel]

    local = threading.local()

    def get_zip():
        if not hasattr(local, "zip"):
            local.zip = with_retries(lambda: RemoteZip(PRETRAIN_ZIP_URL, headers=headers))
        return local.zip

    def fetch(rel):
        data = with_retries(lambda: get_zip().read(by_rel[rel]))
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    failed = []
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(fetch, rel): rel for rel in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="stage1 images"):
            if fut.exception():
                failed.append(futures[fut])
    return missing + failed


def download_stage2_images(examples, out_dir, workers):
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = sorted({ex["image"] for ex in examples})
    session = requests.Session()

    def fetch(name):
        dest = out_dir / name
        if dest.exists():
            return
        r = session.get(COCO_URL.format(name), timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)

    failed = []
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(fetch, name): name for name in wanted}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="stage2 images"):
            if fut.exception():
                failed.append(futures[fut])
    return failed


def verify(examples, image_dir):
    """Return (missing, corrupt) image paths."""
    missing, corrupt = [], []
    for rel in sorted({ex["image"] for ex in examples}):
        path = image_dir / rel
        if not path.exists():
            missing.append(rel)
            continue
        try:
            with Image.open(path) as im:
                im.verify()
        except Exception:
            corrupt.append(rel)
    return missing, corrupt


def dir_size_mb(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000, help="examples per stage")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--workers", type=int, default=16, help="threads for COCO (stage 2) downloads")
    ap.add_argument("--zip-workers", type=int, default=4, help="threads for the Hugging Face zip (stage 1)")
    ap.add_argument("--skip-images", action="store_true", help="only write the pilot JSONs")
    args = ap.parse_args()

    raw_dir = args.data_dir / "raw"
    pilot_dir = args.data_dir / "pilot"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.n // 1000}k" if args.n >= 1000 else str(args.n)

    print("Step 1: downloading annotation JSONs...")
    pretrain = download_json(PRETRAIN_REPO, PRETRAIN_JSON, raw_dir / "pretrain")
    instruct = download_json(INSTRUCT_REPO, INSTRUCT_JSON, raw_dir / "instruct")
    print(f"  LLaVA-Pretrain: {len(pretrain):,} examples")
    print(f"  LLaVA-Instruct-150K: {len(instruct):,} examples")

    print(f"Step 2: sampling {args.n:,} per stage (seed={args.seed})...")
    s1 = sample(pretrain, args.n, args.seed)
    s2 = sample(instruct, args.n, args.seed)
    s1_path = pilot_dir / f"stage1_pilot_{suffix}.json"
    s2_path = pilot_dir / f"stage2_pilot_{suffix}.json"
    s1_path.write_text(json.dumps(s1, indent=1))
    s2_path.write_text(json.dumps(s2, indent=1))
    print(f"  wrote {s1_path} and {s2_path}")

    summary = {"n": args.n, "seed": args.seed, "stage1_json": str(s1_path), "stage2_json": str(s2_path)}

    if not args.skip_images:
        s1_img = pilot_dir / "images" / "stage1"
        s2_img = pilot_dir / "images" / "stage2"

        print("Step 3: downloading pilot images...")
        f1 = download_stage1_images(s1, s1_img, args.zip_workers)
        f2 = download_stage2_images(s2, s2_img, args.workers)
        if f1 or f2:
            print(f"  download failures: stage1={len(f1)}, stage2={len(f2)} (re-run to retry)")

        print("Step 4: verifying images...")
        m1, c1 = verify(s1, s1_img)
        m2, c2 = verify(s2, s2_img)
        summary.update({
            "stage1": {"examples": len(s1), "missing": len(m1), "corrupt": len(c1),
                       "size_mb": round(dir_size_mb(s1_img), 1)},
            "stage2": {"examples": len(s2), "unique_images": len({e["image"] for e in s2}),
                       "missing": len(m2), "corrupt": len(c2), "size_mb": round(dir_size_mb(s2_img), 1)},
            "bad_images": {"stage1": m1 + c1, "stage2": m2 + c2},
        })

    (pilot_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "bad_images"}, indent=2))


if __name__ == "__main__":
    main()
