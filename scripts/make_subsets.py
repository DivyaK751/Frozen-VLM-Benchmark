"""D5: make the small overfit subsets (100 examples per stage).

Used to check the pipeline before real runs: a working model should drive the
training loss on these to near 0 within a few hundred steps. If it cannot,
there is a bug (loss masks, image tokens, chat template).

The subsets are drawn from the 10K pilot (D1), so their images are already
downloaded. A fixed seed means everyone gets the same 100 examples.

Usage:
  python scripts/make_subsets.py
"""

import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", type=Path, default=Path("data/pilot"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for stage in (1, 2):
        src = args.pilot_dir / f"stage{stage}_pilot_10k.json"
        data = json.load(open(src))
        subset = random.Random(args.seed).sample(data, args.n)

        img_dir = args.pilot_dir / "images" / f"stage{stage}"
        missing = [ex["image"] for ex in subset if not (img_dir / ex["image"]).exists()]
        assert not missing, f"stage {stage}: {len(missing)} images missing, e.g. {missing[:3]}"

        out = args.pilot_dir / f"stage{stage}_overfit_{args.n}.json"
        out.write_text(json.dumps(subset, indent=1))
        turns = sum(len(ex["conversations"]) // 2 for ex in subset)
        print(f"stage {stage}: {len(subset)} examples, {turns} Q/A turns, "
              f"{len({ex['image'] for ex in subset})} unique images -> {out}")


if __name__ == "__main__":
    main()
