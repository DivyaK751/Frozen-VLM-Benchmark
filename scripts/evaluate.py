"""Evaluate one trained model (or the zero-shot text-only baseline) on ScienceQA, TextVQA and POPE.

    # E2 / E3 / ablations: point at the run directory that holds final/
    python scripts/evaluate.py --run_dir outputs/e3_stage2_lora_seed0 --name e3_seed0 --modes image gray no_image

    # E0 (text-only, LoRA tuned on the same data): evaluated without images
    python scripts/evaluate.py --run_dir outputs/e0_text_only_lora_seed0 --name e0

    # zero-shot text-only baseline: the untouched Qwen, no --run_dir
    python scripts/evaluate.py --name zeroshot_text_only

    # free-form descriptions for the hand inspection (about 50 per model)
    python scripts/evaluate.py --run_dir outputs/e2_stage2_projector_seed0 --name e2_seed0 --benchmarks describe

    # quick smoke test
    python scripts/evaluate.py --run_dir ... --name smoke --max_samples 64

Writes, per mode:  <out_dir>/<name>/<mode>/<benchmark>.jsonl   one record per question (output, parsed
answer, score, format failure, gold, ...) and  <out_dir>/<name>/<mode>/metrics.json.
Finished benchmarks are skipped on a re-run (use --overwrite to redo), so it is safe on spot machines.
The same questions are used for every model (the subset, if any, is fixed by --subset_seed).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.eval.generate import MODES, load_eval_model, run_task  # noqa: E402
from src.eval.metrics import summarize  # noqa: E402
from src.eval.tasks import TASKS, describe_indices, subset_indices  # noqa: E402
from src.train.trainer import resolve_device  # noqa: E402

DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default=None, help="training run directory (holding final/); omit for zero-shot Qwen")
    ap.add_argument("--name", required=True, help="label for this model, used in output paths")
    ap.add_argument("--modes", nargs="+", choices=MODES, default=None,
                    help="image | gray | no_image. Default: image if the model has a projector, else no_image")
    ap.add_argument("--benchmarks", nargs="+", choices=list(TASKS), default=["scienceqa", "textvqa", "pope"])
    ap.add_argument("--max_samples", type=int, default=None, help="random subset per benchmark (smoke tests)")
    ap.add_argument("--subset_seed", type=int, default=0)
    ap.add_argument("--describe_n", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_dir", default="outputs/eval")
    ap.add_argument("--pope_dir", default="data/raw/pope", help="local POPE copy; falls back to the Hub")
    ap.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = resolve_device(args.device)
    model, has_projector = load_eval_model(args.run_dir, DTYPES[args.dtype], device)
    modes = args.modes or (["image"] if has_projector else ["no_image"])
    if not has_projector and set(modes) != {"no_image"}:
        raise SystemExit("this model has no projector (text-only), so only --modes no_image makes sense")

    loaded = {}                                        # load each dataset once, even across modes
    for mode in modes:
        out = Path(args.out_dir) / args.name / mode
        out.mkdir(parents=True, exist_ok=True)
        metrics_path = out / "metrics.json"
        metrics = json.load(open(metrics_path)) if metrics_path.exists() else {}
        for bench in args.benchmarks:
            path = out / f"{bench}.jsonl"
            if path.exists() and not args.overwrite:
                print(f"[{args.name}/{mode}] {bench}: already done, skipping", flush=True)
                continue
            task = TASKS[bench]
            if task.load not in loaded:
                loaded[task.load] = task.load(pope_dir=args.pope_dir)
            rows = loaded[task.load]
            idx = (describe_indices(rows, args.describe_n, args.subset_seed) if bench == "describe"
                   else subset_indices(len(rows), args.max_samples, args.subset_seed))
            t0 = time.time()
            print(f"[{args.name}/{mode}] {bench}: {len(idx)} questions", flush=True)
            records = run_task(model, task, rows, idx, mode, device, args.batch_size, args.num_workers,
                               progress=lambda done, total: print(f"  {done}/{total}", end="\r", flush=True))
            tmp = path.with_suffix(".jsonl.tmp")
            with open(tmp, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            tmp.replace(path)
            if task.scored:
                metrics[bench] = summarize(bench, records)
                json.dump(metrics, open(metrics_path, "w"), indent=2)
                head = metrics[bench]["all"] if bench == "pope" else metrics[bench]
                keys = ("n", "accuracy", "f1", "yes_rate", "format_failure_rate", "accuracy_on_valid")
                print(f"[{args.name}/{mode}] {bench}: " + ", ".join(
                    f"{k}={head[k]:.4f}" if isinstance(head.get(k), float) else f"{k}={head[k]}"
                    for k in keys if k in head) + f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print("done")


if __name__ == "__main__":
    main()
