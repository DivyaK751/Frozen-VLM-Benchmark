"""Train one run (E1, E2, E3, E0 or an ablation) from a config file.

    python scripts/train.py --config configs/overfit_stage2.json          # debug: memorise 100 examples
    python scripts/train.py --config configs/e1_stage1.json               # Stage 1 alignment
    python scripts/train.py --config configs/e3_stage2_lora.json --seed 1 --output_dir outputs/e3_seed1

Any config field can be overridden on the command line (see src/train/config.py). Re-running the same
command resumes from output_dir/latest.json, so it is safe on spot/preemptible machines.
"""
import argparse
import json
import sys
import warnings
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.train.config import TrainConfig  # noqa: E402
from src.train.trainer import image_shuffle_check, resolve_device, save_final, set_seed, train  # noqa: E402

DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="JSON file in configs/")
    for f in fields(TrainConfig):                      # every field can be overridden
        flag = f"--{f.name}"
        if f.type is bool:
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        else:
            base = {"Optional[int]": int, "Optional[str]": str}.get(str(f.type).replace("typing.", ""), f.type)
            ap.add_argument(flag, type=base, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    overrides = {f.name: getattr(args, f.name) for f in fields(TrainConfig)}
    cfg = TrainConfig.from_json(args.config, **overrides)

    if cfg.text_only and not cfg.lora:
        raise SystemExit("text_only (E0) trains only LoRA: set lora=true")
    if cfg.stage == 2 and not cfg.text_only and not cfg.init_projector and cfg.split != "overfit":
        warnings.warn("Stage 2 without init_projector starts from a random projector; "
                      "pass the Stage 1 final/projector.pt")

    set_seed(cfg.seed)                                  # projector init, LoRA init, dropout

    from src.data.loader import get_dataloader
    from src.model.vlm import build_model

    base = get_dataloader(cfg.stage, cfg.split, subset_size=cfg.subset_size, text_only=cfg.text_only,
                          batch_size=1, shuffle=False, seed=cfg.seed, num_workers=0, data_root=cfg.data_root,
                          bad_images_path=str(Path(cfg.data_root) / "processed" / "bad_images.json"))
    dataset, collate_fn = base.dataset, base.collate_fn

    device = resolve_device(cfg.device)
    model = build_model(lora=cfg.lora, lora_rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
                        lora_dropout=cfg.lora_dropout, dtype=DTYPES[cfg.dtype], pool_grid=cfg.pool_grid,
                        gradient_checkpointing=cfg.gradient_checkpointing, init_projector=cfg.init_projector,
                        freeze_projector=cfg.text_only)

    summary = train(cfg, model, dataset, collate_fn)

    if cfg.image_shuffle_check and not cfg.text_only:
        res = image_shuffle_check(model, dataset, collate_fn, resolve_device(cfg.device))
        summary["image_shuffle_check"] = res
        verdict = "OK: the model uses the image" if res["loss_shuffled"] > 1.2 * res["loss_correct"] \
            else "WARNING: shuffling images barely changes the loss, the image may be ignored"
        print(f"image shuffle check: correct={res['loss_correct']:.4f} shuffled={res['loss_shuffled']:.4f} -> {verdict}")

    final = save_final(model, cfg, summary)
    print(json.dumps(summary, indent=2))
    print(f"saved to {final}")


if __name__ == "__main__":
    main()
