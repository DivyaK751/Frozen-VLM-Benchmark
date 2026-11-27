"""Training loop for Stage 1 (E1), Stage 2 (E2, E3, ablations) and the text-only control (E0).

What it does:
- Only parameters with requires_grad are trained: the projector, plus the LoRA adapters if the model
  has them. SigLIP and Qwen's base weights are frozen by SmallVLM.
- One optimizer step = grad_accum micro-batches. The loss is normalised by the number of trained
  tokens in the whole step (num_items_in_batch), so accumulating gives the same gradient as one big
  batch. Without it, short answers would get as much weight as long ones.
- Learning rate: linear warm-up, then cosine to zero. Separate rates for projector and LoRA.
- Data order depends only on (seed, epoch), so a resumed run sees exactly the batches it would have
  seen, and skipping ahead costs nothing (no images are loaded for skipped batches).
- Checkpoints hold only the trainable weights, the optimizer and the step (small: about 22 MB for the
  projector, plus about 105 MB with LoRA rank 16). A run restarts from output_dir/latest.json.
"""

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import IGNORE_INDEX

# ---------------------------------------------------------------------------------------------
# helpers


def resolve_device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_multiplier(step, total_steps, warmup_ratio):
    """Linear warm-up for warmup_ratio of the run, then cosine decay to 0. `step` is 0-based."""
    warmup = max(1, int(round(total_steps * warmup_ratio)))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def epoch_batches(n_examples, micro_batch_size, seed, epoch):
    """Micro-batches (lists of dataset indices) for one epoch. Depends only on (seed, epoch).
    A final partial micro-batch is dropped."""
    g = torch.Generator().manual_seed(seed * 100003 + epoch)
    perm = torch.randperm(n_examples, generator=g).tolist()
    return [perm[i:i + micro_batch_size] for i in range(0, n_examples - micro_batch_size + 1, micro_batch_size)]


def plan_steps(n_examples, cfg):
    """Returns (steps_per_epoch, total_steps)."""
    n_micro = n_examples // cfg.micro_batch_size
    steps_per_epoch = n_micro // cfg.grad_accum
    if steps_per_epoch < 1:
        raise ValueError(f"{n_examples} examples are too few for micro_batch_size={cfg.micro_batch_size} "
                         f"x grad_accum={cfg.grad_accum}")
    total = cfg.max_steps if cfg.max_steps else cfg.epochs * steps_per_epoch
    return steps_per_epoch, total


def count_trained_tokens(labels):
    """Number of target tokens that contribute to the loss (labels are shifted by one)."""
    return int((labels[:, 1:] != IGNORE_INDEX).sum())


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def make_optimizer(model, cfg):
    """AdamW with one parameter group per trainable module: projector and LoRA."""
    proj, lora = [], []
    for name, p in model.trainable_parameters():
        (proj if name.startswith("projector.") else lora).append(p)
    groups = []
    if proj:
        groups.append({"params": proj, "lr": cfg.lr_projector, "base_lr": cfg.lr_projector, "name": "projector"})
    if lora:
        groups.append({"params": lora, "lr": cfg.lr_lora, "base_lr": cfg.lr_lora, "name": "lora"})
    if not groups:
        raise ValueError("nothing to train: no trainable parameters (text_only needs lora=true)")
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))


# ---------------------------------------------------------------------------------------------
# checkpoints


def save_checkpoint(model, optimizer, step, cfg):
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "trainable": {n: p.detach().cpu() for n, p in model.trainable_parameters()},
        "optimizer": optimizer.state_dict(),
        "rng": {"torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        "config": cfg.to_dict(),
    }
    path = out / f"ckpt_{step:07d}.pt"
    tmp = out / f"ckpt_{step:07d}.pt.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)                                   # a half-written file is never "latest"
    with open(out / "latest.json.tmp", "w") as f:
        json.dump({"step": step, "path": path.name}, f)
    os.replace(out / "latest.json.tmp", out / "latest.json")
    for old in sorted(out.glob("ckpt_*.pt"))[:-cfg.keep_last]:
        old.unlink()
    return path


def load_checkpoint(model, optimizer, cfg):
    """Restore the latest checkpoint in output_dir, if any. Returns the step to continue from."""
    latest = Path(cfg.output_dir) / "latest.json"
    if not latest.exists():
        return 0
    meta = json.load(open(latest))
    state = torch.load(Path(cfg.output_dir) / meta["path"], map_location="cpu", weights_only=False)
    params = dict(model.trainable_parameters())
    if set(state["trainable"]) != set(params):
        raise RuntimeError("checkpoint does not match this model (different LoRA settings or modules?)")
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(state["trainable"][n])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["rng"]["torch"])
    if state["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    return state["step"]


def save_final(model, cfg, summary):
    """Final artefacts: projector.pt (start of Stage 2, and for eval), lora/ (PEFT adapter), summary.json."""
    final = Path(cfg.output_dir) / "final"
    final.mkdir(parents=True, exist_ok=True)
    if any(p.requires_grad for p in model.projector.parameters()):
        torch.save({k: v.detach().cpu() for k, v in model.projector.state_dict().items()}, final / "projector.pt")
    if model.has_lora:
        model.llm.save_pretrained(final / "lora")
    with open(final / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(final / "config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    return final


# ---------------------------------------------------------------------------------------------
# the overfit diagnostic from DATA_PIPELINE.md section 7


@torch.no_grad()
def image_shuffle_check(model, dataset, collate_fn, device, micro_batch_size=8, max_batches=None):
    """Loss on the training examples with their own images vs with the images rolled by one position
    inside each batch. After memorising the overfit set, the shuffled loss should be clearly higher.
    If it is not, the model is ignoring the image (check the image-token replacement).
    Returns {"loss_correct", "loss_shuffled", "tokens"}, both losses averaged per trained token."""
    model.eval()
    loss_sum = {"correct": 0.0, "shuffled": 0.0}
    n_tokens = 0
    batches = [list(range(i, min(i + micro_batch_size, len(dataset)))) for i in range(0, len(dataset), micro_batch_size)]
    for idx in batches[:max_batches]:
        if len(idx) < 2:                                   # rolling one image changes nothing
            continue
        batch = to_device(collate_fn([dataset[i] for i in idx]), device)
        n = count_trained_tokens(batch["labels"])
        for key, pixels in (("correct", batch["pixel_values"]), ("shuffled", batch["pixel_values"].roll(1, dims=0))):
            out = model(**dict(batch, pixel_values=pixels), num_items_in_batch=n)   # loss = batch mean
            loss_sum[key] += float(out.loss) * n
        n_tokens += n
    model.train()
    return {"loss_correct": loss_sum["correct"] / n_tokens, "loss_shuffled": loss_sum["shuffled"] / n_tokens,
            "tokens": n_tokens}


# ---------------------------------------------------------------------------------------------
# the loop


def train(cfg, model, dataset, collate_fn):
    """Train `model` on `dataset` as configured. Returns a summary dict."""
    device = resolve_device(cfg.device)
    model.to(device)
    model.train()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps_per_epoch, total_steps = plan_steps(len(dataset), cfg)
    optimizer = make_optimizer(model, cfg)
    step = load_checkpoint(model, optimizer, cfg)
    resumed_from = step

    trainable, total = model.count_parameters()
    print(f"[{cfg.name}] device={device} examples={len(dataset)} steps/epoch={steps_per_epoch} "
          f"total_steps={total_steps} effective_batch={cfg.effective_batch()} "
          f"trainable={trainable:,}/{total:,} ({100 * trainable / total:.2f}%)"
          + (f" | resumed from step {step}" if step else ""), flush=True)
    if step >= total_steps:
        print("already finished; nothing to do", flush=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    log_file = open(out_dir / "train_log.jsonl", "a")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = t_last_log = t_last_save = time.time()
    tokens_since_log = 0
    loss_window, last_loss = [], float("nan")
    params = [p for _, p in model.trainable_parameters()]

    while step < total_steps:
        epoch, pos = divmod(step, steps_per_epoch)
        micro = epoch_batches(len(dataset), cfg.micro_batch_size, cfg.seed, epoch)[pos * cfg.grad_accum:
                                                                                  steps_per_epoch * cfg.grad_accum]
        loader = DataLoader(dataset, batch_sampler=micro, collate_fn=collate_fn,
                            num_workers=cfg.num_workers, pin_memory=(device == "cuda"))
        it = iter(loader)

        for _ in range(pos, steps_per_epoch):
            if step >= total_steps:
                break
            group = [next(it) for _ in range(cfg.grad_accum)]
            num_items = sum(count_trained_tokens(b["labels"]) for b in group)

            mult = lr_multiplier(step, total_steps, cfg.warmup_ratio)
            for g in optimizer.param_groups:
                g["lr"] = g["base_lr"] * mult

            step_loss = torch.zeros((), device=device)
            for b in group:
                tokens_since_log += int(b["attention_mask"].sum())
                b = to_device(b, device)
                out = model(**b, num_items_in_batch=num_items)     # loss = sum over tokens / num_items
                out.loss.backward()
                step_loss += out.loss.detach().float()

            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            last_loss = float(step_loss)
            if not math.isfinite(last_loss):
                raise FloatingPointError(f"loss is {last_loss} at step {step}; stopping before saving a bad checkpoint")
            loss_window.append(last_loss)

            if step % cfg.log_every == 0 or step == total_steps:
                now = time.time()
                rec = {"step": step, "epoch": round(step / steps_per_epoch, 4),
                       "loss": round(sum(loss_window) / len(loss_window), 5),
                       "grad_norm": round(float(grad_norm), 4),
                       "lr": {g["name"]: g["lr"] for g in optimizer.param_groups},
                       "tokens_per_s": round(tokens_since_log / max(1e-9, now - t_last_log)),
                       "elapsed_min": round((now - t_start) / 60, 2)}
                if device == "cuda":
                    rec["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                print(json.dumps(rec), flush=True)
                log_file.write(json.dumps(rec) + "\n")
                log_file.flush()
                loss_window, tokens_since_log, t_last_log = [], 0, now

            due_steps = cfg.save_every_steps and step % cfg.save_every_steps == 0
            due_time = cfg.save_every_minutes and (time.time() - t_last_save) > cfg.save_every_minutes * 60
            if (due_steps or due_time) and step < total_steps:
                save_checkpoint(model, optimizer, step, cfg)
                t_last_save = time.time()

    hours = (time.time() - t_start) / 3600
    summary = {"name": cfg.name, "steps": step, "resumed_from_step": resumed_from,
               "final_loss": last_loss, "wall_hours_this_session": round(hours, 3),
               "est_cost_usd_this_session": round(hours * cfg.gpu_hourly_usd, 2),
               "trainable_params": trainable, "total_params": total,
               "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None}
    if step > resumed_from:
        save_checkpoint(model, optimizer, step, cfg)          # so a restart after the end does nothing
    log_file.close()
    return summary
