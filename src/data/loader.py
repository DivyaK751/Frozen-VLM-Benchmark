"""D7: collator and get_dataloader.

A batch is a dict:
  pixel_values    float [B, 3, 384, 384]   (absent when text_only=True)
  input_ids       long  [B, L]             padded with PAD_TOKEN_ID
  attention_mask  long  [B, L]             1 = real token, 0 = padding
  labels          long  [B, L]             -100 on padding and non-trained tokens

Padding is on the right (training). Batched generation for eval needs left
padding; the eval pipeline should handle that separately.
"""

import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.config import IGNORE_INDEX, MAX_LEN, PAD_TOKEN_ID
from src.data.dataset import VLMDataset, load_bad_images

# Where each split lives, relative to data_root. "full" image folders are where the
# zips get extracted on the training machine (raw/pretrain/images.zip, raw/coco/train2017.zip).
SPLITS = {
    "overfit": {1: ("pilot/stage1_overfit_100.json", "pilot/images/stage1"),
                2: ("pilot/stage2_overfit_100.json", "pilot/images/stage2")},
    "pilot":   {1: ("pilot/stage1_pilot_10k.json", "pilot/images/stage1"),
                2: ("pilot/stage2_pilot_10k.json", "pilot/images/stage2")},
    "full":    {1: ("raw/pretrain/blip_laion_cc_sbu_558k.json", "images/pretrain"),
                2: ("raw/instruct/llava_instruct_150k.json", "images/coco/train2017")},
}


class Collator:
    def __init__(self, pad_token_id=PAD_TOKEN_ID, max_len=MAX_LEN, pad_to_multiple_of=8):
        self.pad_token_id = pad_token_id
        self.max_len = max_len
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, items):
        ids = [x["input_ids"] for x in items]
        labels = [x["labels"] for x in items]

        # Safety net: format_example should already truncate (task L6). Image tokens
        # sit at the start of the sequence, so cutting from the right never breaks them.
        if any(len(s) > self.max_len for s in ids):
            warnings.warn(f"truncating sequence(s) longer than {self.max_len} in the collator")
            ids = [s[: self.max_len] for s in ids]
            labels = [s[: self.max_len] for s in labels]

        L = max(len(s) for s in ids)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            L = min(-(-L // m) * m, max(self.max_len, L))   # round up, GPU-friendly sizes

        B = len(items)
        batch = {
            "input_ids": torch.full((B, L), self.pad_token_id, dtype=torch.long),
            "attention_mask": torch.zeros((B, L), dtype=torch.long),
            "labels": torch.full((B, L), IGNORE_INDEX, dtype=torch.long),
        }
        for b, (s, lab) in enumerate(zip(ids, labels)):
            batch["input_ids"][b, : len(s)] = s
            batch["attention_mask"][b, : len(s)] = 1
            batch["labels"][b, : len(lab)] = lab

        if "pixel_values" in items[0]:
            batch["pixel_values"] = torch.stack([x["pixel_values"] for x in items])
        return batch


def get_dataloader(stage, split="pilot", subset_size=None, text_only=False, batch_size=8,
                   shuffle=True, seed=0, num_workers=4, data_root="data", format_fn=None,
                   bad_images_path="data/processed/bad_images.json"):
    """Build a DataLoader for one stage and split.

    stage        1 or 2
    split        "overfit" (100), "pilot" (10K) or "full"
    subset_size  optionally keep only the first N examples (fixed by `seed`)
    text_only    no images (E0 and the no-image test)
    format_fn    Lakshya's format_example; None = temporary stub
    """
    json_rel, img_rel = SPLITS[split][stage]
    root = Path(data_root)
    ds = VLMDataset(root / json_rel, root / img_rel, stage, format_fn=format_fn,
                    text_only=text_only, bad_images=load_bad_images(bad_images_path))

    if subset_size is not None and subset_size < len(ds):
        g = torch.Generator().manual_seed(seed)
        ds = Subset(ds, torch.randperm(len(ds), generator=g)[:subset_size].tolist())

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),   # same order for the same seed
        num_workers=num_workers,
        collate_fn=Collator(),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
