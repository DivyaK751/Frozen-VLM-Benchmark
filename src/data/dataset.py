"""D6: Dataset that turns one JSON example into model inputs.

Each item is a dict:
  pixel_values  float tensor [3, 384, 384]   (omitted when text_only=True)
  input_ids     long tensor  [L]
  labels        long tensor  [L]             (-100 = not trained on)

Text formatting is delegated to `format_fn(conversation, stage, text_only=...)`,
which defaults to Lakshya's `format_example` in `src/data/formatting.py`.
"""

import json
import random
import warnings
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.images import load_image, preprocess


def load_bad_images(report_path):
    """Image names listed by scripts/check_images.py, with any top folder stripped
    (e.g. 'train2017/000000000009.jpg' -> '000000000009.jpg')."""
    if not report_path or not Path(report_path).exists():
        return set()
    report = json.load(open(report_path))
    names = set()
    for source in report.values():
        for name in source["bad_images"]:
            names.add(name)
            names.add(name.split("/", 1)[-1])
            names.add(Path(name).name)
    return names


class VLMDataset(Dataset):
    def __init__(self, json_path, image_dir, stage, format_fn=None, text_only=False,
                 bad_images=None, max_retries=10):
        """
        json_path   LLaVA-format JSON (list of {"id", "image", "conversations"})
        image_dir   folder that `example["image"]` is relative to
        stage       1 (alignment) or 2 (instruction tuning); passed to format_fn
        format_fn   formatter; defaults to src.data.formatting.format_example
        text_only   skip images entirely (E0 and the no-image test)
        bad_images  set of image names to drop (see load_bad_images)
        """
        if format_fn is None:
            from src.data.formatting import format_example as format_fn
        self.format_fn = format_fn
        self.image_dir = Path(image_dir)
        self.stage = stage
        self.text_only = text_only
        self.max_retries = max_retries

        data = json.load(open(json_path))
        bad = bad_images or set()
        self.data = [ex for ex in data if ex["image"] not in bad]
        self.dropped = len(data) - len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # If an image fails to load at runtime (not caught by the bad-image scan),
        # warn and use a different random example instead of crashing training.
        for _ in range(self.max_retries):
            item = self._build(self.data[i])
            if item is not None:
                return item
            warnings.warn(f"skipping unreadable image: {self.data[i]['image']}")
            i = random.randrange(len(self.data))
        raise RuntimeError(f"{self.max_retries} unreadable images in a row; check image_dir={self.image_dir}")

    def _build(self, ex):
        item = {}
        if not self.text_only:
            image, err = load_image(self.image_dir / ex["image"])
            if err:
                return None
            item["pixel_values"] = preprocess([image])[0]

        tokens = self.format_fn(ex["conversations"], self.stage, text_only=self.text_only)
        item["input_ids"] = torch.tensor(tokens["input_ids"], dtype=torch.long)
        item["labels"] = torch.tensor(tokens["labels"], dtype=torch.long)
        return item
