"""Run a model over a benchmark and return one record per question.

Three input modes (abstract.md section 7):
  image      the real image                                   (the normal evaluation)
  gray       every image replaced by a uniform gray 384x384 image, passed through the full pipeline,
             so the model still gets 169 visual tokens in the same positions   (main blank-image test)
  no_image   the visual tokens are removed and the model sees only the text prompt   (secondary test;
             also how the text-only models are evaluated: zero-shot Qwen and E0)

Prompts come from the same `format_example(..., inference=True)` used for training, so the chat format,
system prompt and image block match exactly. Batched generation needs LEFT padding, done in collate().
"""

from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.config import PAD_TOKEN_ID, POOLED_GRID
from src.eval.tasks import get_pil

MODES = ("image", "gray", "no_image")


class EvalDataset(Dataset):
    def __init__(self, rows, indices, task, mode, format_fn=None, preprocess_fn=None, blank_fn=None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if format_fn is None:
            from src.data.formatting import format_example as format_fn
        if preprocess_fn is None or blank_fn is None:
            from src.data.images import blank_image, preprocess
            preprocess_fn, blank_fn = preprocess_fn or preprocess, blank_fn or blank_image
        self.rows, self.indices, self.task, self.mode = rows, list(indices), task, mode
        self.format_fn, self.preprocess_fn, self.blank_fn = format_fn, preprocess_fn, blank_fn
        self._gray = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]
        row = self.rows[i]
        prompt, meta = self.task.make_prompt(row)
        text_only = self.mode == "no_image"
        conversation = [{"from": "human", "value": "<image>\n" + prompt}]
        ids = self.format_fn(conversation, 2, text_only=text_only, inference=True)["input_ids"]
        item = {"input_ids": torch.tensor(ids), "meta": {"idx": i, **meta}}
        if self.mode == "image":
            item["pixel_values"] = self.preprocess_fn([get_pil(row["image"])])[0]
        elif self.mode == "gray":
            if self._gray is None:
                self._gray = self.preprocess_fn([self.blank_fn()])[0]
            item["pixel_values"] = self._gray
        return item


def collate(items, pad_token_id=PAD_TOKEN_ID):
    """Left-pad prompts so every row ends at the same position, which generation needs."""
    L = max(len(x["input_ids"]) for x in items)
    ids = torch.full((len(items), L), pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(items), L), dtype=torch.long)
    for b, x in enumerate(items):
        n = len(x["input_ids"])
        ids[b, L - n:] = x["input_ids"]
        mask[b, L - n:] = 1
    batch = {"input_ids": ids, "attention_mask": mask, "metas": [x["meta"] for x in items]}
    if "pixel_values" in items[0]:
        batch["pixel_values"] = torch.stack([x["pixel_values"] for x in items])
    return batch


def run_task(model, task, rows, indices, mode, device, batch_size=16, num_workers=0, decode_fn=None,
             format_fn=None, preprocess_fn=None, blank_fn=None, progress=None, pad_token_id=PAD_TOKEN_ID):
    """Generate an answer for every selected row and score it. Returns records sorted by row index."""
    if decode_fn is None:
        from src.data.formatting import get_tokenizer
        tok = get_tokenizer()
        decode_fn = lambda ids: tok.batch_decode(ids, skip_special_tokens=True)  # noqa: E731

    ds = EvalDataset(rows, indices, task, mode, format_fn, preprocess_fn, blank_fn)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=partial(collate, pad_token_id=pad_token_id))
    model.eval()
    records = []
    for batch in loader:
        pixels = batch.get("pixel_values")
        new_tokens = model.generate(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                                    None if pixels is None else pixels.to(device),
                                    max_new_tokens=task.max_new_tokens)
        for meta, text in zip(batch["metas"], decode_fn(new_tokens.cpu())):
            text = text.strip()
            rec = {"benchmark": task.name, "mode": mode, **meta, "output": text}
            if task.scored:
                rec.update(task.score(text, meta))
            records.append(rec)
        if progress:
            progress(len(records), len(ds))
    return sorted(records, key=lambda r: r["idx"])


def load_eval_model(run_dir=None, dtype=torch.bfloat16, device="cuda", pool_grid=POOLED_GRID, **build_kwargs):
    """Rebuild a trained model from a run directory (the one holding final/), or the untouched base
    Qwen when run_dir is None (the zero-shot text-only baseline).

    final/projector.pt  is loaded if it exists (E1, E2, E3, ablations); E0 and the base have none.
    final/lora/         is merged into the LLM if it exists (E3, E0, ablations).
    build_kwargs (encoder_name, llm_name) go to build_model; tests use them to point at tiny checkpoints.
    Returns (model, has_projector)."""
    from src.model.vlm import build_model

    final, projector = None, None
    if run_dir:
        final = Path(run_dir) / "final"
        final = final if final.exists() else Path(run_dir)
        projector = final / "projector.pt" if (final / "projector.pt").exists() else None
    model = build_model(lora=False, dtype=dtype, pool_grid=pool_grid,
                        init_projector=str(projector) if projector else None, **build_kwargs)
    if final is not None and (final / "lora").exists():
        model.merge_lora(final / "lora")
    return model.to(device).eval(), projector is not None
