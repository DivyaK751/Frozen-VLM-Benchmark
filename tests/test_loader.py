"""D7 checks on the overfit sets. Run from the repo root: python tests/test_loader.py"""
import sys, time, warnings
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
warnings.simplefilter("ignore")
import torch
from src.data.loader import get_dataloader, Collator, PAD_TOKEN_ID

IMG = 151655

def check(batch, text_only=False):
    ids, am, lab = batch["input_ids"], batch["attention_mask"], batch["labels"]
    assert ids.shape == am.shape == lab.shape
    assert ids.shape[1] % 8 == 0
    assert (lab[am == 0] == -100).all(), "padding must not be trained on"
    assert (ids[am == 0] == PAD_TOKEN_ID).all()
    assert (am.sum(1) > 0).all()
    assert ((ids == IMG) & (lab != -100)).sum() == 0, "image tokens must not be trained on"
    n_img = (ids == IMG).sum(1).tolist()
    if text_only:
        assert "pixel_values" not in batch and set(n_img) == {0}
    else:
        assert batch["pixel_values"].shape == (ids.shape[0], 3, 384, 384)
        assert set(n_img) == {169}
    return n_img

if __name__ == "__main__":
    for stage in (1, 2):
        dl = get_dataloader(stage, "overfit", batch_size=8, num_workers=2)
        t = time.time(); batches = list(dl); dt = time.time() - t
        for b in batches: check(b)
        b = batches[0]
        print(f"stage {stage}: {len(batches)} batches in {dt:.1f}s (2 workers) | "
              f"input_ids {tuple(b['input_ids'].shape)} pixel_values {tuple(b['pixel_values'].shape)} "
              f"| real lengths {b['attention_mask'].sum(1).tolist()}")

    b = next(iter(get_dataloader(2, "overfit", text_only=True, num_workers=0)))
    check(b, text_only=True); print("text_only batch keys:", sorted(b))

    a = [x["input_ids"][:3].tolist() for x in get_dataloader(2, "overfit", num_workers=0, seed=0)]
    c = [x["input_ids"][:3].tolist() for x in get_dataloader(2, "overfit", num_workers=0, seed=0)]
    d = [x["input_ids"][:3].tolist() for x in get_dataloader(2, "overfit", num_workers=0, seed=1)]
    print("same seed -> same order:", a == c, "| different seed -> different order:", a != d)

    dl = get_dataloader(1, "pilot", subset_size=50, num_workers=0)
    print("subset_size=50 ->", len(dl.dataset), "examples")

    long_item = {"input_ids": torch.arange(3000), "labels": torch.arange(3000)}
    out = Collator()([long_item])
    print("collator truncation: 3000 ->", out["input_ids"].shape[1])
    print("ALL CHECKS PASSED")
