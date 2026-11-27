"""Checks for SmallVLM on tiny random models (CPU, no downloads).
Run from the repo root: python tests/test_model.py   (or: pytest tests/test_model.py)"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.simplefilter("ignore")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from _tiny import HIDDEN, IMAGE_PAD, N_IMG, POOL_GRID, TinyDataset, collate, tiny_model  # noqa: E402
from src.model.projector import VisualProjector  # noqa: E402
from src.model.vlm import SmallVLM  # noqa: E402


def batch(n=3, text_only=False):
    ds = TinyDataset(n, text_only)
    return collate()([ds[i] for i in range(n)])


def test_projector_param_count():
    p = VisualProjector()                                   # real sizes 1152 -> 896
    expected = 1152 * 896 + 896 + 896 * 896 + 896
    assert sum(x.numel() for x in p.parameters()) == expected == 1_836_800


def test_splice_order_and_values():
    """Image features must land on the placeholder positions in row-major order, per sample."""
    class FakeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))
            self.calls = 0

        def forward(self, pixel_values):
            self.calls += 1
            b = pixel_values.shape[0]
            # patch k of sample s has feature value 100*s + k in every channel (4x4 grid = 16 patches)
            f = (100 * torch.arange(b)[:, None] + torch.arange(16)[None, :]).float()
            return SimpleNamespace(last_hidden_state=f[:, :, None].expand(b, 16, HIDDEN).contiguous())

    base = tiny_model()
    proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
    proj.weight.data = torch.eye(HIDDEN)
    enc = FakeEncoder()
    model = SmallVLM(enc, base.llm, proj, pool_grid=4, image_pad_id=IMAGE_PAD)    # grid 4 = no pooling
    ids = torch.tensor([[1] + [IMAGE_PAD] * 16 + [2, 3], [1] + [IMAGE_PAD] * 16 + [4, 5]])
    embeds = model.build_inputs_embeds(ids, torch.zeros(2, 3, 56, 56))
    text = model.llm.get_input_embeddings()(ids)
    for s in range(2):
        for k in range(16):
            assert torch.allclose(embeds[s, 1 + k], torch.full((HIDDEN,), float(100 * s + k))), (s, k)
        assert torch.equal(embeds[s, 0], text[s, 0]) and torch.equal(embeds[s, -1], text[s, -1])

    # wrong number of placeholders -> error, raised before the encoder runs
    enc.calls = 0
    bad = torch.tensor([[1] + [IMAGE_PAD] * 15 + [2, 3, 4, 5], [1] + [IMAGE_PAD] * 16 + [4, 5, 6]])
    try:
        model.build_inputs_embeds(bad, torch.zeros(2, 3, 56, 56))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    assert enc.calls == 0


def test_text_only_path():
    model = tiny_model()
    b = batch(text_only=True)
    out = model(b["input_ids"], b["attention_mask"], b["labels"])           # no pixel_values
    assert torch.isfinite(out.loss)
    img = batch()
    try:
        model(img["input_ids"], img["attention_mask"], img["labels"])      # image tokens but no image
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_frozen_and_trainable():
    model = tiny_model()
    names = [n for n, _ in model.trainable_parameters()]
    assert names and all(n.startswith("projector.") for n in names)         # only the projector trains
    model.train()
    assert not model.encoder.training                                       # encoder stays in eval mode
    b = batch()
    out = model(**b)
    out.loss.backward()
    assert all(p.grad is not None for p in model.projector.parameters())
    assert all(p.grad is None for p in model.llm.parameters())              # gradient flows *through* the LLM, not into it
    assert all(p.grad is None for p in model.encoder.parameters())


def test_lora_starts_as_identity_and_targets_seven_layers():
    base = tiny_model(seed=3)
    b = batch()
    base.eval()
    ref = base(**b).logits.detach()
    lora = tiny_model(seed=3, lora=True)                                    # same weights, LoRA added
    lora.eval()
    assert torch.allclose(lora(**b).logits.detach(), ref, atol=1e-6), "LoRA must not change the output at init"
    a_names = [n for n, _ in lora.trainable_parameters() if "lora_A" in n]
    layers = lora.llm.config.num_hidden_layers
    assert len(a_names) == 7 * layers, a_names                              # q k v o gate up down in every block
    for t in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        assert sum(t in n for n in a_names) == layers
    other = [n for n, _ in lora.trainable_parameters() if not n.startswith("projector.") and "lora_" not in n]
    assert not other, other                                                 # nothing else is trainable
    trainable, total = lora.count_parameters()
    assert trainable < 0.5 * total


def test_e0_text_only_lora_trains_only_lora():
    model = tiny_model(lora=True, freeze_projector=True)
    assert all("lora_" in n for n, _ in model.trainable_parameters())
    b = batch(text_only=True)
    model.train()
    model(b["input_ids"], b["attention_mask"], b["labels"]).loss.backward()
    assert any(p.grad is not None for _, p in model.trainable_parameters())


def test_gradient_checkpointing_matches():
    """Same loss and same gradients with checkpointing, including the E0 case (frozen embeddings)."""
    for text_only in (False, True):
        b = batch(text_only=text_only)
        args = (b["input_ids"], b["attention_mask"], b["labels"]) + (() if text_only else (b["pixel_values"],))
        kw = lambda a: dict(zip(["input_ids", "attention_mask", "labels"] + ([] if text_only else ["pixel_values"]), a))
        grads = []
        for ckpt in (False, True):
            m = tiny_model(seed=5, lora=True, freeze_projector=text_only)
            if ckpt:
                m.enable_gradient_checkpointing()
            m.train()
            m(**kw(args)).loss.backward()
            grads.append({n: p.grad.clone() for n, p in m.trainable_parameters() if p.grad is not None})
        assert set(grads[0]) == set(grads[1]) and grads[0], text_only
        for n in grads[0]:
            assert torch.allclose(grads[0][n], grads[1][n], atol=1e-5), n


def test_bf16_llm_fp32_projector():
    model = tiny_model(dtype=torch.bfloat16, lora=True)
    assert next(model.projector.parameters()).dtype == torch.float32
    b = batch()
    model.train()
    out = model(**b)
    assert torch.isfinite(out.loss.float())
    out.loss.backward()
    assert all(p.grad is not None and p.grad.dtype == torch.float32 for p in model.projector.parameters())


def test_real_pooling_is_used():
    """The tiny encoder has a 4x4 grid pooled to 2x2: the model must produce exactly N_IMG tokens."""
    model = tiny_model()
    assert model.num_image_tokens == N_IMG == POOL_GRID ** 2
    assert model.encode_images(torch.randn(2, 3, 56, 56)).shape == (2, N_IMG, HIDDEN)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print("ALL CHECKS PASSED")
