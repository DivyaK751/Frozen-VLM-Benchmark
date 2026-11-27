"""Tiny random-weight stand-ins so the model and trainer can be tested on a CPU without downloads.
Not a test file itself (the leading underscore keeps pytest from collecting it)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from transformers import Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel  # noqa: E402

from src.data.loader import Collator  # noqa: E402
from src.model.projector import VisualProjector  # noqa: E402
from src.model.vlm import SmallVLM  # noqa: E402

IMAGE_PAD = 7          # stand-in for <|image_pad|> (the real one is 151655)
POOL_GRID = 2          # 4 visual tokens per image; the tiny encoder has a 4x4 patch grid
N_IMG = POOL_GRID ** 2
HIDDEN = 32
VOCAB = 300


def tiny_model(seed=0, lora=False, lora_dropout=0.0, dtype=torch.float32, freeze_projector=False):
    torch.manual_seed(seed)
    enc = SiglipVisionModel(SiglipVisionConfig(hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=2,
                                               num_attention_heads=4, image_size=56, patch_size=14)).to(dtype)
    llm = Qwen2ForCausalLM(Qwen2Config(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=64,
                                       num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                       max_position_embeddings=256, tie_word_embeddings=True)).to(dtype)
    # Random init gives tiny logits (embedding std 0.02), which caps how low the loss can go. A larger
    # embedding scale lets the tiny model memorise, as a real pretrained model would.
    llm.get_input_embeddings().weight.data.normal_(0, 0.5)
    model = SmallVLM(enc, llm, VisualProjector(HIDDEN, HIDDEN).float(), pool_grid=POOL_GRID, image_pad_id=IMAGE_PAD)
    if lora:
        model.add_lora(rank=4, alpha=8, dropout=lora_dropout)
    if freeze_projector:
        model.projector.requires_grad_(False)
    return model


class TinyDataset(torch.utils.data.Dataset):
    """Every prompt is identical, and the answer depends only on the image, so the model can only
    fit the data by using the image. Items look like VLMDataset items."""

    def __init__(self, n=8, text_only=False, fail_after=None):
        self.n, self.text_only, self.fail_after, self.calls = n, text_only, fail_after, 0

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise KeyboardInterrupt("simulated crash")            # not an Exception, like a killed job
        prefix = [1] + ([] if self.text_only else [IMAGE_PAD] * N_IMG) + [2, 3, 4]
        answer_len = 2 + i % 3                                    # different answer lengths per example
        answer = [100 + i] * answer_len + [5]
        ids = prefix + answer
        labels = [-100] * len(prefix) + answer
        item = {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels)}
        if not self.text_only:
            g = torch.Generator().manual_seed(1000 + i)
            item["pixel_values"] = torch.randn(3, 56, 56, generator=g)
        return item


def collate():
    return Collator(pad_token_id=0, max_len=256, pad_to_multiple_of=0)
