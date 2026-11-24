"""TEMPORARY stand-in for Lakshya's `format_example` (tasks L3-L7).

It has the same interface as the real function so the Dataset (D6) and
collator (D7) can be built and tested now. Replace it with the real one by
passing `format_fn=format_example` to `VLMDataset`.

It follows the agreed format (see data_and_lora_plan.md, "Image-token format")
but is deliberately simple. Known shortcuts the real version must handle:
- labels are not masked per turn correctly for every edge case
- no truncation to MAX_LEN
- `inference=True` is not implemented
"""

from functools import lru_cache

LLM_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
NUM_IMAGE_TOKENS = 169
IGNORE_INDEX = -100


@lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(LLM_NAME)


def format_example(conversation, stage, text_only=False, inference=False):
    """LLaVA conversation -> {"input_ids": [...], "labels": [...]} (Qwen chat format)."""
    if inference:
        raise NotImplementedError("stub: inference mode comes with Lakshya's version")
    tok = get_tokenizer()
    image = "" if text_only else "<|vision_start|>" + "<|image_pad|>" * NUM_IMAGE_TOKENS + "<|vision_end|>\n"

    input_ids, labels = [], []

    def add(text, train):
        ids = tok.encode(text, add_special_tokens=False)
        input_ids.extend(ids)
        labels.extend(ids if train else [IGNORE_INDEX] * len(ids))

    add("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", train=False)
    for turn in conversation:
        text = turn["value"].replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
        if turn["from"] == "human":
            add(f"<|im_start|>user\n{image}{text}<|im_end|>\n", train=False)
            image = ""  # image only in the first user turn
        else:
            add("<|im_start|>assistant\n", train=False)
            add(f"{text}<|im_end|>", train=True)
            add("\n", train=False)
    return {"input_ids": input_ids, "labels": labels}
