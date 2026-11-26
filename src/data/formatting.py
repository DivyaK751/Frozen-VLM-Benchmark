"""Text and token formatting (tasks L3-L7): LLaVA conversation -> Qwen chat tokens + labels.

    format_example(conversation, stage, text_only=False, inference=False)
        -> {"input_ids": [int, ...], "labels": [int, ...]}

Output layout (see data_and_lora_plan.md, "Image-token format"):

    <|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n
    <|im_start|>user\n<|vision_start|><|image_pad|>x169<|vision_end|>\n{question}<|im_end|>\n
    <|im_start|>assistant\n{answer}<|im_end|>\n
    ... further user/assistant turns (Stage 2) ...

- Stage 1: the user turn is the image + STAGE1_PROMPT; the assistant turn is the caption.
- Stage 2: the multi-turn conversation is kept as is; the image goes in the first user turn.
- labels: IGNORE_INDEX everywhere except assistant replies and their closing <|im_end|>
  (so the model learns to stop). The "<|im_start|>assistant\n" header is not trained on.
- text_only=True: no image tokens; everything else is identical (E0, no-image test).
- inference=True: the final assistant answer is dropped and the prompt ends with
  "<|im_start|>assistant\n"; all labels are IGNORE_INDEX. Used for eval.
- Sequences longer than MAX_LEN are cut from the right. The image block sits near the
  start, so it is never cut.
"""

import logging
from functools import lru_cache

from src.config import (IGNORE_INDEX, IMAGE_PAD_ID, IMAGE_PAD_TOKEN, LLM_NAME, MAX_LEN,
                        NUM_IMAGE_TOKENS, STAGE1_PROMPT, SYSTEM_PROMPT, VISION_END_TOKEN,
                        VISION_START_TOKEN)

log = logging.getLogger(__name__)

IMAGE_PLACEHOLDER = "<image>"
IMAGE_BLOCK = VISION_START_TOKEN + IMAGE_PAD_TOKEN * NUM_IMAGE_TOKENS + VISION_END_TOKEN + "\n"


@lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(LLM_NAME)


def strip_image_placeholder(text):
    """Remove LLaVA's '<image>' placeholder together with its separating newline."""
    if text.startswith(IMAGE_PLACEHOLDER + "\n"):
        return text[len(IMAGE_PLACEHOLDER) + 1:]
    if text.endswith("\n" + IMAGE_PLACEHOLDER):
        return text[: -len(IMAGE_PLACEHOLDER) - 1]
    return text.replace(IMAGE_PLACEHOLDER, "")


def to_messages(conversation, stage):
    """LLaVA turns -> [(role, text)] with the image placeholder removed (L3).

    Checks the conversation shape so malformed data fails loudly instead of silently
    producing bad masks: turns alternate human/gpt starting with human, and the only
    '<image>' is in the first human turn.
    """
    if stage not in (1, 2):
        raise ValueError(f"stage must be 1 or 2, got {stage!r}")
    if not conversation:
        raise ValueError("empty conversation")
    for i, turn in enumerate(conversation):
        expected = "human" if i % 2 == 0 else "gpt"
        if turn["from"] != expected:
            raise ValueError(f"turn {i} is from {turn['from']!r}, expected {expected!r}")
        if i > 0 and IMAGE_PLACEHOLDER in turn["value"]:
            raise ValueError(f"'<image>' found in turn {i}; only the first user turn may hold it")

    messages = [("user" if t["from"] == "human" else "assistant", t["value"]) for t in conversation]
    messages[0] = ("user", strip_image_placeholder(messages[0][1]))
    if stage == 1:
        if len(messages) != 2:
            raise ValueError(f"stage 1 expects one question and one caption, got {len(messages)} turns")
        messages[0] = ("user", STAGE1_PROMPT)
    return messages


def format_example(conversation, stage, text_only=False, inference=False, max_len=MAX_LEN):
    """LLaVA conversation -> {"input_ids": [...], "labels": [...]} in Qwen's chat format."""
    tok = get_tokenizer()
    messages = to_messages(conversation, stage)
    if inference and messages[-1][0] == "assistant":
        messages = messages[:-1]

    input_ids, labels = [], []

    def add(text, train):
        ids = tok.encode(text, add_special_tokens=False)
        input_ids.extend(ids)
        labels.extend(ids if train and not inference else [IGNORE_INDEX] * len(ids))

    add(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n", train=False)
    image = "" if text_only else IMAGE_BLOCK
    for role, text in messages:
        if role == "user":
            add(f"<|im_start|>user\n{image}{text}<|im_end|>\n", train=False)
            image = ""  # the image goes in the first user turn only
        else:
            add("<|im_start|>assistant\n", train=False)
            add(f"{text}<|im_end|>", train=True)
            add("\n", train=False)
    if inference:
        add("<|im_start|>assistant\n", train=False)

    n_image = input_ids.count(IMAGE_PAD_ID)
    if n_image != (0 if text_only else NUM_IMAGE_TOKENS):
        raise ValueError(f"expected {0 if text_only else NUM_IMAGE_TOKENS} image tokens, got {n_image} "
                         "(does the text itself contain image special tokens?)")

    if len(input_ids) > max_len:
        if inference:
            # Cutting would remove the generation prompt; leave it to the caller.
            log.warning("inference prompt has %d tokens (> max_len %d); not truncated",
                        len(input_ids), max_len)
        else:
            input_ids, labels = truncate(input_ids, labels, max_len)
    return {"input_ids": input_ids, "labels": labels}


def truncate(input_ids, labels, max_len=MAX_LEN):
    """Cut from the right to max_len (L6). Raises if the cut would split the image block."""
    if IMAGE_PAD_ID in input_ids:
        last_image = len(input_ids) - 1 - input_ids[::-1].index(IMAGE_PAD_ID)
        if last_image + 2 > max_len:  # keep every image pad and the <|vision_end|> after it
            raise ValueError(f"max_len {max_len} would cut the image tokens")
    log.info("truncated example from %d to %d tokens", len(input_ids), max_len)
    input_ids, labels = input_ids[:max_len], labels[:max_len]
    if all(l == IGNORE_INDEX for l in labels):
        log.warning("truncated example has no trained tokens left")
    return input_ids, labels
