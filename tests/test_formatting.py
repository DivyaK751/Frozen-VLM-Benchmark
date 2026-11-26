"""L3-L7 checks for format_example. Run from the repo root: python tests/test_formatting.py

Runs on every example of the 10K pilots (both stages) if present, else the overfit sets.
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.simplefilter("ignore")

from src.config import (IGNORE_INDEX, IMAGE_PAD_ID, MAX_LEN, NUM_IMAGE_TOKENS, STAGE1_PROMPT,  # noqa: E402
                        SYSTEM_PROMPT, VISION_END_ID, VISION_START_ID)
from src.data.formatting import (IMAGE_BLOCK, format_example, get_tokenizer,  # noqa: E402
                                 to_messages)

tok = get_tokenizer()


def load(stage):
    for name in (f"stage{stage}_pilot_10k.json", f"stage{stage}_overfit_100.json"):
        p = Path("data/pilot") / name
        if p.exists():
            return json.load(open(p))
    raise FileNotFoundError(f"no pilot or overfit JSON for stage {stage}")


def reference_ids(conversation, stage, text_only=False, inference=False):
    """Qwen's own chat template applied to the same messages (independent of our code path,
    except the shared message extraction)."""
    msgs = to_messages(conversation, stage)
    if inference:
        msgs = msgs[:-1]
    chat = [{"role": r, "content": t} for r, t in msgs]
    if not text_only:
        chat[0]["content"] = IMAGE_BLOCK + chat[0]["content"]
    text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=inference)
    return tok.encode(text, add_special_tokens=False)


def check_example(conversation, stage):
    out = format_example(conversation, stage)
    ids, labels = out["input_ids"], out["labels"]
    assert len(ids) == len(labels) <= MAX_LEN

    # L3: identical to Qwen's chat template (also proves segment-wise tokenization is exact).
    assert ids == reference_ids(conversation, stage), "differs from Qwen's chat template"
    assert SYSTEM_PROMPT in tok.decode(ids[:30])

    # L4: exactly 169 contiguous image tokens, wrapped in vision start/end, in the first user turn.
    pos = [i for i, t in enumerate(ids) if t == IMAGE_PAD_ID]
    assert len(pos) == NUM_IMAGE_TOKENS and pos[-1] - pos[0] == NUM_IMAGE_TOKENS - 1
    assert ids[pos[0] - 1] == VISION_START_ID and ids[pos[-1] + 1] == VISION_END_ID
    assert tok.decode(ids[: pos[0]]).endswith("<|im_start|>user\n<|vision_start|>")

    # L5: trained tokens decode to exactly the assistant answers, each followed by <|im_end|>.
    trained = [t for t, l in zip(ids, labels) if l != IGNORE_INDEX]
    assert all(l in (IGNORE_INDEX, t) for t, l in zip(ids, labels))
    answers = [t["value"] for t in conversation if t["from"] == "gpt"]
    assert tok.decode(trained) == "".join(a + "<|im_end|>" for a in answers), "mask is wrong"

    # L7: text-only is the same sequence minus the image block.
    txt = format_example(conversation, stage, text_only=True)
    assert IMAGE_PAD_ID not in txt["input_ids"] and VISION_START_ID not in txt["input_ids"]
    assert txt["input_ids"] == reference_ids(conversation, stage, text_only=True)
    assert len(ids) - len(txt["input_ids"]) == NUM_IMAGE_TOKENS + 3  # start, pads, end, "\n"
    assert [l for l in txt["labels"] if l != IGNORE_INDEX] == trained

    # inference: prompt only, ends with the assistant header, nothing trained.
    inf = format_example(conversation, stage, inference=True)
    assert inf["input_ids"] == reference_ids(conversation, stage, inference=True)
    assert tok.decode(inf["input_ids"]).endswith("<|im_start|>assistant\n")
    assert set(inf["labels"]) == {IGNORE_INDEX}
    return len(ids), len(trained)


def main():
    for stage in (1, 2):
        data = load(stage)
        lengths, trained = zip(*(check_example(ex["conversations"], stage) for ex in data))
        s = sorted(lengths)
        print(f"stage {stage}: {len(data)} examples OK | length min/median/max "
              f"{s[0]}/{s[len(s) // 2]}/{s[-1]} | trained {sum(trained) / sum(lengths):.1%}")

    # Stage 1 uses the fixed prompt, whatever the original request was.
    ex = load(1)[0]["conversations"]
    assert tok.decode(format_example(ex, 1)["input_ids"]).count(STAGE1_PROMPT) == 1

    # L6: truncation keeps the image block and cuts the answer from the right.
    long_conv = [{"from": "human", "value": "<image>\nTell me a story."},
                 {"from": "gpt", "value": "word " * 5000}]
    out = format_example(long_conv, 2)
    assert len(out["input_ids"]) == MAX_LEN == len(out["labels"])
    assert out["input_ids"].count(IMAGE_PAD_ID) == NUM_IMAGE_TOKENS
    assert out["labels"][-1] != IGNORE_INDEX
    print("truncation: 5000-word answer ->", len(out["input_ids"]), "tokens, image intact")

    # Malformed conversations fail loudly.
    bad = [
        [{"from": "gpt", "value": "hi"}],
        [{"from": "human", "value": "<image>\nq"}, {"from": "human", "value": "q2"}],
        [{"from": "human", "value": "<image>\nq"}, {"from": "gpt", "value": "a"},
         {"from": "human", "value": "<image>\nq2"}, {"from": "gpt", "value": "a2"}],
    ]
    for conv in bad:
        try:
            format_example(conv, 2)
        except ValueError:
            continue
        raise AssertionError(f"accepted malformed conversation: {conv}")
    print("malformed conversations rejected")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
