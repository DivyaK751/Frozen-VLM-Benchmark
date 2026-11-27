"""Checks for the eval harness on tiny random models and synthetic rows (CPU, no downloads).
Run from the repo root: python tests/test_eval.py   (or: pytest tests/test_eval.py)"""
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.simplefilter("ignore")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from _tiny import HIDDEN, IMAGE_PAD, N_IMG, POOL_GRID, VOCAB, tiny_model  # noqa: E402
from src.eval import stats  # noqa: E402
from src.eval.coco_objects import OBJECT_TO_SUPERCATEGORY, SUPERCATEGORIES, supercategory  # noqa: E402
from src.eval.generate import EvalDataset, collate, load_eval_model, run_task  # noqa: E402
from src.eval.metrics import aggregate_pope, aggregate_scienceqa, aggregate_textvqa, confusion_metrics, pope_breakdown, summarize  # noqa: E402
from src.eval.parsing import is_textvqa_format_failure, parse_choice, parse_yes_no  # noqa: E402
from src.eval.tasks import (TASKS, describe_indices, get_pil, load_pope, pope_prompt, pope_score, sqa_prompt,  # noqa: E402
                            sqa_score, subset_indices, textvqa_prompt, textvqa_score, vqa_accuracy)


# ---------------------------------------------------------------------------------------------
# parsing and prompts (expected values were checked against lmms-eval 0.7.4)


def test_parse_choice():
    ok = {"B": 1, "B.": 1, "B. West Virginia": 1, "(B)": 1, "B)": 1, "Answer: B": 1, "The answer is B.": 1,
          "  c  ": 2, "a": 0, "b.": 1, "A\nB": 0}
    for text, idx in ok.items():
        assert parse_choice(text, 4) == idx, text
    bad = ["", "A cat is on the table.", "a cat", "The image shows a map.", "E", "Option A", "AB", "42", "I think B"]
    for text in bad:
        assert parse_choice(text, 4) is None, text
    assert parse_choice("D", 3) is None and parse_choice("C", 3) == 2     # letter must be a valid option


def test_parse_yes_no():
    for text, ans in {"Yes": "yes", "yes.": "yes", "No": "no", "NO, there is not": "no", "\"Yes\"": "yes", "no.": "no"}.items():
        assert parse_yes_no(text) == ans, text
    for text in ["", "None", "Nothing is there", "Yesterday", "There is a dog", "Not sure", "Maybe yes"]:
        assert parse_yes_no(text) is None, text


def test_textvqa_format_failure():
    assert is_textvqa_format_failure("") and is_textvqa_format_failure("   ")
    assert not is_textvqa_format_failure("coca cola") and not is_textvqa_format_failure("one two three four five six seven eight nine ten")
    assert is_textvqa_format_failure("one two three four five six seven eight nine ten eleven")


def test_prompts_match_lmms_eval():
    p, meta = sqa_prompt({"hint": "", "question": "Which is farthest north?", "choices": ["West Virginia", "Louisiana"], "answer": 0})
    assert p == "Which is farthest north?\nA. West Virginia\nB. Louisiana\nAnswer with the option's letter from the given choices directly."
    assert meta == {"gold": 0, "n_choices": 2}
    p, _ = sqa_prompt({"hint": "A map.", "question": "q", "choices": ["x", "y"], "answer": 1})
    assert p.startswith("Context: A map.\nq\nA. x")
    assert textvqa_prompt({"question": "what is the brand name?", "answers": []})[0] == \
        "What is the brand name?\nAnswer the question using a single word or phrase."
    p, meta = pope_prompt({"question": " Is there a dining table in the image? ", "answer": "Yes", "category": "adversarial"})
    assert p == "Is there a dining table in the image?\nAnswer the question using a single word or phrase."
    assert meta == {"gold": "yes", "split": "adversarial", "object": "dining table", "supercategory": "furniture"}


def test_vqa_accuracy():
    assert vqa_accuracy("stop", ["stop"] * 10)[0] == 1.0
    assert abs(vqa_accuracy("stop", ["stop"] * 2 + ["go"] * 8)[0] - 0.6) < 1e-12   # (2*1/3 + 8*2/3) / 10
    assert vqa_accuracy("stop", ["go"] * 10)[0] == 0.0
    assert vqa_accuracy("The Stop.", ["stop"] * 10)[0] == 1.0                      # normalisation: case, article, period
    assert vqa_accuracy("three", ["3"] * 10)[0] == 1.0                             # number words
    assert vqa_accuracy("", ["stop"] * 10)[0] == 0.0


def test_scoring_and_failure_convention():
    meta = {"gold": 1, "n_choices": 3}
    assert sqa_score("B", meta) == {"pred": "B", "score": 1.0, "format_failure": False}
    assert sqa_score("A", meta)["score"] == 0.0 and not sqa_score("A", meta)["format_failure"]
    s = sqa_score("The image shows a map.", meta)
    assert s["score"] == 0.0 and s["format_failure"] and s["pred"] is None
    # POPE: a format failure is scored as the wrong answer = the opposite of gold in the confusion matrix
    for gold in ("yes", "no"):
        s = pope_score("There might be one.", {"gold": gold})
        assert s["format_failure"] and s["score"] == 0.0 and s["pred_yes"] == (gold == "no") and not s["said_yes"]
    s = pope_score("Yes, there is.", {"gold": "no"})
    assert s["pred"] == "yes" and s["score"] == 0.0 and s["pred_yes"] and s["said_yes"] and not s["format_failure"]
    t = textvqa_score("a very long answer that goes on and on and on past ten words", {"answers": ["x"] * 10})
    assert t["format_failure"]


def test_coco_supercategories():
    assert len(OBJECT_TO_SUPERCATEGORY) == 80 and len(SUPERCATEGORIES) == 12          # all of COCO, no duplicates
    assert supercategory("skis") == "sports" and supercategory("tv") == "electronic" and supercategory("x") == "unknown"


# ---------------------------------------------------------------------------------------------
# metrics


def pope_rec(gold, out, split="random", obj="car"):
    row = {"question": f"Is there a {obj} in the image?", "answer": gold, "category": split}
    meta = pope_prompt(row)[1]
    return {**meta, **pope_score(out, meta), "idx": 0}


def test_confusion_metrics_and_pope_aggregate():
    m = confusion_metrics([True, True, False, False], [True, False, True, False])
    assert m == {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5}
    recs = ([pope_rec("yes", "Yes")] * 3 + [pope_rec("yes", "No")] + [pope_rec("no", "No")] * 2 + [pope_rec("no", "Yes")] * 2
            + [pope_rec("no", "Maybe")])
    agg = aggregate_pope(recs)["all"]
    assert agg["n"] == 9 and abs(agg["format_failure_rate"] - 1 / 9) < 1e-12
    # unparseable on a "no" question counts as a (wrong) "yes" in the confusion matrix: tp=3 fp=3 fn=1 tn=2
    assert abs(agg["precision"] - 0.5) < 1e-12 and abs(agg["recall"] - 0.75) < 1e-12
    assert abs(agg["yes_rate"] - 5 / 9) < 1e-12                   # said yes: 3 + 2; the failure is not a "yes"
    assert agg["on_valid"]["accuracy"] == 5 / 8                  # parseable only: 3 + 2 right of 8


def test_pope_breakdown():
    recs = [pope_rec("no", "Yes", "adversarial", "dog"), pope_rec("no", "No", "adversarial", "dog"),
            pope_rec("no", "No", "random", "tv"), pope_rec("yes", "Yes", "random", "tv"),
            pope_rec("no", "Yes", "popular", "dog")]
    b = pope_breakdown(recs, by="object")
    assert list(b) == ["dog", "tv"]                                # worst first
    assert b["dog"]["hallucination_rate"] == 2 / 3 and b["dog"]["n_absent"] == 3
    assert b["tv"]["hallucination_rate"] == 0.0 and b["tv"]["accuracy"] == 1.0
    assert pope_breakdown(recs, by="object", split="adversarial")["dog"]["hallucination_rate"] == 0.5
    assert set(pope_breakdown(recs, by="supercategory")) == {"animal", "electronic"}


def test_scienceqa_textvqa_aggregate():
    recs = [{"score": 1.0, "format_failure": False, "n_choices": 4}, {"score": 0.0, "format_failure": False, "n_choices": 2},
            {"score": 0.0, "format_failure": True, "n_choices": 4}, {"score": 0.0, "format_failure": True, "n_choices": 2}]
    a = aggregate_scienceqa(recs)
    assert a["accuracy"] == 0.25 and a["format_failure_rate"] == 0.5 and a["accuracy_on_valid"] == 0.5
    assert abs(a["chance"] - 0.375) < 1e-12
    t = aggregate_textvqa([{"score": 0.6, "format_failure": False}, {"score": 0.0, "format_failure": True}])
    assert t["accuracy"] == 0.3 and t["accuracy_on_valid"] == 0.6 and t["format_failure_rate"] == 0.5
    assert "breakdown_object" in summarize("pope", [pope_rec("no", "Yes")])


# ---------------------------------------------------------------------------------------------
# dataset, generation and model loading


def fake_format(conversation, stage, text_only=False, inference=False):
    """Stand-in for format_example (the real one needs the Qwen tokenizer from the Hub)."""
    assert stage == 2 and inference and conversation[0]["value"].startswith("<image>\n")
    n = 3 + len(conversation[0]["value"]) % 4                     # prompt length varies by row
    return {"input_ids": [1] + ([] if text_only else [IMAGE_PAD] * N_IMG) + [2] * n + [3]}


def fake_preprocess(images):
    return torch.stack([torch.from_numpy(np.asarray(im.resize((56, 56)), dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)
                        for im in images])


def fake_blank():
    return Image.new("RGB", (56, 56), (128, 128, 128))


def pope_rows(n=6):
    rows = []
    for i in range(n):
        rows.append({"question": f"Is there a {['dog', 'car', 'tv'][i % 3]} in the image?", "answer": ["yes", "no"][i % 2],
                     "category": ["random", "popular", "adversarial"][i % 3], "image_source": f"COCO_val2014_{i // 2:012d}",
                     "image": Image.new("RGB", (80 + i, 64), (10 * i, 20, 30))})
    return rows


def test_eval_dataset_modes_and_left_padding():
    rows, task = pope_rows(), TASKS["pope"]
    mk = lambda mode: EvalDataset(rows, [0, 1, 2, 3], task, mode, fake_format, fake_preprocess, fake_blank)  # noqa: E731
    img, gray, none = mk("image"), mk("gray"), mk("no_image")
    assert img[0]["pixel_values"].shape == (3, 56, 56) and not torch.equal(img[0]["pixel_values"], img[1]["pixel_values"])
    assert torch.equal(gray[0]["pixel_values"], gray[3]["pixel_values"])           # every image replaced by the same gray one
    assert abs(float(gray[0]["pixel_values"].abs().max())) < 0.01                  # ~0 after normalisation
    assert (img[0]["input_ids"] == IMAGE_PAD).sum() == N_IMG and (gray[0]["input_ids"] == IMAGE_PAD).sum() == N_IMG
    assert "pixel_values" not in none[0] and (none[0]["input_ids"] == IMAGE_PAD).sum() == 0   # tokens removed
    batch = collate([img[i] for i in range(4)], pad_token_id=0)
    lens = (batch["attention_mask"]).sum(1).tolist()
    assert len(set(lens)) > 1, "the test needs prompts of different lengths"
    for b, n in enumerate(lens):                                                   # padding on the LEFT
        assert batch["attention_mask"][b, -n:].all() and not batch["attention_mask"][b, :-n].any() if n < batch["input_ids"].shape[1] else True
        assert batch["input_ids"][b, -1] == 3
    try:
        EvalDataset(rows, [0], task, "blurry", fake_format, fake_preprocess, fake_blank)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_generation_is_batch_invariant():
    """Left-padded batched greedy decoding must give the same tokens as one prompt at a time."""
    model = tiny_model(seed=7).eval()
    ds = EvalDataset(pope_rows(), range(6), TASKS["pope"], "image", fake_format, fake_preprocess, fake_blank)
    items = [ds[i] for i in range(6)]
    assert len({len(x["input_ids"]) for x in items}) > 1
    batch = collate(items, pad_token_id=0)
    together = model.generate(batch["input_ids"], batch["attention_mask"], batch["pixel_values"], max_new_tokens=6,
                              eos_token_id=VOCAB - 1, pad_token_id=0)
    for b, x in enumerate(items):
        one = collate([x], pad_token_id=0)
        alone = model.generate(one["input_ids"], one["attention_mask"], one["pixel_values"], max_new_tokens=6,
                               eos_token_id=VOCAB - 1, pad_token_id=0)
        assert torch.equal(together[b, :alone.shape[1]], alone[0]), b
    # the image must matter: a different image changes the logits
    other = collate([items[0]], pad_token_id=0)
    flipped = dict(other, pixel_values=-other["pixel_values"])
    a = model(other["input_ids"], other["attention_mask"], pixel_values=other["pixel_values"]).logits
    b = model(flipped["input_ids"], flipped["attention_mask"], pixel_values=flipped["pixel_values"]).logits
    assert not torch.allclose(a, b)


def test_run_task_records():
    model, rows, task = tiny_model(seed=3), pope_rows(), TASKS["pope"]
    outputs = iter(["Yes.", "No", "I am not sure", "yes", "no", "Yes"] * 2)
    decode = lambda ids: [next(outputs) for _ in range(len(ids))]  # noqa: E731
    seen = {}
    for mode in ("image", "gray", "no_image"):
        recs = run_task(model, task, rows, [4, 1, 2, 0], mode, "cpu", batch_size=3, decode_fn=decode, format_fn=fake_format,
                        preprocess_fn=fake_preprocess, blank_fn=fake_blank, pad_token_id=0)
        assert [r["idx"] for r in recs] == [0, 1, 2, 4]                           # sorted by row index
        assert all(r["mode"] == mode and r["benchmark"] == "pope" for r in recs)
        assert {"output", "pred", "score", "format_failure", "gold", "split", "object", "supercategory", "pred_yes", "said_yes"} <= set(recs[0])
        seen[mode] = recs
        outputs = iter(["Yes.", "No", "I am not sure", "yes", "no", "Yes"] * 2)
    assert sum(r["format_failure"] for r in seen["image"]) == 1
    # describe: not scored, only outputs
    d = TASKS["describe"]
    idx = describe_indices(rows, n=2, seed=0)
    assert len(idx) == 2 and len({rows[i]["image_source"] for i in idx}) == 2
    recs = run_task(model, d, rows, idx, "image", "cpu", decode_fn=lambda ids: ["A photo."] * len(ids), format_fn=fake_format,
                    preprocess_fn=fake_preprocess, blank_fn=fake_blank, pad_token_id=0)
    assert all("score" not in r and r["output"] == "A photo." for r in recs)


def test_subset_indices():
    assert subset_indices(10) == list(range(10)) and subset_indices(10, 20) == list(range(10))
    a = subset_indices(1000, 50, seed=0)
    assert a == subset_indices(1000, 50, seed=0) and a != subset_indices(1000, 50, seed=1) and a == sorted(a) and len(set(a)) == 50


def test_merge_lora_and_load_eval_model():
    """A saved run directory (final/projector.pt + final/lora) reloads into a model that behaves like the trained one."""
    from transformers import Qwen2Config, Qwen2ForCausalLM, SiglipConfig, SiglipModel, SiglipTextConfig, SiglipVisionConfig
    tmp = Path(tempfile.mkdtemp())
    try:
        # tiny "pretrained" checkpoints on disk
        torch.manual_seed(0)
        vcfg = SiglipVisionConfig(hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                                  image_size=56, patch_size=14)
        tcfg = SiglipTextConfig(hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
                                vocab_size=100, max_position_embeddings=16)
        SiglipModel(SiglipConfig(text_config=tcfg.to_dict(), vision_config=vcfg.to_dict())).save_pretrained(tmp / "enc")
        llm = Qwen2ForCausalLM(Qwen2Config(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=2,
                                           num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                                           tie_word_embeddings=True))
        llm.get_input_embeddings().weight.data.normal_(0, 0.5)
        llm.save_pretrained(tmp / "llm")
        paths = dict(encoder_name=str(tmp / "enc"), llm_name=str(tmp / "llm"))

        # a "trained" model: same base, random non-zero LoRA and projector
        from src.model.vlm import build_model
        trained = build_model(lora=True, lora_rank=4, lora_alpha=8, lora_dropout=0.0, dtype=torch.float32, pool_grid=POOL_GRID, **paths)
        trained.image_pad_id = IMAGE_PAD
        with torch.no_grad():
            for n, p in trained.trainable_parameters():
                p.add_(torch.randn_like(p) * 0.1)
        final = tmp / "run" / "final"
        final.mkdir(parents=True)
        torch.save(trained.projector.state_dict(), final / "projector.pt")
        trained.llm.save_pretrained(final / "lora")

        reloaded, has_proj = load_eval_model(tmp / "run", torch.float32, "cpu", pool_grid=POOL_GRID, **paths)
        reloaded.image_pad_id = IMAGE_PAD
        assert has_proj and not reloaded.has_lora
        assert not any("lora_" in n for n, _ in reloaded.named_parameters())      # merged into plain weights
        ds = EvalDataset(pope_rows(), [0, 1, 2], TASKS["pope"], "image", fake_format, fake_preprocess, fake_blank)
        b = collate([ds[i] for i in range(3)], pad_token_id=0)
        trained.eval()
        a = trained(b["input_ids"], b["attention_mask"], pixel_values=b["pixel_values"]).logits
        r = reloaded(b["input_ids"], b["attention_mask"], pixel_values=b["pixel_values"]).logits
        assert torch.allclose(a, r, atol=1e-4), float((a - r).abs().max())

        # no run_dir = zero-shot base model: no projector, no LoRA
        base, has_proj = load_eval_model(None, torch.float32, "cpu", pool_grid=POOL_GRID, **paths)
        assert not has_proj
        # E0-style directory: LoRA only, no projector.pt
        (tmp / "e0" / "final").mkdir(parents=True)
        trained.llm.save_pretrained(tmp / "e0" / "final" / "lora")
        _, has_proj = load_eval_model(tmp / "e0", torch.float32, "cpu", pool_grid=POOL_GRID, **paths)
        assert not has_proj
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_huggingface_dataset_rows_and_local_pope():
    """The loaders hand us Hugging Face datasets; check the row types the harness relies on."""
    from datasets import Dataset, Features, Image as HFImage, Value
    tmp = Path(tempfile.mkdtemp())
    try:
        rows = pope_rows(4)
        feats = Features({"question": Value("string"), "answer": Value("string"), "category": Value("string"),
                          "image_source": Value("string"), "image": HFImage()})
        ds = Dataset.from_list(rows, features=feats)
        assert isinstance(ds[0]["image"], Image.Image) and get_pil(ds[0]["image"]).mode == "RGB"
        assert describe_indices(ds, n=2) == describe_indices(rows, n=2)            # column access works for both
        (tmp / "data").mkdir()
        ds.to_parquet(str(tmp / "data" / "test-00000.parquet"))
        local = load_pope(pope_dir=str(tmp))                                       # local parquet, like data/raw/pope
        assert len(local) == 4 and get_pil(local[0]["image"]).size == (80, 64)
        import io
        buf = io.BytesIO(); Image.new("RGB", (5, 5)).save(buf, format="PNG")
        assert get_pil({"bytes": buf.getvalue(), "path": None}).size == (5, 5)     # raw parquet struct
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_evaluate_cli_end_to_end():
    """scripts/evaluate.py: writes JSONL + metrics.json per mode, skips finished work, honours --overwrite."""
    import dataclasses
    import json
    import runpy
    tmp = Path(tempfile.mkdtemp())
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import evaluate as ev
        from src.eval import generate as gen

        sqa_rows = [{"image": Image.new("RGB", (40, 40), (i * 20, 0, 0)), "hint": "", "question": f"Question {i}?",
                     "choices": ["x", "y", "z"], "answer": i % 3} for i in range(6)]
        rows = {"scienceqa": sqa_rows, "pope": pope_rows(12)}
        tasks = {name: dataclasses.replace(TASKS[name], load=(lambda name=name, **_: rows[name])) for name in ("scienceqa", "pope")}
        tasks["describe"] = dataclasses.replace(TASKS["describe"], load=lambda **_: rows["pope"])
        model = tiny_model(seed=1)
        calls = []

        def fake_run_task(model, task, rows, idx, mode, device, batch_size, num_workers, progress=None):
            calls.append((task.name, mode, len(idx)))
            return gen.run_task(model, task, rows, idx, mode, "cpu", batch_size, 0, progress=progress, pad_token_id=0,
                                decode_fn=lambda ids: ["B"] * len(ids) if task.name == "scienceqa" else ["Yes, there is."] * len(ids),
                                format_fn=fake_format, preprocess_fn=fake_preprocess, blank_fn=fake_blank)

        ev.TASKS, ev.run_task, ev.load_eval_model = tasks, fake_run_task, lambda *a, **k: (model, True)
        argv = ["evaluate.py", "--name", "m", "--out_dir", str(tmp), "--device", "cpu", "--num_workers", "0",
                "--modes", "image", "gray", "no_image", "--benchmarks", "scienceqa", "pope", "--max_samples", "4"]
        sys.argv = argv
        ev.main()
        assert len(calls) == 6 and all(n == 4 for _, _, n in calls)                 # 2 benchmarks x 3 modes, 4 questions each
        for mode in ("image", "gray", "no_image"):
            recs = [json.loads(l) for l in open(tmp / "m" / mode / "pope.jsonl")]
            assert len(recs) == 4 and recs[0]["mode"] == mode and "output" in recs[0]
            metrics = json.load(open(tmp / "m" / mode / "metrics.json"))
            assert set(metrics) == {"scienceqa", "pope"} and metrics["pope"]["all"]["n"] == 4
            assert metrics["scienceqa"]["format_failure_rate"] == 0.0
        # same questions in every mode and benchmark run (the subset is fixed by --subset_seed)
        ids = [[json.loads(l)["idx"] for l in open(tmp / "m" / m / "pope.jsonl")] for m in ("image", "gray", "no_image")]
        assert ids[0] == ids[1] == ids[2]
        calls.clear()
        ev.main()                                                                   # finished work is skipped
        assert calls == []
        sys.argv = argv + ["--overwrite"]
        ev.main()
        assert len(calls) == 6
        # text-only model: only no_image allowed
        ev.load_eval_model = lambda *a, **k: (model, False)
        sys.argv = ["evaluate.py", "--name", "t", "--out_dir", str(tmp), "--device", "cpu", "--benchmarks", "pope", "--max_samples", "4"]
        calls.clear()
        ev.main()
        assert calls == [("pope", "no_image", 4)]
        sys.argv = ["evaluate.py", "--name", "t2", "--out_dir", str(tmp), "--device", "cpu", "--modes", "image"]
        try:
            ev.main()
            raise AssertionError("expected SystemExit for image mode without a projector")
        except SystemExit:
            pass
        # describe: outputs only, no metrics entry
        ev.load_eval_model = lambda *a, **k: (model, True)
        sys.argv = ["evaluate.py", "--name", "d", "--out_dir", str(tmp), "--device", "cpu", "--benchmarks", "describe", "--describe_n", "3"]
        ev.main()
        assert len(open(tmp / "d" / "image" / "describe.jsonl").readlines()) == 3 and not (tmp / "d" / "image" / "metrics.json").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        sys.argv = ["x"]


# ---------------------------------------------------------------------------------------------
# statistics (definitions from abstract.md section 5)


def sim_records(scores, extra=None):
    return [{"idx": i, "score": float(s), **(extra[i] if extra else {})} for i, s in enumerate(scores)]


def test_bootstrap_and_paired_difference():
    rng = np.random.default_rng(0)
    n = 400
    base = (rng.random(n) < 0.5).astype(float)
    a = stats.Results.from_records([sim_records(base)], "textvqa")
    ci = stats.bootstrap(a, "accuracy", n_boot=500, seed=0)
    assert abs(ci.point - base.mean()) < 1e-12 and ci.lo < ci.point < ci.hi and ci.hi - ci.lo < 0.15
    assert stats.bootstrap(a, "accuracy", 500, 0).lo == ci.lo                       # reproducible
    same = stats.paired_difference(a, a, "accuracy", 500)
    assert (same.point, same.lo, same.hi) == (0.0, 0.0, 0.0) and not same.excludes_zero
    better = np.maximum(base, (rng.random(n) < 0.3).astype(float))                 # B is right wherever A is, and more
    b = stats.Results.from_records([sim_records(better)], "textvqa")
    d = stats.paired_difference(a, b, "accuracy", 500)
    assert d.excludes_zero and d.point == better.mean() - base.mean() and d.lo > 0
    noise = stats.Results.from_records([sim_records((rng.random(n) < 0.5).astype(float))], "textvqa")
    assert not stats.paired_difference(a, noise, "accuracy", 500).excludes_zero      # same rate, different items
    try:
        stats.paired_difference(a, stats.Results.from_records([sim_records(base[:100])], "textvqa"), "accuracy")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_two_seeds_are_averaged():
    s0, s1 = np.array([1, 1, 0, 0.0]), np.array([1, 0, 0, 0.0])
    r = stats.Results.from_records([sim_records(s0), sim_records(s1)], "textvqa")
    assert abs(stats.bootstrap(r, "accuracy", 100).point - 0.375) < 1e-12           # mean over seeds of the mean over items
    assert r.score.shape == (2, 4)


def pope_sim(n, p_yes_given_yes, p_yes_given_no, seed):
    rng = np.random.default_rng(seed)
    gold = np.arange(n) % 2 == 0
    said = np.where(gold, rng.random(n) < p_yes_given_yes, rng.random(n) < p_yes_given_no)
    return [{"idx": i, "gold": "yes" if g else "no", "score": float(s == g), "pred_yes": bool(s), "said_yes": bool(s)}
            for i, (g, s) in enumerate(zip(gold, said))]


def test_f1_matches_metrics_module_and_detects_hallucination_gap():
    recs = pope_sim(600, 0.85, 0.40, 0)
    res = stats.Results.from_records([recs], "pope")
    m = confusion_metrics([r["pred_yes"] for r in recs], [r["gold"] == "yes" for r in recs])
    assert abs(stats.bootstrap(res, "f1", 200).point - m["f1"]) < 1e-12
    assert abs(stats.bootstrap(res, "yes_rate", 200).point - np.mean([r["said_yes"] for r in recs])) < 1e-12
    careful = stats.Results.from_records([pope_sim(600, 0.85, 0.10, 0)], "pope")    # E3-like: fewer false "yes"
    gap = stats.paired_difference(res, careful, "f1", 300)
    assert gap.point > 0 and gap.excludes_zero
    # split filter
    for r in recs:
        r["split"] = "adversarial" if r["idx"] % 3 == 0 else "random"
    sub = stats.Results.from_records([recs], "pope", split="adversarial")
    assert sub.n == 200


def test_difference_of_gains():
    n = 500
    rng = np.random.default_rng(1)
    def pair(gain):
        a = (rng.random(n) < 0.4).astype(float)
        b = np.where(rng.random(n) < gain, 1.0, a)                                 # B fixes `gain` of A's errors
        return (stats.Results.from_records([sim_records(a)], "textvqa"), stats.Results.from_records([sim_records(b)], "textvqa"))
    big, small = pair(0.40), pair(0.05)
    d = stats.difference_of_gains((*big, "accuracy"), (*small, "accuracy"), n_boot=500)
    assert d.point > 0.1 and d.excludes_zero                                       # a clearly larger gain
    d2 = stats.difference_of_gains((*small, "accuracy"), (*small, "accuracy"), n_boot=500)
    assert d2.point == 0 and d2.lo < 0 < d2.hi                                     # equal gains: "no clear largest gain"


def test_relative_drop():
    n = 600
    rng = np.random.default_rng(2)
    meta = [{"n_choices": 4} for _ in range(n)]                                    # chance = 0.25
    img = (rng.random(n) < 0.85).astype(float)
    blank_at_chance = (rng.random(n) < 0.25).astype(float)
    r_img = stats.Results.from_records([sim_records(img, meta)], "scienceqa")
    r_blank = stats.Results.from_records([sim_records(blank_at_chance, meta)], "scienceqa")
    ci, dropped = stats.relative_drop(r_img, r_blank, n_boot=300)
    expected = (img.mean() - blank_at_chance.mean()) / (img.mean() - 0.25)
    assert abs(ci.point - expected) < 1e-9 and 0.9 < ci.point < 1.1 and ci.lo < ci.point < ci.hi and dropped == 0
    same, _ = stats.relative_drop(r_img, r_img, n_boot=100)
    assert same.point == 0.0 and same.lo == 0.0 and same.hi == 0.0                 # blanking changed nothing
    # a model at chance has no drop to normalise: undefined, reported as nan
    at_chance = stats.Results.from_records([sim_records(blank_at_chance, meta)], "scienceqa")
    ci, _ = stats.relative_drop(at_chance, r_blank, n_boot=100)
    assert np.isnan(ci.point) or abs(ci.point) < 1.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print("ALL CHECKS PASSED")
