"""Checks for the training loop on tiny random models (CPU, no downloads).
Run from the repo root: python tests/test_train.py   (or: pytest tests/test_train.py)"""
import json
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.simplefilter("ignore")

import torch  # noqa: E402

from _tiny import TinyDataset, collate, tiny_model  # noqa: E402
from src.train.config import TrainConfig  # noqa: E402
from src.train.trainer import (count_trained_tokens, epoch_batches, image_shuffle_check, lr_multiplier,  # noqa: E402
                               plan_steps, save_final, train)

REPO = Path(__file__).resolve().parents[1]


def cfg_for(out, **kw):
    base = dict(name="t", output_dir=str(out), device="cpu", dtype="float32", num_workers=0, micro_batch_size=2,
                grad_accum=2, max_steps=6, log_every=100, save_every_steps=3, save_every_minutes=0,
                warmup_ratio=0.2, lr_projector=1e-3, lr_lora=1e-3, lora=True, lora_rank=4, lora_alpha=8,
                lora_dropout=0.0)
    base.update(kw)
    return TrainConfig(**base)


def test_lr_schedule():
    total, warm = 100, 0.1
    lrs = [lr_multiplier(s, total, warm) for s in range(total)]
    assert abs(lrs[0] - 0.1) < 1e-9 and abs(lrs[9] - 1.0) < 1e-9          # linear warm-up to the peak
    assert all(a >= b for a, b in zip(lrs[9:], lrs[10:]))                  # then monotone decay
    assert lrs[-1] < 0.01                                                  # to ~0 at the end


def test_epoch_batches():
    a = epoch_batches(50, 4, seed=0, epoch=0)
    assert a == epoch_batches(50, 4, seed=0, epoch=0)                      # deterministic
    assert a != epoch_batches(50, 4, seed=0, epoch=1) and a != epoch_batches(50, 4, seed=1, epoch=0)
    flat = [i for b in a for i in b]
    assert len(a) == 12 and all(len(b) == 4 for b in a)                    # partial micro-batch dropped
    assert len(set(flat)) == len(flat)                                     # no example twice in an epoch
    cfg = TrainConfig(micro_batch_size=4, grad_accum=3, epochs=2)
    assert plan_steps(50, cfg) == (4, 8)                                   # 12 micro-batches -> 4 steps per epoch
    assert plan_steps(50, TrainConfig(micro_batch_size=4, grad_accum=3, max_steps=5)) == (4, 5)


def test_accumulation_equals_one_big_batch():
    """num_items_in_batch makes 2 accumulated micro-batches give the same gradient as 1 big batch,
    even though the micro-batches hold different numbers of trained tokens."""
    ds, col = TinyDataset(4), collate()
    items = [ds[i] for i in range(4)]
    big = col(items)
    m = tiny_model(seed=2)
    m.eval()
    m(**big).loss.backward()
    ref = {n: p.grad.clone() for n, p in m.trainable_parameters()}
    ref_loss = float(m(**big).loss)

    m2 = tiny_model(seed=2)
    m2.eval()
    micro = [col(items[:2]), col(items[2:])]
    n = sum(count_trained_tokens(b["labels"]) for b in micro)
    assert n == count_trained_tokens(big["labels"])
    total = 0.0
    for b in micro:
        out = m2(**b, num_items_in_batch=n)
        out.loss.backward()
        total += float(out.loss)
    assert abs(total - ref_loss) < 1e-5, (total, ref_loss)
    for name, p in m2.trainable_parameters():
        assert torch.allclose(p.grad, ref[name], atol=1e-6, rtol=1e-4), name


def test_resume_gives_identical_result():
    """Kill a run mid-way, restart it, and end with exactly the same weights as an uninterrupted run."""
    tmp = Path(tempfile.mkdtemp())
    try:
        # uninterrupted: 6 steps
        m_a = tiny_model(seed=4, lora=True)
        train(cfg_for(tmp / "a"), m_a, TinyDataset(16), collate())

        # interrupted: the data source "crashes" after 3 steps (3 * 4 examples) + part of the 4th
        m_b = tiny_model(seed=4, lora=True)
        try:
            train(cfg_for(tmp / "b"), m_b, TinyDataset(16, fail_after=14), collate())
            raise AssertionError("expected the simulated crash")
        except KeyboardInterrupt:
            pass
        assert json.load(open(tmp / "b" / "latest.json"))["step"] == 3

        # restart with a fresh model: it must pick up from step 3
        m_c = tiny_model(seed=4, lora=True)
        summary = train(cfg_for(tmp / "b"), m_c, TinyDataset(16), collate())
        assert summary["resumed_from_step"] == 3 and summary["steps"] == 6
        pa, pc = dict(m_a.trainable_parameters()), dict(m_c.trainable_parameters())
        assert set(pa) == set(pc)
        worst = max(float((pa[n] - pc[n]).abs().max()) for n in pa)
        assert worst < 1e-6, worst
        start = {n: p for n, p in tiny_model(seed=4, lora=True).trainable_parameters()}
        assert max(float((pa[n] - start[n]).abs().max()) for n in pa) > 1e-4, "training should have changed the weights"
    finally:
        shutil.rmtree(tmp)


def test_checkpoint_pruning_and_final_files():
    tmp = Path(tempfile.mkdtemp())
    try:
        cfg = cfg_for(tmp, max_steps=8, save_every_steps=2, keep_last=2)
        m = tiny_model(lora=True)
        summary = train(cfg, m, TinyDataset(16), collate())
        assert len(list(tmp.glob("ckpt_*.pt"))) == 2                       # old ones pruned
        final = save_final(m, cfg, summary)
        assert (final / "projector.pt").exists() and (final / "lora" / "adapter_config.json").exists()
        assert (final / "summary.json").exists()
        sd = torch.load(final / "projector.pt")
        assert set(sd) == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
        # E0: projector frozen -> no projector.pt
        cfg0 = cfg_for(tmp / "e0", max_steps=2, text_only=True)
        m0 = tiny_model(lora=True, freeze_projector=True)
        s0 = train(cfg0, m0, TinyDataset(16, text_only=True), collate())
        assert not (save_final(m0, cfg0, s0) / "projector.pt").exists()
    finally:
        shutil.rmtree(tmp)


def test_overfit_and_image_shuffle_check():
    """The DATA_PIPELINE.md overfit test: loss falls on a tiny set, and shuffling the images raises it."""
    tmp = Path(tempfile.mkdtemp())
    try:
        ds, col = TinyDataset(8), collate()
        m = tiny_model(seed=1, lora=True)
        cfg = cfg_for(tmp, max_steps=150, micro_batch_size=4, grad_accum=2, lr_projector=3e-3, lr_lora=3e-3,
                      warmup_ratio=0.05, save_every_steps=0, log_every=10)
        before = image_shuffle_check(m, ds, col, "cpu")
        train(cfg, m, ds, col)
        after = image_shuffle_check(m, ds, col, "cpu")
        rows = [json.loads(l) for l in open(tmp / "train_log.jsonl")]
        assert rows[-1]["loss"] < 0.3 * rows[0]["loss"], (rows[0]["loss"], rows[-1]["loss"])
        assert after["loss_correct"] < 0.3 * before["loss_correct"], (before, after)
        assert after["loss_shuffled"] > 3 * after["loss_correct"], after   # the model really uses the image
    finally:
        shutil.rmtree(tmp)


def test_nan_loss_stops_before_saving():
    tmp = Path(tempfile.mkdtemp())
    try:
        m = tiny_model(lora=True)
        with torch.no_grad():
            m.projector.fc1.weight.fill_(float("nan"))
        try:
            train(cfg_for(tmp, max_steps=2), m, TinyDataset(16), collate())
            raise AssertionError("expected FloatingPointError")
        except FloatingPointError:
            pass
        assert not list(tmp.glob("ckpt_*.pt"))
    finally:
        shutil.rmtree(tmp)


def test_configs_load_and_cli_overrides():
    for path in sorted((REPO / "configs").glob("*.json")):
        cfg = TrainConfig.from_json(path)                                  # unknown keys would raise
        assert cfg.effective_batch() > 0, path
        if cfg.text_only:
            assert cfg.lora, path
    e1, e2, e3 = (TrainConfig.from_json(REPO / "configs" / f) for f in
                  ("e1_stage1.json", "e2_stage2_projector.json", "e3_stage2_lora.json"))
    assert e1.stage == 1 and e2.stage == e3.stage == 2
    # E2 and E3 must differ only in LoRA settings, output dir and name
    d2, d3 = e2.to_dict(), e3.to_dict()
    diff = {k for k in d2 if d2[k] != d3[k]}
    assert diff <= {"name", "output_dir", "lora", "lora_rank", "lora_alpha", "lora_dropout", "lr_lora"}, diff
    assert e2.effective_batch() == e3.effective_batch()
    # command-line override path
    sys.path.insert(0, str(REPO / "scripts"))
    import train as cli
    sys.argv = ["train.py", "--config", str(REPO / "configs" / "e3_stage2_lora.json"), "--seed", "1",
                "--lora_rank", "64", "--subset_size", "1000", "--lora", "true"]
    args = cli.parse_args()
    assert args.seed == 1 and args.lora_rank == 64 and args.subset_size == 1000 and args.lora is True
    assert args.output_dir is None and args.lr_lora is None               # not given -> config value is kept


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print("ALL CHECKS PASSED")
