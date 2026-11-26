"""Checks for pool_patches. Run from the repo root: python tests/test_pooling.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.config import ENCODER_GRID, NUM_IMAGE_TOKENS, POOLED_GRID  # noqa: E402
from src.model.pooling import pool_patches  # noqa: E402

S, D = ENCODER_GRID, 1152


def patches_used(pool_fn):
    """How many of the S*S input patches affect the output (non-zero gradient)."""
    x = torch.randn(1, S * S, D, requires_grad=True)
    pool_fn(x).sum().backward()
    return int((x.grad.abs().sum(-1) > 0).sum())


def old_2x2(x):
    """The original spec: 2x2 average pooling with stride 2 (drops the last row/column)."""
    g = F.avg_pool2d(x.transpose(1, 2).reshape(1, D, S, S), 2)
    return g.flatten(2).transpose(1, 2)


def main():
    x = torch.randn(2, S * S, D)
    out = pool_patches(x)
    assert POOLED_GRID ** 2 == NUM_IMAGE_TOKENS == 169
    assert out.shape == (2, NUM_IMAGE_TOKENS, D), out.shape
    print(f"shape: {tuple(x.shape)} -> {tuple(out.shape)}")

    used_new, used_old = patches_used(pool_patches), patches_used(old_2x2)
    assert used_new == S * S and used_old == S * S - (2 * S - 1)
    print(f"patches used: adaptive 13x13 {used_new}/{S * S}, plain 2x2 stride 2 {used_old}/{S * S}")

    # Window layout along one axis, from PyTorch's adaptive pooling rule.
    wins = [(i * S // POOLED_GRID, -(-(i + 1) * S // POOLED_GRID)) for i in range(POOLED_GRID)]
    assert wins[0][0] == 0 and wins[-1][1] == S
    assert all(a <= prev_end for (a, _), (_, prev_end) in zip(wins[1:], wins[:-1])), "gap between windows"
    print("windows per axis:", " ".join(f"[{a},{b})" for a, b in wins))
    k3s2 = F.avg_pool2d(x.transpose(1, 2).reshape(2, D, S, S), 3, stride=2).flatten(2).transpose(1, 2)
    assert torch.allclose(out, k3s2, atol=1e-6)
    print("adaptive 27->13 == 3x3 average pooling with stride 2")

    # Row-major order: output token k covers grid cell (k // 13, k % 13).
    rows = torch.arange(S).repeat_interleave(S).float()          # patch -> its row index
    cols = torch.arange(S).repeat(S).float()
    pooled = pool_patches(torch.stack([rows, cols], -1)[None])[0]   # [169, 2] mean row/col
    k = torch.arange(NUM_IMAGE_TOKENS)
    assert (pooled[:, 0].diff().reshape(-1)[k[:-1] % POOLED_GRID != POOLED_GRID - 1] == 0).all()
    assert torch.equal(pooled[:, 0].argsort(stable=True), k) and pooled[0, 1] < pooled[1, 1]
    print("order: row-major (left to right, top to bottom)")

    # Variants: 81 tokens = exact 3x3 pooling; 729 tokens = identity.
    g3 = F.avg_pool2d(x.transpose(1, 2).reshape(2, D, S, S), 3).flatten(2).transpose(1, 2)
    assert torch.allclose(pool_patches(x, 9), g3, atol=1e-6)
    assert torch.equal(pool_patches(x, S), x)
    print("variants: grid=9 == 3x3 pooling (81 tokens), grid=27 == identity (729 tokens)")

    out_bf16 = pool_patches(x.bfloat16())
    assert out_bf16.dtype == torch.bfloat16 and out_bf16.shape == out.shape
    for bad in (torch.randn(1, 730, D), torch.randn(1, S * S, D)):
        try:
            pool_patches(bad, 13 if bad.shape[1] == 730 else 28)
        except ValueError:
            continue
        raise AssertionError("bad input accepted")
    print("bf16 OK; non-square grids and upsampling rejected")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
