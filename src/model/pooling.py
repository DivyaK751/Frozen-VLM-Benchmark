"""Visual-token pooling: SigLIP patch features -> NUM_IMAGE_TOKENS features for the projector.

SigLIP 2 So400m (patch 14, 384 px) outputs a 27x27 grid of patch features. Plain 2x2
pooling with stride 2 would give 13x13 but drop the last row and column (53 of 729
patches, a ~14 px strip on the right and bottom edges), because 27 is odd.

Adaptive average pooling to 13x13 instead picks window boundaries so the windows cover
the whole grid. For 27 -> 13 the windows are exactly 3x3 with stride 2 (neighbours
overlap by one patch), so every patch contributes. The same function gives the variants:
  grid=13 -> 169 tokens (main), grid=9 -> 81 (identical to 3x3 pooling, since 27 = 3x9),
  grid=27 -> 729 (identity, no pooling).

    feats = siglip(pixel_values).last_hidden_state      # [B, 729, 1152]
    tokens = pool_patches(feats)                         # [B, 169, 1152]
    projected = projector(tokens)                        # [B, 169, 896]

Output order is row-major (left to right, top to bottom), matching the order of the
<|image_pad|> positions in input_ids.
"""

import math

import torch.nn.functional as F

from src.config import POOLED_GRID


def pool_patches(features, grid=POOLED_GRID):
    """[B, N, D] patch features on a square grid (N = S*S) -> [B, grid*grid, D]."""
    b, n, d = features.shape
    s = math.isqrt(n)
    if s * s != n:
        raise ValueError(f"expected a square grid of patches, got {n} tokens")
    if grid > s:
        raise ValueError(f"cannot pool a {s}x{s} grid up to {grid}x{grid}")
    x = features.transpose(1, 2).reshape(b, d, s, s)          # [B, D, S, S], row-major patches
    x = F.adaptive_avg_pool2d(x, grid)                         # [B, D, grid, grid]
    return x.flatten(2).transpose(1, 2)                        # [B, grid*grid, D]
