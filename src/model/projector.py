"""The trained connector: a 2-layer MLP that maps pooled SigLIP features into Qwen's input space.

    [B, 169, 1152]  ->  Linear(1152, 896)  ->  GELU  ->  Linear(896, 896)  ->  [B, 169, 896]

1152 is SigLIP 2 So400m's hidden size, 896 is Qwen2.5-0.5B's. The parameter count is
1152*896 + 896 + 896*896 + 896 = 1,836,800 (about 1.8M).
"""

import torch.nn as nn


class VisualProjector(nn.Module):
    def __init__(self, in_dim=1152, out_dim=896):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
