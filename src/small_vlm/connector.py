import torch.nn.functional as F
import torch.nn as nn

def pool_visual_tokens(tokens, grid_size = 27, output_size = 14):
    """ [B, grid_size ** 2, C] -> [B, output_size ** 2, C] """


    batch_size, num_patches, feature_dim = tokens.shape

    assert num_patches == grid_size**2, f"expected {grid_size**2} tokens, got {num_patches}"

    grid_tokens = tokens.reshape(batch_size,grid_size, grid_size, feature_dim)
    grid_tokens = grid_tokens.permute(0,3, 1, 2)
    pooled = F.adaptive_avg_pool2d(grid_tokens, (output_size, output_size))
    pooled = pooled.permute(0, 2, 3, 1)
    pooled_tokens = pooled.reshape(batch_size, output_size ** 2, feature_dim)

    return pooled_tokens

class VisualProjector(nn.Module):
    """Two-layer MLP that maps visual tokens into the LLM embedding space."""

    def __init__(self, in_dim = 1152, out_dim = 896):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, tokens):
        """ [B, N, in_dim] -> [B, N, out_dim] """

        hidden = self.fc1(tokens)
        hidden = self.act(hidden)
        hidden = self.fc2(hidden)

        return hidden


