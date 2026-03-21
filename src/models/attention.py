import torch
import torch.nn as nn
import torch.nn.functional as F


class StemAttentionPool(nn.Module):
    """
    Single-query attention pooling:
      score_i = v^T tanh(W h_i)
      weight_i = softmax(score_i)
      out = sum_i(weight_i * h_i)

    This is a lightweight learnable alternative to multi-head self-attention
    that works well when N is small (4 stems).

    Parameters
    ----------
    embed_dim : dimension of each stem embedding
    """

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.W = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x       : (B, N, embed_dim)
        returns : attended (B, embed_dim),  weights (B, N)
        """
        scores = self.v(torch.tanh(self.W(x))).squeeze(-1)   # (B, N)
        weights = F.softmax(scores, dim=-1)                   # (B, N)
        out = (weights.unsqueeze(-1) * x).sum(dim=1)          # (B, embed_dim)
        return out, weights