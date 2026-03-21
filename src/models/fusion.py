import torch
import torch.nn as nn


class FusionModule(nn.Module):
    """
    Parameters
    ----------
    stem_embed_dim  : embed_dim of stem attention output
    mix_embed_dim   : embed_dim of mix branch output
    projected_dim   : output dimension after fusion projection
    """

    def __init__(
        self,
        stem_embed_dim: int = 128,
        mix_embed_dim: int = 128,
        projected_dim: int = 256,
    ) -> None:
        super().__init__()
        in_dim = stem_embed_dim + mix_embed_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, projected_dim),
            nn.LayerNorm(projected_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, stem_feat: torch.Tensor, mix_feat: torch.Tensor) -> torch.Tensor:
        """
        stem_feat : (B, stem_embed_dim)
        mix_feat  : (B, mix_embed_dim)
        returns   : (B, projected_dim)
        """
        x = torch.cat([stem_feat, mix_feat], dim=-1)
        return self.proj(x)