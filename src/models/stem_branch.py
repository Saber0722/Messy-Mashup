import torch
import torch.nn as nn


class StemEncoder(nn.Module):
    """
    Parameters
    ----------
    channels    : output channels for each of the 3 conv blocks
    embed_dim   : size of the output embedding vector
    dropout     : dropout probability after the linear projection
    """

    def __init__(
        self,
        channels: list[int] | None = None,
        embed_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]

        assert len(channels) == 3, "StemEncoder expects exactly 3 channel sizes"

        c1, c2, c3 = channels

        self.conv = nn.Sequential(
            # Block 1
            nn.Conv2d(1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 4 * 4, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, n_mels, T)
        returns: (B, embed_dim)
        """
        x = self.conv(x)
        x = self.projector(x)
        return x