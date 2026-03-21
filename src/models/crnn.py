import torch
import torch.nn as nn


class TemporalCRNN(nn.Module):
    """
    Parameters
    ----------
    input_size   : input feature dimension
    hidden_size  : GRU hidden units (doubled by bidirectional)
    num_layers   : number of stacked GRU layers
    dropout      : dropout between GRU layers (ignored if num_layers == 1)
    """

    def __init__(
        self,
        input_size: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Project bidirectional output back to hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, input_size)  or  (B, T, input_size)
        returns: (B, hidden_size)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)          # treat as single time-step: (B, 1, input_size)
        _, h_n = self.gru(x)            # h_n: (num_layers*2, B, hidden_size)
        # Take last layer, concat fwd + bwd
        h_fwd = h_n[-2]                 # (B, hidden_size)
        h_bwd = h_n[-1]                 # (B, hidden_size)
        h = torch.cat([h_fwd, h_bwd], dim=-1)   # (B, hidden_size*2)
        return self.proj(h)             # (B, hidden_size)