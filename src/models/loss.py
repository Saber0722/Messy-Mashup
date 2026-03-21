import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing.

    Equivalent to nn.CrossEntropyLoss(label_smoothing=...) in PyTorch ≥ 1.10,
    but kept here explicitly for clarity.
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C)
        targets : (B,) long
        """
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Smooth targets
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_criterion(cfg: dict) -> nn.Module:
    """Build the loss from the training_config loss section."""
    name = cfg.get("name", "CrossEntropyLoss")
    smoothing = cfg.get("label_smoothing", 0.0)

    if name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss(label_smoothing=smoothing)
    elif name == "LabelSmoothingCrossEntropy":
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    else:
        raise ValueError(f"Unknown loss: {name}")