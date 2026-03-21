import logging
import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score

logger = logging.getLogger(__name__)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def compute_metrics(
    all_preds: list[int],
    all_targets: list[int],
    idx2label: dict[int, str],
) -> dict:
    """
    Compute accuracy, macro F1, weighted F1, and full sklearn report.

    Returns a dict with keys: accuracy, macro_f1, weighted_f1, report_str
    """
    labels = [idx2label[i] for i in sorted(idx2label)]
    label_indices = sorted(idx2label.keys())

    correct = sum(p == t for p, t in zip(all_preds, all_targets))
    acc = correct / len(all_targets)

    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    report = classification_report(
        all_targets, all_preds,
        labels=label_indices,
        target_names=labels,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report_str": report,
    }