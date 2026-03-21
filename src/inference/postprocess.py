import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


@torch.no_grad()
def tta_predict(
    model: nn.Module,
    loader: DataLoader,
    augment_fn,
    device: torch.device,
    n_views: int = 5,
) -> torch.Tensor:
    """
    Run TTA: average softmax probabilities over *n_views* augmented passes.

    Returns a (N, num_classes) probability tensor.
    """
    model.eval()
    all_probs: list[torch.Tensor] | None = None

    for view_idx in range(n_views):
        view_probs: list[torch.Tensor] = []
        for batch in tqdm(loader, desc=f"TTA view {view_idx+1}/{n_views}", leave=False):
            # Apply augmentation in the sample dict space
            if augment_fn is not None:
                batch = {k: augment_fn({k: v})[k] if k != "label" else v for k, v in batch.items()}
            batch = _to_device(batch, device)
            out = model(batch)
            probs = F.softmax(out["logits"], dim=-1).cpu()
            view_probs.append(probs)
        view_probs_t = torch.cat(view_probs, dim=0)   # (N, C)

        if all_probs is None:
            all_probs = view_probs_t
        else:
            all_probs = all_probs + view_probs_t

    assert all_probs is not None
    return all_probs / n_views    # (N, C) averaged