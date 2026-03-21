import logging
import pickle
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.audio_loader import MultiBranchDataset

logger = logging.getLogger(__name__)

STEMS = ["bass", "drums", "other", "vocals", "mix"]


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


@torch.no_grad()
def run_inference(
    model: nn.Module,
    test_csv: str | Path,
    mel_path: str | Path,
    label_encoder_path: str | Path,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 4,
) -> list[tuple[str, str]]:
    """
    Run inference on the test CSV and return (file_base, predicted_genre) pairs.
    """
    with open(label_encoder_path, "rb") as f:
        le = pickle.load(f)

    label2idx = {g: i for i, g in enumerate(le.classes_)}
    idx2label = {i: g for g, i in label2idx.items()}

    dataset = MultiBranchDataset(
        csv_file=test_csv,
        mel_path=mel_path,
        label2idx=label2idx,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model.eval()
    results: list[tuple[str, str]] = []
    file_bases: list[str] = dataset.df["file_base"].tolist()
    pred_idx = 0

    for batch in tqdm(loader, desc="Inference", dynamic_ncols=True):
        batch = _to_device(batch, device)
        out = model(batch)
        preds = out["logits"].argmax(dim=1).cpu().tolist()
        for p in preds:
            results.append((file_bases[pred_idx], idx2label[p]))
            pred_idx += 1

    assert pred_idx == len(file_bases), "Mismatch between predictions and dataset length"
    logger.info(f"Inference complete: {len(results)} predictions")
    return results