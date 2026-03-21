"""
Evaluate the best checkpoint on the held-out test set.

Run from project root:
    python scripts/evaluate.py
"""

import logging
import pickle
import sys
from pathlib import Path

import torch
import yaml
from rich.console import Console
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.audio_loader import MultiBranchDataset
from src.models.messy_mashup_model import build_model
from src.training.metrics import compute_metrics
from src.utils.label_encoder import LabelEncoder  # noqa: F401 — needed for pickle
from src.utils.logger import get_logger
from src.utils.visualization import plot_confusion_matrix

console = Console()
logger = get_logger(__name__, log_file=str(PROJECT_ROOT / "experiments/logs/evaluate.log"))


def load_cfg(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    base_cfg = load_cfg(PROJECT_ROOT / "configs/base_config.yaml")
    model_cfg = load_cfg(PROJECT_ROOT / "configs/model_config.yaml")
    inf_cfg = load_cfg(PROJECT_ROOT / "configs/inference_config.yaml")

    paths = base_cfg["paths"]
    mel_path = PROJECT_ROOT / paths["mel_path"]
    splits_path = PROJECT_ROOT / paths["splits_path"]
    checkpoint_path = PROJECT_ROOT / inf_cfg["inference"]["model_checkpoint"]
    le_path = PROJECT_ROOT / inf_cfg["inference"]["label_encoder"]

    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
    assert le_path.exists(), f"Label encoder not found: {le_path}"

    # Device
    cfg_device = inf_cfg["inference"].get("device", "auto")
    if cfg_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg_device)
    console.print(f"[bold]Device:[/bold] {device}")

    # Label encoder
    with open(le_path, "rb") as f:
        le = pickle.load(f)
    label2idx = {g: i for i, g in enumerate(le.classes_)}
    idx2label = {i: g for g, i in label2idx.items()}

    # Dataset
    test_ds = MultiBranchDataset(
        csv_file=splits_path / "test.csv",
        mel_path=mel_path,
        label2idx=label2idx,
        target_frames=base_cfg["audio"]["target_frames"],
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=inf_cfg["inference"]["batch_size"],
        shuffle=False,
        num_workers=inf_cfg["inference"]["num_workers"],
    )

    # Model
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = len(label2idx)
    model = build_model(num_classes=num_classes, model_cfg=model_cfg["model"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Inference
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", dynamic_ncols=True):
            targets = batch["label"]
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(batch)
            all_preds.extend(out["logits"].argmax(dim=1).cpu().tolist())
            all_targets.extend(targets.tolist())

    metrics = compute_metrics(all_preds, all_targets, idx2label)

    console.rule("[bold green]Test Set Results")
    console.print(f"Accuracy  : {metrics['accuracy']:.4f}")
    console.print(f"Macro F1  : {metrics['macro_f1']:.4f}")
    console.print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    console.print("\n" + metrics["report_str"])

    # Save confusion matrix
    exp_dir = PROJECT_ROOT / "experiments/exp_002_stem_multi_branch"
    exp_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        all_preds, all_targets,
        labels=list(range(num_classes)),
        save_path=exp_dir / "test_confusion_matrix.png",
    )
    logger.info(f"Confusion matrix saved to {exp_dir / 'test_confusion_matrix.png'}")


if __name__ == "__main__":
    main()