"""
Train the MultiBranchCRNN model.

Run from project root:
    python scripts/train_multibranch.py
"""

import logging
import pickle
import sys
from pathlib import Path

import torch
import yaml
from rich.console import Console
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.augmentation import build_augmentation
from src.data.audio_loader import MultiBranchDataset
from src.models.loss import build_criterion
from src.models.messy_mashup_model import build_model
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.scheduler import build_scheduler
from src.training.trainer import print_epoch_summary, train_one_epoch, validate
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.utils.visualization import plot_confusion_matrix, plot_training_curves

console = Console()
logger = get_logger(
    __name__,
    log_file=str(PROJECT_ROOT / "experiments/logs/train_multibranch.log"),
)


def load_cfg(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(cfg_str: str = "auto") -> torch.device:
    if cfg_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg_str)


from src.utils.label_encoder import LabelEncoder as _LabelEncoder


def main() -> None:
    base_cfg = load_cfg(PROJECT_ROOT / "configs/base_config.yaml")
    model_cfg = load_cfg(PROJECT_ROOT / "configs/model_config.yaml")
    train_cfg = load_cfg(PROJECT_ROOT / "configs/training_config.yaml")
    aug_cfg = load_cfg(PROJECT_ROOT / "configs/augmentation_config.yaml")

    seed = base_cfg["project"]["seed"]
    set_seed(seed)

    paths = base_cfg["paths"]
    t = train_cfg["training"]

    mel_path = PROJECT_ROOT / paths["mel_path"]
    splits_path = PROJECT_ROOT / paths["splits_path"]
    checkpoint_dir = PROJECT_ROOT / paths["checkpoints_dir"]

    assert mel_path.exists(), f"mel_path not found: {mel_path} — run build_dataset.py first"
    assert (splits_path / "train.csv").exists(), "train.csv not found — run build_dataset.py first"

    device = resolve_device()
    console.print(f"[bold]Device:[/bold] {device}")

    # Datasets
    train_aug = build_augmentation(aug_cfg["augmentation"], mel_path, train=True)

    train_ds = MultiBranchDataset(
        csv_file=splits_path / "train.csv",
        mel_path=mel_path,
        target_frames=base_cfg["audio"]["target_frames"],
        augment=train_aug,
    )
    val_ds = MultiBranchDataset(
        csv_file=splits_path / "val.csv",
        mel_path=mel_path,
        label2idx=train_ds.label2idx,
        target_frames=base_cfg["audio"]["target_frames"],
        augment=None,
    )
    test_ds = MultiBranchDataset(
        csv_file=splits_path / "test.csv",
        mel_path=mel_path,
        label2idx=train_ds.label2idx,
        target_frames=base_cfg["audio"]["target_frames"],
        augment=None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=t["batch_size"],
        shuffle=True,
        num_workers=t["num_workers"],
        pin_memory=t.get("pin_memory", True) and device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=t["batch_size"],
        shuffle=False,
        num_workers=t["num_workers"],
    )

    idx2label = train_ds.idx2label
    num_classes = len(train_ds.label2idx)

    # Save label encoder alongside checkpoints
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    le = _LabelEncoder(sorted(train_ds.label2idx, key=train_ds.label2idx.get))
    with open(checkpoint_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    logger.info(f"Label encoder saved to {checkpoint_dir / 'label_encoder.pkl'}")

    # Model
    model = build_model(num_classes=num_classes, model_cfg=model_cfg["model"])
    model = model.to(device)

    # Optimizer
    opt_cfg = t["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg.get("weight_decay", 1e-4),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
    )

    # Scheduler
    scheduler = build_scheduler(optimizer, t["scheduler"], steps_per_epoch=len(train_loader))

    # Loss
    criterion = build_criterion(t["loss"])

    # AMP
    use_amp = t.get("amp", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    logger.info(f"AMP enabled: {use_amp}")

    # Callbacks
    es_cfg = t["early_stopping"]
    early_stop = EarlyStopping(
        patience=es_cfg["patience"],
        min_delta=es_cfg["min_delta"],
        mode="min" if es_cfg["monitor"] == "val_loss" else "max",
    )
    ckpt_cfg = t["checkpoint"]
    checkpointer = ModelCheckpoint(
        checkpoint_dir=checkpoint_dir,
        metric=ckpt_cfg["metric"],
        mode=ckpt_cfg["mode"],
        save_last=ckpt_cfg.get("save_last", True),
    )

    # History
    train_losses, val_losses, val_accs = [], [], []

    console.rule("[bold cyan]Training")

    for epoch in range(1, t["epochs"] + 1):
        train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            grad_clip=t.get("grad_clip", 1.0), epoch=epoch,
        )
        val_m = validate(model, val_loader, criterion, device, idx2label, epoch=epoch)

        # Step scheduler (ReduceLROnPlateau needs the metric)
        sched_name = t["scheduler"].get("name", "")
        if sched_name == "ReduceLROnPlateau":
            scheduler.step(val_m["loss"])
        else:
            scheduler.step()

        print_epoch_summary(epoch, train_m, val_m)

        train_losses.append(train_m["loss"])
        val_losses.append(val_m["loss"])
        val_accs.append(val_m["accuracy"])

        # Checkpoint
        monitor_val = val_m["accuracy"] if ckpt_cfg["metric"] == "val_acc" else val_m["loss"]
        checkpointer.step(
            value=monitor_val,
            model=model,
            epoch=epoch,
            extra={"label2idx": train_ds.label2idx},
        )

        # Early stopping
        es_value = val_m["loss"] if es_cfg["monitor"] == "val_loss" else val_m["accuracy"]
        if early_stop.step(es_value):
            console.print(f"[yellow]Early stopping at epoch {epoch}[/yellow]")
            break

    console.rule("[bold cyan]Training complete — final evaluation on val set")
    console.print(val_m.get("report_str", ""))

    # Plots
    exp_dir = PROJECT_ROOT / "experiments/exp_002_stem_multi_branch"
    exp_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(train_losses, val_losses, val_accs, exp_dir / "training_curves.png")
    # Full confusion matrix is generated by evaluate.py on the test set.
    # Here we just log that training is done.
    logger.info("Run 'python scripts/evaluate.py' for full test-set confusion matrix.")
    logger.info(f"Artifacts saved to {exp_dir}")


if __name__ == "__main__":
    main()