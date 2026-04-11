"""
Fine-tune the MultiBranchCRNN checkpoint for mix-only inference.

The competition test set has only mixed audio (no separated stems).
This script fine-tunes the saved checkpoint by feeding the mix mel
to ALL 5 branches, so the model learns to classify from mixed audio.

The training data already has `genre__track__mix.npy` — we reuse those.

Run from project root:
    python scripts/finetune_mix.py
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from rich.console import Console
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.messy_mashup_model import build_model
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.trainer import print_epoch_summary, train_one_epoch, validate
from src.utils.audio_utils import fix_length, normalise_mel
from src.utils.logger import get_logger
from src.utils.seed import set_seed

console = Console()
logger = get_logger(
    __name__,
    log_file=str(PROJECT_ROOT / "experiments/logs/finetune_mix.log"),
)

STEMS = ["bass", "drums", "other", "vocals", "mix"]


# ── Mix-only Dataset ──────────────────────────────────────────────────────────

class MixOnlyDataset(Dataset):
    """
    Loads ONLY the mix mel for each track and broadcasts it to all 5 branches.
    This mimics exactly what competition inference does with the flat .wav files.
    """

    def __init__(
        self,
        csv_file: str | Path,
        mel_path: str | Path,
        label2idx: dict[str, int] | None = None,
        target_frames: int = 1300,
    ) -> None:
        self.mel_path = Path(mel_path)
        self.target_frames = target_frames

        self.df = pd.read_csv(csv_file)
        assert {"file_base", "label"}.issubset(self.df.columns)

        if label2idx is None:
            labels = sorted(self.df["label"].unique())
            self.label2idx = {g: i for i, g in enumerate(labels)}
        else:
            self.label2idx = label2idx

        self.idx2label = {v: k for k, v in self.label2idx.items()}
        logger.info(f"MixOnlyDataset: {len(self.df)} tracks from {csv_file}")

    def __len__(self) -> int:
        return len(self.df)

    def _load_mix_mel(self, file_base: str) -> torch.Tensor:
        path = self.mel_path / f"{file_base}__mix.npy"
        if not path.exists():
            # fallback: try old naming
            parts = file_base.split("__", 1)
            if len(parts) == 2:
                genre, track = parts
                for candidate in [
                    f"{genre}_{track}_mix.wav.npy",
                    f"{genre}_{track}_mix.npy",
                ]:
                    p = self.mel_path / candidate
                    if p.exists():
                        path = p
                        break
        if not path.exists():
            logger.warning(f"Mix mel not found for {file_base}, using zeros")
            return torch.zeros(1, 128, self.target_frames)

        mel = np.load(path).astype(np.float32)
        mel = fix_length(mel, self.target_frames)
        mel = normalise_mel(mel)
        return torch.from_numpy(mel).float().unsqueeze(0)  # (1, n_mels, T)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        file_base: str = row["file_base"]
        label: int = self.label2idx[row["label"]]

        mel = self._load_mix_mel(file_base)

        # Broadcast mix to ALL branches — identical to competition inference
        return {
            "bass":   mel,
            "drums":  mel,
            "other":  mel,
            "vocals": mel,
            "mix":    mel,
            "label":  label,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_cfg(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    base_cfg = load_cfg(PROJECT_ROOT / "configs/base_config.yaml")
    model_cfg = load_cfg(PROJECT_ROOT / "configs/model_config.yaml")
    train_cfg = load_cfg(PROJECT_ROOT / "configs/training_config.yaml")

    set_seed(base_cfg["project"]["seed"])

    paths = base_cfg["paths"]
    t = train_cfg["training"]

    mel_path       = PROJECT_ROOT / paths["mel_path"]
    splits_path    = PROJECT_ROOT / paths["splits_path"]
    checkpoint_dir = PROJECT_ROOT / paths["checkpoints_dir"]
    # Resume from mix checkpoint if it exists, otherwise start from original
    mix_ckpt     = PROJECT_ROOT / "checkpoints_mix/best_multibranch_model.pth"
    src_ckpt     = mix_ckpt if mix_ckpt.exists() else checkpoint_dir / "best_multibranch_model.pth"
    out_ckpt_dir = PROJECT_ROOT / "checkpoints_mix"
    out_ckpt_dir.mkdir(parents=True, exist_ok=True)
    le_path      = checkpoint_dir / "label_encoder.pkl"

    assert src_ckpt.exists(), f"Source checkpoint not found: {src_ckpt}"
    console.print(f"[dim]Resuming from: {src_ckpt}[/dim]")
    assert le_path.exists(),  f"Label encoder not found: {le_path}"

    device = resolve_device()
    console.print(f"[bold]Device:[/bold] {device}")

    # Free any leftover CUDA memory from previous runs
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        console.print(f"[dim]CUDA memory cleared. Free: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GiB[/dim]")

    console.print(f"[cyan]Loading checkpoint:[/cyan] {src_ckpt}")

    # ── Label encoder ─────────────────────────────────────────────────────────
    with open(le_path, "rb") as f:
        le = pickle.load(f)
    label2idx = {g: i for i, g in enumerate(le.classes_)}
    idx2label = {i: g for g, i in label2idx.items()}
    num_classes = len(label2idx)

    # ── Datasets (mix-only) ───────────────────────────────────────────────────
    # Combine ALL labeled splits — val accuracy is noisy with only 150 samples,
    # and we select the final model via Kaggle score, not internal val.
    all_df = pd.concat([
        pd.read_csv(splits_path / "train.csv"),
        pd.read_csv(splits_path / "val.csv"),
        pd.read_csv(splits_path / "test.csv"),
    ], ignore_index=True)
    # Hold out 15% as an internal val to track training (stratified)
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        all_df, test_size=0.15, stratify=all_df["label"], random_state=base_cfg["project"]["seed"]
    )
    # Write temp CSVs
    import tempfile
    _train_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    _val_tmp   = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    train_df.to_csv(_train_tmp.name, index=False); _train_tmp.close()
    val_df.to_csv(_val_tmp.name,   index=False); _val_tmp.close()
    console.print(f"[dim]All-data split: {len(train_df)} train / {len(val_df)} val[/dim]")

    train_ds = MixOnlyDataset(
        csv_file=_train_tmp.name,
        mel_path=mel_path,
        label2idx=label2idx,
        target_frames=base_cfg["audio"]["target_frames"],
    )
    val_ds = MixOnlyDataset(
        csv_file=_val_tmp.name,
        mel_path=mel_path,
        label2idx=label2idx,
        target_frames=base_cfg["audio"]["target_frames"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=t["batch_size"],
        shuffle=True,
        num_workers=t["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=t["batch_size"],
        shuffle=False,
        num_workers=t["num_workers"],
    )

    # ── Model — load from checkpoint ──────────────────────────────────────────
    ckpt = torch.load(src_ckpt, map_location=device, weights_only=False)
    model = build_model(num_classes=num_classes, model_cfg=model_cfg["model"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    console.print("[green]Checkpoint loaded ✓[/green]")

    # ── Fine-tuning config ────────────────────────────────────────────────────
    # Lower LR than original training — we're adapting, not learning from scratch
    FINETUNE_EPOCHS = 150
    FINETUNE_LR     = 1e-4
    PATIENCE        = 15

    optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FINETUNE_EPOCHS, eta_min=1e-6
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    early_stop = EarlyStopping(patience=PATIENCE, min_delta=1e-4, mode="max")
    checkpointer = ModelCheckpoint(
        checkpoint_dir=out_ckpt_dir,
        metric="val_acc",
        mode="max",
        save_last=True,
    )

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    console.rule("[bold cyan]Fine-tuning — mix-only")
    console.print(f"Epochs: {FINETUNE_EPOCHS}  |  LR: {FINETUNE_LR}  |  Patience: {PATIENCE}")

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            grad_clip=1.0, epoch=epoch,
        )
        val_m = validate(model, val_loader, criterion, device, idx2label, epoch=epoch)
        scheduler.step()

        print_epoch_summary(epoch, train_m, val_m)

        checkpointer.step(
            value=val_m["accuracy"],
            model=model,
            epoch=epoch,
            extra={"label2idx": label2idx},
        )

        if early_stop.step(val_m["accuracy"]):
            console.print(f"[yellow]Early stopping at epoch {epoch}[/yellow]")
            break

    best_path = out_ckpt_dir / "best_multibranch_model.pth"
    console.print(f"\n[bold green]Fine-tuning complete![/bold green]")
    console.print(f"Best checkpoint → [cyan]{best_path}[/cyan]")
    console.print(
        "\n[yellow]Update inference_config.yaml:[/yellow]\n"
        f"  model_checkpoint: checkpoints_mix/best_multibranch_model.pth"
    )


if __name__ == "__main__":
    main()