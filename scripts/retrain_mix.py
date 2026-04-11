"""
Retrain the MultiBranchCRNN model from scratch for mix-only inference.

The competition test set has only mixed audio (no separated stems).
This script retrains the model by feeding the mix mel to ALL 5 branches,
so the model learns to classify from mixed audio from the beginning.

It uses ALL available training data combined from train/val/test splits.

Run from project root:
    python scripts/retrain_mix_from_scratch.py
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
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split

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
    log_file=str(PROJECT_ROOT / "experiments/logs/retrain_mix_from_scratch.log"),
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
        dataframe: pd.DataFrame, # Accept a DataFrame directly
        mel_path: str | Path,
        label2idx: dict[str, int] | None = None,
        target_frames: int = 1300,
    ) -> None:
        self.mel_path = Path(mel_path)
        self.target_frames = target_frames
        self.df = dataframe.reset_index(drop=True) # Reset index for safe integer indexing

        if label2idx is None:
            labels = sorted(self.df["label"].unique())
            self.label2idx = {g: i for i, g in enumerate(labels)}
        else:
            self.label2idx = label2idx

        self.idx2label = {v: k for k, v in self.label2idx.items()}
        logger.info(f"MixOnlyDataset: {len(self.df)} tracks loaded.")

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
    # For retraining from scratch, always use a fresh directory
    out_ckpt_dir = PROJECT_ROOT / "checkpoints_mix_retrained"
    out_ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device()
    console.print(f"[bold]Device:[/bold] {device}")

    # Free any leftover CUDA memory from previous runs
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        console.print(f"[dim]CUDA memory cleared. Free: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GiB[/dim]")

    # ── Load ALL data and perform new split ─────────────────────────────────────────
    # Read all original splits
    train_df = pd.read_csv(splits_path / "train.csv")
    val_df = pd.read_csv(splits_path / "val.csv")
    test_df = pd.read_csv(splits_path / "test.csv")

    # Combine all available data
    all_df_combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    console.print(f"[dim]Total available data points: {len(all_df_combined)}[/dim]")

    # Perform stratified train/validation split on the combined data
    # Use the original validation ratio for the internal validation set
    orig_val_ratio = 0.15 # You can adjust this if needed
    train_df_new, val_df_new = train_test_split(
        all_df_combined,
        test_size=orig_val_ratio,
        stratify=all_df_combined["label"],
        random_state=base_cfg["project"]["seed"]
    )

    console.print(f"[dim]New Split - Train: {len(train_df_new)}, Val: {len(val_df_new)}[/dim]")

    # ── Create Label Encoder and Save ──────────────────────────────────────────
    # Determine label mappings from the full combined dataset
    labels = sorted(all_df_combined["label"].unique())
    label2idx = {g: i for i, g in enumerate(labels)}
    idx2label = {v: k for k, v in label2idx.items()}
    num_classes = len(label2idx)

    # Save label encoder for inference
    le_path = out_ckpt_dir / "label_encoder.pkl"
    import pickle
    from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder # Use standard LE
    le = SklearnLabelEncoder()
    le.fit(labels) # Fit on the sorted unique labels
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    console.print(f"[green]Label encoder saved to {le_path}[/green]")


    # ── Create Single Dataset Instance ───────────────────────────────────────────
    full_dataset = MixOnlyDataset(
        dataframe=all_df_combined, # Pass the full combined DataFrame
        mel_path=mel_path,
        label2idx=label2idx, # Pass the pre-computed mapping
        target_frames=base_cfg["audio"]["target_frames"],
    )

    # ── Get indices for the new train/val split ────────────────────────────────────
    # Identify the indices corresponding to the new train and val sets within the full dataset
    all_file_bases = all_df_combined['file_base'].values
    train_file_bases = set(train_df_new['file_base'].values)
    val_file_bases = set(val_df_new['file_base'].values)

    train_indices = [i for i, fb in enumerate(all_file_bases) if fb in train_file_bases]
    val_indices = [i for i, fb in enumerate(all_file_bases) if fb in val_file_bases]

    # ── Create Subsets and DataLoaders ─────────────────────────────────────────────
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_subset, # Use the subset
        batch_size=t["batch_size"],
        shuffle=True, # Shuffle the training subset
        num_workers=t["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_subset, # Use the subset
        batch_size=t["batch_size"],
        shuffle=False, # Usually no shuffle for validation
        num_workers=t["num_workers"],
    )

    # ── Model — Build NEW model from scratch ──────────────────────────────────────────
    console.print("[cyan]Building new model from scratch...[/cyan]")
    model = build_model(num_classes=num_classes, model_cfg=model_cfg["model"])
    model = model.to(device)
    console.print("[green]New model built and moved to device ✓[/green]")

    # ── Training config (Adjusted for Retraining) ────────────────────────────────────
    # Increase epochs significantly since we are training from scratch on more data
    # Adjust patience relative to the new total epochs if needed
    RETRAIN_EPOCHS = 150 # Increased epochs for full retraining
    RETRAIN_LR     = 1e-3 # Use initial training LR, not fine-tuning LR
    PATIENCE       = 15 # Increased patience, maybe make it ~10% of epochs?

    optimizer = torch.optim.AdamW(model.parameters(), lr=RETRAIN_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=RETRAIN_EPOCHS, eta_min=1e-6
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

    console.rule("[bold cyan]Full Retraining — mix-only (using ALL data)")
    console.print(f"Epochs: {RETRAIN_EPOCHS}  |  LR: {RETRAIN_LR}  |  Patience: {PATIENCE}")
    console.print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    for epoch in range(1, RETRAIN_EPOCHS + 1):
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

    best_path = out_ckpt_dir / "best_multibranch_model_from_scratch.pth"
    console.print(f"\n[bold green]Full retraining complete![/bold green]")
    console.print(f"Best checkpoint → [cyan]{best_path}[/cyan]")
    console.print(
        "\n[yellow]Update inference_config.yaml:[/yellow]\n"
        f"  model_checkpoint: checkpoints_mix_retrained/best_multibranch_model_from_scratch.pth"
    )


if __name__ == "__main__":
    main()