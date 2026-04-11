"""
Fine-tune the MultiBranchCRNN checkpoint for mix-only inference.

The competition test set has only mixed audio (no separated stems).
This script fine-tunes the saved checkpoint by feeding the mix mel
to ALL 5 branches, so the model learns to classify from mixed audio.

It uses ALL available training data combined from train/val/test splits.

Run from project root:
    python scripts/finetune_mix_all_data.py
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
    log_file=str(PROJECT_ROOT / "experiments/logs/finetune_mix_all_data.log"),
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
    # Resume from the ORIGINAL checkpoint (the one trained on stems)
    src_ckpt     = checkpoint_dir / "best_multibranch_model.pth"
    # Use a dedicated directory for this specific fine-tuning run
    out_ckpt_dir = PROJECT_ROOT / "checkpoints_finetune_mix_all_data"
    out_ckpt_dir.mkdir(parents=True, exist_ok=True)
    le_path      = checkpoint_dir / "label_encoder.pkl" # Load from original dir

    assert src_ckpt.exists(), f"Original checkpoint not found: {src_ckpt}"
    console.print(f"[dim]Resuming from ORIGINAL checkpoint: {src_ckpt}[/dim]")
    assert le_path.exists(),  f"Label encoder not found: {le_path}"

    device = resolve_device()
    console.print(f"[bold]Device:[/bold] {device}")

    # Free any leftover CUDA memory from previous runs
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        console.print(f"[dim]CUDA memory cleared. Free: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GiB[/dim]")

    console.print(f"[cyan]Loading original checkpoint:[/cyan] {src_ckpt}")

    # ── Label encoder ─────────────────────────────────────────────────────────
    with open(le_path, "rb") as f:
        le = pickle.load(f)
    label2idx = {g: i for i, g in enumerate(le.classes_)}
    idx2label = {i: g for g, i in label2idx.items()}
    num_classes = len(label2idx)

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

    # ── Create Single Dataset Instance ───────────────────────────────────────────
    full_dataset = MixOnlyDataset(
        dataframe=all_df_combined, # Pass the full combined DataFrame
        mel_path=mel_path,
        label2idx=label2idx, # Pass the pre-computed mapping from original LE
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

    # ── Model — load from ORIGINAL checkpoint ──────────────────────────────────────────
    console.print("[cyan]Loading original model for fine-tuning...[/cyan]")
    ckpt = torch.load(src_ckpt, map_location=device, weights_only=False)
    model = build_model(num_classes=num_classes, model_cfg=model_cfg["model"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    console.print("[green]Original checkpoint loaded and model moved to device ✓[/green]")

    # ── Fine-tuning config (Adjusted for More Data) ────────────────────────────────────
    # Increase epochs significantly since we now have more training data
    FINETUNE_EPOCHS = 150 # Increased epochs for larger training set
    FINETUNE_LR     = 1e-4 # Keep lower LR for fine-tuning
    PATIENCE        = 15 # Increased patience to match longer epochs

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

    console.rule("[bold cyan]Fine-tuning (ALL data) — mix-only")
    console.print(f"Epochs: {FINETUNE_EPOCHS}  |  LR: {FINETUNE_LR}  |  Patience: {PATIENCE}")
    console.print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

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
    console.print(f"\n[bold green]Fine-tuning (ALL data) complete![/bold green]")
    console.print(f"Best checkpoint → [cyan]{best_path}[/cyan]")
    console.print(
        "\n[yellow]Update inference_config.yaml:[/yellow]\n"
        f"  model_checkpoint: checkpoints_finetune_mix_all_data/best_multibranch_model.pth"
    )


if __name__ == "__main__":
    main()