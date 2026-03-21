import logging
import sys
from pathlib import Path

import yaml
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.mel_spectrogram import extract_and_save
from src.data.dataset_builder import build_splits
from src.utils.logger import get_logger
from src.utils.seed import set_seed

console = Console()
logger = get_logger(__name__, log_file=str(PROJECT_ROOT / "experiments/logs/build_dataset.log"))


def load_cfg(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    base_cfg = load_cfg(PROJECT_ROOT / "configs/base_config.yaml")
    train_cfg = load_cfg(PROJECT_ROOT / "configs/training_config.yaml")

    set_seed(base_cfg["project"]["seed"])

    audio = base_cfg["audio"]
    paths = base_cfg["paths"]

    genres_path = PROJECT_ROOT / paths["genres_path"]
    mel_path = PROJECT_ROOT / paths["mel_path"]
    splits_path = PROJECT_ROOT / paths["splits_path"]

    assert genres_path.exists(), f"genres_path not found: {genres_path}"

    # Step 1: Extract mel spectrograms for all stems
    console.rule("[bold cyan]Step 1 — Mel spectrogram extraction")
    extract_and_save(
        genres_path=genres_path,
        mel_save_path=mel_path,
        sample_rate=audio["sample_rate"],
        n_mels=audio["n_mels"],
        n_fft=audio["n_fft"],
        hop_length=audio["hop_length"],
        fmax=audio["fmax"],
        target_frames=audio["target_frames"],
        duration=audio["duration"],
    )

    # Step 2: Build train/val/test splits
    console.rule("[bold cyan]Step 2 — Building splits")
    split_cfg = train_cfg["training"]["splits"]
    build_splits(
        mel_path=mel_path,
        splits_path=splits_path,
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
        seed=base_cfg["project"]["seed"],
    )

    console.rule("[bold green]Dataset build complete")


if __name__ == "__main__":
    main()