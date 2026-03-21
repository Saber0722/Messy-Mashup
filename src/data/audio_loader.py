"""
PyTorch Dataset for the multi-branch genre classifier.

Each sample returns a dict:
    {
        "bass":   FloatTensor[1, n_mels, T],
        "drums":  FloatTensor[1, n_mels, T],
        "other":  FloatTensor[1, n_mels, T],
        "vocals": FloatTensor[1, n_mels, T],
        "mix":    FloatTensor[1, n_mels, T],
        "label":  int,
    }
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.audio_utils import fix_length, normalise_mel

logger = logging.getLogger(__name__)

STEMS = ["bass", "drums", "other", "vocals", "mix"]


class MultiBranchDataset(Dataset):
    """
    Loads pre-computed mel spectrograms from *mel_path* for every stem
    of every track listed in *csv_file*.

    Parameters
    ----------
    csv_file    : path to train/val/test CSV produced by dataset_builder.py
    mel_path    : directory containing <genre>__<track>__<stem>.npy files
    label2idx   : optional fixed label → int mapping (use during val/test)
    target_frames: expected time-axis length (pad/trim if needed)
    augment     : augmentation callable (see src/augmentation/) or None
    """

    def __init__(
        self,
        csv_file: str | Path,
        mel_path: str | Path,
        label2idx: dict[str, int] | None = None,
        target_frames: int = 1300,
        augment=None,
    ) -> None:
        self.mel_path = Path(mel_path)
        self.target_frames = target_frames
        self.augment = augment

        assert self.mel_path.exists(), f"mel_path does not exist: {self.mel_path}"

        self.df = pd.read_csv(csv_file)
        assert {"file_base", "label"}.issubset(self.df.columns), (
            f"CSV must contain 'file_base' and 'label' columns, got: {list(self.df.columns)}"
        )

        # Build or reuse label mapping
        if label2idx is None:
            labels = sorted(self.df["label"].unique())
            self.label2idx = {g: i for i, g in enumerate(labels)}
        else:
            self.label2idx = label2idx

        self.idx2label = {v: k for k, v in self.label2idx.items()}
        logger.info(
            f"Dataset from {csv_file}: {len(self.df)} tracks, "
            f"{len(self.label2idx)} classes"
        )

    def __len__(self) -> int:
        return len(self.df)

    def _load_mel(self, file_base: str, stem: str) -> torch.Tensor:
        # Try new naming first (genre__track__stem.npy)
        path = self.mel_path / f"{file_base}__{stem}.npy"
        if not path.exists():
            # Fall back to old notebook naming (genre_track_stem.wav.npy)
            genre, track = file_base.split("__", 1)
            old_base = f"{genre}_{track}_{stem}"
            for candidate in [f"{old_base}.wav.npy", f"{old_base}.npy"]:
                p = self.mel_path / candidate
                if p.exists():
                    path = p
                    break
        assert path.exists(), f"Missing mel file (tried both naming conventions): {self.mel_path / file_base}*__{stem}.npy"
        mel = np.load(path).astype(np.float32)
        mel = fix_length(mel, self.target_frames)
        mel = normalise_mel(mel)
        t = torch.from_numpy(mel).float().unsqueeze(0)  # (1, n_mels, T)
        return t

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        file_base: str = row["file_base"]
        label_str: str = row["label"]
        label = self.label2idx[label_str]

        sample: dict = {}
        for stem in STEMS:
            try:
                mel = self._load_mel(file_base, stem)
            except Exception as exc:
                logger.error(f"Error loading {file_base} / {stem}: {exc}")
                n_mels = 128
                mel = torch.zeros(1, n_mels, self.target_frames)
            sample[stem] = mel

        if self.augment is not None:
            sample = self.augment(sample)

        sample["label"] = label
        return sample