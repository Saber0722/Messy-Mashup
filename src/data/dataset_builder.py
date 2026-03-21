"""
Build train / val / test CSV splits from processed mel spectrograms.

The CSV schema (one row per *track*, not per stem):
    file_base   : "<genre>__<track>" (prefix shared by all 5 npy files for this track)
    label       : genre string
    split       : train | val | test

Usage:
    python -m src.data.dataset_builder
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

STEMS = ["bass", "drums", "other", "vocals", "mix"]


def build_splits(
    mel_path: str | Path,
    splits_path: str | Path,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> None:
    """
    Scan *mel_path* for .npy files, group by track, create a balanced
    train/val/test split stratified by genre, and save three CSVs.
    """
    mel_path = Path(mel_path)
    splits_path = Path(splits_path)
    splits_path.mkdir(parents=True, exist_ok=True)

    assert mel_path.exists(), f"mel_path does not exist: {mel_path}"

    # Collect unique track keys (genre__track)
    npy_files = sorted(mel_path.glob("*.npy"))
    assert len(npy_files) > 0, f"No .npy files found in {mel_path}"

    track_set: set[str] = set()
    skipped = 0
    for f in npy_files:
        # Support both naming conventions:
        #   new (ours):      genre__track__stem.npy
        #   old (notebook):  genre_track_stem.wav.npy
        if "__" in f.stem:
            parts = f.stem.split("__")
            if len(parts) != 3:
                logger.warning(f"Skipping unexpected filename: {f.name}")
                skipped += 1
                continue
            genre, track, stem = parts
        else:
            # Old format: blues_blues.00000_bass.wav
            # Strip trailing .wav if present
            stem_raw = f.stem[:-4] if f.stem.endswith(".wav") else f.stem
            parts = stem_raw.split("_")
            if len(parts) < 3:
                logger.warning(f"Skipping unrecognised filename: {f.name}")
                skipped += 1
                continue
            genre = parts[0]
            stem = parts[-1]
            track = "_".join(parts[1:-1])

        # Normalise stem name (strip .wav suffix if it slipped through)
        stem = stem.replace(".wav", "")

        if stem not in STEMS:
            logger.debug(f"Skipping non-stem file: {f.name} (stem={stem!r})")
            skipped += 1
            continue
        track_set.add(f"{genre}__{track}")

    if skipped:
        logger.warning(f"Skipped {skipped} files during scan")
    assert len(track_set) > 0, "No valid tracks found — check mel_path contents"

    def _stem_exists(key: str, stem: str) -> bool:
        genre, track = key.split("__", 1)
        if (mel_path / f"{key}__{stem}.npy").exists():
            return True
        old_base = f"{genre}_{track}_{stem}"
        if (mel_path / f"{old_base}.wav.npy").exists():
            return True
        if (mel_path / f"{old_base}.npy").exists():
            return True
        return False

    records = []
    for key in sorted(track_set):
        genre = key.split("__")[0]
        # Verify all stems exist for this track
        missing = [s for s in STEMS if not _stem_exists(key, s)]
        if missing:
            logger.warning(f"Track {key!r} missing stems: {missing} — skipping")
            continue
        records.append({"file_base": key, "label": genre})

    df = pd.DataFrame(records)
    assert len(df) > 0, "No valid tracks found after stem completeness check"
    logger.info(f"Total valid tracks: {len(df)} across {df['label'].nunique()} genres")

    # Stratified split: first carve out test, then split remainder into train/val
    train_val_df, test_df = train_test_split(
        df, test_size=test_ratio, stratify=df["label"], random_state=seed
    )
    adjusted_val_ratio = val_ratio / (1 - test_ratio)
    train_df, val_df = train_test_split(
        train_val_df, test_size=adjusted_val_ratio, stratify=train_val_df["label"], random_state=seed
    )

    train_df = train_df.assign(split="train").reset_index(drop=True)
    val_df = val_df.assign(split="val").reset_index(drop=True)
    test_df = test_df.assign(split="test").reset_index(drop=True)

    for name, frame in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = splits_path / f"{name}.csv"
        frame.to_csv(out, index=False)
        logger.info(f"Saved {name}.csv: {len(frame)} tracks — {frame['label'].value_counts().to_dict()}")

    # Sanity checks
    all_splits = pd.concat([train_df, val_df, test_df])
    overlap = set(train_df["file_base"]) & set(val_df["file_base"]) & set(test_df["file_base"])
    assert len(overlap) == 0, f"Split overlap detected: {overlap}"
    assert len(all_splits) == len(df), "Row count mismatch after splitting"
    logger.info("Split sanity checks passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    build_splits(
        mel_path=PROJECT_ROOT / "data/processed/mel_spectrograms",
        splits_path=PROJECT_ROOT / "data/splits",
    )