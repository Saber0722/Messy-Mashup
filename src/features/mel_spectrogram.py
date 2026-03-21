import logging
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.utils.audio_utils import compute_mel, fix_length, load_audio

logger = logging.getLogger(__name__)

STEMS = ["bass", "drums", "other", "vocals"]


def extract_and_save(
    genres_path: str | Path,
    mel_save_path: str | Path,
    sample_rate: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmax: float = 8000.0,
    target_frames: int = 1300,
    duration: float = 30.0,
) -> None:
    """
    Walk *genres_path*, extract one mel per stem + one for the mixture per track,
    and save each as a .npy file under *mel_save_path*.

    Naming convention:
        <genre>__<track>__<stem>.npy
        <genre>__<track>__mix.npy
    """
    genres_path = Path(genres_path)
    mel_save_path = Path(mel_save_path)
    mel_save_path.mkdir(parents=True, exist_ok=True)

    assert genres_path.exists(), f"genres_path does not exist: {genres_path}"

    genres = sorted(os.listdir(genres_path))
    logger.info(f"Found {len(genres)} genres: {genres}")

    errors = 0

    for genre in genres:
        genre_dir = genres_path / genre
        tracks = sorted(os.listdir(genre_dir))

        for track in tqdm(tracks, desc=genre, leave=False):
            track_dir = genre_dir / track
            if not track_dir.is_dir():
                continue

            stem_waves: dict[str, np.ndarray] = {}

            for stem in STEMS:
                wav_path = track_dir / f"{stem}.wav"
                if not wav_path.exists():
                    logger.warning(f"Missing stem: {wav_path}")
                    continue
                try:
                    y, sr = load_audio(wav_path, sr=sample_rate, duration=duration)
                    stem_waves[stem] = y
                    mel = compute_mel(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=fmax)
                    mel = fix_length(mel, target_frames)
                    save_name = f"{genre}__{track}__{stem}.npy"
                    np.save(mel_save_path / save_name, mel)
                except Exception as exc:
                    logger.error(f"Error processing {wav_path}: {exc}")
                    errors += 1

            # Build mixture = mean of available stems
            if stem_waves:
                try:
                    mix = np.mean(list(stem_waves.values()), axis=0)
                    mel_mix = compute_mel(mix, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=fmax)
                    mel_mix = fix_length(mel_mix, target_frames)
                    mix_name = f"{genre}__{track}__mix.npy"
                    np.save(mel_save_path / mix_name, mel_mix)
                except Exception as exc:
                    logger.error(f"Error building mix for {genre}/{track}: {exc}")
                    errors += 1

    logger.info(f"Done. Total errors: {errors}")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    extract_and_save(
        genres_path=PROJECT_ROOT / "data/raw/messy_mashup/genres_stems",
        mel_save_path=PROJECT_ROOT / "data/processed/mel_spectrograms",
    )