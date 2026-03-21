import logging
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)


def load_audio(path: str | Path, sr: int = 22050, duration: float | None = 30.0) -> tuple[np.ndarray, int]:
    """
    Load a wav/mp3 file, resampling to *sr* and trimming/padding to *duration* seconds.

    Returns
    -------
    y   : float32 mono waveform
    sr  : sample rate (same as input *sr*)
    """
    path = str(path)
    try:
        y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    except Exception as exc:
        logger.error(f"Failed to load audio: {path!r} — {exc}")
        raise

    if duration is not None:
        target_len = int(sr * duration)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode="constant")
        else:
            y = y[:target_len]

    return y.astype(np.float32), sr


def compute_mel(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmax: float = 8000.0,
) -> np.ndarray:
    """
    Compute log-mel spectrogram from a waveform.

    Returns float32 array of shape (n_mels, T).
    """
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmax=fmax,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def fix_length(mel: np.ndarray, target_frames: int = 1300) -> np.ndarray:
    """Pad or trim a mel spectrogram to *target_frames* along the time axis."""
    n_mels, t = mel.shape
    if t < target_frames:
        pad = target_frames - t
        mel = np.pad(mel, ((0, 0), (0, pad)), mode="constant")
    else:
        mel = mel[:, :target_frames]
    assert mel.shape == (n_mels, target_frames), (
        f"fix_length: expected ({n_mels}, {target_frames}), got {mel.shape}"
    )
    return mel


def normalise_mel(mel: np.ndarray) -> np.ndarray:
    """Instance-wise mean/std normalisation."""
    mean = mel.mean()
    std = mel.std() + 1e-6
    return ((mel - mean) / std).astype(np.float32)