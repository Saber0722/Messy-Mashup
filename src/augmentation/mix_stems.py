import logging
import random
from pathlib import Path

import numpy as np
import torch

from src.utils.audio_utils import fix_length, normalise_mel

logger = logging.getLogger(__name__)

STEMS = ["bass", "drums", "other", "vocals", "mix"]


class StemMixer:
    """
    Parameters
    ----------
    mel_path   : directory of pre-computed .npy mel files
    mix_prob   : probability of applying mixing to a given stem
    alpha/beta : mixing coefficient drawn from Uniform(alpha, beta)
    apply_prob : probability that any mixing happens at all this sample
    """

    def __init__(
        self,
        mel_path: str | Path,
        mix_prob: float = 0.3,
        alpha: float = 0.1,
        beta: float = 0.4,
        apply_prob: float = 0.5,
        target_frames: int = 1300,
    ) -> None:
        self.mel_path = Path(mel_path)
        self.mix_prob = mix_prob
        self.alpha = alpha
        self.beta = beta
        self.apply_prob = apply_prob
        self.target_frames = target_frames

        # Build index: stem -> list of .npy paths
        self._index: dict[str, list[Path]] = {s: [] for s in STEMS}
        for f in self.mel_path.glob("*.npy"):
            parts = f.stem.split("__")
            if len(parts) == 3:
                stem = parts[2]
                if stem in self._index:
                    self._index[stem].append(f)

        for stem, paths in self._index.items():
            logger.debug(f"StemMixer index — {stem}: {len(paths)} files")

    def _load_random(self, stem: str) -> torch.Tensor | None:
        paths = self._index.get(stem, [])
        if not paths:
            return None
        path = random.choice(paths)
        try:
            mel = np.load(path).astype(np.float32)
            mel = fix_length(mel, self.target_frames)
            mel = normalise_mel(mel)
            return torch.from_numpy(mel).float().unsqueeze(0)
        except Exception as exc:
            logger.error(f"StemMixer load error {path}: {exc}")
            return None

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.apply_prob:
            return sample
        out = dict(sample)
        for stem in STEMS:
            if stem not in out:
                continue
            if random.random() > self.mix_prob:
                continue
            other = self._load_random(stem)
            if other is None:
                continue
            lam = random.uniform(self.alpha, self.beta)
            out[stem] = (1 - lam) * out[stem] + lam * other
        return out