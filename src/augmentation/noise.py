import logging
import random

import torch

logger = logging.getLogger(__name__)

STEMS = ["bass", "drums", "other", "vocals", "mix"]


class AddNoise:
    def __init__(
        self,
        min_snr_db: float = 15.0,
        max_snr_db: float = 40.0,
        apply_prob: float = 0.5,
    ) -> None:
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.apply_prob = apply_prob

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.apply_prob:
            return sample
        snr_db = random.uniform(self.min_snr_db, self.max_snr_db)
        out = dict(sample)
        for stem in STEMS:
            if stem in out and isinstance(out[stem], torch.Tensor):
                mel = out[stem]
                signal_power = mel.pow(2).mean()
                snr_linear = 10 ** (snr_db / 10.0)
                noise_power = signal_power / (snr_linear + 1e-8)
                noise = torch.randn_like(mel) * noise_power.sqrt()
                out[stem] = mel + noise
        return out