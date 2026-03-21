import logging
import random

import torch

logger = logging.getLogger(__name__)

STEMS = ["bass", "drums", "other", "vocals", "mix"]


class GainJitter:
    def __init__(
        self,
        min_gain_db: float = -6.0,
        max_gain_db: float = 6.0,
        apply_prob: float = 0.5,
    ) -> None:
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db
        self.apply_prob = apply_prob

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.apply_prob:
            return sample
        gain = random.uniform(self.min_gain_db, self.max_gain_db)
        out = dict(sample)
        for stem in STEMS:
            if stem in out and isinstance(out[stem], torch.Tensor):
                out[stem] = out[stem] + gain
        return out