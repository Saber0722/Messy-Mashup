import logging
import random

import torch

logger = logging.getLogger(__name__)

STEMS = ["bass", "drums", "other", "vocals", "mix"]


class SpecAugment:
    """
    Apply time and frequency masking to each stem mel independently.

    Parameters follow the augmentation_config.yaml spec_augment section.
    """

    def __init__(
        self,
        time_mask_param: int = 80,
        freq_mask_param: int = 20,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
        apply_prob: float = 0.5,
    ) -> None:
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.apply_prob = apply_prob

    def _mask(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (1, F, T)"""
        mel = mel.clone()
        _, F, T = mel.shape

        for _ in range(self.num_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(0, F - f))
            mel[:, f0 : f0 + f, :] = 0.0

        for _ in range(self.num_time_masks):
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, max(0, T - t))
            mel[:, :, t0 : t0 + t] = 0.0

        return mel

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.apply_prob:
            return sample
        out = dict(sample)
        for stem in STEMS:
            if stem in out and isinstance(out[stem], torch.Tensor):
                out[stem] = self._mask(out[stem])
        return out