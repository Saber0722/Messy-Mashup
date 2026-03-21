import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def build_augmentation(aug_cfg: dict, mel_path: str | Path, train: bool = True) -> Callable | None:
    """
    Build and compose the augmentation pipeline from the augmentation_config
    dict.  Returns None if augmentation is disabled or train=False.
    """
    if not train or not aug_cfg.get("enabled", False):
        return None

    from src.augmentation.gain import GainJitter
    from src.augmentation.mix_stems import StemMixer
    from src.augmentation.noise import AddNoise
    from src.augmentation.spec_augment import SpecAugment

    transforms: list[Callable] = []
    apply_prob = aug_cfg.get("apply_prob", 0.5)

    if aug_cfg.get("gain", {}).get("enabled", False):
        g = aug_cfg["gain"]
        transforms.append(
            GainJitter(
                min_gain_db=g.get("min_gain_db", -6.0),
                max_gain_db=g.get("max_gain_db", 6.0),
                apply_prob=apply_prob,
            )
        )

    if aug_cfg.get("noise", {}).get("enabled", False):
        n = aug_cfg["noise"]
        transforms.append(
            AddNoise(
                min_snr_db=n.get("min_snr_db", 15.0),
                max_snr_db=n.get("max_snr_db", 40.0),
                apply_prob=apply_prob,
            )
        )

    if aug_cfg.get("spec_augment", {}).get("enabled", False):
        s = aug_cfg["spec_augment"]
        transforms.append(
            SpecAugment(
                time_mask_param=s.get("time_mask_param", 80),
                freq_mask_param=s.get("freq_mask_param", 20),
                num_time_masks=s.get("num_time_masks", 2),
                num_freq_masks=s.get("num_freq_masks", 2),
                apply_prob=apply_prob,
            )
        )

    if aug_cfg.get("mix_stems", {}).get("enabled", False):
        m = aug_cfg["mix_stems"]
        transforms.append(
            StemMixer(
                mel_path=mel_path,
                mix_prob=m.get("mix_prob", 0.3),
                alpha=m.get("alpha", 0.1),
                beta=m.get("beta", 0.4),
                apply_prob=apply_prob,
            )
        )

    if not transforms:
        logger.warning("Augmentation enabled but no transforms configured — returning None")
        return None

    def pipeline(sample: dict) -> dict:
        for t in transforms:
            sample = t(sample)
        return sample

    logger.info(f"Augmentation pipeline: {[type(t).__name__ for t in transforms]}")
    return pipeline