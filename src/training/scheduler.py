import logging

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau, StepLR

logger = logging.getLogger(__name__)


def build_scheduler(optimizer: Optimizer, cfg: dict, steps_per_epoch: int = 0):
    """
    Build a scheduler from the training_config scheduler section.

    Supported:
        CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR, StepLR
    """
    name = cfg.get("name", "CosineAnnealingLR")

    if name == "CosineAnnealingLR":
        sched = CosineAnnealingLR(
            optimizer,
            T_max=cfg.get("T_max", 50),
            eta_min=cfg.get("eta_min", 1e-6),
        )
    elif name == "ReduceLROnPlateau":
        sched = ReduceLROnPlateau(
            optimizer,
            mode=cfg.get("mode", "min"),
            factor=cfg.get("factor", 0.5),
            patience=cfg.get("patience", 4),
            min_lr=cfg.get("min_lr", 1e-6),
        )
    elif name == "OneCycleLR":
        assert steps_per_epoch > 0, "OneCycleLR requires steps_per_epoch"
        sched = OneCycleLR(
            optimizer,
            max_lr=cfg.get("max_lr", 1e-3),
            epochs=cfg.get("epochs", 50),
            steps_per_epoch=steps_per_epoch,
            pct_start=cfg.get("pct_start", 0.3),
        )
    elif name == "StepLR":
        sched = StepLR(
            optimizer,
            step_size=cfg.get("step_size", 10),
            gamma=cfg.get("gamma", 0.5),
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}")

    logger.info(f"Scheduler: {name}")
    return sched