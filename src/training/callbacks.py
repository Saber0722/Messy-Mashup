import logging
import shutil
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience    : epochs to wait after last improvement
    min_delta   : minimum change to qualify as an improvement
    mode        : 'min' (loss) or 'max' (accuracy)
    """

    def __init__(self, patience: int = 8, min_delta: float = 1e-4, mode: str = "min") -> None:
        assert mode in ("min", "max"), f"mode must be 'min' or 'max', got {mode!r}"
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best: float | None = None
        self.triggered = False

    def step(self, value: float) -> bool:
        """
        Returns True if training should stop.
        """
        if self.best is None:
            self.best = value
            return False

        improved = (
            (value < self.best - self.min_delta) if self.mode == "min"
            else (value > self.best + self.min_delta)
        )

        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping: no improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                logger.info("EarlyStopping triggered.")
                self.triggered = True
                return True
        return False


class ModelCheckpoint:
    """
    Save the best model and optionally the last model.

    Parameters
    ----------
    checkpoint_dir  : directory to save .pth files
    metric          : metric name (for logging only)
    mode            : 'min' or 'max'
    save_last       : also save the most recent epoch
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        metric: str = "val_acc",
        mode: str = "max",
        save_last: bool = True,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metric = metric
        self.mode = mode
        self.save_last = save_last
        self.best: float | None = None

    def step(
        self,
        value: float,
        model: nn.Module,
        epoch: int,
        extra: dict | None = None,
    ) -> bool:
        """
        Save checkpoint if *value* improves.  Returns True if a new best was saved.
        """
        improved = self.best is None or (
            (value < self.best) if self.mode == "min" else (value > self.best)
        )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            self.metric: value,
            **(extra or {}),
        }

        if self.save_last:
            last_path = self.checkpoint_dir / "last_model.pth"
            torch.save(state, last_path)

        if improved:
            self.best = value
            best_path = self.checkpoint_dir / "best_multibranch_model.pth"
            torch.save(state, best_path)
            logger.info(
                f"Checkpoint saved — epoch {epoch}, {self.metric}={value:.4f} → {best_path}"
            )
            return True

        return False