import logging
from pathlib import Path

import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.metrics import accuracy, compute_metrics

logger = logging.getLogger(__name__)
console = Console()

STEM_KEYS = ["bass", "drums", "other", "vocals"]


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler | None,
    grad_clip: float = 1.0,
    epoch: int = 0,
) -> dict:
    """Run one training epoch. Returns dict of metrics."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False, dynamic_ncols=True)

    for batch in pbar:
        batch = _to_device(batch, device)
        targets = batch["label"]

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                out = model(batch)
                loss = criterion(out["logits"], targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(batch)
            loss = criterion(out["logits"], targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_acc = accuracy(out["logits"].detach(), targets)
        total_loss += loss.item()
        total_acc += batch_acc

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.3f}")

    return {
        "loss": total_loss / n_batches,
        "acc": total_acc / n_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    idx2label: dict[int, str],
    epoch: int = 0,
) -> dict:
    """Run validation. Returns dict of metrics including full classification report."""
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_targets: list[int] = []
    all_stem_weights: list[torch.Tensor] = []

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [val]  ", leave=False, dynamic_ncols=True)

    for batch in pbar:
        batch = _to_device(batch, device)
        targets = batch["label"]

        out = model(batch)
        loss = criterion(out["logits"], targets)

        total_loss += loss.item()
        all_preds.extend(out["logits"].argmax(dim=1).cpu().tolist())
        all_targets.extend(targets.cpu().tolist())
        all_stem_weights.append(out["stem_weights"].cpu())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    metrics = compute_metrics(all_preds, all_targets, idx2label)
    metrics["loss"] = total_loss / len(loader)

    # Average stem attention weights
    sw = torch.cat(all_stem_weights, dim=0).mean(dim=0)  # (4,)
    metrics["stem_weights"] = {s: sw[i].item() for i, s in enumerate(STEM_KEYS)}

    return metrics


def print_epoch_summary(epoch: int, train_m: dict, val_m: dict) -> None:
    """Print a rich table summarising the epoch."""
    table = Table(title=f"Epoch {epoch}", show_header=True, header_style="bold cyan")
    table.add_column("Split", style="dim")
    table.add_column("Loss", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Macro F1", justify="right")

    table.add_row(
        "Train",
        f"{train_m['loss']:.4f}",
        f"{train_m['acc']:.4f}",
        "—",
    )
    table.add_row(
        "Val",
        f"{val_m['loss']:.4f}",
        f"{val_m['accuracy']:.4f}",
        f"{val_m['macro_f1']:.4f}",
    )
    console.print(table)

    # Stem weights
    sw = val_m.get("stem_weights", {})
    if sw:
        weight_str = "  ".join(f"{s}={w:.3f}" for s, w in sw.items())
        console.print(f"  [dim]Stem attention:[/dim] {weight_str}")