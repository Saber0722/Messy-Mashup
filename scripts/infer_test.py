"""
Generate submission.csv from the test set.

Run from project root:
    python scripts/infer_test.py
"""

import pickle
import sys
from pathlib import Path

import torch
import yaml
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import run_inference
from src.inference.submission_writer import write_submission
from src.models.messy_mashup_model import build_model
from src.utils.logger import get_logger

console = Console()
logger = get_logger(__name__, log_file=str(PROJECT_ROOT / "experiments/logs/infer_test.log"))


def load_cfg(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    base_cfg = load_cfg(PROJECT_ROOT / "configs/base_config.yaml")
    model_cfg = load_cfg(PROJECT_ROOT / "configs/model_config.yaml")
    inf_cfg = load_cfg(PROJECT_ROOT / "configs/inference_config.yaml")["inference"]

    paths = base_cfg["paths"]
    mel_path = PROJECT_ROOT / paths["mel_path"]
    splits_path = PROJECT_ROOT / paths["splits_path"]
    checkpoint_path = PROJECT_ROOT / inf_cfg["model_checkpoint"]
    le_path = PROJECT_ROOT / inf_cfg["label_encoder"]

    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
    assert le_path.exists(), f"Label encoder not found: {le_path}"

    cfg_device = inf_cfg.get("device", "auto")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg_device == "auto" else torch.device(cfg_device)
    )

    with open(le_path, "rb") as f:
        le = pickle.load(f)
    num_classes = len(le.classes_)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(num_classes=num_classes, model_cfg=model_cfg["model"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    predictions = run_inference(
        model=model,
        test_csv=splits_path / "test.csv",
        mel_path=mel_path,
        label_encoder_path=le_path,
        device=device,
        batch_size=inf_cfg["batch_size"],
        num_workers=inf_cfg["num_workers"],
    )

    sample_sub = PROJECT_ROOT / "data/raw/messy_mashup/sample_submission.csv"
    write_submission(
        predictions,
        submission_path=PROJECT_ROOT / inf_cfg["submission_path"],
        sample_submission_path=sample_sub if sample_sub.exists() else None,
    )

    console.print(f"[bold green]Submission written:[/bold green] {inf_cfg['submission_path']}")


if __name__ == "__main__":
    main()