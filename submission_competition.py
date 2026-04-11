"""
Standalone competition inference script for the fine-tuned transformer model (10sec audio, 90/10 split).

Reads the competition test.csv (id, filename), extracts raw audio
on-the-fly from the raw mashup audio, runs the saved writes
submissions/submission_competition_transformer_10sec.csv in the required (id, genre) format.

Run from project root:
    python scripts/infer_competition_transformer_10sec.py
"""

import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from rich.console import Console
from torch.utils.data import DataLoader, Dataset
from tqdm import main, tqdm
import librosa # Use librosa for loading audio

# --- ADD RICH IMPORTS ---
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
console = Console()
# -----------------------

PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger(
    __name__,
    log_file=str(PROJECT_ROOT / "infer_competition_transformer_10sec.log"),
)

ID_DIGITS = 4  # zero-pad width for output id column


# ── Inline Dataset ──────────────────────────────────────────────────────

class CompetitionDataset(Dataset):
    """
    Loads raw wav files from mashup_path based on filenames in test CSV,
    prepares audio for the transformer model (10 sec clips).
    Expected files: <mashup_path>/<filename_from_csv>.wav (e.g., mashups/song0001.wav)
    """

    def __init__(
        self,
        file_ids: list[str],          # e.g. ["song0001", "song0002", ...] from CSV
        mashup_path: Path,
        sample_rate: int = 16000,     # AST default
        duration_sec: float = 10.0,   # Match new training script (was 5.0)
    ) -> None:
        self.file_ids = file_ids
        self.mashup_path = mashup_path
        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * duration_sec) # 160000 for 10 sec

    def __len__(self) -> int:
        return len(self.file_ids)

    def _load_and_process_audio(self, song_id: str) -> np.ndarray | None:
        """Load and process the wav file."""
        wav_path = self.mashup_path / f"{song_id}.wav"
        if not wav_path.exists():
            logger.warning(f"Missing: {wav_path}")
            return None
        try:
            # Load audio
            y, sr = librosa.load(wav_path, sr=self.sample_rate, mono=True)
            # Fix length (pad or truncate to 10 seconds)
            if len(y) < self.target_samples:
                y = np.pad(y, (0, self.target_samples - len(y)), mode='constant')
            else:
                y = y[:self.target_samples]
            return y
        except Exception as exc:
            logger.error(f"Error loading {exc}")
            return None

    def __getitem__(self, idx: int) -> dict:
        song_id = self.file_ids[idx]

        y = self._load_and_process_audio(song_id)
        if y is not None:
            audio_tensor = torch.from_numpy(y).float()
        else:
            # Return zeros if loading failed
            audio_tensor = torch.zeros(self.target_samples, dtype=torch.float)

        return {
            "input_values": audio_tensor,
            "song_id": song_id # Include for potential debugging
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # --- Configuration (Hardcoded for New Transformer Model) ---
    # --- Update these paths to match your setup ---
    # Use the checkpoint directory from the NEW training run
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints_transformer_90_10_es_wandb_rich"
    MODEL_PATH = CHECKPOINT_DIR / "pytorch_model.bin" # Or "model.safetensors"
    CONFIG_PATH = CHECKPOINT_DIR / "config.json"
    LABEL_ENCODER_PATH = CHECKPOINT_DIR / "label_encoder.pkl"

    MASHUP_PATH = PROJECT_ROOT / "data" / "raw" / "messy_mashup" / "mashups" # Path to your test wav files
    TEST_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "messy_mashup" / "test.csv" # Path to the provided test.csv
    SUBMISSION_PATH = PROJECT_ROOT / "submissions/submission_competition_transformer_10sec.csv"

    # Audio params (must match NEW training: 10 sec)
    SAMPLE_RATE_AST = 16000
    DURATION_SEC = 10.0 # NEW: Match training script
    BATCH_SIZE = 16 # Match training batch size or adjust as needed

    # Ensure directories exist
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    assert MASHUP_PATH.exists(), f"mashup_path not found: {MASHUP_PATH}"
    assert TEST_CSV_PATH.exists(), f"Test CSV not found: {TEST_CSV_PATH}"
    assert MODEL_PATH.exists() or (CHECKPOINT_DIR / "model.safetensors").exists(), \
           f"Model weights not found in {CHECKPOINT_DIR}. Expected 'pytorch_model.bin' or 'model.safetensors'."
    assert CONFIG_PATH.exists(), f"Config not found: {CONFIG_PATH}"
    assert LABEL_ENCODER_PATH.exists(), f"Label encoder not found: {LABEL_ENCODER_PATH}"

    # Print paths using Rich
    path_table = Table(title="Inference Configuration Paths", show_header=True, header_style="bold magenta")
    path_table.add_column("Component", style="dim", width=20)
    path_table.add_column("Path", style="green")
    path_table.add_row("Model Checkpoint Dir", str(CHECKPOINT_DIR))
    path_table.add_row("Model Weights", str(MODEL_PATH))
    path_table.add_row("Config", str(CONFIG_PATH))
    path_table.add_row("Label Encoder", str(LABEL_ENCODER_PATH))
    path_table.add_row("Mashup Files", str(MASHUP_PATH))
    path_table.add_row("Test CSV", str(TEST_CSV_PATH))
    path_table.add_row("Submission Output", str(SUBMISSION_PATH))
    console.print(Panel(path_table, border_style="blue"))

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[cyan]Device:[/cyan] {device}")

    # ── Label encoder ─────────────────────────────────────────────────────────
    with open(LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    idx2label = {i: g for i, g in enumerate(le.classes_)}
    console.print(f"[cyan]Classes:[/cyan] {list(le.classes_)}")

# ── Feature Extractor & Model ─────────────────────────────────────────────────────────
    console.print(f"[cyan]Loading model from:[/cyan] {CHECKPOINT_DIR}")
    feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    # Load config and instantiate model
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(str(CONFIG_PATH))
    model = AutoModelForAudioClassification.from_config(config=config)

    # Load weights
    if (CHECKPOINT_DIR / "pytorch_model.bin").exists():
        MODEL_WEIGHTS_PATH = CHECKPOINT_DIR / "pytorch_model.bin"
        state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device, weights_only=True)
    elif (CHECKPOINT_DIR / "model.safetensors").exists():
        MODEL_WEIGHTS_PATH = CHECKPOINT_DIR / "model.safetensors"
        from safetensors.torch import load_file
        state_dict = load_file(MODEL_WEIGHTS_PATH, device=str(device))
    else:
        raise FileNotFoundError("No model weights found (.bin or .safetensors)")

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    console.print(f"[cyan]Model loaded:[/cyan] {CHECKPOINT_DIR}")

    # ── Test CSV ──────────────────────────────────────────────────────────────
    test_df = pd.read_csv(TEST_CSV_PATH)
    console.print(f"[cyan]Test CSV:[/cyan] {TEST_CSV_PATH} — {len(test_df)} rows")
    console.print(f"[dim]Columns: {list(test_df.columns)}[/dim]")

    # filename column: "mashups/song1510.wav" → song_id = "song1510"
    test_df["song_id"] = test_df["filename"].apply(lambda x: Path(x).stem)  # strip .wav extension

    # ── Dataset / Loader ──────────────────────────────────────────────────────
    dataset = CompetitionDataset(
        file_ids=test_df["song_id"].tolist(),
        mashup_path=MASHUP_PATH,
        sample_rate=SAMPLE_RATE_AST,
        duration_sec=DURATION_SEC, # NEW: Use 10 sec
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # Keep order consistent with test_df
        num_workers=4, # Adjust as needed
        pin_memory=(device.type == "cuda"),
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    predictions: list[str] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Competition inference (10sec)", dynamic_ncols=True):
            audio_batch = batch["input_values"]
            # Preprocess batch using feature extractor
            inputs = feature_extractor(
                audio_batch.numpy(), sampling_rate=SAMPLE_RATE_AST, return_tensors="pt", padding=True
            )
            x = inputs["input_values"].to(device)

            outputs = model(x)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().tolist()

            predictions.extend(idx2label[p] for p in preds)

    assert len(predictions) == len(test_df), f"Prediction count mismatch: {len(predictions)} vs {len(test_df)}"

    # ── Write submission ──────────────────────────────────────────────────────
    # Use the id column directly from test.csv
    out_df = pd.DataFrame({
        "id":    test_df["id"].astype(str).str.zfill(ID_DIGITS).tolist(),
        "genre": predictions,
    })
    out_df.to_csv(SUBMISSION_PATH, index=False)

    console.print(f"\n[bold green]Submission written:[/bold green] {SUBMISSION_PATH}")
    console.print(f"[green] {len(out_df)} predictions generated for submission.[/green]")
    console.print(f"[green]ID range:[/green] {out_df['id'].iloc[0]} → {out_df['id'].iloc[-1]}")
    console.print(f"\n[green]Genre distribution:[/green]")
    genre_counts = out_df["genre"].value_counts()
    genre_dist_table = Table(show_header=True, header_style="bold cyan")
    genre_dist_table.add_column("Genre", style="dim")
    genre_dist_table.add_column("Count", justify="right")
    for genre, count in genre_counts.items():
        genre_dist_table.add_row(genre, str(count))
    console.print(genre_dist_table)

    console.print(f"\n[bold blue]Number of rows written: {len(out_df)}[/bold blue]")
    console.print(Panel(Text("Ready for Kaggle Submission!", style="bold yellow"), border_style="green"))


if __name__ == "__main__":
    main()