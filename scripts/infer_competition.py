"""
Standalone competition inference script.

Reads the competition test.csv (id, filename), extracts mel spectrograms
on-the-fly from the raw mashup audio, runs the saved model, and writes
submissions/submission_competition.csv in the required (id, genre) format.

Run from project root:
    python scripts/infer_competition.py
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
import yaml
from rich.console import Console
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.messy_mashup_model import build_model
from src.utils.audio_utils import compute_mel, fix_length, load_audio, normalise_mel
from src.utils.logger import get_logger

console = Console()
logger = get_logger(
    __name__,
    log_file=str(PROJECT_ROOT / "experiments/logs/infer_competition.log"),
)

STEMS = ["bass", "drums", "other", "vocals"]
ID_DIGITS = 4  # zero-pad width for output id column


# ── Mix-only forward pass ─────────────────────────────────────────────────────

def mix_only_forward(model, batch: dict) -> dict:
    """
    Override forward for competition inference where only mix audio is available.
    Routes the mix mel through ALL encoders (stem + mix) so the model sees real
    audio signal everywhere instead of zeros. Stem encoders produce imperfect
    embeddings from mixed audio, but far better than silence.
    """
    import torch
    mix = batch["mix"]  # (B, 1, n_mels, T)

    # Run mix through each stem encoder (they'll see mix instead of stems)
    stem_embeds = []
    for stem in model.stem_encoders:
        emb = model.stem_encoders[stem](mix)
        stem_embeds.append(emb)

    stem_stack = torch.stack(stem_embeds, dim=1)          # (B, 4, embed_dim)
    attended, stem_weights = model.attention(stem_stack)  # (B, embed_dim)

    mix_emb = model.mix_encoder(mix)                      # (B, embed_dim)
    fused = model.fusion(attended, mix_emb)               # (B, projected_dim)
    temporal = model.crnn(fused)                          # (B, hidden_size)
    logits = model.classifier(temporal)                   # (B, num_classes)

    return {"logits": logits, "stem_weights": stem_weights}


# ── Inline Dataset (no dependency on MultiBranchDataset) ──────────────────────

class CompetitionDataset(Dataset):
    """
    Loads raw stems from mashup_path, extracts mels on-the-fly.

    Expected directory layout:
        <mashup_path>/<song_id>/bass.wav
        <mashup_path>/<song_id>/drums.wav
        <mashup_path>/<song_id>/other.wav
        <mashup_path>/<song_id>/vocals.wav
    """

    def __init__(
        self,
        file_ids: list[str],          # e.g. ["song0001", "song0002", ...]
        mashup_path: Path,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        fmax: float = 8000.0,
        duration: float = 30.0,
        target_frames: int = 1300,
    ) -> None:
        self.file_ids = file_ids
        self.mashup_path = mashup_path
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmax = fmax
        self.duration = duration
        self.target_frames = target_frames

    def __len__(self) -> int:
        return len(self.file_ids)

    def _load_mix(self, song_id: str) -> np.ndarray | None:
        """Load the single mixed wav file (e.g. mashups/song0001.wav)."""
        wav_path = self.mashup_path / f"{song_id}.wav"
        if not wav_path.exists():
            logger.warning(f"Missing: {wav_path}")
            return None
        try:
            y, _ = load_audio(wav_path, sr=self.sample_rate, duration=self.duration)
            return y
        except Exception as exc:
            logger.error(f"Error loading {wav_path}: {exc}")
            return None

    def _to_mel_tensor(self, y: np.ndarray) -> torch.Tensor:
        mel = compute_mel(
            y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmax=self.fmax,
        )
        mel = fix_length(mel, self.target_frames)
        mel = normalise_mel(mel)  # identical to training pipeline
        return torch.from_numpy(mel).float().unsqueeze(0)  # (1, n_mels, T)

    def _zeros(self) -> torch.Tensor:
        return torch.zeros(1, self.n_mels, self.target_frames)

    def __getitem__(self, idx: int) -> dict:
        song_id = self.file_ids[idx]

        # Competition test set has only the mixed wav (no separated stems).
        # Feed the mix to all 5 branches — better than zeros.
        y = self._load_mix(song_id)
        if y is not None:
            mel = self._to_mel_tensor(y)
        else:
            mel = self._zeros()

        return {
            "bass":   mel,
            "drums":  mel,
            "other":  mel,
            "vocals": mel,
            "mix":    mel,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_cfg(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    base_cfg = load_cfg(PROJECT_ROOT / "configs/base_config.yaml")
    model_cfg = load_cfg(PROJECT_ROOT / "configs/model_config.yaml")
    inf_cfg = load_cfg(PROJECT_ROOT / "configs/inference_config.yaml")["inference"]

    audio = base_cfg["audio"]
    paths = base_cfg["paths"]

    mashup_path = PROJECT_ROOT / paths["mashup_path"]
    checkpoint_path = PROJECT_ROOT / inf_cfg["model_checkpoint"]
    le_path = PROJECT_ROOT / inf_cfg["label_encoder"]
    test_csv_path = PROJECT_ROOT / inf_cfg["test_csv"]
    submission_path = PROJECT_ROOT / "submissions/submission_competition.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    assert mashup_path.exists(), f"mashup_path not found: {mashup_path}"
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
    assert le_path.exists(), f"Label encoder not found: {le_path}"
    assert test_csv_path.exists(), f"Test CSV not found: {test_csv_path}"

    # ── Device ────────────────────────────────────────────────────────────────
    cfg_device = inf_cfg.get("device", "auto")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg_device == "auto" else torch.device(cfg_device)
    )
    console.print(f"[cyan]Device:[/cyan] {device}")

    # ── Label encoder ─────────────────────────────────────────────────────────
    with open(le_path, "rb") as f:
        le = pickle.load(f)
    idx2label = {i: g for i, g in enumerate(le.classes_)}
    console.print(f"[cyan]Classes:[/cyan] {list(le.classes_)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(num_classes=len(le.classes_), model_cfg=model_cfg["model"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    console.print(f"[cyan]Checkpoint:[/cyan] {checkpoint_path}")

    # ── Test CSV ──────────────────────────────────────────────────────────────
    test_df = pd.read_csv(test_csv_path)
    console.print(f"[cyan]Test CSV:[/cyan] {test_csv_path} — {len(test_df)} rows")
    console.print(f"[dim]Columns: {list(test_df.columns)}[/dim]")

    # filename column: "mashups/song1510" → song_id = "song1510"
    test_df["song_id"] = test_df["filename"].apply(lambda x: Path(x).stem)  # strip .wav extension

    # ── Dataset / Loader ──────────────────────────────────────────────────────
    dataset = CompetitionDataset(
        file_ids=test_df["song_id"].tolist(),
        mashup_path=mashup_path,
        sample_rate=audio["sample_rate"],
        n_mels=audio["n_mels"],
        n_fft=audio["n_fft"],
        hop_length=audio["hop_length"],
        fmax=audio["fmax"],
        duration=audio["duration"],
        target_frames=audio["target_frames"],
    )

    loader = DataLoader(
        dataset,
        batch_size=inf_cfg["batch_size"],
        shuffle=False,
        num_workers=inf_cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    # Use mix_only_forward: routes mix audio through all encoders since
    # the competition test set has no separated stems.
    predictions: list[str] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Competition inference", dynamic_ncols=True):
            batch = _to_device(batch, device)
            out = mix_only_forward(model, batch)
            preds = out["logits"].argmax(dim=1).cpu().tolist()
            predictions.extend(idx2label[p] for p in preds)

    assert len(predictions) == len(test_df), (
        f"Prediction count mismatch: {len(predictions)} vs {len(test_df)}"
    )

    # ── Write submission ──────────────────────────────────────────────────────
    # Use the id column directly from test.csv — never regenerate ids
    out_df = pd.DataFrame({
        "id":    test_df["id"].astype(str).str.zfill(ID_DIGITS).tolist(),
        "genre": predictions,
    })
    out_df.to_csv(submission_path, index=False)

    console.print(f"\n[bold green]Submission written:[/bold green] {submission_path}")
    console.print(f"[green]Rows:[/green] {len(out_df)}")
    console.print(f"[green]ID range:[/green] {out_df['id'].iloc[0]} → {out_df['id'].iloc[-1]}")
    console.print(f"\n[green]Genre distribution:[/green]")
    console.print(out_df["genre"].value_counts().to_string())


if __name__ == "__main__":
    main()