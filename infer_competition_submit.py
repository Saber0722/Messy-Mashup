"""
infer_competition_transformer_v2.py

Inference script for the v2 transformer model with:
  - Multi-seed model ensemble (averages logits across all available seeds)
  - 15-run TTA (1 clean + 14 augmented passes per sample)
  - Confidence summary (useful for pseudo-labelling follow-up)

Expects checkpoints at:
    checkpoints_transformer_v2/best_model_s{seed}/   ← one per seed
    checkpoints_transformer_v2/label_encoder.pkl

If only a single model exists at checkpoints_transformer_v2/best_model/
it falls back gracefully to single-model inference.

Run from project root:
    python scripts/infer_competition_transformer_v2.py
"""

import logging
import pickle
import random
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

warnings.filterwarnings("ignore")
console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[0]
CKPT_ROOT    = PROJECT_ROOT / "checkpoints_transformer_v3"

# List every seed you trained. Script auto-discovers which checkpoints exist.
# Naming convention: checkpoints_transformer_v2/best_model_s42/  etc.
ENSEMBLE_SEEDS = [42, 123, 7]

LABEL_ENCODER_PATH = CKPT_ROOT / "label_encoder.pkl"
MASHUP_PATH        = PROJECT_ROOT / "data" / "raw" / "messy_mashup" / "mashups"
TEST_CSV_PATH      = PROJECT_ROOT / "data" / "raw" / "messy_mashup" / "test.csv"
SUBMISSION_PATH    = PROJECT_ROOT / "submissions" / "submission_transformer_v2.csv"

SAMPLE_RATE  = 16000
DURATION_SEC = 10
TARGET_LEN   = SAMPLE_RATE * DURATION_SEC

# TTA: total passes per sample per model (1 clean + rest augmented)
TTA_RUNS  = 15
BATCH_SIZE = 8    # reduce to 4 if OOM

# Augmentation params — must mirror train_transformer_v2.py exactly
TEMPO_RANGE     = (0.85, 1.15)
PITCH_SEMITONES = (-2, 2)
NOISE_SNR_RANGE = (5, 25)
NOISE_PROB      = 0.8

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Audio helpers ─────────────────────────────────────────────────────────────

def fix_len(y: np.ndarray, target: int = TARGET_LEN) -> np.ndarray:
    if len(y) < target:
        return np.pad(y, (0, target - len(y)))
    return y[:target]


def load_audio(path: Path) -> np.ndarray:
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=DURATION_SEC)
    return fix_len(y).astype(np.float32)


def augment_audio(y: np.ndarray, noise_files: list) -> np.ndarray:
    """
    Mirrors the training augmentation pipeline exactly so TTA samples
    match the distribution the model was trained on.
    """
    # Random start offset (simulate different mashup entry points)
    offset = random.randint(0, TARGET_LEN // 4)
    y = fix_len(y[offset:])

    # Tempo stretch
    if random.random() < 0.7:
        try:
            y = librosa.effects.time_stretch(y, rate=random.uniform(*TEMPO_RANGE))
        except Exception:
            pass
        y = fix_len(y)

    # Pitch shift
    if random.random() < 0.4:
        try:
            y = librosa.effects.pitch_shift(
                y, sr=SAMPLE_RATE, n_steps=random.uniform(*PITCH_SEMITONES)
            )
        except Exception:
            pass
        y = fix_len(y)

    # Random gain
    y = y * random.uniform(0.7, 1.0)

    # Additive environmental noise
    if noise_files and random.random() < NOISE_PROB:
        snr_db = random.uniform(*NOISE_SNR_RANGE)
        try:
            noise, _ = librosa.load(random.choice(noise_files), sr=SAMPLE_RATE, mono=True)
            noise     = fix_len(noise)
            sig_pow   = np.mean(y ** 2) + 1e-10
            noise_pow = np.mean(noise ** 2) + 1e-10
            noise    *= np.sqrt(sig_pow / (noise_pow * 10 ** (snr_db / 10)))
            y         = y + noise
        except Exception:
            pass

    peak = np.max(np.abs(y)) + 1e-8
    return (y / peak).astype(np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class TestDataset(Dataset):
    """Loads raw mashup audio once. TTA augmentation is applied in the loop."""

    def __init__(self, song_ids: list, mashup_path: Path):
        self.song_ids    = song_ids
        self.mashup_path = mashup_path

    def __len__(self):
        return len(self.song_ids)

    def __getitem__(self, idx):
        song_id  = self.song_ids[idx]
        wav_path = self.mashup_path / f"{song_id}.wav"
        if wav_path.exists():
            try:
                return load_audio(wav_path), song_id
            except Exception:
                console.print(f"[red]Error loading {wav_path}[/red]")
        else:
            console.print(f"[red]Missing: {wav_path}[/red]")
        return np.zeros(TARGET_LEN, dtype=np.float32), song_id


def collate_audio(batch):
    audios, ids = zip(*batch)
    return np.stack(audios), list(ids)


# ── Checkpoint discovery ──────────────────────────────────────────────────────

def discover_checkpoints(ckpt_root: Path, seeds: list) -> list[Path]:
    """
    Returns checkpoint directories to ensemble, in priority order:
      1. best_model_s{seed}/ for each seed found on disk
      2. best_model/ as single-model fallback
    """
    found = []
    for seed in seeds:
        p = ckpt_root / f"best_model_s{seed}"
        if p.exists():
            found.append(p)
            console.print(f"  [green]✓[/green] seed-{seed}: {p.name}")
        else:
            console.print(f"  [yellow]✗[/yellow] seed-{seed}: not found, skipping")

    if not found:
        fallback = ckpt_root / "best_model"
        if fallback.exists():
            console.print(f"  [yellow]No seed checkpoints found — falling back to:[/yellow] {fallback.name}")
            found.append(fallback)
        else:
            raise FileNotFoundError(
                f"No model checkpoints found under {ckpt_root}.\n"
                "Expected  best_model_s{{seed}}/  or  best_model/"
            )
    return found


# ── TTA inference for a single model ─────────────────────────────────────────

@torch.no_grad()
def run_tta_for_model(
    model,
    feature_extractor,
    all_audio: list,
    noise_files: list,
    device: torch.device,
    n_runs: int,
    label: str,
) -> torch.Tensor:
    """
    Runs n_runs passes (1 clean + n_runs-1 augmented) over all_audio.
    Returns summed logit tensor of shape (n_samples, n_classes).
    Caller divides by the total number of passes to get the average.
    """
    model.eval()
    n = len(all_audio)
    accumulated = None

    for run_idx in range(n_runs):
        is_clean = (run_idx == 0)
        desc = f"[{label}] {'clean ' if is_clean else f'aug {run_idx:2d}/{n_runs-1}'}"
        run_logits = []

        for start in tqdm(range(0, n, BATCH_SIZE), desc=desc, leave=False):
            batch = all_audio[start : start + BATCH_SIZE]
            processed = (
                list(batch) if is_clean
                else [augment_audio(a.copy(), noise_files) for a in batch]
            )
            inputs = feature_extractor(
                processed,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.cuda.amp.autocast():
                outputs = model(**inputs)
            run_logits.append(outputs.logits.cpu().float())

        run_tensor  = torch.cat(run_logits, dim=0)          # (n, C)
        accumulated = run_tensor if accumulated is None else accumulated + run_tensor

        # Per-run snapshot so you can watch convergence in the logs
        snap_preds  = run_tensor.argmax(dim=1)
        console.print(
            f"  [{label}] run {run_idx:2d} done | "
            f"top class counts: { dict(zip(*snap_preds.unique(return_counts=True))) }"
        )

    return accumulated   # summed, not averaged yet


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Validate fixed paths ──────────────────────────────────────────────────
    for name, p in [
        ("Label encoder", LABEL_ENCODER_PATH),
        ("Mashup dir",    MASHUP_PATH),
        ("Test CSV",      TEST_CSV_PATH),
    ]:
        assert p.exists(), f"{name} not found: {p}"

    # ── Discover ensemble checkpoints ─────────────────────────────────────────
    console.rule("[bold cyan]Discovering checkpoints")
    ckpt_dirs = discover_checkpoints(CKPT_ROOT, ENSEMBLE_SEEDS)
    n_models  = len(ckpt_dirs)

    # ── Config table ──────────────────────────────────────────────────────────
    cfg = Table(title="Inference Config", show_header=True, header_style="bold magenta")
    cfg.add_column("Parameter", style="dim", width=26)
    cfg.add_column("Value", style="green")
    for k, v in [
        ("Models in ensemble",   n_models),
        ("TTA runs / model",     TTA_RUNS),
        ("Total inference passes", n_models * TTA_RUNS),
        ("Batch size",           BATCH_SIZE),
        ("Sample rate",          SAMPLE_RATE),
        ("Duration (sec)",       DURATION_SEC),
        ("Noise prob",           NOISE_PROB),
        ("Tempo range",          str(TEMPO_RANGE)),
        ("Pitch range (semitones)", str(PITCH_SEMITONES)),
        ("Submission out",       str(SUBMISSION_PATH)),
    ]:
        cfg.add_row(str(k), str(v))
    console.print(Panel(cfg, border_style="blue"))

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[cyan]Device:[/cyan] {device}")

    # ── Label encoder ─────────────────────────────────────────────────────────
    with open(LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    idx2label = {i: g for i, g in enumerate(le.classes_)}
    console.print(f"[cyan]Classes ({len(le.classes_)}):[/cyan] {list(le.classes_)}")

    # ── Noise files ───────────────────────────────────────────────────────────
    noise_dir   = PROJECT_ROOT / "data" / "raw" / "messy_mashup" / "ESC-50-master" / "audio"
    noise_files = sorted(noise_dir.glob("*.wav")) if noise_dir.exists() else []
    console.print(f"[cyan]Noise files available:[/cyan] {len(noise_files)}")
    if not noise_files:
        console.print("[yellow]Warning: no noise files found — noise augmentation disabled for TTA[/yellow]")

    # ── Feature extractor (shared — all checkpoints use the same base arch) ──
    feature_extractor = AutoFeatureExtractor.from_pretrained(str(ckpt_dirs[0]))

    # ── Test CSV ──────────────────────────────────────────────────────────────
    test_df = pd.read_csv(TEST_CSV_PATH)
    console.print(f"[cyan]Test samples:[/cyan] {len(test_df)}")
    test_df["song_id"] = test_df["filename"].apply(lambda x: Path(x).stem)

    # ── Pre-load all audio into RAM once ──────────────────────────────────────
    console.rule("[bold cyan]Pre-loading test audio")
    dataset = TestDataset(test_df["song_id"].tolist(), MASHUP_PATH)
    loader  = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, collate_fn=collate_audio, pin_memory=False,
    )
    all_audio, all_ids = [], []
    for audios, ids in tqdm(loader, desc="Loading audio"):
        all_audio.extend(audios)
        all_ids.extend(ids)
    console.print(f"Loaded {len(all_audio)} clips  (~{len(all_audio)*TARGET_LEN*4/1e6:.0f} MB in RAM)")

    # ── Ensemble: iterate models, accumulate logits ───────────────────────────
    console.rule("[bold cyan]Running ensemble TTA inference")
    grand_sum = None   # accumulates logit sums across all models × all runs

    for ckpt_dir in ckpt_dirs:
        label = ckpt_dir.name
        console.print(f"\n[bold]── Model: {label}[/bold]")
        model    = AutoModelForAudioClassification.from_pretrained(str(ckpt_dir)).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        console.print(f"  Params: {n_params:,}")

        model_sum = run_tta_for_model(
            model, feature_extractor, all_audio,
            noise_files, device,
            n_runs=TTA_RUNS,
            label=label,
        )
        grand_sum = model_sum if grand_sum is None else grand_sum + model_sum

        del model
        torch.cuda.empty_cache()

    # ── Average logits and take argmax ────────────────────────────────────────
    total_passes = n_models * TTA_RUNS
    avg_logits   = grand_sum / total_passes         # (n, C)
    final_preds  = avg_logits.argmax(dim=1).tolist()
    predictions  = [idx2label[p] for p in final_preds]

    # ── Confidence summary (save for pseudo-labelling) ────────────────────────
    probs     = torch.softmax(avg_logits, dim=1)
    max_probs = probs.max(dim=1).values
    console.print(
        f"\n[bold cyan]Confidence summary[/bold cyan]\n"
        f"  Mean:          {max_probs.mean():.4f}\n"
        f"  Min:           {max_probs.min():.4f}\n"
        f"  High-conf >0.97: {(max_probs > 0.97).sum().item()}/{len(final_preds)} samples\n"
        f"  High-conf >0.90: {(max_probs > 0.90).sum().item()}/{len(final_preds)} samples"
    )

    # Optionally save high-confidence pseudo-labels for future retraining
    pseudo_path = SUBMISSION_PATH.parent / "pseudo_labels_v2.csv"
    pseudo_df   = pd.DataFrame({
        "song_id":    all_ids,
        "genre":      predictions,
        "confidence": max_probs.tolist(),
    })
    pseudo_df.to_csv(pseudo_path, index=False)
    console.print(f"[dim]Pseudo-label file (all samples + confidence) → {pseudo_path}[/dim]")

    # ── Align predictions with test_df order ─────────────────────────────────
    id_to_genre  = dict(zip(all_ids, predictions))
    test_df["genre"] = test_df["song_id"].map(id_to_genre)

    # ── Write submission ──────────────────────────────────────────────────────
    out_df = pd.DataFrame({"id": test_df["id"], "genre": test_df["genre"]})
    out_df.to_csv(SUBMISSION_PATH, index=False)

    console.print(f"\n[bold green]Submission written → {SUBMISSION_PATH}[/bold green]")
    console.print(f"[green]{len(out_df)} predictions[/green]")

    dist_table = Table(title="Genre Distribution", show_header=True, header_style="bold cyan")
    dist_table.add_column("Genre", style="dim")
    dist_table.add_column("Count", justify="right")
    dist_table.add_column("Pct",   justify="right")
    for genre, count in out_df["genre"].value_counts().items():
        dist_table.add_row(genre, str(count), f"{100*count/len(out_df):.1f}%")
    console.print(dist_table)

    console.print(Panel(
        Text(
            f"✓ {SUBMISSION_PATH.name} ready  |  "
            f"{n_models} model(s) × {TTA_RUNS} TTA runs = {total_passes} total passes",
            style="bold yellow",
        ),
        border_style="green",
    ))


if __name__ == "__main__":
    main()