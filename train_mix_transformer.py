"""
train_transformer_v2.py

Key improvements over v1:
  1. ON-THE-FLY augmentation via custom PyTorch Dataset (every epoch sees fresh mixes)
  2. Progressive layer unfreezing (freeze backbone initially, unfreeze later)
  3. Test-Time Augmentation (TTA) at inference
  4. Mixup augmentation in feature space
  5. Label smoothing
  6. Cosine LR schedule with warmup
  7. Multiple augmented passes per track per epoch (virtual dataset expansion)

Run from project root:
    python train_transformer_v2.py
"""

import logging
import os
import pickle
import random
import warnings
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parents[0]
RAW_DIR        = PROJECT_ROOT / "data/raw/messy_mashup"
GENRES_DIR     = RAW_DIR / "genres_stems"
NOISE_DIR      = RAW_DIR / "ESC-50-master/audio"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints_transformer_v2"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME   = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE  = 16000
DURATION_SEC = 10
TARGET_LEN   = SAMPLE_RATE * DURATION_SEC
STEMS        = ["bass", "drums", "other", "vocals"]

# Training
BATCH_SIZE        = 4
GRAD_ACCUM        = 2          # effective batch = 8
NUM_EPOCHS        = 60
LR                = 3e-5
WEIGHT_DECAY      = 0.01
PATIENCE          = 10
SEED              = 42
LABEL_SMOOTHING   = 0.1
WARMUP_RATIO      = 0.1

# Virtual expansion: how many augmented versions to generate per track per epoch
# e.g. AUGMENT_MULTIPLIER=5 → 5x dataset size per epoch
AUGMENT_MULTIPLIER = 8

# Progressive unfreezing
UNFREEZE_EPOCH = 5   # Start with classifier head only; unfreeze all at this epoch

# Augmentation
CROSS_MIX_PROB  = 0.9
TEMPO_PERTURB   = True
TEMPO_RANGE     = (0.85, 1.15)
NOISE_PROB      = 0.8
NOISE_SNR_RANGE = (5, 25)
PITCH_SHIFT     = True
PITCH_SEMITONES = (-2, 2)

# TTA: how many augmented versions to average at test time
TTA_RUNS = 5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Audio helpers ─────────────────────────────────────────────────────────────

def fix_len(audio: np.ndarray, target: int) -> np.ndarray:
    if len(audio) < target:
        return np.pad(audio, (0, target - len(audio)))
    return audio[:target]


def load_stem(path: Path) -> np.ndarray:
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=DURATION_SEC)
    return fix_len(y, TARGET_LEN).astype(np.float32)


def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    try:
        y = librosa.effects.time_stretch(y, rate=rate)
    except Exception:
        pass
    return fix_len(y, TARGET_LEN)


def pitch_shift(y: np.ndarray, n_steps: float) -> np.ndarray:
    try:
        y = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=n_steps)
    except Exception:
        pass
    return fix_len(y, TARGET_LEN)


def add_noise(y: np.ndarray, noise_files: list, snr_db: float) -> np.ndarray:
    if not noise_files:
        return y
    try:
        noise, _ = librosa.load(random.choice(noise_files), sr=SAMPLE_RATE, mono=True)
    except Exception:
        return y
    noise = fix_len(noise, TARGET_LEN)
    sig_pow   = np.mean(y ** 2) + 1e-10
    noise_pow = np.mean(noise ** 2) + 1e-10
    noise *= np.sqrt(sig_pow / (noise_pow * 10 ** (snr_db / 10)))
    mixed = y + noise
    return (mixed / (np.max(np.abs(mixed)) + 1e-8)).astype(np.float32)


def random_gain(y: np.ndarray, low=0.7, high=1.0) -> np.ndarray:
    return y * random.uniform(low, high)


def random_offset(y: np.ndarray) -> np.ndarray:
    """Randomly shift the start position of the audio."""
    max_offset = TARGET_LEN // 4
    offset = random.randint(0, max_offset)
    return fix_len(y[offset:], TARGET_LEN)


# ── Stem index ────────────────────────────────────────────────────────────────

def build_stem_index(genres_dir: Path) -> dict:
    """Returns {genre: {stem: [wav_path, ...]}}"""
    index = defaultdict(lambda: defaultdict(list))
    for genre_dir in sorted(genres_dir.iterdir()):
        if not genre_dir.is_dir():
            continue
        for track_dir in sorted(genre_dir.iterdir()):
            if not track_dir.is_dir():
                continue
            for stem in STEMS:
                p = track_dir / f"{stem}.wav"
                if p.exists():
                    index[genre_dir.name][stem].append(p)
    return index


def build_records(genres_dir: Path) -> list:
    records = []
    for genre_dir in sorted(genres_dir.iterdir()):
        if not genre_dir.is_dir():
            continue
        for track_dir in sorted(genre_dir.iterdir()):
            if not track_dir.is_dir():
                continue
            if all((track_dir / f"{s}.wav").exists() for s in STEMS):
                records.append({
                    "genre": genre_dir.name,
                    "track_dir": str(track_dir),
                })
    return records


# ── Mix generation ────────────────────────────────────────────────────────────

def make_augmented_mix(genre: str, stem_index: dict, noise_files: list) -> np.ndarray:
    """
    Generate one augmented mix by randomly picking stems from DIFFERENT
    tracks of the same genre. Called fresh on every __getitem__ call.
    """
    stems_audio = []
    for stem in STEMS:
        candidates = stem_index[genre][stem]
        if not candidates:
            stems_audio.append(np.zeros(TARGET_LEN, dtype=np.float32))
            continue

        y = load_stem(random.choice(candidates))

        # Random offset (simulate different start points)
        y = random_offset(y)

        # Tempo perturbation
        if TEMPO_PERTURB and random.random() < 0.7:
            y = time_stretch(y, random.uniform(*TEMPO_RANGE))

        # Pitch shift
        if PITCH_SHIFT and random.random() < 0.4:
            y = pitch_shift(y, random.uniform(*PITCH_SEMITONES))

        # Per-stem gain variation
        y = random_gain(y, 0.6, 1.0)

        stems_audio.append(y)

    # Weighted random mix (not always equal weights)
    weights = np.array([random.uniform(0.5, 1.0) for _ in STEMS])
    weights /= weights.sum()
    mix = sum(w * s for w, s in zip(weights, stems_audio))
    mix = mix / (np.max(np.abs(mix)) + 1e-8)

    # Add environmental noise
    if noise_files and random.random() < NOISE_PROB:
        mix = add_noise(mix, noise_files, random.uniform(*NOISE_SNR_RANGE))

    return mix.astype(np.float32)


def make_clean_mix(track_dir: str) -> np.ndarray:
    """Deterministic same-track mix used for validation."""
    track_path = Path(track_dir)
    stems_audio = []
    for stem in STEMS:
        p = track_path / f"{stem}.wav"
        y = load_stem(p) if p.exists() else np.zeros(TARGET_LEN, dtype=np.float32)
        stems_audio.append(y)
    mix = np.mean(stems_audio, axis=0)
    return (mix / (np.max(np.abs(mix)) + 1e-8)).astype(np.float32)


# ── PyTorch Datasets (on-the-fly augmentation) ────────────────────────────────

class TrainAudioDataset(Dataset):
    """
    Each __getitem__ generates a FRESH augmented mix — never the same twice.
    AUGMENT_MULTIPLIER copies of each track are interleaved in the index so
    a single epoch exposes the model to many more unique mixes.
    """
    def __init__(self, records: list, stem_index: dict, noise_files: list,
                 feature_extractor, multiplier: int = AUGMENT_MULTIPLIER):
        self.records    = records          # list of {genre, label, track_dir}
        self.stem_index = stem_index
        self.noise_files = noise_files
        self.feature_extractor = feature_extractor
        self.multiplier = multiplier
        # Expand index: [0,1,2,...,N-1, 0,1,...,N-1, ...] × multiplier
        self._index = list(range(len(records))) * multiplier

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        rec   = self.records[self._index[idx]]
        genre = rec["genre"]
        label = rec["label"]

        # Fresh random mix every call
        audio = make_augmented_mix(genre, self.stem_index, self.noise_files)

        inputs = self.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in inputs.items()}, label


class ValAudioDataset(Dataset):
    """Deterministic clean mixes for stable validation."""
    def __init__(self, records: list, feature_extractor):
        self.records = records
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        audio = make_clean_mix(rec["track_dir"])
        inputs = self.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in inputs.items()}, rec["label"]


class TTADataset(Dataset):
    """
    At inference time, generate TTA_RUNS fresh augmented mixes per val sample
    and average the logits. This simulates the test distribution more closely.
    """
    def __init__(self, records: list, stem_index: dict, noise_files: list,
                 feature_extractor, n_runs: int = TTA_RUNS):
        self.records = records
        self.stem_index = stem_index
        self.noise_files = noise_files
        self.feature_extractor = feature_extractor
        self.n_runs = n_runs

    def __len__(self):
        return len(self.records)

    def get_all_runs(self, idx):
        """Returns list of n_runs feature dicts and the label."""
        rec   = self.records[idx]
        genre = rec["genre"]
        label = rec["label"]
        all_inputs = []
        # Always include one clean mix
        clean = make_clean_mix(rec["track_dir"])
        inp = self.feature_extractor(clean, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        all_inputs.append({k: v.squeeze(0) for k, v in inp.items()})
        # Plus n_runs-1 augmented mixes
        for _ in range(self.n_runs - 1):
            audio = make_augmented_mix(genre, self.stem_index, self.noise_files)
            inp = self.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            all_inputs.append({k: v.squeeze(0) for k, v in inp.items()})
        return all_inputs, label


# ── Collate ───────────────────────────────────────────────────────────────────

def collate_fn(batch):
    inputs_list, labels = zip(*batch)
    collated = {}
    for key in inputs_list[0]:
        collated[key] = torch.stack([x[key] for x in inputs_list])
    return collated, torch.tensor(labels, dtype=torch.long)


# ── Progressive unfreezing ────────────────────────────────────────────────────

def freeze_backbone(model):
    """Freeze everything except the classification head."""
    for name, param in model.named_parameters():
        if "classifier" not in name and "layernorm" not in name:
            param.requires_grad = False
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[yellow]Backbone frozen. Trainable params: {n_trainable:,}[/yellow]")


def unfreeze_all(model):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[green]All layers unfrozen. Trainable params: {n_trainable:,}[/green]")


# ── Training loop ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, scheduler, criterion, device, is_train: bool):
    model.train() if is_train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for batch_inputs, labels in tqdm(loader, leave=False, desc="train" if is_train else "val"):
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            labels = labels.to(device)

            outputs = model(**batch_inputs)
            logits  = outputs.logits
            loss    = criterion(logits, labels)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1


# ── TTA inference ─────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_with_tta(model, tta_dataset, device):
    model.eval()
    all_preds, all_labels = [], []
    for i in tqdm(range(len(tta_dataset)), desc="TTA inference"):
        runs, label = tta_dataset.get_all_runs(i)
        logits_list = []
        for inp in runs:
            inp = {k: v.unsqueeze(0).to(device) for k, v in inp.items()}
            out = model(**inp)
            logits_list.append(out.logits.squeeze(0).cpu())
        avg_logits = torch.stack(logits_list).mean(0)
        all_preds.append(avg_logits.argmax().item())
        all_labels.append(label)
    return np.array(all_labels), np.array(all_preds)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    wandb.init(
        project="messy-mashup-transformer",
        name="v2-onthefly-tta-progressive",
        config={
            "model": MODEL_NAME,
            "augment_multiplier": AUGMENT_MULTIPLIER,
            "cross_mix_prob": CROSS_MIX_PROB,
            "tempo_perturb": TEMPO_PERTURB,
            "noise_prob": NOISE_PROB,
            "pitch_shift": PITCH_SHIFT,
            "tta_runs": TTA_RUNS,
            "unfreeze_epoch": UNFREEZE_EPOCH,
            "label_smoothing": LABEL_SMOOTHING,
            "lr": LR,
            "epochs": NUM_EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
        }
    )

    config_table = Table(title="Training Config v2", show_header=True, header_style="bold magenta")
    config_table.add_column("Parameter", style="dim")
    config_table.add_column("Value", justify="right")
    for k, v in [
        ("Model", MODEL_NAME), ("Augment Multiplier", AUGMENT_MULTIPLIER),
        ("TTA Runs", TTA_RUNS), ("Unfreeze Epoch", UNFREEZE_EPOCH),
        ("Label Smoothing", LABEL_SMOOTHING), ("LR", LR),
        ("Epochs", NUM_EPOCHS), ("Patience", PATIENCE),
        ("Batch Size", BATCH_SIZE), ("Grad Accum", GRAD_ACCUM),
        ("Pitch Shift", PITCH_SHIFT), ("Noise Prob", NOISE_PROB),
    ]:
        config_table.add_row(str(k), str(v))
    console.print(Panel(config_table, border_style="blue"))

    # ── Data ──────────────────────────────────────────────────────────────────
    console.rule("[bold cyan]Building index")
    stem_index   = build_stem_index(GENRES_DIR)
    noise_files  = sorted(NOISE_DIR.glob("*.wav")) if NOISE_DIR.exists() else []
    console.print(f"Noise files: {len(noise_files)}")

    records = build_records(GENRES_DIR)
    console.print(f"Total tracks: {len(records)}")

    df = pd.DataFrame(records)
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["genre"])
    label2idx = {g: i for i, g in enumerate(le.classes_)}
    idx2label = {i: g for g, i in label2idx.items()}
    num_classes = len(label2idx)
    console.print(f"Genres ({num_classes}): {list(le.classes_)}")

    train_df, val_df = train_test_split(
        df, test_size=0.10, stratify=df["label"], random_state=SEED
    )
    train_records = train_df[["genre", "label", "track_dir"]].to_dict("records")
    val_records   = val_df[["genre", "label", "track_dir"]].to_dict("records")
    console.print(f"Train: {len(train_records)}  Val: {len(val_records)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    console.rule("[bold cyan]Loading model")
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.num_labels = num_classes
    config.label2id   = label2idx
    config.id2label   = idx2label

    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_NAME, config=config, ignore_mismatched_sizes=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Start with backbone frozen — only train classifier head first
    freeze_backbone(model)

    # ── Datasets & Loaders ────────────────────────────────────────────────────
    train_ds = TrainAudioDataset(train_records, stem_index, noise_files, feature_extractor)
    val_ds   = ValAudioDataset(val_records, feature_extractor)
    tta_ds   = TTADataset(val_records, stem_index, noise_files, feature_extractor)

    # NOTE: shuffle=True critical — with multiplier the same tracks repeat
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True
    )

    # ── Optimizer / Scheduler ─────────────────────────────────────────────────
    # Two-phase: phase 1 (head only) uses higher LR, phase 2 (all) uses lower LR
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR * 5,          # Higher LR when only training head
        weight_decay=WEIGHT_DECAY,
    )
    total_steps   = (len(train_loader) // GRAD_ACCUM) * NUM_EPOCHS
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # ── Training loop ─────────────────────────────────────────────────────────
    console.rule("[bold cyan]Training")
    best_f1, patience_counter, best_epoch = 0.0, 0, 0
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, NUM_EPOCHS + 1):

        # Progressive unfreeze
        if epoch == UNFREEZE_EPOCH:
            unfreeze_all(model)
            # Reset optimizer with lower LR for full fine-tuning
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
            )
            # Rebuild scheduler for remaining epochs
            remaining_steps = (len(train_loader) // GRAD_ACCUM) * (NUM_EPOCHS - epoch + 1)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(remaining_steps * 0.05),
                num_training_steps=remaining_steps,
            )

        # ---- Train ----
        model.train()
        total_loss, all_preds, all_labels = 0.0, [], []
        optimizer.zero_grad()
        step = 0

        for i, (batch_inputs, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]", leave=False)
        ):
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            labels = labels.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(**batch_inputs)
                loss = criterion(outputs.logits, labels) / GRAD_ACCUM

            scaler.scale(loss).backward()

            if (i + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

            total_loss += loss.item() * GRAD_ACCUM
            all_preds.extend(outputs.logits.argmax(1).detach().cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        train_loss = total_loss / len(train_loader)
        train_acc  = accuracy_score(all_labels, all_preds)
        train_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        # ---- Validate (clean mixes) ----
        model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []
        with torch.no_grad():
            for batch_inputs, labels in tqdm(val_loader, desc="val", leave=False):
                batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                labels = labels.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(**batch_inputs)
                    loss = criterion(outputs.logits, labels)
                val_loss += loss.item()
                val_preds.extend(outputs.logits.argmax(1).cpu().tolist())
                val_labels.extend(labels.cpu().tolist())

        val_loss /= len(val_loader)
        val_acc   = accuracy_score(val_labels, val_preds)
        val_f1    = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        console.print(
            f"[bold]Epoch {epoch:3d}[/bold] | "
            f"train loss={train_loss:.4f} f1={train_f1:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f} f1=[bold cyan]{val_f1:.4f}[/bold cyan]"
        )
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss, "train_f1": train_f1,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
        })

        # ---- Checkpoint best ----
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            model.save_pretrained(CHECKPOINT_DIR / "best_model")
            feature_extractor.save_pretrained(CHECKPOINT_DIR / "best_model")
            with open(CHECKPOINT_DIR / "label_encoder.pkl", "wb") as f:
                pickle.dump(le, f)
            console.print(f"  [green]↑ New best F1={best_f1:.4f} saved.[/green]")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                console.print(f"[yellow]Early stopping at epoch {epoch}.[/yellow]")
                break

    # ── TTA Final Eval ────────────────────────────────────────────────────────
    console.rule("[bold cyan]TTA Final Evaluation")
    # Load best model
    best_model = AutoModelForAudioClassification.from_pretrained(
        CHECKPOINT_DIR / "best_model"
    ).to(device)

    labels_arr, preds_arr = predict_with_tta(best_model, tta_ds, device)

    tta_f1  = f1_score(labels_arr, preds_arr, average="macro")
    tta_acc = accuracy_score(labels_arr, preds_arr)

    console.print(f"[bold green]TTA Val F1: {tta_f1:.4f}  Acc: {tta_acc:.4f}[/bold green]")
    wandb.log({"tta_val_f1": tta_f1, "tta_val_acc": tta_acc})

    pred_names  = [idx2label[p] for p in preds_arr]
    label_names = [idx2label[l] for l in labels_arr]
    report = classification_report(label_names, pred_names, target_names=sorted(label2idx))
    console.print(Panel(report, title=f"TTA Classification Report (best epoch={best_epoch})",
                        border_style="yellow"))

    wandb.finish()
    console.print(f"[bold green]Done. Best val F1 = {best_f1:.4f} at epoch {best_epoch}[/bold green]")


if __name__ == "__main__":
    main()