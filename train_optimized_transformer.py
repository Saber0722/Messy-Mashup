"""
train_transformer_v3.py

Improvements over v2:
  1. SEED passed via CLI arg → run 3× with different seeds for free ensemble
     python train_transformer_v3.py --seed 42
     python train_transformer_v3.py --seed 123
     python train_transformer_v3.py --seed 7
  2. Second model support: HTS-AT (hierarchical audio transformer) alongside AST
     Set MODEL_NAME / run script twice with different MODEL_NAME to build a
     cross-architecture ensemble
  3. Stem dropout augmentation — randomly silence 0-2 stems per mix, forces
     the model to classify from partial information (matches real test variability)
  4. Augmented validation — val uses cross-track mixes (same distribution as test)
     instead of clean same-track mixes, closing the val/Kaggle gap
  5. Checkpoint saved as best_model_s{SEED}/ → drop-in compatible with
     infer_competition_transformer_v2.py multi-seed ensemble

Run from project root:
    python train_transformer_v3.py --seed 42
    python train_transformer_v3.py --seed 123 --model-name MIT/ast-finetuned-audioset-10-10-0.4593
    python train_transformer_v3.py --seed 7   --model-name nvidia/mit-b0  # any HF audio clf model
"""

import argparse
import logging
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
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    get_cosine_schedule_with_warmup,
)

warnings.filterwarnings("ignore")
console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",       type=int,  default=42,
                   help="Random seed. Train 42/123/7 for a 3-model ensemble.")
    p.add_argument("--model-name", type=str,
                   default="MIT/ast-finetuned-audioset-10-10-0.4593",
                   help="HuggingFace model identifier.")
    p.add_argument("--epochs",     type=int,  default=60)
    p.add_argument("--lr",         type=float, default=3e-5)
    p.add_argument("--batch-size", type=int,  default=4)
    p.add_argument("--no-wandb",   action="store_true",
                   help="Disable W&B logging.")
    return p.parse_args()

# ── Static config ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[0]
RAW_DIR      = PROJECT_ROOT / "data/raw/messy_mashup"
GENRES_DIR   = RAW_DIR / "genres_stems"
NOISE_DIR    = RAW_DIR / "ESC-50-master/audio"

SAMPLE_RATE  = 16000
DURATION_SEC = 10
TARGET_LEN   = SAMPLE_RATE * DURATION_SEC
STEMS        = ["bass", "drums", "other", "vocals"]

GRAD_ACCUM        = 2
WEIGHT_DECAY      = 0.01
PATIENCE          = 10
LABEL_SMOOTHING   = 0.1
WARMUP_RATIO      = 0.1
AUGMENT_MULTIPLIER = 8
UNFREEZE_EPOCH    = 5
TTA_RUNS          = 5

# Augmentation
TEMPO_RANGE     = (0.85, 1.15)
NOISE_SNR_RANGE = (5, 25)
NOISE_PROB      = 0.8
PITCH_SEMITONES = (-2, 2)

# Stem dropout: probability of silencing 1 or 2 stems per mix
STEM_DROP_PROB  = 0.3   # prob of dropping any stems at all
STEM_DROP_N     = [1, 2]  # how many stems to drop (chosen uniformly)


# ── Audio helpers ─────────────────────────────────────────────────────────────

def fix_len(audio: np.ndarray, target: int = TARGET_LEN) -> np.ndarray:
    if len(audio) < target:
        return np.pad(audio, (0, target - len(audio)))
    return audio[:target]


def load_stem(path: Path) -> np.ndarray:
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=DURATION_SEC)
    return fix_len(y).astype(np.float32)


def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    try:
        y = librosa.effects.time_stretch(y, rate=rate)
    except Exception:
        pass
    return fix_len(y)


def pitch_shift_fn(y: np.ndarray, n_steps: float) -> np.ndarray:
    try:
        y = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=n_steps)
    except Exception:
        pass
    return fix_len(y)


def add_noise(y: np.ndarray, noise_files: list, snr_db: float) -> np.ndarray:
    if not noise_files:
        return y
    try:
        noise, _ = librosa.load(random.choice(noise_files), sr=SAMPLE_RATE, mono=True)
    except Exception:
        return y
    noise     = fix_len(noise)
    sig_pow   = np.mean(y ** 2) + 1e-10
    noise_pow = np.mean(noise ** 2) + 1e-10
    noise    *= np.sqrt(sig_pow / (noise_pow * 10 ** (snr_db / 10)))
    mixed     = y + noise
    return (mixed / (np.max(np.abs(mixed)) + 1e-8)).astype(np.float32)


def random_offset(y: np.ndarray) -> np.ndarray:
    offset = random.randint(0, TARGET_LEN // 4)
    return fix_len(y[offset:])


# ── Stem index ────────────────────────────────────────────────────────────────

def build_stem_index(genres_dir: Path) -> dict:
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
                    "genre":     genre_dir.name,
                    "track_dir": str(track_dir),
                })
    return records


# ── Mix generation ────────────────────────────────────────────────────────────

def make_augmented_mix(genre: str, stem_index: dict, noise_files: list) -> np.ndarray:
    """
    Cross-track augmented mix — every call produces a unique sample.
    Includes stem dropout to force robustness to missing instruments.
    """
    stems_audio = []
    for stem in STEMS:
        candidates = stem_index[genre][stem]
        if not candidates:
            stems_audio.append(np.zeros(TARGET_LEN, dtype=np.float32))
            continue

        y = load_stem(random.choice(candidates))
        y = random_offset(y)

        if random.random() < 0.7:
            y = time_stretch(y, random.uniform(*TEMPO_RANGE))

        if random.random() < 0.4:
            y = pitch_shift_fn(y, random.uniform(*PITCH_SEMITONES))

        y = y * random.uniform(0.6, 1.0)   # per-stem gain
        stems_audio.append(y)

    # ── Stem dropout: silence 1–2 stems with probability STEM_DROP_PROB ──────
    if random.random() < STEM_DROP_PROB:
        n_drop   = random.choice(STEM_DROP_N)
        drop_idx = random.sample(range(len(STEMS)), min(n_drop, len(STEMS) - 1))
        for i in drop_idx:
            stems_audio[i] = np.zeros(TARGET_LEN, dtype=np.float32)

    # Weighted mix
    weights = np.array([random.uniform(0.5, 1.0) for _ in STEMS])
    weights /= weights.sum()
    mix = sum(w * s for w, s in zip(weights, stems_audio))
    mix = mix / (np.max(np.abs(mix)) + 1e-8)

    if noise_files and random.random() < NOISE_PROB:
        mix = add_noise(mix, noise_files, random.uniform(*NOISE_SNR_RANGE))

    return mix.astype(np.float32)


def make_clean_mix(track_dir: str) -> np.ndarray:
    """Deterministic same-track mix — used only for TTA baseline pass."""
    track_path  = Path(track_dir)
    stems_audio = []
    for stem in STEMS:
        p = track_path / f"{stem}.wav"
        y = load_stem(p) if p.exists() else np.zeros(TARGET_LEN, dtype=np.float32)
        stems_audio.append(y)
    mix = np.mean(stems_audio, axis=0)
    return (mix / (np.max(np.abs(mix)) + 1e-8)).astype(np.float32)


# ── Datasets ──────────────────────────────────────────────────────────────────

class TrainAudioDataset(Dataset):
    """Fresh augmented cross-track mix on every __getitem__."""

    def __init__(self, records, stem_index, noise_files, feature_extractor,
                 multiplier=AUGMENT_MULTIPLIER):
        self.records           = records
        self.stem_index        = stem_index
        self.noise_files       = noise_files
        self.feature_extractor = feature_extractor
        self._index            = list(range(len(records))) * multiplier

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        rec   = self.records[self._index[idx]]
        audio = make_augmented_mix(rec["genre"], self.stem_index, self.noise_files)
        inputs = self.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in inputs.items()}, rec["label"]


class ValAudioDataset(Dataset):
    """
    Validation uses augmented cross-track mixes — same distribution as the
    competition test set. This closes the val/Kaggle F1 gap.
    """

    def __init__(self, records, stem_index, noise_files, feature_extractor):
        self.records           = records
        self.stem_index        = stem_index
        self.noise_files       = noise_files
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        audio = make_augmented_mix(rec["genre"], self.stem_index, self.noise_files)
        inputs = self.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in inputs.items()}, rec["label"]


class TTADataset(Dataset):
    """Used only for end-of-training TTA evaluation."""

    def __init__(self, records, stem_index, noise_files, feature_extractor,
                 n_runs=TTA_RUNS):
        self.records           = records
        self.stem_index        = stem_index
        self.noise_files       = noise_files
        self.feature_extractor = feature_extractor
        self.n_runs            = n_runs

    def __len__(self):
        return len(self.records)

    def get_all_runs(self, idx):
        rec   = self.records[idx]
        label = rec["label"]
        runs  = []

        # Run 0: clean same-track mix (stable baseline)
        clean = make_clean_mix(rec["track_dir"])
        inp   = self.feature_extractor(clean, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        runs.append({k: v.squeeze(0) for k, v in inp.items()})

        # Runs 1…n_runs-1: fresh augmented cross-track mixes
        for _ in range(self.n_runs - 1):
            audio = make_augmented_mix(rec["genre"], self.stem_index, self.noise_files)
            inp   = self.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            runs.append({k: v.squeeze(0) for k, v in inp.items()})

        return runs, label


# ── Collate ───────────────────────────────────────────────────────────────────

def collate_fn(batch):
    inputs_list, labels = zip(*batch)
    collated = {k: torch.stack([x[k] for x in inputs_list]) for k in inputs_list[0]}
    return collated, torch.tensor(labels, dtype=torch.long)


# ── Freezing helpers ──────────────────────────────────────────────────────────

def freeze_backbone(model):
    for name, param in model.named_parameters():
        if "classifier" not in name and "layernorm" not in name:
            param.requires_grad = False
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[yellow]Backbone frozen. Trainable: {n:,}[/yellow]")


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[green]All layers unfrozen. Trainable: {n:,}[/green]")


# ── TTA eval ──────────────────────────────────────────────────────────────────

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
    args = parse_args()

    SEED       = args.seed
    MODEL_NAME = args.model_name
    LR         = args.lr
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs

    # Checkpoint dir includes seed so multiple runs don't overwrite each other
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints_transformer_v3"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_DIR = CHECKPOINT_DIR / f"best_model_s{SEED}"

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ── W&B ───────────────────────────────────────────────────────────────────
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project="messy-mashup-transformer",
            name=f"v3-seed{SEED}-{MODEL_NAME.split('/')[-1]}",
            config=dict(
                model=MODEL_NAME, seed=SEED,
                augment_multiplier=AUGMENT_MULTIPLIER,
                stem_drop_prob=STEM_DROP_PROB,
                tempo_range=TEMPO_RANGE,
                noise_prob=NOISE_PROB,
                pitch_semitones=PITCH_SEMITONES,
                tta_runs=TTA_RUNS,
                unfreeze_epoch=UNFREEZE_EPOCH,
                label_smoothing=LABEL_SMOOTHING,
                lr=LR, epochs=NUM_EPOCHS,
                patience=PATIENCE, batch_size=BATCH_SIZE,
            ),
        )

    # ── Config table ──────────────────────────────────────────────────────────
    cfg = Table(title=f"Training Config v3 — seed={SEED}", show_header=True,
                header_style="bold magenta")
    cfg.add_column("Parameter", style="dim")
    cfg.add_column("Value", justify="right")
    for k, v in [
        ("Model",             MODEL_NAME),
        ("Seed",              SEED),
        ("Augment multiplier", AUGMENT_MULTIPLIER),
        ("Stem drop prob",    STEM_DROP_PROB),
        ("TTA runs",          TTA_RUNS),
        ("Unfreeze epoch",    UNFREEZE_EPOCH),
        ("Label smoothing",   LABEL_SMOOTHING),
        ("LR",                LR),
        ("Epochs",            NUM_EPOCHS),
        ("Patience",          PATIENCE),
        ("Batch size",        BATCH_SIZE),
        ("Grad accum",        GRAD_ACCUM),
        ("Effective batch",   BATCH_SIZE * GRAD_ACCUM),
        ("Checkpoint dir",    str(BEST_MODEL_DIR)),
    ]:
        cfg.add_row(str(k), str(v))
    console.print(Panel(cfg, border_style="blue"))

    # ── Data ──────────────────────────────────────────────────────────────────
    console.rule("[bold cyan]Building stem index")
    stem_index  = build_stem_index(GENRES_DIR)
    noise_files = sorted(NOISE_DIR.glob("*.wav")) if NOISE_DIR.exists() else []
    console.print(f"Noise files: {len(noise_files)}")

    records = build_records(GENRES_DIR)
    console.print(f"Total tracks: {len(records)}")

    df = pd.DataFrame(records)
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["genre"])
    label2idx   = {g: i for i, g in enumerate(le.classes_)}
    idx2label   = {i: g for g, i in label2idx.items()}
    num_classes = len(label2idx)
    console.print(f"Genres ({num_classes}): {list(le.classes_)}")

    train_df, val_df = train_test_split(
        df, test_size=0.10, stratify=df["label"], random_state=SEED
    )
    train_records = train_df[["genre", "label", "track_dir"]].to_dict("records")
    val_records   = val_df[["genre", "label", "track_dir"]].to_dict("records")
    console.print(f"Train tracks: {len(train_records)}  Val tracks: {len(val_records)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    console.rule("[bold cyan]Loading model")
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    config            = AutoConfig.from_pretrained(MODEL_NAME)
    config.num_labels = num_classes
    config.label2id   = label2idx
    config.id2label   = idx2label

    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_NAME, config=config, ignore_mismatched_sizes=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    console.print(f"Device: {device}  |  Params: {sum(p.numel() for p in model.parameters()):,}")

    freeze_backbone(model)

    # ── Datasets & loaders ────────────────────────────────────────────────────
    train_ds = TrainAudioDataset(
        train_records, stem_index, noise_files, feature_extractor
    )
    val_ds = ValAudioDataset(
        val_records, stem_index, noise_files, feature_extractor
    )
    tta_ds = TTADataset(
        val_records, stem_index, noise_files, feature_extractor
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    # ── Optimizer / Scheduler (phase 1 — head only) ───────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR * 5,
        weight_decay=WEIGHT_DECAY,
    )
    total_steps  = (len(train_loader) // GRAD_ACCUM) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler    = torch.cuda.amp.GradScaler()

    # ── Training loop ─────────────────────────────────────────────────────────
    console.rule("[bold cyan]Training")
    best_f1, patience_counter, best_epoch = 0.0, 0, 0

    for epoch in range(1, NUM_EPOCHS + 1):

        # Progressive unfreeze at UNFREEZE_EPOCH
        if epoch == UNFREEZE_EPOCH:
            unfreeze_all(model)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
            )
            remaining = (len(train_loader) // GRAD_ACCUM) * (NUM_EPOCHS - epoch + 1)
            scheduler  = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(remaining * 0.05),
                num_training_steps=remaining,
            )

        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        total_loss, all_preds, all_labels = 0.0, [], []
        optimizer.zero_grad()

        for i, (batch_inputs, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} train", leave=False)
        ):
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            labels       = labels.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(**batch_inputs)
                loss    = criterion(outputs.logits, labels) / GRAD_ACCUM

            scaler.scale(loss).backward()

            if (i + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM
            all_preds.extend(outputs.logits.argmax(1).detach().cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        train_loss = total_loss / len(train_loader)
        train_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        train_acc  = accuracy_score(all_labels, all_preds)

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []
        with torch.no_grad():
            for batch_inputs, labels in tqdm(val_loader, desc="val", leave=False):
                batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                labels       = labels.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(**batch_inputs)
                    loss    = criterion(outputs.logits, labels)
                val_loss += loss.item()
                val_preds.extend(outputs.logits.argmax(1).cpu().tolist())
                val_labels.extend(labels.cpu().tolist())

        val_loss /= len(val_loader)
        val_f1    = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        val_acc   = accuracy_score(val_labels, val_preds)

        console.print(
            f"[bold]Epoch {epoch:3d}[/bold] | "
            f"train loss={train_loss:.4f} f1={train_f1:.4f} | "
            f"val   loss={val_loss:.4f} acc={val_acc:.4f} "
            f"f1=[bold cyan]{val_f1:.4f}[/bold cyan]"
        )

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss, "train_f1": train_f1, "train_acc": train_acc,
                "val_loss":   val_loss,   "val_f1":   val_f1,   "val_acc":   val_acc,
            })

        # ── Checkpoint ───────────────────────────────────────────────────────
        if val_f1 > best_f1:
            best_f1      = val_f1
            best_epoch   = epoch
            patience_counter = 0
            model.save_pretrained(BEST_MODEL_DIR)
            feature_extractor.save_pretrained(BEST_MODEL_DIR)
            with open(CHECKPOINT_DIR / "label_encoder.pkl", "wb") as f:
                pickle.dump(le, f)
            console.print(
                f"  [green]↑ New best F1={best_f1:.4f} → saved to {BEST_MODEL_DIR.name}[/green]"
            )
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                console.print(f"[yellow]Early stopping at epoch {epoch}.[/yellow]")
                break

    # ── TTA final evaluation ──────────────────────────────────────────────────
    console.rule("[bold cyan]TTA Final Evaluation")
    best_model = AutoModelForAudioClassification.from_pretrained(
        str(BEST_MODEL_DIR)
    ).to(device)

    labels_arr, preds_arr = predict_with_tta(best_model, tta_ds, device)
    tta_f1  = f1_score(labels_arr, preds_arr, average="macro")
    tta_acc = accuracy_score(labels_arr, preds_arr)
    console.print(f"[bold green]TTA Val F1: {tta_f1:.4f}  Acc: {tta_acc:.4f}[/bold green]")

    if use_wandb:
        wandb.log({"tta_val_f1": tta_f1, "tta_val_acc": tta_acc})

    pred_names  = [idx2label[p] for p in preds_arr]
    label_names = [idx2label[l] for l in labels_arr]
    report = classification_report(
        label_names, pred_names, target_names=sorted(label2idx)
    )
    console.print(Panel(
        report,
        title=f"TTA Classification Report  (best epoch={best_epoch}, seed={SEED})",
        border_style="yellow",
    ))

    if use_wandb:
        wandb.finish()

    console.print(
        f"[bold green]Done. Best val F1={best_f1:.4f} at epoch {best_epoch} | "
        f"checkpoint → {BEST_MODEL_DIR.name}[/bold green]"
    )
    console.print(
        "\n[dim]To build a 3-model ensemble, run:[/dim]\n"
        "  python train_transformer_v3.py --seed 42\n"
        "  python train_transformer_v3.py --seed 123\n"
        "  python train_transformer_v3.py --seed 7\n"
        "[dim]Then run infer_competition_transformer_v2.py — "
        "it auto-discovers all best_model_s*/ checkpoints.[/dim]"
    )


if __name__ == "__main__":
    main()