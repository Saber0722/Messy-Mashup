# fresh_training_from_scratch_corrected_v2.py
# A self-contained script to train a genre classifier from scratch on mixed audio,
# simulating the Messy Mashup competition test distribution.
# NOW includes generating mix.wav from the 4 individual stems.
# FIXED: Simplified augmentations to avoid errors.

import os
import sys
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import librosa
import soundfile as sf # Need this to save the mixed wav file
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 1. CONFIGURATION (HARDCODED)
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[0]  # assuming script is in = PROJECT_ROOT / "data" / "raw" / "messy_mashup"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "messy_mashup"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MEL_DIR = PROCESSED_DIR / "mel_spectrograms"
SPLITS_DIR = PROCESSED_DIR / "splits"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Audio params
SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMAX = SAMPLE_RATE // 2
TARGET_FRAMES = 1300  # ~12.9s at hop=512, sr=22050
DURATION = TARGET_FRAMES * HOP_LENGTH / SAMPLE_RATE

# Training params
BATCH_SIZE = 32
NUM_EPOCHS = 120
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dirs
MEL_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------
# 2. HELPER FUNCTIONS
# ---------------------------
def fix_length(audio, target_len):
    """Pad or truncate audio to target length."""
    if len(audio) < target_len:
        return np.pad(audio, (0, target_len - len(audio)), mode='constant')
    else:
        return audio[:target_len]

def normalise_mel(mel):
    """Min-max & freq."""
    mel = mel.astype(np.float32)
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
    return mel

def extract_mel(wav_path, sr=SAMPLE_RATE):
    """Extract mel spectrogram from .wav file."""
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    y = fix_length(y, int(DURATION * sr))
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmax=FMAX
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = normalise_mel(mel)
    return mel  # (n_mels, T)

# ---------------------------
# 2.5 GENERATE MIX.WAV FILES
# ---------------------------
def generate_mix_files(genres_stems_dir: Path):
    """Generate mix.wav files by summing the 4 individual stems."""
    logger.info("Starting to generate mix.wav files...")
    stems_to_sum = ["bass.wav", "drums.wav", "other.wav", "vocals.wav"]
    mix_count = 0
    for genre_dir in genres_stems_dir.iterdir():
        if not genre_dir.is_dir():
            continue
        for track_dir in genre_dir.iterdir():
            if not track_dir.is_dir():
                continue
            # Check if all 4 stems exist
            stem_paths = [track_dir / stem for stem in stems_to_sum]
            if all(p.exists() for p in stem_paths):
                mix_path = track_dir / "mix.wav"
                # Only regenerate if mix.wav doesn't exist
                if not mix_path.exists():
                    logger.debug(f"Generating mix for {track_dir.name}")
                    # Load all 4 stems
                    y_mixed = None
                    for stem_path in stem_paths:
                        y_stem, sr = librosa.load(stem_path, sr=SAMPLE_RATE, mono=True)
                        y_stem = fix_length(y_stem, int(DURATION * SAMPLE_RATE))
                        if y_mixed is None:
                            y_mixed = y_stem
                        else:
                            y_mixed += y_stem
                    # Normalize the mixed audio to prevent clipping
                    y_mixed = y_mixed / (np.max(np.abs(y_mixed)) + 1e-8)
                    # Save the mixed audio file
                    sf.write(mix_path, y_mixed, SAMPLE_RATE)
                    logger.debug(f"Saved mix: {mix_path}")
                    mix_count += 1
                else:
                    logger.debug(f"Mix already exists, skipping: {mix_path}")
            else:
                logger.warning(f"Missing stems in {track_dir}, skipping mix generation.")
    logger.info(f"Generated {mix_count} mix.wav files.")


# ---------------------------
# 3. AUGMENTATIONS (simplified, simulate mashup/noise)
# ---------------------------
class MixupAugmentation:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, mix_mel):
        if random.random() > self.p:
            return mix_mel
        # Simulate cross-song stem recombination: add scaled noise/gain variation
        noise = np.random.normal(0, 0.05, mix_mel.shape)
        gain = np.random.uniform(0.8, 1.2)
        return np.clip(mix_mel * gain + noise, 0, 1)

class NoiseInjection:
    def __init__(self, max_noise_level=0.1, p=0.4):
        self.max_noise_level = max_noise_level
        self.p = p

    def __call__(self, mel):
        if random.random() > self.p:
            return mel
        noise = np.random.normal(0, np.random.uniform(0, self.max_noise_level), mel.shape)
        return np.clip(mel + noise, 0, 1)

# Combined augmentation
def apply_augmentations(mel):
    aug = Compose([
        # TempoStretch(), # Removed due to error
        # PitchShift(),   # Removed due to potential error
        NoiseInjection(),
        MixupAugmentation()
    ])
    return aug(mel)

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

# ---------------------------
# 4. DATASET SCAN & SPLIT BUILDING
# ---------------------------
def scan_genres_stems(genres_stems_dir: Path):
    """Scan genres_stems/ and return list of (genre, track_id, mix_path)"""
    records = []
    for genre_dir in genres_stems_dir.iterdir():
        if not genre_dir.is_dir():
            continue
        for track_dir in genre_dir.iterdir():
            if not track_dir.is_dir():
                continue
            # NOW expect: bass.wav, drums.wav, other.wav, vocals.wav, mix.wav
            stems = ["bass.wav", "drums.wav", "other.wav", "vocals.wav", "mix.wav"]
            if all((track_dir / s).exists() for s in stems):
                mix_path = track_dir / "mix.wav" # Use the generated mix.wav
                records.append({
                    "genre": genre_dir.name,
                    "track_id": track_dir.name,
                    "mix_path": str(mix_path),
                    "file_base": f"{genre_dir.name}__{track_dir.name}"
                })
    logger.info(f"Found {len(records)} valid tracks in genres_stems/ (with mix.wav)")
    return records

# Generate mix files first
genres_stems_path = RAW_DIR / "genres_stems"
generate_mix_files(genres_stems_path)

# Scan data (should now find mix.wav files)
records = scan_genres_stems(genres_stems_path)

if not records:
    raise RuntimeError("No valid tracks found after attempting to generate mix.wav! Check data structure and generation.")

df = pd.DataFrame(records)
le = LabelEncoder()
df["label"] = le.fit_transform(df["genre"])
label2idx = {cls: idx for idx, cls in enumerate(le.classes_)}
idx2label = {idx: cls for cls, idx in label2idx.items()}
num_classes = len(label2idx)
logger.info(f"Genres: {list(le.classes_)}")

# Stratified split
train_df, temp_df = train_test_split(df, test_size=0.25, stratify=df["label"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
    d.to_csv(SPLITS_DIR / f"{name}.csv", index=False)
    logger.info(f" {name}.csv: {len(d)} samples")

# ---------------------------
# 5. DATASET CLASS (Mix-Only + Augmentation)
# ---------------------------
class MixOnlyDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mix_path = row["mix_path"]

        # Load and extract mel
        mel = extract_mel(mix_path)
        mel = torch.from_numpy(mel).float()  # (128, T)

        # Augment if training
        if self.augment:
            mel_np = mel.numpy()
            mel_np = apply_augmentations(mel_np)
            mel = torch.from_numpy(mel_np).float()

        # Ensure fixed shape
        if mel.shape[1] < TARGET_FRAMES:
            pad = torch.zeros(128, TARGET_FRAMES - mel.shape[1])
            mel = torch.cat([mel, pad], dim=1)
        else:
            mel = mel[:, :TARGET_FRAMES]

        label = torch.tensor(row["label"], dtype=torch.long)
        return {"mix": mel.unsqueeze(0), "label": label}  # (1, 128, T)

# ---------------------------
# 6. MODEL ARCHITECTURE
# ---------------------------
class GenreClassifier(nn.Module):
    def __init__(self, num_classes, n_mels=128, hidden_size=128):
        super().__init__()
        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            
            nn.Conv2d(32, 64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            
            nn.Conv2d(64, 128, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((None, 1)),  # (B, 128, T', 1)
        )
        
        # GRU
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Attention
        self.attention = nn.Sequential(
            nn.Linear(2 * hidden_size, 1),
            nn.Tanh()
        )
        
        # Classifier
        self.classifier = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x):
        # x: (B, 1, 128, T)
        B, C, H, W = x.shape
        x = self.cnn(x)  # (B, 128, T_out, 1) -> (B, 128, T_out)
        x = x.squeeze(-1).permute(0, 2, 1)  # (B, T_out, 128)

        x, _ = self.gru(x)  # (B, T_out, 2*hidden)
        
        # Attention
        attn_weights = self.attention(x)  # (B, T_out, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        x = (x * attn_weights).sum(dim=1)  # (B, 2*hidden)

        logits = self.classifier(x)  # (B, num_classes)
        return logits

# ---------------------------
# 7. TRAINING LOOP
# ---------------------------
def train():
    train_ds = MixOnlyDataset(train_df, augment=True)
    val_ds = MixOnlyDataset(val_df, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = GenreClassifier(num_classes=num_classes).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # LR Scheduler: Warmup + Cosine Annealing
    warmup_steps = 5
    total_steps = len(train_loader) * NUM_EPOCHS
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler() if DEVICE.type == "cuda" else None

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            x = batch["mix"].to(DEVICE)
            y = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            if scaler:
                with autocast():
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            scheduler.step()

        train_acc = correct / total
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["mix"].to(DEVICE)
                y = batch["label"].to(DEVICE)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))

        # Macro F1
        from sklearn.metrics import f1_score
        val_f1 = f1_score(val_labels, val_preds, average='macro')

        logger.info(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        # Early stopping & checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_f1': val_f1,
            }, CHECKPOINT_DIR / "best_model.pth")
            logger.info(f"✅ New best F1: {val_f1:.4f} at epoch {epoch}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"🛑 Early stopping triggered after {patience_counter} epochs without improvement.")
                break

    logger.info(f"Training finished. Best Val Macro F1: {best_val_f1:.4f}")

    # Save label encoder
    import pickle
    with open(CHECKPOINT_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    logger.info("Label encoder saved.")

if __name__ == "__main__":
    train()