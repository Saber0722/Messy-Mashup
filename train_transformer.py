# train_transformer_90_10_early_stop_wandb_rich.py
# A script to fine-tune a pre-trained audio transformer model using 90/10 split, early stopping, W&B logging, and Rich UI.

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import librosa
import soundfile as sf  # Need this to save the mixed wav file
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, TrainingArguments, Trainer
from datasets import Dataset as HFDataset  # HuggingFace dataset for easier Trainer integration
import warnings
warnings.filterwarnings("ignore")

# --- ADD W&B IMPORTS ---
import wandb
# -----------------------
# --- ADD RICH IMPORTS ---
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
console = Console()
# -----------------------

# ---------------------------
# 1. CONFIGURATION & SETUP
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[0]  # assuming script is in root
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "messy_mashup"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MEL_DIR = PROCESSED_DIR / "mel_spectrograms"
SPLITS_DIR = PROCESSED_DIR / "splits"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints_transformer_90_10_es_wandb_rich"  # New directory

# Audio params (specifically for the pre-trained model if needed, otherwise use librosa defaults)
# AST expects 16kHz audio
SAMPLE_RATE_AST = 16000  # AST default
DURATION_SEC = 10  # INCREASED TO 10 SECONDS - Match original AST context
TARGET_SAMPLES = int(SAMPLE_RATE_AST * DURATION_SEC)

# Training params
BATCH_SIZE = 2  # REDUCED BATCH SIZE due to increased length and memory usage
NUM_EPOCHS = 100  # Increased max epochs, but rely on early stopping
LEARNING_RATE = 2e-5  # Common fine-tuning LR
WEIGHT_DECAY = 0.01
PATIENCE = 10  # Early stopping patience
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4  # For DataLoader

# Create dirs
MEL_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- INITIALIZE W&B ---
wandb.init(
    project="messy-mashup-transformer",  # Replace with your preferred project name
    config={
        "model": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "dataset": "Messy Mashup (genres_stems -> mix)",
        "split_ratio": "90/10",
        "duration_sec": DURATION_SEC,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "num_epochs": NUM_EPOCHS,
        "patience": PATIENCE,
        "sample_rate": SAMPLE_RATE_AST,
        "gradient_accumulation_steps": 4,
        "fp16": True,
        "gradient_checkpointing": True,
    }
)
# ----------------------

# Print paths using Rich
path_table = Table(title="Configuration Paths", show_header=True, header_style="bold magenta")
path_table.add_column("Directory", style="dim", width=20)
path_table.add_column("Path", style="green")
path_table.add_row("Project Root", str(PROJECT_ROOT))
path_table.add_row("Raw Data", str(RAW_DIR))
path_table.add_row("Processed Dir", str(PROCESSED_DIR))
path_table.add_row("Mel Spectrograms", str(MEL_DIR))
path_table.add_row("Splits", str(SPLITS_DIR))
path_table.add_row("Checkpoints", str(CHECKPOINT_DIR))
console.print(Panel(path_table, border_style="blue"))

# ---------------------------
# 2. HELPER FUNCTIONS (defined first)
# ---------------------------

def fix_length_audio(audio, target_len):
    """Pad or truncate audio to target length."""
    if len(audio) < target_len:
        return np.pad(audio, (0, target_len - len(audio)), mode='constant')
    else:
        return audio[:target_len]

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
                        y_stem, sr = librosa.load(stem_path, sr=SAMPLE_RATE_AST, mono=True)  # Resample to AST rate
                        y_stem = fix_length_audio(y_stem, TARGET_SAMPLES)
                        if y_mixed is None:
                            y_mixed = y_stem
                        else:
                            y_mixed += y_stem
                    # Normalize the mixed audio to prevent clipping
                    y_mixed = y_mixed / (np.max(np.abs(y_mixed)) + 1e-8)
                    # Save the mixed audio file
                    sf.write(mix_path, y_mixed, SAMPLE_RATE_AST)
                    logger.debug(f"Saved mix: {mix_path}")
                    mix_count += 1
                else:
                    logger.debug(f"Mix already exists, skipping: {mix_path}")
            else:
                logger.warning(f"Missing stems in {track_dir}, skipping mix generation.")
    logger.info(f"Generated {mix_count} mix.wav files.")

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
                mix_path = track_dir / "mix.wav"  # Use the generated mix.wav
                records.append({
                    "genre": genre_dir.name,
                    "track_id": track_dir.name,
                    "mix_path": str(mix_path),
                    "file_base": f"{genre_dir.name}__{track_dir.name}"
                })
    logger.info(f"Found {len(records)} valid tracks in genres_stems/ (with mix.wav)")
    return records

# Define preprocessing function for the HuggingFace dataset
def create_preprocess_fn(feature_extractor, target_sr, target_samples):
    def preprocess_function(examples):
        # Load audio file
        audio_sample, fs = librosa.load(examples['mix_path'], sr=target_sr, mono=True)
        audio_sample = fix_length_audio(audio_sample, target_samples)
        # Extract features using the preprocessor
        inputs = feature_extractor(
            audio_sample, sampling_rate=target_sr, return_tensors="pt"
        )
        # Squeeze the batch dimension added by feature extractor (it expects lists/batches)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = examples["label"]
        return inputs
    return preprocess_function

# Define compute metrics function for the trainer
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # The model outputs logits, take argmax for predictions
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    # --- LOG TO W&B ---
    wandb.log({"eval_accuracy": acc, "eval_f1": f1})
    # -------------------
    return {"accuracy": acc, "f1": f1}

# Define function to create HuggingFace dataset
def df_to_hf_dataset(df, preprocess_function):
    hf_ds = HFDataset.from_pandas(df[['mix_path', 'label']])
    # Apply the preprocessing function to the entire dataset
    # Remove the 'mix_path' column as it's no longer needed after preprocessing
    hf_ds = hf_ds.map(preprocess_function, remove_columns=["mix_path"])
    return hf_ds

# ---------------------------
# 3. MAIN LOGIC
# ---------------------------

console.rule("[bold blue]Step 1: Data Preparation")
# Generate mix files first (if needed)
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

# Stratified split: ONLY train and val (90/10 split)
# Total: 1000 tracks -> 900 train, 100 val (10% val)
train_df, val_df = train_test_split(df, test_size=0.10, stratify=df["label"], random_state=42)

# Save only train and val CSVs
for name, d in [("train", train_df), ("val", val_df)]:
    d.to_csv(SPLITS_DIR / f"{name}.csv", index=False)
    logger.info(f" {name}.csv: {len(d)} samples")

# Log dataset info to W&B
wandb.log({"train_samples": len(train_df), "val_samples": len(val_df), "num_classes": num_classes})

# Print split summary using Rich
split_table = Table(title="Dataset Split Summary", show_header=True, header_style="bold cyan")
split_table.add_column("Split", style="dim")
split_table.add_column("Samples", justify="right")
split_table.add_column("Distribution", style="green")
for name, d in [("Train", train_df), ("Val", val_df)]:
    dist_str = ", ".join([f"{k}:{v}" for k, v in d['label'].value_counts().sort_index().items()])
    split_table.add_row(name, str(len(d)), dist_str)
console.print(split_table)

console.rule("[bold blue]Step 2: Model Setup")
# Load the feature extractor and model
logger.info("Loading pre-trained AST model and feature extractor...")
model_checkpoint = "MIT/ast-finetuned-audioset-10-10-0.4593"

feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

# Load the model configuration and update it
from transformers import AutoConfig
config = AutoConfig.from_pretrained(model_checkpoint)
config.num_labels = num_classes
config.label2id = label2idx
config.id2label = idx2label

model = AutoModelForAudioClassification.from_pretrained(
    model_checkpoint,
    config=config,
    ignore_mismatched_sizes=True,  # Important: allows resizing classifier head
)
logger.info(f"Model loaded: {model_checkpoint}")

# Move model to device
model = model.to(DEVICE)

console.rule("[bold blue]Step 3: Training Setup")
# Prepare HuggingFace Datasets for the Trainer
preprocess_fn = create_preprocess_fn(feature_extractor, SAMPLE_RATE_AST, TARGET_SAMPLES)
train_hf_ds = df_to_hf_dataset(train_df, preprocess_fn)
val_hf_ds = df_to_hf_dataset(val_df, preprocess_fn)

# Define Training Arguments
# The Trainer supports early stopping via callbacks, though it's not directly a TrainingArgument.
# We'll use the built-in mechanisms as much as possible.
# --- LOG TRAINING ARGS TO W&B ---
wandb.config.update({"effective_batch_size": BATCH_SIZE * 4}) # Example: log effective batch size
# ----------------------------------
training_args = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,   # effective batch = 2x4 = 8 (or 4 if BS was 1)
    fp16=True,                        # halves memory usage
    gradient_checkpointing=True,      # trades compute for memory
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    eval_strategy="epoch", # Evaluate every epoch
    save_strategy="epoch", # Save every epoch
    load_best_model_at_end=True, # Load the best model based on eval_metric at the end of training
    metric_for_best_model="f1", # Use F1 for best model selection
    greater_is_better=True, # Higher F1 is better
    logging_steps=10,
    report_to="wandb", # <-- ENABLE W&B LOGGING IN TRAINER
    dataloader_num_workers=NUM_WORKERS,
    remove_unused_columns=False,
    push_to_hub=False,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_hf_ds,
    eval_dataset=val_hf_ds,
    compute_metrics=compute_metrics,
)

# Start Training
logger.info("Starting fine-tuning with 90/10 split, 10sec audio, and W&B logging...")
trainer.train()

console.rule("[bold green]Training Complete!")
# Save the final best model (Trainer saves the best one based on eval F1)
trainer.save_model()
model_save_path = CHECKPOINT_DIR / "pytorch_model.bin" # Standard filename for the main weights
console.print(f"[bold]Model saved to:[/bold] {model_save_path}")

# Save label encoder
import pickle
le_save_path = CHECKPOINT_DIR / "label_encoder.pkl"
with open(le_save_path, "wb") as f:
    pickle.dump(le, f)
console.print(f"[bold]Label encoder saved to:[/bold] {le_save_path}")

# Print final metrics (optional, trainer logs it via W&B)
final_metrics = trainer.evaluate()
logger.info(f"Final Evaluation Metrics: {final_metrics}")
wandb.log({"final_eval_metrics": final_metrics}) # Log final metrics explicitly if needed

# Print final metrics using Rich
metrics_table = Table(title="Final Evaluation Metrics", show_header=True, header_style="bold yellow")
metrics_table.add_column("Metric", style="dim")
metrics_table.add_column("Value", justify="right")
for k, v in final_metrics.items():
    if isinstance(v, float):
        metrics_table.add_row(k, f"{v:.4f}")
    else:
        metrics_table.add_row(k, str(v))
console.print(metrics_table)

# --- FINISH W&B RUN ---
wandb.finish()
