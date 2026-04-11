# infer_transformers_val.py
# A script to load the fine-tuned transformer model and print a detailed classification report on the validation set.

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from pathlib import Path
import librosa
import pickle
import warnings
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
warnings.filterwarnings("ignore")

# ---------------------------
# 1. CONFIGURATION
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "messy_mashup"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"

# Audio params (must match training)
SAMPLE_RATE_AST = 16000
DURATION_SEC = 5
TARGET_SAMPLES = int(SAMPLE_RATE_AST * DURATION_SEC)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
console = Console()

# ---------------------------
# 2. HELPER FUNCTIONS
# ---------------------------
def fix_length_audio(audio, target_len):
    """Pad or truncate audio to target length."""
    if len(audio) < target_len:
        return np.pad(audio, (0, target_len - len(audio)), mode='constant')
    else:
        return audio[:target_len]

# ---------------------------
# 3. DATASET CLASS
# ---------------------------
class InferenceDataset(Dataset):
    def __init__(self, df, label_encoder):
        self.df = df.reset_index(drop=True)
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mix_path = row["mix_path"]
        label_str = row["genre"]
        label_int = self.label_encoder.transform([label_str])[0]

        y, sr = librosa.load(mix_path, sr=SAMPLE_RATE_AST, mono=True)
        y = fix_length_audio(y, TARGET_SAMPLES)

        return {
            "input_values": torch.from_numpy(y).float(),
            "labels": torch.tensor(label_int, dtype=torch.long),
            "file_base": row["file_base"]
        }

# ---------------------------
# 4. FIND AND LOAD MODEL
# ---------------------------
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints_transformer_minimal"
logger.info(f"Searching for model files in: {CHECKPOINT_DIR}")

# Step 1: Find the model weights file (pytorch_model.bin or model.safetensors)
model_bin_path = CHECKPOINT_DIR / "pytorch_model.bin"
# model_safetensors_path = CHECKPOINT_DIR / "checkpoint-3780/model.safetensors" # 0.87 f1 on val
model_safetensors_path = CHECKPOINT_DIR / "model.safetensors" # Check top level first


MODEL_PATH = None
if model_bin_path.exists():
    MODEL_PATH = model_bin_path
    logger.info(f"Using PyTorch model weights: {MODEL_PATH}")
elif model_safetensors_path.exists():
    MODEL_PATH = model_safetensors_path
    logger.info(f"Using Safetensors model weights: {MODEL_PATH}")
else:
    # If not in top level, look for latest checkpoint subfolder
    checkpoint_dirs = [d for d in CHECKPOINT_DIR.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if checkpoint_dirs:
        latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.name.split('-')[1]))[-1]
        model_bin_in_cp = latest_checkpoint / "pytorch_model.bin"
        model_safetensors_in_cp = latest_checkpoint / "model.safetensors"
        if model_bin_in_cp.exists():
            MODEL_PATH = model_bin_in_cp
            logger.info(f"Using PyTorch model weights from checkpoint: {MODEL_PATH}")
        elif model_safetensors_in_cp.exists():
            MODEL_PATH = model_safetensors_in_cp
            logger.info(f"Using Safetensors model weights from checkpoint: {MODEL_PATH}")
    else:
        logger.error(f"No checkpoint directories found in {CHECKPOINT_DIR}")

console.print(f"[green] Loading model from {MODEL_PATH} [/green]")

if not MODEL_PATH:
    raise RuntimeError(f"No model weights found in {CHECKPOINT_DIR} or its subdirectories. Expected 'pytorch_model.bin' or 'model.safetensors'.")

# Step 2: Load the config and label encoder
CONFIG_PATH = CHECKPOINT_DIR / "config.json"
LABEL_ENCODER_PATH = CHECKPOINT_DIR / "label_encoder.pkl"

logger.info(f"Loading label encoder from {LABEL_ENCODER_PATH}")
with open(LABEL_ENCODER_PATH, "rb") as f:
    le = pickle.load(f)
idx2label = {i: l for i, l in enumerate(le.classes_)}
logger.info(f"Loaded label encoder for classes: {list(le.classes_)}")

# Load validation data instead of test data
VAL_CSV_PATH = SPLITS_DIR / "val.csv"
logger.info(f"Loading validation data from {VAL_CSV_PATH}")
val_df = pd.read_csv(VAL_CSV_PATH)
logger.info(f"Loaded validation dataframe with {len(val_df)} samples.")

# Step 3: Load the model
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, AutoConfig

logger.info(f"Loading model configuration from {CONFIG_PATH}")

# Load config
config = AutoConfig.from_pretrained(str(CONFIG_PATH))
# Instantiate model with config
model = AutoModelForAudioClassification.from_config(config=config)

# Load the weights
logger.info(f"Loading model weights from {MODEL_PATH}")
if MODEL_PATH.suffix == ".bin":
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
elif MODEL_PATH.suffix == ".safetensors":
    from safetensors.torch import load_file
    state_dict = load_file(MODEL_PATH, device=str(DEVICE))
    model.load_state_dict(state_dict)
else:
    raise ValueError("Unsupported model file format.")

model = model.to(DEVICE)
model.eval()

# Load feature extractor (use the original checkpoint name for consistency)
feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

# ---------------------------
# 5. RUN INFERENCE ON VAL SET
# ---------------------------
console.rule("[bold blue]Running Inference on Validation Set")
val_dataset = InferenceDataset(val_df, le)
all_preds = []
all_true = []

model.eval()
with torch.no_grad():
    for i in range(len(val_dataset)):
        item = val_dataset[i]
        audio_input = item["input_values"].numpy()
        label_true = item["labels"].item()

        # Preprocess using the feature extractor
        inputs = feature_extractor(
            audio_input, sampling_rate=SAMPLE_RATE_AST, return_tensors="pt"
        )
        x = inputs["input_values"].to(DEVICE)

        outputs = model(x)
        logits = outputs.logits
        pred_idx = torch.argmax(logits, dim=1).item()

        all_preds.append(pred_idx)
        all_true.append(label_true)

logger.info("Inference completed.")

# ---------------------------
# 6. CALCULATE AND PRINT METRICS (Rich UI)
# ---------------------------
true_labels_str = le.inverse_transform(all_true)
pred_labels_str = le.inverse_transform(all_preds)

macro_f1 = f1_score(all_true, all_preds, average='macro')
accuracy = np.mean(np.array(all_true) == np.array(all_preds))

# Overall Metrics Panel
overall_metrics_table = Table(title="Overall Metrics", box=None, show_header=False, title_style="bold magenta")
overall_metrics_table.add_row("Accuracy", f"{accuracy:.4f}")
overall_metrics_table.add_row("Macro F1", f"{macro_f1:.4f}")
overall_metrics_panel = Panel(overall_metrics_table, border_style="green")
console.print(overall_metrics_panel)

# Detailed Classification Report
report_str = classification_report(true_labels_str, pred_labels_str, target_names=le.classes_, output_dict=False)
console.print(Panel(Text(report_str, style="default"), title="[bold yellow]Detailed Classification Report (Validation Set)", border_style="blue"))

# Save report (text version)
report_filename = CHECKPOINT_DIR / "val_classification_report.txt"
with open(report_filename, "w") as f:
    f.write("Classification Report (Validation Set):\n")
    f.write("="*50 + "\n")
    f.write(classification_report(true_labels_str, pred_labels_str, target_names=le.classes_))
    f.write(f"\nOverall Accuracy: {accuracy:.4f}\n")
    f.write(f"Overall Macro F1 Score: {macro_f1:.4f}\n")
logger.info(f"Report saved to: {report_filename}")

console.print(f"\n[bold green]Validation report saved to:[/bold green] {report_filename}")