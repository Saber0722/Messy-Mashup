# imports

import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pathlib import Path

# basic config


PROJECT_ROOT = Path("..").resolve()

RAW_PATH = PROJECT_ROOT / "data/raw/messy_mashup"

GENRES_PATH = RAW_PATH / "genres_stems"

MASHUP_PATH = RAW_PATH / "mashups"

PROCESSED_MEL = PROJECT_ROOT / "data/processed/mel_spectrograms"

SPLIT_PATH = PROJECT_ROOT / "data/splits"

os.makedirs(PROCESSED_MEL, exist_ok=True)
os.makedirs(SPLIT_PATH, exist_ok=True)

# mel spectorgram function

def compute_mel(file_path, sr=22050, n_mels=128):

    y, sr = librosa.load(file_path, sr=sr)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels
    )

    mel_db = librosa.power_to_db(mel)

    return mel_db

# process genre stems

def process_genre_stems():

    records = []

    genres = sorted(os.listdir(GENRES_PATH))

    for genre in genres:

        print(f"Processing genre: {genre}")
        genre_path = os.path.join(GENRES_PATH, genre)

        for track in tqdm(os.listdir(genre_path), desc=genre):

            track_path = os.path.join(genre_path, track)

            if not os.path.isdir(track_path):
                continue

            for stem in os.listdir(track_path):

                if not stem.endswith(".wav"):
                    continue

                stem_path = os.path.join(track_path, stem)

                try:

                    mel = compute_mel(stem_path)

                    save_name = f"{genre}_{track}_{stem}.npy"
                    save_path = os.path.join(PROCESSED_MEL, save_name)

                    np.save(save_path, mel)

                    records.append({
                        "file": save_name,
                        "label": genre,
                        "type": "stem"
                    })

                except Exception as e:
                    print("Error:", stem_path)

    return records

# process mashups

def process_mashups():

    records = []

    mashup_files = [f for f in os.listdir(MASHUP_PATH) if f.endswith(".wav")]

    for file in tqdm(mashup_files, desc="Mashups"):

        path = os.path.join(MASHUP_PATH, file)

        try:

            mel = compute_mel(path)

            save_name = f"mashup_{file.replace('.wav','.npy')}"
            save_path = os.path.join(PROCESSED_MEL, save_name)

            np.save(save_path, mel)

            records.append({
                "file": save_name,
                "label": "mashup",
                "type": "mashup"
            })

        except Exception as e:
            print("Error:", file)

    return records

# create train/val splits

def create_splits(records):

    df = pd.DataFrame(records)

    train, val = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    train.to_csv(os.path.join(SPLIT_PATH, "train.csv"), index=False)
    val.to_csv(os.path.join(SPLIT_PATH, "val.csv"), index=False)

    print("Saved splits.")

# main function

def main():

    print("Processing genre stems...")
    stem_records = process_genre_stems()

    print("Processing mashups...")
    mashup_records = process_mashups()

    records = stem_records + mashup_records

    print("Creating dataset splits...")
    create_splits(records)

    print("Dataset build complete.")

if __name__ == "__main__":
    main()