# count_tracks_from_mel.py
import pathlib
import re

MEL_PATH = pathlib.Path("data/processed/mel_spectrograms")
STEMS = ["bass", "drums", "other", "vocals", "mix"]

track_set = set()
for npy_file in MEL_PATH.glob("*.npy"):
    stem = npy_file.stem
    # Try new format: genre__track__stem.npy
    if "__" in stem:
        parts = stem.split("__")
        if len(parts) == 3:
            genre, track, s = parts
            if s in STEMS:
                track_set.add(f"{genre}__{track}")
        else:
            # fallback: maybe old format slipped through?
            pass
    else:
        # Old format: genre_track_stem.wav.npy or genre_track_stem.npy
        # Example: blues_blues.00000_bass.wav.npy
        if "_" in stem:
            parts = stem.split("_")
            if len(parts) >= 3:
                genre = parts[0]
                track = "_".join(parts[1:-1])
                s = parts[-1].replace(".wav", "").replace(".npy", "")
                if s in STEMS:
                    track_set.add(f"{genre}__{track}")

# Now verify completeness: for each track, ensure all 5 stems exist
complete_tracks = []
for key in track_set:
    genre, track = key.split("__", 1)
    missing = []
    for stem in STEMS:
        candidates = [
            f"{key}__{stem}.npy",
            f"{genre}_{track}_{stem}.wav.npy",
            f"{genre}_{track}_{stem}.npy",
        ]
        if not any((MEL_PATH / cand).exists() for cand in candidates):
            missing.append(stem)
    if not missing:
        complete_tracks.append(key)

print(f"✅ Total complete tracks (with all 5 stems in mel_spectrograms): {len(complete_tracks)}")