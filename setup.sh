#!/bin/bash

# Core files
touch README.md
touch requirements.txt
touch pyproject.toml
touch setup.py
touch .gitignore

# Configs
mkdir -p configs
touch configs/base_config.yaml
touch configs/model_config.yaml
touch configs/augmentation_config.yaml
touch configs/training_config.yaml
touch configs/inference_config.yaml

# Notebooks
mkdir -p notebooks
touch notebooks/01_data_exploration.ipynb
touch notebooks/02_feature_analysis.ipynb
touch notebooks/03_augmentation_visualization.ipynb
touch notebooks/04_model_debugging.ipynb

# Data folders
mkdir -p data/raw/messy_mashup
mkdir -p data/interim/simulated_mashups
mkdir -p data/interim/cached_stems
mkdir -p data/interim/augmented_samples
mkdir -p data/processed/mel_spectrograms
mkdir -p data/processed/stem_embeddings
mkdir -p data/processed/tfrecords
mkdir -p data/splits
touch data/splits/train.csv
touch data/splits/val.csv
touch data/splits/fold_indices.pkl

# Source code structure
mkdir -p src/data
mkdir -p src/augmentation
mkdir -p src/features
mkdir -p src/models
mkdir -p src/training
mkdir -p src/inference
mkdir -p src/utils

touch src/__init__.py

touch src/data/dataset_builder.py
touch src/data/tfrecord_writer.py
touch src/data/tf_dataset_loader.py
touch src/data/audio_loader.py
touch src/data/stem_sampler.py

touch src/augmentation/tempo.py
touch src/augmentation/noise.py
touch src/augmentation/gain.py
touch src/augmentation/mix_stems.py
touch src/augmentation/spec_augment.py

touch src/features/mel_spectrogram.py
touch src/features/chroma.py
touch src/features/spectral_features.py
touch src/features/feature_fusion.py

touch src/models/stem_branch.py
touch src/models/fusion.py
touch src/models/crnn.py
touch src/models/attention.py
touch src/models/messy_mashup_model.py
touch src/models/loss.py

touch src/training/trainer.py
touch src/training/metrics.py
touch src/training/callbacks.py
touch src/training/scheduler.py

touch src/inference/predict.py
touch src/inference/postprocess.py
touch src/inference/submission_writer.py

touch src/utils/logger.py
touch src/utils/seed.py
touch src/utils/audio_utils.py
touch src/utils/visualization.py

# Experiments
mkdir -p experiments/exp_001_baseline
mkdir -p experiments/exp_002_stem_multi_branch
mkdir -p experiments/exp_003_noise_robust
mkdir -p experiments/logs

# Checkpoints
mkdir -p checkpoints

# Submissions
mkdir -p submissions

# Scripts
mkdir -p scripts
touch scripts/build_dataset.py
touch scripts/train.py
touch scripts/evaluate.py
touch scripts/infer_test.py

echo "Project structure created successfully!"
