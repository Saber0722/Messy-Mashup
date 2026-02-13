# 🎵 Messy Mashup: Robust Music Genre Classification

---

## 📌 Project Title

**Robust Stem-Aware Music Genre Classification under Noisy Mashup Conditions**

---

## 👤 Student Details

- **Name:** Aakash Chilakamarri
- **Roll Number:** 24f1001512
- **Course:** DL-GenAI-Project
- **Institution:** IIT Madras
- **Competition:** Messy Mashup (Kaggle Competition)

---

## 📖 Project Overview

The *Messy Mashup* project focuses on robust music genre classification under realistic and noisy mixing conditions.

Unlike traditional genre classification tasks that operate on clean audio tracks, this challenge introduces significant distribution shift between training and testing data.

### 🔹 Training Data
- 10 music genres
- 100 songs per genre
- Each song decomposed into 4 instrument stems:
  - drums.wav
  - vocals.wav
  - bass.wav
  - others.wav

### 🔹 Test Data
- Cross-song stem mashups
- Tempo-adjusted stems
- Random instrument balance
- Additive environmental noise (ESC-50 dataset)

The objective is to design a model that learns genre-specific musical characteristics invariant to:
- Cross-song recombination
- Tempo variations
- Noise corruption
- Instrument amplitude shifts

---

## 🎯 Evaluation Metric

The competition is evaluated using **Macro F1 Score** across the 10 genre classes.

Macro F1 computes the F1 score independently for each genre and then averages them, ensuring equal weight to all classes.

---

## 🧠 Proposed Approach (Initial Plan)

This project adopts a **stem-aware multi-branch deep learning architecture** built using TensorFlow.

Key components include:

- Log-Mel Spectrogram feature extraction
- On-the-fly simulated mashup generation
- Noise augmentation using ESC-50 dataset
- Multi-branch CNN for stem-level representation learning
- Cross-stem feature fusion
- Temporal modeling using CRNN/Attention
- Macro-F1 optimized training

The goal is to build a robust audio representation model capable of generalizing under distribution shift.

---

## 🗂 Project Structure

The repository is organized into:

- `src/` → Core source code
- `data/` → Raw and processed datasets
- `configs/` → Experiment configuration files
- `experiments/` → Versioned experiment runs
- `checkpoints/` → Model weights
- `submissions/` → Kaggle submission files
- `scripts/` → Training and inference scripts

---

## 🚀 Status

Project initialization complete.  
Data exploration and preprocessing pipeline development in progress.

---

## 📜 License

Subject to competition rules.
