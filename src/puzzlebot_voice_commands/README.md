# puzzlebot_voice_commands

Offline voice command recognition package for the Puzzlebot ROS 2 workspace.

**Current phase:** 9 — ROS 2 inference node (in progress).  
**Selected models:** KMeans (97.74%) + HMM librosa (92.01%), trained on 4-speaker augmented dataset.  
**Artifacts:** `artifacts_final/` — production-ready pkl files.

## Models

| Model | Accuracy (test) | Inference | Approach |
|-------|----------------|-----------|----------|
| `KMeansCodebookClassifier` | **97.74%** | 0.53 ms | One K-Means codebook per class (VQ distance) |
| `HiddenMarkovModelClassifier` | **92.01%** | 53 ms | Left-to-right HMM per class, librosa features |
| `GaussianNaiveBayesClassifier` | 89.41% | 0.22 ms | MFCC summary vector + Gaussian log-likelihood |

Evaluated on 576 test samples (augmented dataset, 4 speakers × 20 clips/command × 4x aug).

## Commands

`avanzar`, `retroceder`, `izquierda`, `derecha`, `alto`, `inicio`

## Dataset

4 speakers (rodo, jorge, valeria, jesus) × 20 clips/command = **480 original clips**.  
Augmented 4x (time stretch ×1.1, time stretch ×0.9, pitch shift +1 semitone) → **1920 clips**.

```
datasets/
├── data_rodo/              — 20/cmd ✓
├── data_jorge/             — 20/cmd ✓
├── data_valeria/           — 20/cmd ✓
├── data_jesus/             — 20/cmd ✓
├── voice_commands_dataset/ — merged original (480 clips)
└── voice_commands_dataset_aug/ — augmented 4x (1920 clips)
```

## Artifacts

```
artifacts_final/               ← production models (use these in ROS node)
├── hmm_model.pkl              — HMM librosa + syllable-states
├── hmm_config.json            — HMM params + MFCC config
├── kmeans_model.pkl           — KMeans codebook
├── kmeans_feature_config.json — KMeans MFCC config
├── labels.json                — class list
└── train_metadata.json        — split metadata

artifacts_pre_aug/             ← snapshot before augmentation
artifacts_hmm_manual/          ← manual HMM (no librosa) on augmented data
artifacts_hmm_manual_pre_aug/  ← manual HMM before augmentation
```

## HMM feature configuration

```
n_mfcc=20, delta=True, cmvn=True
librosa backend: MFCC + ZCR + RMS + spectral contrast
per-command states: alto=3, avanzar=4, derecha=4, inicio=4, izquierda=5, retroceder=6
n_symbols=32, n_iter=20
```

## Quick start (Windows, no ROS)

```powershell
cd C:\path\to\puzzlebot_sim\src\puzzlebot_voice_commands

# Train KMeans + GNB on augmented dataset
python -m puzzlebot_voice_commands.scripts.train_models `
  --dataset datasets\voice_commands_dataset_aug `
  --model both --output-dir artifacts `
  --n-mfcc 20 --delta --min-max --kmeans-delta --kmeans-cmvn

# Train HMM librosa + syllable-states on augmented dataset
python -m puzzlebot_voice_commands.scripts.train_hmm `
  --dataset datasets\voice_commands_dataset_aug `
  --output-dir artifacts `
  --n-mfcc 20 --n-states 5 --n-symbols 32 --n-iter 20 `
  --cmvn --delta --librosa --include-zcr --include-rms --include-contrast `
  --syllable-states

# Evaluate all three models
python -m puzzlebot_voice_commands.scripts.evaluate_models `
  --dataset datasets\voice_commands_dataset_aug `
  --artifact-dir artifacts --output-dir reports --model all

# Live test (mic -> prediction)
python -m puzzlebot_voice_commands.scripts.live_test `
  --artifact-dir artifacts_final --models kmeans hmm

# Data augmentation
python -m puzzlebot_voice_commands.scripts.augment_dataset `
  --input-dir  datasets\voice_commands_dataset `
  --output-dir datasets\voice_commands_dataset_aug `
  --factor 3
```

## Quick start (ROS 2 / WSL2)

```bash
colcon build --packages-select puzzlebot_voice_commands
source install/setup.bash

ros2 run puzzlebot_voice_commands voice_commands_node \
  --ros-args -p artifact_dir:=src/puzzlebot_voice_commands/artifacts_final
```

## ROS 2 topics

| Topic | Type | Description |
|-------|------|-------------|
| `/voice/trigger` | `std_msgs/String` | Receive to start a recording |
| `/voice/command` | `std_msgs/String` | Predicted command |
| `/voice/confidence` | `std_msgs/Float32` | Log-likelihood margin (HMM) or distance margin (KMeans) |
| `/voice/status` | `std_msgs/String` | `listening` / `processing` / `idle` |
| `/voice/ranked_predictions` | `std_msgs/String` | JSON with top-3 predictions |
| `/voice/inference_time_ms` | `std_msgs/Float32` | Inference latency in ms |

## Package structure

```
puzzlebot_voice_commands/
├── package.xml
├── setup.py
├── setup.cfg
├── VALIDATION.md
├── resource/puzzlebot_voice_commands
├── puzzlebot_voice_commands/
│   ├── config.py               — MFCCConfig, HMMConfig, DatasetConfig, ...
│   ├── audio_io.py             — WAV loading, mono, normalization
│   ├── mfcc.py                 — Manual MFCC pipeline (NumPy/SciPy)
│   ├── librosa_features.py     — librosa-based feature extraction (HMM)
│   ├── dataset.py              — Dataset discovery and stratified split
│   ├── metrics.py              — All metrics from scratch (no sklearn)
│   ├── serialization.py        — pickle and JSON save/load helpers
│   ├── reports.py              — CSV, JSON, Markdown report writers
│   ├── voice_commands_node.py  — ROS 2 inference node (Phase 9)
│   ├── models/
│   │   ├── kmeans_codebook.py  — KMeansCodebookClassifier
│   │   ├── gaussian_nb.py      — GaussianNaiveBayesClassifier
│   │   └── hmm.py              — HiddenMarkovModelClassifier
│   └── scripts/
│       ├── grabar.py           — Record samples interactively
│       ├── merge_datasets.py   — Merge per-person folders
│       ├── prepare_dataset.py  — prepare_voice_dataset
│       ├── train_models.py     — train_voice_models (KMeans + GNB)
│       ├── train_hmm.py        — train_hmm_models
│       ├── tune_hmm.py         — Grid search over HMM hyperparameters
│       ├── evaluate_models.py  — evaluate_voice_models (--model all)
│       ├── predict_file.py     — predict_voice_file
│       ├── augment_dataset.py  — augment_voice_dataset
│       ├── live_test.py        — live_test_voice (mic → prediction)
│       ├── cross_validate.py   — k-fold CV
│       ├── learning_curve.py   — accuracy vs training size
│       └── speaker_test.py     — per-speaker evaluation
├── datasets/                   — Audio clips (not committed)
├── artifacts_final/            — Production models (pkl files)
└── reports/                    — Evaluation outputs
```

## Allowed libraries

NumPy, SciPy, librosa, and standard Python only.  
No scikit-learn, PyTorch, TensorFlow, or any prebuilt ML classifier.

## Implementation phases

| Phase | Content | Status |
|-------|---------|--------|
| 1 | Package structure, stubs, buildable skeleton | Done |
| 2 | Audio I/O, MFCC extraction, dataset split | Done |
| 3 | KMeansCodebookClassifier + training script | Done |
| 4 | GaussianNaiveBayesClassifier + training script | Done |
| 5 | Full metrics, report generation, model comparison | Done |
| 6 | Documentation cleanup, validation checklist | Done |
| 7 | HMM classifier from scratch (Baum-Welch + Viterbi) | Done |
| 8 | Feature engineering: delta, CMVN, librosa, per-model configs, grid search | Done |
| 8b | 4th speaker (jesus), data augmentation 4x, syllable-states HMM | Done |
| **9** | **ROS 2 inference node, trigger subscriber, /voice/* publishers** | **In progress** |
