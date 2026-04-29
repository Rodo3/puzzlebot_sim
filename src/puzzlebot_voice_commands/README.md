# puzzlebot_voice_commands

Offline voice command recognition package for the Puzzlebot ROS 2 workspace.

**Current phase:** 6 complete — all offline phases done.
**Status:** Full pipeline functional. Dataset collection in progress (2/4 speakers recorded).
Recommended model: **GaussianNB** (98.1% accuracy, 0 safety errors, 0.17 ms inference).

## Purpose

Train and evaluate voice command classifiers using `.wav` audio files and
hand-crafted MFCC features. Two models are implemented from scratch:

| Model | Feature input | Approach |
|-------|---------------|----------|
| `KMeansCodebookClassifier` | Frame-level MFCCs | One K-Means codebook per class (VQ-style) |
| `GaussianNaiveBayesClassifier` | MFCC summary vector | Gaussian log-likelihood + class prior |

This package is **offline only** — it does not connect to the robot or publish
to `/cmd_vel`. Integration with the Puzzlebot control stack is a future phase.

The system is designed for **exactly 4 known speakers**. The model intentionally
learns each team member's voice — it is not expected to generalise to unknown speakers.

## Target commands

`avanzar`, `retroceder`, `izquierda`, `derecha`, `alto`, `inicio`

Classes are auto-discovered from dataset subfolders.

## Quick start (Windows, no ROS)

```powershell
cd C:\path\to\puzzlebot_sim
$env:PYTHONPATH = "src\puzzlebot_voice_commands"

# 1. Record samples (one person at a time)
python -m puzzlebot_voice_commands.scripts.grabar
# rename the generated data/ folder to data_<name>/

# 2. Merge all per-person folders into one dataset
python -m puzzlebot_voice_commands.scripts.merge_datasets `
  --inputs  datasets\data_jorge datasets\data_valeria ... `
  --output  datasets\voice_commands_dataset

# 3. Train both models
python -m puzzlebot_voice_commands.scripts.train_models `
  --dataset    datasets\voice_commands_dataset `
  --model      both `
  --output-dir artifacts

# 4. Evaluate and generate reports
python -m puzzlebot_voice_commands.scripts.evaluate_models `
  --dataset      datasets\voice_commands_dataset `
  --artifact-dir artifacts `
  --output-dir   reports

# 5. Cross-validate (verify results are not lucky split)
python -m puzzlebot_voice_commands.scripts.cross_validate `
  --dataset datasets\voice_commands_dataset --model both --k 5

# 6. Learning curve (check if more data is needed)
python -m puzzlebot_voice_commands.scripts.learning_curve `
  --dataset    datasets\voice_commands_dataset `
  --model      both `
  --output-dir reports

# 7. Per-speaker evaluation (verify each team member is recognized)
python -m puzzlebot_voice_commands.scripts.speaker_test `
  --dataset    datasets\voice_commands_dataset `
  --model      gnb `
  --mode       all-train `
  --output-dir reports

# 8. Predict a single file
python -m puzzlebot_voice_commands.scripts.predict_file `
  --model-type gnb `
  --model-path artifacts\gnb_model.pkl `
  --audio      path\to\audio.wav
```

## Quick start (ROS 2 / WSL2)

```bash
colcon build --packages-select puzzlebot_voice_commands
source install/setup.bash

ros2 run puzzlebot_voice_commands train_voice_models \
  --dataset    src/puzzlebot_voice_commands/datasets/voice_commands_dataset \
  --model      both \
  --output-dir src/puzzlebot_voice_commands/artifacts

ros2 run puzzlebot_voice_commands evaluate_voice_models \
  --dataset      src/puzzlebot_voice_commands/datasets/voice_commands_dataset \
  --artifact-dir src/puzzlebot_voice_commands/artifacts \
  --output-dir   src/puzzlebot_voice_commands/reports
```

## Package structure

```
puzzlebot_voice_commands/
├── package.xml
├── setup.py
├── setup.cfg
├── VALIDATION.md           — step-by-step build/train/evaluate checklist
├── resource/puzzlebot_voice_commands
├── puzzlebot_voice_commands/
│   ├── config.py           — MFCCConfig, DatasetConfig, KMeansConfig, GNBConfig
│   ├── audio_io.py         — WAV loading, mono conversion, normalization
│   ├── mfcc.py             — Manual MFCC pipeline (NumPy/SciPy)
│   ├── dataset.py          — Dataset discovery and stratified split
│   ├── metrics.py          — All metrics from scratch (no sklearn)
│   ├── serialization.py    — pickle and JSON save/load helpers
│   ├── reports.py          — CSV, JSON, and Markdown report writers
│   ├── models/
│   │   ├── kmeans_codebook.py  — KMeansCodebookClassifier
│   │   └── gaussian_nb.py      — GaussianNaiveBayesClassifier
│   └── scripts/
│       ├── grabar.py           — CLI: record samples interactively
│       ├── merge_datasets.py   — CLI: merge per-person folders into one dataset
│       ├── prepare_dataset.py  — CLI: prepare_voice_dataset
│       ├── train_models.py     — CLI: train_voice_models
│       ├── evaluate_models.py  — CLI: evaluate_voice_models
│       ├── predict_file.py     — CLI: predict_voice_file
│       ├── cross_validate.py   — CLI: k-fold cross-validation
│       ├── learning_curve.py   — CLI: accuracy vs training size curve
│       └── speaker_test.py     — CLI: per-speaker evaluation
├── datasets/               — Per-person folders + merged dataset (not committed)
├── artifacts/              — Trained models and configs (not committed)
└── reports/                — Evaluation outputs (not committed)
```

## Allowed libraries

NumPy, SciPy, and standard Python only.
No scikit-learn, PyTorch, TensorFlow, or any prebuilt ML classifier.

## Dataset collection workflow

Each team member records using `grabar.py` (20 clips per command):

```
datasets/
├── data_jorge/       — 15 clips/class (needs 5 more per class)
├── data_valeria/     — 15 clips/class (needs 5 more per class)
├── data_<person3>/   — pending
├── data_<person4>/   — pending
└── voice_commands_dataset/  — merged output (auto-generated by merge_datasets)
```

**Recommended clip count:** 20 per person per command.
After recording all 4, run `speaker_test --mode leave-one-out` to check if any
person needs more recordings (target: recall ≥ 0.90 per class per speaker).

## Evaluation results (2 speakers, 2026-04-28)

Dataset: Jorge + Valeria, 30 clips/class, 6 classes, 180 total samples.

### Model comparison

| Metric | KMeans | GaussianNB |
|--------|--------|------------|
| Global accuracy | 92.6% | **98.1%** |
| Macro recall | 92.6% | **98.1%** |
| Macro F1 | 92.8% | **98.1%** |
| Top-2 accuracy | 96.3% | **100%** |
| Safety errors (`alto`) | 1 | **0** |
| Avg inference | 0.58 ms | **0.17 ms** |
| Artifact size | 5.5 KB | 3.3 KB |

### Cross-validation (k=5)

| Metric | KMeans | GaussianNB |
|--------|--------|------------|
| Acc mean ± std | 92.8% ± 3.3% | **97.2% ± 5.6%** |
| Macro recall mean | 93.7% | **98.5%** |
| Safety errors (total) | 3 | **0** |

High std on GNB (5.6%) is expected with only 2 speakers — will decrease with 4.

### Learning curve

- **KMeans:** large train/test gap at all fractions → memorises, needs more data.
- **GaussianNB:** plateaus at ~50% of training data → sufficient data per speaker.

### Per-speaker test (all-train mode)

| Speaker | KMeans acc | GNB acc | KMeans safety err | GNB safety err |
|---------|-----------|---------|-------------------|----------------|
| Jorge | 98.9% | **100%** | 0 | 0 |
| Valeria | 98.9% | **100%** | 1 (`alto`) | **0** |

**GaussianNB is the recommended model for ROS 2 integration.**

## Implementation phases

| Phase | Content | Status |
|-------|---------|--------|
| 1 | Package structure, stubs, buildable skeleton | **Done** |
| 2 | Audio I/O, MFCC extraction, dataset split | **Done** |
| 3 | KMeansCodebookClassifier + training script | **Done** |
| 4 | GaussianNaiveBayesClassifier + training script | **Done** |
| 5 | Full metrics, report generation, model comparison | **Done** |
| 6 | Documentation cleanup, validation checklist | **Done** |
| 7+ | ROS 2 inference node, Puzzlebot integration | Future |

## CLI scripts

| Command | Mode | Description |
|---------|------|-------------|
| `grabar` | standalone | Record 20 clips per command interactively |
| `merge_voice_datasets` | standalone | Merge per-person folders into one dataset |
| `prepare_voice_dataset` | ROS 2 / standalone | Discover → split → extract MFCCs → JSON |
| `train_voice_models` | ROS 2 / standalone | Train KMeans and/or GNB, save artifacts |
| `evaluate_voice_models` | ROS 2 / standalone | Evaluate on test split, generate 7 reports |
| `predict_voice_file` | ROS 2 / standalone | Single-file inference with ranked output |
| `cross_validate_voice` | standalone | K-fold CV: mean/std of accuracy and recall |
| `learning_curve_voice` | standalone | Accuracy vs training size, detects overfitting |
| `speaker_test_voice` | standalone | Per-speaker recall, all-train or leave-one-out |
