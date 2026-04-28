# puzzlebot_voice_commands

Offline voice command recognition package for the Puzzlebot ROS 2 workspace.

**Phase:** 1 — package structure only. ML logic is not yet implemented.

## Purpose

Train and evaluate voice command classifiers using `.wav` audio files and
hand-crafted MFCC features. Two models are implemented from scratch:

| Model                       | Feature input         | Approach                                      |
|-----------------------------|-----------------------|-----------------------------------------------|
| `KMeansCodebookClassifier`  | Frame-level MFCCs     | One K-Means codebook per class (VQ-style)     |
| `GaussianNaiveBayesClassifier` | MFCC summary vector | Gaussian log-likelihood + class prior         |

This package is **offline only** — it does not connect to the robot or publish
to `/cmd_vel`. Integration with the Puzzlebot control stack is a future phase.

## Target commands

`adelante`, `atras`, `izquierda`, `derecha`, `alto`, `inicio`

Additional classes are auto-discovered from dataset subfolders.

## Quick start (after Phase 2+)

```bash
# Build the workspace
make build
source install/setup.bash

# 1. (Optional) pre-extract features
ros2 run puzzlebot_voice_commands prepare_voice_dataset \
  --dataset src/puzzlebot_voice_commands/datasets/voice_commands_dataset \
  --output  src/puzzlebot_voice_commands/artifacts/features.json

# 2. Train both models
ros2 run puzzlebot_voice_commands train_voice_models \
  --dataset    src/puzzlebot_voice_commands/datasets/voice_commands_dataset \
  --model      both \
  --output-dir src/puzzlebot_voice_commands/artifacts

# 3. Evaluate and generate reports
ros2 run puzzlebot_voice_commands evaluate_voice_models \
  --dataset      src/puzzlebot_voice_commands/datasets/voice_commands_dataset \
  --artifact-dir src/puzzlebot_voice_commands/artifacts \
  --output-dir   src/puzzlebot_voice_commands/reports

# 4. Predict a single file
ros2 run puzzlebot_voice_commands predict_voice_file \
  --model-type gnb \
  --model-path src/puzzlebot_voice_commands/artifacts/gnb_model.pkl \
  --audio      path/to/audio.wav
```

## Package structure

```
puzzlebot_voice_commands/
├── package.xml
├── setup.py
├── setup.cfg
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
│       ├── prepare_dataset.py  — CLI: prepare_voice_dataset
│       ├── train_models.py     — CLI: train_voice_models
│       ├── evaluate_models.py  — CLI: evaluate_voice_models
│       └── predict_file.py     — CLI: predict_voice_file
├── datasets/               — Place voice_commands_dataset/ here (not committed)
├── artifacts/              — Trained models and configs (not committed)
└── reports/                — Evaluation outputs (not committed)
```

## Allowed libraries

NumPy, SciPy, Matplotlib (optional plots), and standard Python only.
No scikit-learn, PyTorch, TensorFlow, or any prebuilt ML classifier.

## Implementation phases

| Phase | Content                                                  | Status      |
|-------|----------------------------------------------------------|-------------|
| 1     | Package structure, stubs, buildable skeleton             | Done        |
| 2     | Audio I/O, MFCC extraction, dataset split                | Pending     |
| 3     | KMeansCodebookClassifier + training script               | Pending     |
| 4     | GaussianNaiveBayesClassifier + training script           | Pending     |
| 5     | Full metrics, report generation, model comparison        | Pending     |
| 6     | Documentation cleanup, validation checklist              | Pending     |
| 7+    | ROS 2 inference node, Puzzlebot integration              | Future      |
