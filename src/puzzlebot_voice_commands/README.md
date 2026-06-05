# puzzlebot_voice_commands

Offline voice command recognition for the Puzzlebot ROS 2 workspace.

**Model:** HMM librosa + syllable-states — **87.40% accuracy** (10 commands, 4 speakers, leakage-free split).

## Commands

`avanzar`, `retroceder`, `izquierda`, `derecha`, `alto`, `inicio`, `subir`, `bajar`, `tomar`, `soltar`

## Model

| Model | Accuracy | Inference | Notes |
|-------|----------|-----------|-------|
| `HiddenMarkovModelClassifier` | **87.40%** | ~55 ms | Left-to-right HMM, librosa features, n_symbols=64 |

Evaluated on 960 test samples (leakage-free split, 3200 total augmented clips).

## Dataset

4 speakers (rodo, jorge, valeria, jesus) × 20 clips/command × 10 commands = **800 original clips**.
Augmented 4x → **3200 clips** in `voice_commands_dataset_aug/`.

```
datasets/
├── data_rodo/
├── data_jorge/
├── data_valeria/
├── data_jesus/
├── voice_commands_dataset/     — merged originals (800 clips)
└── voice_commands_dataset_aug/ — augmented 4x (3200 clips)
```

## Artifacts

```
artifacts_final/               ← production model
├── hmm_model.pkl              — HMM classifier
├── hmm_config.json            — HMM params + MFCC config
├── labels.json                — class list
└── train_metadata.json        — split metadata

artifacts_v2_85pct/            ← backup (85.31% model)
```

## HMM configuration

```
n_mfcc=20, delta=True, cmvn=True
librosa backend: MFCC + ZCR + RMS + spectral contrast
n_symbols=64, n_iter=5
per-command states:
  alto=4, avanzar=6, bajar=5, derecha=6, inicio=4,
  izquierda=5, retroceder=6, soltar=5, subir=6, tomar=6
```

## Quick start (Windows, no ROS)

```powershell
cd src\puzzlebot_voice_commands

# Train HMM
python -m puzzlebot_voice_commands.scripts.train_hmm `
  --dataset datasets\voice_commands_dataset_aug `
  --output-dir artifacts_final --n-iter 5 --syllable-states `
  --n-mfcc 20 --n-symbols 64 --cmvn --delta --librosa `
  --include-zcr --include-rms --include-contrast

# Evaluate
python -m puzzlebot_voice_commands.scripts.evaluate_models `
  --dataset datasets\voice_commands_dataset_aug `
  --artifact-dir artifacts_final --output-dir reports --model hmm

# Live test
python -m puzzlebot_voice_commands.scripts.live_test `
  --artifact-dir artifacts_final --models hmm

# Record new samples
python -m puzzlebot_voice_commands.scripts.grabar

# Augment dataset
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
| `/voice/confidence` | `std_msgs/Float32` | Log-likelihood margin |
| `/voice/status` | `std_msgs/String` | `listening` / `processing` / `idle` |
| `/voice/ranked_predictions` | `std_msgs/String` | JSON top-3 predictions |
| `/voice/inference_time_ms` | `std_msgs/Float32` | Inference latency in ms |

## Package structure

```
puzzlebot_voice_commands/
├── puzzlebot_voice_commands/
│   ├── config.py               — MFCCConfig, HMMConfig, DatasetConfig
│   ├── audio_io.py             — WAV loading, mono, normalization
│   ├── librosa_features.py     — librosa feature extraction (HMM)
│   ├── dataset.py              — leakage-free dataset split
│   ├── metrics.py              — metrics from scratch (no sklearn)
│   ├── voice_inference.py      — VoiceInferenceEngine (bridge integration)
│   ├── voice_commands_node.py  — ROS 2 inference node
│   ├── models/
│   │   └── hmm.py              — HiddenMarkovModelClassifier
│   └── scripts/
│       ├── grabar.py               — Record samples
│       ├── augment_dataset.py      — Data augmentation 4x
│       ├── train_hmm.py            — HMM training
│       ├── tune_hmm_per_class.py   — Per-class n_states+n_symbols gridsearch
│       ├── evaluate_models.py      — Evaluation + reports
│       └── live_test.py            — Mic → prediction
├── datasets/
├── artifacts_final/
├── artifacts_v2_85pct/
└── reports/
```

## HMM Parameter Report

Generates a PDF (max 3 pages) with A and B heatmaps at three training stages
(initial / mid / final) for up to 3 selected words. Does **not** touch `artifacts_final/`.

```powershell
python -m puzzlebot_voice_commands.scripts.generate_hmm_parameter_report `
  --dataset datasets\voice_commands_dataset_aug `
  --output-dir reports_hmm_parameters `
  --words alto avanzar retroceder `
  --n-symbols 256 --n-iter 20 `
  --n-mfcc 20 --delta --cmvn --librosa `
  --include-zcr --include-rms --include-contrast `
  --syllable-states --smoothing-eps 1e-6
```

Outputs: `reports_hmm_parameters/hmm_parameter_report.pdf`,
`figures/`, and `report_metadata.json`.

---

## Allowed libraries

NumPy, SciPy, librosa, and standard Python only. No scikit-learn, PyTorch, or TensorFlow.

## Implementation phases

| Phase | Content | Status |
|-------|---------|--------|
| 1–6 | Package structure, MFCC, KMeans, GNB, metrics, docs | Done |
| 7 | HMM from scratch (Baum-Welch + Viterbi) | Done |
| 8–8b | Feature engineering, augmentation, 4 speakers | Done |
| 9 | ROS 2 inference node + dashboard bridge integration | Done |
| **10** | **10 words, leakage-free split, gridsearch n_states×n_symbols** | **Done** |
