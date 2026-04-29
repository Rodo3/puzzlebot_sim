# puzzlebot_voice_commands

Offline voice command recognition package for the Puzzlebot ROS 2 workspace.

**Current phase:** 6 complete — all offline phases done.
**Status:** Full offline pipeline functional. VALIDATION.md checklist added. Phase 7+ (ROS 2 node, robot integration) is future work.

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

## Quick start

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
| 1     | Package structure, stubs, buildable skeleton             | **Done**    |
| 2     | Audio I/O, MFCC extraction, dataset split                | **Done**    |
| 3     | KMeansCodebookClassifier + training script               | **Done**    |
| 4     | GaussianNaiveBayesClassifier + training script           | **Done**    |
| 5     | Full metrics, report generation, model comparison        | **Done**    |
| 6     | Documentation cleanup, validation checklist              | **Done**    |
| 7+    | ROS 2 inference node, Puzzlebot integration              | Future      |

## What is implemented (Phases 1–5)

### `audio_io.py`
- `load_wav(path, target_sr)` — reads WAV, converts int16/int32/uint8 → float32, resamples with `scipy.signal.resample_poly`.
- `to_mono(signal)` — averages channels.
- `normalize(signal)` — peak-normalise to [-1, 1]; no-op on silent signals.

### `mfcc.py`
Full manual pipeline (no librosa):
1. Pre-emphasis `y[t] = x[t] - 0.97·x[t-1]`
2. Overlapping frames via NumPy stride tricks (zero-copy)
3. Hamming window
4. FFT + power spectrum `|FFT|² / n_fft`
5. Triangular Mel filterbank from scratch (26 filters, 0–8 kHz)
6. Log Mel energies (floor 1e-10)
7. DCT-II (`scipy.fft.dct`) → first 13 coefficients
8. `extract_mfcc_frames` → `(n_frames, 13)` — used by KMeans
9. `extract_mfcc_summary` → `(26,)` mean+std — used by GNB
10. Optional delta / delta-delta (off by default)

### `dataset.py`
- `discover_dataset(root)` — auto-discovers classes from subdirectories.
- `split_dataset` — stratified manual split with seeded RNG; ≥1 train and ≥1 test per class.
- `DatasetSplit.summary()` and `to_metadata_dict()`.

### `serialization.py`
- `save_pickle` / `load_pickle`, `save_json` / `load_json`, `artifact_size_kb`.

### `models/kmeans_codebook.py`
- `KMeansCodebookClassifier` — one K-Means codebook per class on frame-level MFCCs.
- Manual K-Means: random init → assign → recompute → empty-cluster reinit → convergence.
- Vectorised `_pairwise_sq_distances` via `||x-c||² = ||x||² + ||c||² - 2x·cᵀ`.
- `predict` → `(label, margin)`, `predict_ranked` → sorted list.
- `save` / `load` via pickle.

### `models/gaussian_nb.py`
- `GaussianNaiveBayesClassifier` — fixed-length MFCC summary vectors.
- Stores log priors, per-class means, per-class variances + epsilon smoothing.
- Inference: `log P(c|x) ∝ log P(c) + Σ log N(x_j | μ_cj, σ²_cj)`.
- `predict` → `(label, softmax_scores)`, `predict_ranked` → sorted by log-posterior.
- `save` / `load` via pickle.

### `metrics.py` (all from scratch, no sklearn)
`accuracy`, `confusion_matrix`, `precision_recall_f1`, `macro_f1`,
`per_command_accuracy`, `safety_critical_errors` (stop→movement + opposite directions),
`top2_accuracy`, `confidence_stats`.

### `reports.py`
- `save_confusion_matrix_csv` — CSV with class headers.
- `save_metrics_json` — pretty JSON.
- `generate_comparison_report` — 12-section Markdown: dataset, MFCC config, model configs,
  metrics table, per-class F1, confusion matrices, safety errors, inference times,
  artifact sizes, automated recommendation, limitations, next steps.

### CLI scripts

| Command | Status | Description |
|---|---|---|
| `prepare_voice_dataset` | Done | Discover → split → extract MFCCs → save JSON artifact |
| `train_voice_models` | Done | Train KMeans and/or GNB, save all artifacts |
| `evaluate_voice_models` | Done | Evaluate on test split, generate all 7 report files |
| `predict_voice_file` | Done | Single-file inference with ranked output |
