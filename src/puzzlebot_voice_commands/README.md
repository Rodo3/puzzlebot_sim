# puzzlebot_voice_commands

Offline voice command recognition package for the Puzzlebot ROS 2 workspace.

**Current phase:** 2 complete — audio I/O, MFCC pipeline, dataset split, and `prepare_voice_dataset` CLI are implemented.
**Next phase:** 3 — KMeansCodebookClassifier from scratch + training script.

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
| 3     | KMeansCodebookClassifier + training script               | **Next**    |
| 4     | GaussianNaiveBayesClassifier + training script           | Pending     |
| 5     | Full metrics, report generation, model comparison        | Pending     |
| 6     | Documentation cleanup, validation checklist              | Pending     |
| 7+    | ROS 2 inference node, Puzzlebot integration              | Future      |

## Phase 2 — What was implemented

### `audio_io.py`
- `load_wav(path, target_sr)` — reads WAV via `scipy.io.wavfile`, converts int16/int32/uint8 to float32, resamples to 16 kHz with `scipy.signal.resample_poly` if needed.
- `to_mono(signal)` — averages channels for stereo or multi-channel input.
- `normalize(signal)` — divides by peak absolute value; no-op on silent signals.

### `mfcc.py`
Full manual pipeline (no librosa):
1. Pre-emphasis `y[t] = x[t] - 0.97·x[t-1]`
2. Overlapping frames via NumPy stride tricks (zero-copy)
3. Hamming window per frame
4. FFT + power spectrum `|FFT|² / n_fft`
5. Triangular Mel filterbank built from scratch (26 filters, 0 Hz – 8 kHz)
6. Log of Mel energies (floor 1e-10)
7. DCT-II via `scipy.fft.dct` → first 13 coefficients
8. `extract_mfcc_frames(signal, config)` → `(n_frames, 13)` — used by KMeans
9. `extract_mfcc_summary(signal, config)` → `(26,)` mean+std — used by GNB
10. Optional delta / delta-delta (disabled by default)

### `dataset.py`
- `discover_dataset(root)` — auto-discovers classes from subdirectories, warns on empty folders.
- `split_dataset(samples_by_class, config)` — stratified manual split with seeded `random.Random`; guarantees ≥1 train and ≥1 test per class; warns on classes with <2 samples.
- `DatasetSplit.summary()` and `DatasetSplit.to_metadata_dict()` for reporting.

### `serialization.py`
- `save_pickle` / `load_pickle` — highest-protocol pickle with auto `mkdir`.
- `save_json` / `load_json` — pretty-printed UTF-8 JSON with auto `mkdir`.
- `artifact_size_kb` — file size helper.

### `scripts/prepare_dataset.py` (CLI: `prepare_voice_dataset`)
Full working CLI:
- Discovers dataset, prints per-class counts.
- Runs stratified split, prints summary table.
- Extracts MFCC mean+std for every sample (optionally includes raw frames with `--include-frames`).
- Saves a structured JSON artifact with config, split metadata, and feature vectors.
- Prints a validation summary (file size, record count, vector dimension).

## Phase 3 — What to implement next

**File:** `puzzlebot_voice_commands/models/kmeans_codebook.py`

**Class:** `KMeansCodebookClassifier`

Key points:
- One K-Means codebook trained per class on **frame-level** MFCC vectors `(N_frames, 13)`.
- K-Means implemented manually with NumPy (no sklearn): random init → assign → update → convergence check.
- Inference: average minimum Euclidean distance from each input frame to the nearest centroid in each codebook → predict the class with the lowest average distance.
- `decision_margin = second_best_distance - best_distance` (higher = more confident).
- `predict_ranked()` returns all classes sorted by distance.
- `save(path)` / `load(path)` via pickle (use `serialization.py`).
- Config: `KMeansConfig(n_clusters=16, max_iter=300, tolerance=1e-4, random_state=42)`.

**File:** `puzzlebot_voice_commands/scripts/train_models.py` (CLI: `train_voice_models`)

For Phase 3, implement the `--model kmeans` path:
- Discover dataset → split → for each train sample: `load_wav` → `normalize` → `extract_mfcc_frames`.
- Concatenate all frames per class into one large array.
- Call `KMeansCodebookClassifier.fit(frames_by_class)`.
- Save `kmeans_model.pkl`, `labels.json`, `feature_config.json`, `train_metadata.json` to `--output-dir`.

The `--model gnb` and `--model both` paths remain as stubs until Phase 4.
