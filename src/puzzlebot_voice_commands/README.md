# puzzlebot_voice_commands

Offline voice command recognition package for the Puzzlebot ROS 2 workspace.

**Current phase:** 8 complete — feature engineering (delta MFCCs, CMVN, min/max stats, per-model configs).
**Status:** Full pipeline functional. Dataset collection in progress (3/4 speakers recorded).
Recommended model: **GaussianNB** (97.7% CV accuracy, 0 safety errors, 0.17 ms inference).

## Purpose

Train and evaluate voice command classifiers using `.wav` audio files and
hand-crafted MFCC features. Two models are implemented from scratch:

| Model | Feature input | Approach | Config flag |
|-------|---------------|----------|-------------|
| `KMeansCodebookClassifier` | Frame-level MFCCs | One K-Means codebook per class (VQ-style) | `--kmeans-delta --kmeans-cmvn` |
| `GaussianNaiveBayesClassifier` | MFCC summary vector | Gaussian log-likelihood + class prior | `--delta --min-max` |
| `HiddenMarkovModelClassifier` | Frame-level MFCCs | Left-to-right HMM per class, Baum-Welch + Viterbi | `--cmvn` |

Each model uses its own MFCC feature configuration, stored independently in `artifacts/`.

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

# 3. Train both models (independent feature configs per model)
python -m puzzlebot_voice_commands.scripts.train_models `
  --dataset    datasets\voice_commands_dataset `
  --model      both `
  --output-dir artifacts `
  --delta --min-max `
  --kmeans-delta --kmeans-cmvn

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

# 8. (Optional) Grid search for best HMM hyperparameters
python -m puzzlebot_voice_commands.scripts.tune_hmm `
  --dataset   datasets\voice_commands_dataset `
  --n-states  5 8 10 --n-symbols 32 64 --n-iter 20 50 `
  --k 3 --cmvn

# 9. Train HMM with best config (default: n_states=8, n_symbols=64, n_iter=50)
python -m puzzlebot_voice_commands.scripts.train_hmm `
  --dataset    datasets\voice_commands_dataset `
  --output-dir artifacts `
  --n-states 8 --n-symbols 64 --n-iter 50 --cmvn

# 10. Evaluate all three models
python -m puzzlebot_voice_commands.scripts.evaluate_models `
  --dataset      datasets\voice_commands_dataset `
  --artifact-dir artifacts `
  --output-dir   reports `
  --model        all

# 11. Predict a single file
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
│   │   ├── gaussian_nb.py      — GaussianNaiveBayesClassifier
│   │   └── hmm.py              — HiddenMarkovModelClassifier
│   └── scripts/
│       ├── grabar.py           — CLI: record samples interactively
│       ├── merge_datasets.py   — CLI: merge per-person folders into one dataset
│       ├── prepare_dataset.py  — CLI: prepare_voice_dataset
│       ├── train_models.py     — CLI: train_voice_models  (--delta --min-max --kmeans-delta --kmeans-cmvn)
│       ├── train_hmm.py        — CLI: train_hmm_models    (--cmvn --n-states --n-symbols --n-iter)
│       ├── tune_hmm.py         — CLI: tune_hmm_models     (grid search over HMM hyperparameters)
│       ├── evaluate_models.py  — CLI: evaluate_voice_models (loads per-model MFCC configs)
│       ├── predict_file.py     — CLI: predict_voice_file
│       ├── cross_validate.py   — CLI: k-fold cross-validation (--delta --min-max --kmeans-delta --kmeans-cmvn)
│       ├── learning_curve.py   — CLI: accuracy vs training size curve
│       └── speaker_test.py     — CLI: per-speaker evaluation
├── datasets/               — Per-person folders + merged dataset (not committed)
├── artifacts/              — Trained models and configs (not committed)
│   ├── feature_config.json         — GNB MFCC config (delta + min/max)
│   ├── kmeans_feature_config.json  — KMeans MFCC config (delta + cmvn)
│   └── hmm_config.json             — HMM params + MFCC config (cmvn)
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
├── data_rodo/        — 20 clips/class ✓
├── data_<person4>/   — pending
└── voice_commands_dataset/  — merged output (auto-generated by merge_datasets)
```

**Recommended clip count:** 20 per person per command.
After recording all 4, run `speaker_test --mode leave-one-out` to check if any
person needs more recordings (target: recall ≥ 0.90 per class per speaker).

## Evaluation results (3 speakers, 2026-05-12)

Dataset: Jorge + Valeria + Rodo, 50 clips/class, 6 classes, 300 total samples.
Each model uses its own MFCC feature config (see Phase 8).

### Model comparison

| Metric | KMeans | **GaussianNB** | HMM |
|--------|--------|----------------|-----|
| Test accuracy | **100%** | **100%** | 70.0% |
| Macro recall | **100%** | **100%** | 70.0% |
| Macro F1 | **100%** | **100%** | 70.3% |
| Top-2 accuracy | **100%** | **100%** | 81.1% |
| Safety errors (`alto`) | **0** | **0** | 5 |
| Avg inference | 0.46 ms | **0.17 ms** | 39.7 ms |
| Artifact size | 10.3 KB | 10.6 KB | 31.9 KB |
| Feature config | delta + cmvn | delta + min/max | cmvn |

### Cross-validation (k=5)

| Metric | KMeans | **GaussianNB** |
|--------|--------|----------------|
| Acc mean ± std | **99.7% ± 0.7%** | 97.7% ± 2.0% |
| Macro recall mean | **99.7%** | 97.6% |
| Safety errors (total) | **0** | **0** |

Std on GNB (2.0%) is expected with only 3 speakers — will decrease with 4.

**GaussianNB is the recommended model for ROS 2 integration** (fastest inference, 0 safety errors).

## Phase 7 — HMM (Hidden Markov Model)

A third classifier will be added using a discrete-observation HMM trained on
frame-level MFCCs — implemented entirely from scratch with NumPy only.

### Why HMM?
- KMeans and GNB treat each audio sample as a static feature vector, ignoring
  temporal dynamics (how a word evolves over time).
- HMM explicitly models the sequence of MFCC frames as transitions between
  hidden states, which is the classical approach for speech recognition.
- Expected to outperform KMeans and GNB on commands with similar spectral
  content but different duration/rhythm (e.g. `avanzar` vs `retroceder`).

### Design (from scratch, NumPy only)

| Component | Description |
|-----------|-------------|
| Observation quantization | K-Means codebook (reuse existing) to map MFCC frames → discrete symbols |
| HMM topology | Left-to-right (Bakis) — states flow forward only, matching speech progression |
| Training | Baum-Welch algorithm (EM) — forward-backward to estimate A, B, π |
| Inference | Viterbi algorithm — most likely state sequence → log-likelihood score |
| Classifier | One HMM per class; argmax of log-likelihoods across all models |
| Parameters | `n_states` (default 5), `n_iter` (default 20), `n_symbols` from codebook |

### Files added

```
models/
└── hmm.py              — HiddenMarkovModel + Baum-Welch + Viterbi
scripts/
└── train_hmm.py        — CLI: train_hmm_models  (saves hmm_model.pkl)
```

`evaluate_models.py`, `predict_file.py`, `cross_validate.py`, `learning_curve.py`,
and `speaker_test.py` now support `--model hmm` and `--model all`.

### Results (3 speakers, tuned params: n_states=8, n_symbols=64, n_iter=50, --cmvn)

| Metric | KMeans | GaussianNB | HMM |
|--------|--------|------------|-----|
| Test accuracy | 100% | **100%** | 70.0% |
| Safety errors | 0 | **0** | 5 |
| Avg inference | 0.46 ms | **0.17 ms** | 39.7 ms |

HMM still underperforms; the primary bottleneck is the 4th speaker (more training diversity).
Use `tune_hmm_models` to find the best `n_states`/`n_symbols`/`n_iter` for your dataset.

## Phase 8 — Feature engineering

Per-model MFCC configurations and new feature options implemented in this phase.

### New MFCCConfig flags

| Flag | Affects | Effect |
|------|---------|--------|
| `--delta` / `--kmeans-delta` | GNB / KMeans | Append velocity (Δ) coefficients to MFCC frames |
| `--delta-delta` / `--kmeans-delta-delta` | GNB / KMeans | Append acceleration (ΔΔ) coefficients |
| `--cmvn` / `--kmeans-cmvn` | GNB / KMeans | Per-utterance cepstral mean-variance normalization |
| `--min-max` | GNB only | Append per-coefficient min and max to summary vector |

### Why separate configs?

CMVN normalizes each utterance to zero mean — useful for frame-level models (KMeans, HMM)
where speaker-level offsets cause codebook mismatch. For GNB, which classifies from a
summary vector of [mean, std, min, max], CMVN collapses mean≈0 and std≈1 for all classes,
destroying discriminative signal.

### Per-model optimal config (3 speakers)

| Model | Flags | Feature dim | CV acc (k=5) |
|-------|-------|-------------|-------------|
| **GNB** | `--delta --min-max` | 104 (mean+std+min+max × 26) | 97.7% ± 2.0% |
| **KMeans** | `--kmeans-delta --kmeans-cmvn` | 26 frames | 99.7% ± 0.7% |
| **HMM** | `--cmvn` | 13 frames | — |

### Artifact files

Each model's config is saved independently so `evaluate_models` uses the right features:

```
artifacts/
├── feature_config.json         — GNB MFCC config
├── kmeans_feature_config.json  — KMeans MFCC config (may differ from GNB)
└── hmm_config.json             — HMM params + embedded MFCC config
```

### New script: `tune_hmm_models`

Grid search over `n_states × n_symbols × n_iter` using k-fold CV:

```powershell
python -m puzzlebot_voice_commands.scripts.tune_hmm `
  --dataset   datasets\voice_commands_dataset `
  --n-states  5 8 10 --n-symbols 32 64 --n-iter 20 50 `
  --k 3 --cmvn
```

Prints results sorted by mean accuracy and shows the exact `train_hmm` command for the best config.

## Implementation phases

| Phase | Content | Status |
|-------|---------|--------|
| 1 | Package structure, stubs, buildable skeleton | **Done** |
| 2 | Audio I/O, MFCC extraction, dataset split | **Done** |
| 3 | KMeansCodebookClassifier + training script | **Done** |
| 4 | GaussianNaiveBayesClassifier + training script | **Done** |
| 5 | Full metrics, report generation, model comparison | **Done** |
| 6 | Documentation cleanup, validation checklist | **Done** |
| 7 | Hidden Markov Model (HMM) classifier from scratch | **Done** |
| 8 | Feature engineering: delta, CMVN, min/max, per-model configs, HMM grid search | **Done** |
| 9+ | ROS 2 inference node, Puzzlebot integration | Future |

## CLI scripts

| Command | Mode | Description |
|---------|------|-------------|
| `grabar` | standalone | Record 20 clips per command interactively |
| `merge_voice_datasets` | standalone | Merge per-person folders into one dataset |
| `prepare_voice_dataset` | ROS 2 / standalone | Discover → split → extract MFCCs → JSON |
| `train_voice_models` | ROS 2 / standalone | Train KMeans and/or GNB with independent feature configs |
| `train_hmm_models` | ROS 2 / standalone | Train HMM classifiers, save `hmm_model.pkl` and MFCC config |
| `tune_hmm_models` | standalone | Grid search over n_states × n_symbols × n_iter via k-fold CV |
| `evaluate_voice_models` | ROS 2 / standalone | Evaluate all models, each with its own MFCC config |
| `predict_voice_file` | ROS 2 / standalone | Single-file inference with ranked output |
| `cross_validate_voice` | standalone | K-fold CV with independent KMeans/GNB feature flags |
| `learning_curve_voice` | standalone | Accuracy vs training size, detects overfitting |
| `speaker_test_voice` | standalone | Per-speaker recall, all-train or leave-one-out |
