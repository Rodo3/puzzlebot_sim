# Validation Checklist — puzzlebot_voice_commands

Run all commands inside WSL2 from the workspace root (`~/puzzlebot_sim` or wherever
you cloned the repo). Source the ROS 2 environment first:

```bash
source /opt/ros/humble/setup.bash
```

---

## 1. Build

```bash
colcon build --packages-select puzzlebot_voice_commands
source install/setup.bash
```

**Expected:** build succeeds with no errors; four executables appear under
`install/puzzlebot_voice_commands/lib/puzzlebot_voice_commands/`.

---

## 2. Dataset layout

Place recordings under:

```
src/puzzlebot_voice_commands/datasets/voice_commands_dataset/
├── adelante/   *.wav
├── atras/      *.wav
├── izquierda/  *.wav
├── derecha/    *.wav
├── alto/       *.wav
└── inicio/     *.wav
```

Each class needs **at least 2 `.wav` files** (one train, one test).
Recommended minimum: 10 files per class for meaningful metrics.

---

## 3. Prepare dataset (optional feature pre-extraction)

```bash
ros2 run puzzlebot_voice_commands prepare_voice_dataset \
  --dataset src/puzzlebot_voice_commands/datasets/voice_commands_dataset \
  --output  src/puzzlebot_voice_commands/artifacts/features.json
```

**Expected output (last lines):**
```
--- Validation summary ---
  Output file   : .../features.json
  File size     : <N> KB
  Total records : <N>
  Labels        : ['adelante', 'alto', 'atras', 'derecha', 'inicio', 'izquierda']
  MFCC vector   : 26D  (mean + std of 13 coefficients)
OK
```

**Check:** `artifacts/features.json` exists and `"labels"` contains all 6 commands.

---

## 4. Train both models

```bash
ros2 run puzzlebot_voice_commands train_voice_models \
  --dataset    src/puzzlebot_voice_commands/datasets/voice_commands_dataset \
  --model      both \
  --output-dir src/puzzlebot_voice_commands/artifacts
```

**Expected output (last lines):**
```
--- Training summary ---
  Dataset       : ...
  Labels        : ['adelante', 'alto', 'atras', 'derecha', 'inicio', 'izquierda']
  Train samples : <N>
  Test samples  : <N>
  Output dir    : .../artifacts
  kmeans_model  : .../artifacts/kmeans_model.pkl
  gnb_model     : .../artifacts/gnb_model.pkl
Done.
```

**Check:** the following files exist in `artifacts/`:

| File | Required |
|------|----------|
| `kmeans_model.pkl` | yes |
| `gnb_model.pkl` | yes |
| `labels.json` | yes |
| `feature_config.json` | yes |
| `train_metadata.json` | yes |

---

## 5. Evaluate models and generate reports

```bash
ros2 run puzzlebot_voice_commands evaluate_voice_models \
  --dataset      src/puzzlebot_voice_commands/datasets/voice_commands_dataset \
  --artifact-dir src/puzzlebot_voice_commands/artifacts \
  --output-dir   src/puzzlebot_voice_commands/reports
```

**Expected output (last lines):**
```
--- Evaluation complete ---
  Output dir : .../reports
    confusion_matrix_gnb.csv
    confusion_matrix_kmeans.csv
    inference_time.json
    metrics_gnb.json
    metrics_kmeans.json
    model_comparison.md
    safety_metrics.json
```

**Check:** all 7 report files exist in `reports/`.

**Quality gates (adjust thresholds to your dataset size):**

| Metric | Minimum acceptable |
|--------|--------------------|
| Global accuracy (either model) | ≥ 0.70 |
| `alto` recall (stop command) | ≥ 0.90 |
| Safety-critical errors | 0 (ideally) |
| Opposite-direction errors | 0 (ideally) |

Open `reports/model_comparison.md` and verify:
- Section 4 (Metrics Comparison) has numeric values for both models.
- Section 7 (Safety-Critical Errors) lists zero or few cases.
- Section 10 (Recommendation) produces a concrete recommendation text.

---

## 6. Single-file prediction — KMeans

```bash
ros2 run puzzlebot_voice_commands predict_voice_file \
  --model-type kmeans \
  --model-path src/puzzlebot_voice_commands/artifacts/kmeans_model.pkl \
  --audio      <path/to/any/adelante_XX.wav>
```

**Expected output:**
```
Audio             : ...
Model             : KMeans  (kmeans_model.pkl)
Predicted command : adelante
Decision margin   : <positive float>  (higher = more confident)
Inference time    : <N> ms

Ranked predictions:
  1. adelante         avg_min_dist=<lowest>
  2. ...
```

**Check:** predicted command matches the file's label; inference time is < 100 ms.

---

## 7. Single-file prediction — GaussianNB

```bash
ros2 run puzzlebot_voice_commands predict_voice_file \
  --model-type gnb \
  --model-path src/puzzlebot_voice_commands/artifacts/gnb_model.pkl \
  --audio      <path/to/any/alto_XX.wav>
```

**Expected output:**
```
Audio             : ...
Model             : GaussianNB  (gnb_model.pkl)
Predicted command : alto
Confidence        : <float close to 1.0>  (softmax-normalised)
Inference time    : <N> ms

Ranked predictions:
  1. alto             log_posterior=<highest>  score=<highest>
  2. ...
```

**Check:** confidence for the correct label is the highest score; the ranked list
contains all 6 labels.

---

## 8. Reproducibility check

Re-run training a second time and verify:
- `train_metadata.json` contains the same `test_ratio`, `random_state`, and
  `train_per_class` / `test_per_class` counts as the first run.
- Evaluation accuracy is identical to the first run (deterministic split + seed).

---

## 9. Before Puzzlebot integration (pre-requisites)

These items must be true before wiring the package into the robot:

- [ ] `alto` recall ≥ 0.95 on a held-out test set with diverse speakers.
- [ ] Safety-critical errors = 0 on the same test set.
- [ ] Dataset collected from the actual deployment microphone / environment.
- [ ] `model_comparison.md` Section 10 recommends a specific model.
- [ ] ROS 2 inference node (`voice_command_node.py`) implemented and reviewed.
- [ ] Node publishes `std_msgs/String` on `/voice_command` topic.
- [ ] Confidence threshold wired in (commands below threshold are ignored).
- [ ] Node added to a `puzzlebot_bringup` launch file.
