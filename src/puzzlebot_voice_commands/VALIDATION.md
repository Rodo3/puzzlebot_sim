# Validation Checklist — puzzlebot_voice_commands

Run all commands from the package root (`src/puzzlebot_voice_commands/`) on Windows,
or from the workspace root on WSL2 after sourcing ROS 2.

---

## 1. Build (WSL2)

```bash
colcon build --packages-select puzzlebot_voice_commands
source install/setup.bash
```

Expected: build succeeds; `voice_commands_node` appears under
`install/puzzlebot_voice_commands/lib/puzzlebot_voice_commands/`.

---

## 2. Dataset layout

```
datasets/
├── voice_commands_dataset/        — original (480 clips, 4 speakers × 20/cmd)
│   ├── alto/        *.wav
│   ├── avanzar/     *.wav
│   ├── derecha/     *.wav
│   ├── inicio/      *.wav
│   ├── izquierda/   *.wav
│   └── retroceder/  *.wav
└── voice_commands_dataset_aug/    — augmented 4x (1920 clips)
```

---

## 3. Train KMeans + GNB on augmented dataset

```powershell
python -m puzzlebot_voice_commands.scripts.train_models `
  --dataset datasets\voice_commands_dataset_aug `
  --model both --output-dir artifacts `
  --n-mfcc 20 --delta --min-max --kmeans-delta --kmeans-cmvn
```

Expected artifacts: `kmeans_model.pkl`, `gnb_model.pkl`, `feature_config.json`,
`kmeans_feature_config.json`, `labels.json`, `train_metadata.json`.

---

## 4. Train HMM librosa + syllable-states on augmented dataset

```powershell
python -m puzzlebot_voice_commands.scripts.train_hmm `
  --dataset datasets\voice_commands_dataset_aug `
  --output-dir artifacts `
  --n-mfcc 20 --n-states 5 --n-symbols 32 --n-iter 20 `
  --cmvn --delta --librosa --include-zcr --include-rms --include-contrast `
  --syllable-states
```

Expected: `hmm_model.pkl`, `hmm_config.json`. Train sanity accuracy ≥ 0.90.

---

## 5. Evaluate all models

```powershell
python -m puzzlebot_voice_commands.scripts.evaluate_models `
  --dataset datasets\voice_commands_dataset_aug `
  --artifact-dir artifacts --output-dir reports --model all
```

Quality gates:

| Metric | Minimum |
|--------|---------|
| KMeans accuracy | ≥ 0.95 |
| HMM accuracy | ≥ 0.88 |
| `alto` recall (any model) | ≥ 0.95 |
| KMeans safety errors | 0 |

---

## 6. Live test (mic → prediction)

```powershell
python -m puzzlebot_voice_commands.scripts.live_test `
  --artifact-dir artifacts_final --models kmeans hmm
```

Press Enter to record 1.5 s. Say one command clearly.
Expected: correct label predicted with positive margin. Inference < 100 ms for KMeans.

---

## 7. Data augmentation (regenerate if needed)

```powershell
python -m puzzlebot_voice_commands.scripts.augment_dataset `
  --input-dir  datasets\voice_commands_dataset `
  --output-dir datasets\voice_commands_dataset_aug `
  --factor 3
```

Expected: 480 originals → 1920 total (4x).

---

## 8. ROS 2 node (Phase 9)

```bash
# WSL2
ros2 run puzzlebot_voice_commands voice_commands_node \
  --ros-args -p artifact_dir:=src/puzzlebot_voice_commands/artifacts_final

# Trigger a recording from another terminal
ros2 topic pub --once /voice/trigger std_msgs/String "data: 'record'"

# Monitor predictions
ros2 topic echo /voice/command
ros2 topic echo /voice/confidence
ros2 topic echo /voice/ranked_predictions
```

Quality gates:
- [ ] `/voice/command` publishes within 3 s of trigger
- [ ] `/voice/status` transitions: `idle` → `listening` → `processing` → `idle`
- [ ] `alto` not predicted when silence is recorded
- [ ] Confidence threshold filters out low-confidence predictions

---

## 9. Pre-integration checklist

- [ ] `alto` recall ≥ 0.95 on held-out test set
- [ ] Safety-critical errors = 0
- [ ] `artifacts_final/` contains `hmm_model.pkl` + `kmeans_model.pkl`
- [ ] `voice_commands_node` builds and runs without error
- [ ] Node added to a `puzzlebot_bringup` launch file
- [ ] Confidence threshold configured (commands below threshold → no publish)
