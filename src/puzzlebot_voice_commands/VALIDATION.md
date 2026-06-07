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
├── voice_commands_dataset/        — originals (800 clips, 4 speakers × 20/cmd × 10 cmds)
│   ├── alto/        *.wav
│   ├── avanzar/     *.wav
│   ├── bajar/       *.wav
│   ├── derecha/     *.wav
│   ├── inicio/      *.wav
│   ├── izquierda/   *.wav
│   ├── retroceder/  *.wav
│   ├── soltar/      *.wav
│   ├── subir/       *.wav
│   └── tomar/       *.wav
└── voice_commands_dataset_aug/    — augmented 4x (3200 clips)
```

---

## 3. Train HMM on augmented dataset

```powershell
python -m puzzlebot_voice_commands.scripts.train_hmm `
  --dataset datasets\voice_commands_dataset_aug `
  --output-dir artifacts_final --n-iter 5 --syllable-states `
  --n-mfcc 20 --n-symbols 64 --cmvn --delta --librosa `
  --include-zcr --include-rms --include-contrast
```

Expected artifacts: `hmm_model.pkl`, `hmm_config.json`. Train sanity accuracy ≥ 0.90.

---

## 4. Evaluate HMM

```powershell
python -m puzzlebot_voice_commands.scripts.evaluate_models `
  --dataset datasets\voice_commands_dataset_aug `
  --artifact-dir artifacts_final --output-dir reports --model hmm
```

Quality gates:

| Metric | Minimum |
|--------|---------|
| HMM accuracy | ≥ 0.87 |
| `alto` recall | ≥ 0.93 |
| Safety errors (alto→avanzar/retroceder) | ≤ 5 |

---

## 5. Live test (mic → prediction)

```powershell
python -m puzzlebot_voice_commands.scripts.live_test `
  --artifact-dir artifacts_final --models hmm
```

Press Enter to record 1.5 s. Say one command clearly.
Expected: correct label predicted. Inference < 100 ms.

---

## 6. Data augmentation (regenerate if needed)

```powershell
python -m puzzlebot_voice_commands.scripts.augment_dataset `
  --input-dir  datasets\voice_commands_dataset `
  --output-dir datasets\voice_commands_dataset_aug `
  --factor 3
```

Expected: 800 originals → 3200 total (4x).

---

## 7. ROS 2 node (WSL2)

```bash
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

---

## 8. Pre-integration checklist

- [ ] HMM accuracy ≥ 0.87 on held-out test set
- [ ] Safety-critical errors ≤ 5
- [ ] `artifacts_final/` contains `hmm_model.pkl` + `hmm_config.json`
- [ ] `voice_commands_node` builds and runs without error
- [ ] Node integrated to a `puzzlebot_bringup` launch file
