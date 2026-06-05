# Voice Command Recognition — Model Comparison Report

## 1. Dataset Summary

- **Root path:** `datasets\voice_commands_dataset_aug`
- **Classes:** 10
- **Total samples:** 3200
- **Train samples:** 2240
- **Test samples:** 960
- **Test ratio:** 0.3
- **Random state:** 42

### Samples per class

| Class | Total | Train | Test |
|-------|-------|-------|------|
| alto | 320 | 224 | 96 |
| avanzar | 320 | 224 | 96 |
| bajar | 320 | 224 | 96 |
| derecha | 320 | 224 | 96 |
| inicio | 320 | 224 | 96 |
| izquierda | 320 | 224 | 96 |
| retroceder | 320 | 224 | 96 |
| soltar | 320 | 224 | 96 |
| subir | 320 | 224 | 96 |
| tomar | 320 | 224 | 96 |

## 2. MFCC Feature Configuration

- sample_rate: 16000 Hz
- pre_emphasis: 0.97
- frame_size: 0.025 s  (400 samples)
- frame_stride: 0.01 s  (160 samples)
- n_fft: 512
- n_filters: 26
- n_mfcc: 13
- include_delta: False
- include_delta_delta: False
- **Feature vector size:** 26 dimensions (mean + std)

## 3. Model Configurations

## 4. Metrics Comparison

| Metric | KMeans | GaussianNB |
|--------|--------|-----------|
| Global accuracy | N/A | N/A |
| Macro recall | N/A | N/A |
| Macro F1 | N/A | N/A |
| Top-2 accuracy | N/A | N/A |
| Safety-critical errors | N/A | N/A |
| Safety-critical rate | N/A | N/A |
| Opposite-dir errors | N/A | N/A |
| Avg inference time (ms) | N/A | N/A |
| Artifact size (KB) | N/A | N/A |

## 5. Per-class Precision / Recall / F1

| Class | KM-P | KM-R | KM-F1 | GNB-P | GNB-R | GNB-F1 |
|-------|------|------|-------|-------|-------|--------|
| alto | N/A | N/A | N/A | N/A | N/A | N/A |
| avanzar | N/A | N/A | N/A | N/A | N/A | N/A |
| bajar | N/A | N/A | N/A | N/A | N/A | N/A |
| derecha | N/A | N/A | N/A | N/A | N/A | N/A |
| inicio | N/A | N/A | N/A | N/A | N/A | N/A |
| izquierda | N/A | N/A | N/A | N/A | N/A | N/A |
| retroceder | N/A | N/A | N/A | N/A | N/A | N/A |
| soltar | N/A | N/A | N/A | N/A | N/A | N/A |
| subir | N/A | N/A | N/A | N/A | N/A | N/A |
| tomar | N/A | N/A | N/A | N/A | N/A | N/A |

## 6. Confusion Matrix Summary

## 7. Safety-Critical Errors

Safety-critical: a stop command (`alto`, `stop`) predicted as a movement command.
Opposite-direction: `adelante↔atras`, `izquierda↔derecha`.

## 8. Inference Time

| Model | Avg (ms) | Std (ms) | Min (ms) | Max (ms) |
|-------|----------|----------|----------|----------|
| KMeans | N/A | N/A | N/A | N/A |
| GaussianNB | N/A | N/A | N/A | N/A |

## 9. Model Artifact Size

| Model | Size (KB) |
|-------|-----------|
| KMeans | N/A |
| GaussianNB | N/A |

## 10. Recommendation for ROS 2 Integration

No models evaluated — run both models before generating this report.

## 11. Known Limitations

- Dataset size is small; metrics may not generalise well to unseen speakers.
- MFCC features capture spectral shape but not prosody or duration.
- KMeans codebook quality depends on having enough frames per class.
- GNB assumes feature independence given the class (Naive Bayes assumption).
- No data augmentation (noise, speed perturbation) was applied.
- Models were trained and tested on the same recording conditions.

## 12. Next Steps Before Connecting to Puzzlebot

1. Collect more diverse recordings (different speakers, microphones, distances).
2. Add noise augmentation to improve robustness in real environments.
3. Validate safety-critical recall ≥ 0.95 for `alto` before integration.
4. Implement a ROS 2 inference node (`voice_command_node.py`) in this package.
5. Publish recognised commands as `std_msgs/String` on `/voice_command` topic.
6. Add a confidence threshold: commands below threshold are ignored.
7. Wire the node into `puzzlebot_bringup` launch files.
