# Voice Command Recognition — Model Comparison Report

## 1. Dataset Summary

- **Root path:** `datasets\voice_commands_dataset_aug`
- **Classes:** 6
- **Total samples:** 1920
- **Train samples:** 1344
- **Test samples:** 576
- **Test ratio:** 0.3
- **Random state:** 42

### Samples per class

| Class | Total | Train | Test |
|-------|-------|-------|------|
| alto | 320 | 224 | 96 |
| avanzar | 320 | 224 | 96 |
| derecha | 320 | 224 | 96 |
| inicio | 320 | 224 | 96 |
| izquierda | 320 | 224 | 96 |
| retroceder | 320 | 224 | 96 |

## 2. MFCC Feature Configuration

- sample_rate: 16000 Hz
- pre_emphasis: 0.97
- frame_size: 0.025 s  (400 samples)
- frame_stride: 0.01 s  (160 samples)
- n_fft: 512
- n_filters: 26
- n_mfcc: 20
- include_delta: True
- include_delta_delta: False
- **Feature vector size:** 40 dimensions (mean + std)

## 3. Model Configurations

### KMeansCodebookClassifier
- n_clusters: 16 per class
- max_iter: 300
- tolerance: 0.0001
- random_state: 42
- Feature input: frame-level MFCCs  `(n_frames × 20)`

### GaussianNaiveBayesClassifier
- var_epsilon: 1e-09
- Feature input: MFCC summary vector  `(40,)`

## 4. Metrics Comparison

| Metric | KMeans | GaussianNB |
|--------|--------|-----------|
| Global accuracy | 0.9774 | 0.8941 |
| Macro recall | 0.9774 | 0.8941 |
| Macro F1 | 0.9774 | 0.8950 |
| Top-2 accuracy | 0.9983 | 0.9774 |
| Safety-critical errors | 0 | 5 |
| Safety-critical rate | 0.0000 | 0.0087 |
| Opposite-dir errors | 1 | 19 |
| Avg inference time (ms) | 0.5317 | 0.2190 |
| Artifact size (KB) | 15.5900 | 15.8800 |

## 5. Per-class Precision / Recall / F1

| Class | KM-P | KM-R | KM-F1 | GNB-P | GNB-R | GNB-F1 |
|-------|------|------|-------|-------|-------|--------|
| alto | 1.000 | 1.000 | 1.000 | 0.978 | 0.948 | 0.963 |
| avanzar | 1.000 | 1.000 | 1.000 | 0.959 | 0.979 | 0.969 |
| derecha | 1.000 | 0.990 | 0.995 | 0.750 | 0.875 | 0.808 |
| inicio | 1.000 | 0.875 | 0.933 | 0.924 | 0.885 | 0.904 |
| izquierda | 0.881 | 1.000 | 0.937 | 0.865 | 0.802 | 0.832 |
| retroceder | 1.000 | 1.000 | 1.000 | 0.913 | 0.875 | 0.894 |

## 6. Confusion Matrix Summary

### KMeans

| true \ pred | alto | avanzar | derecha | inicio | izquierda | retroceder |
|---|---|---|---|---|---|---|
| **alto** | 96 | 0 | 0 | 0 | 0 | 0 |
| **avanzar** | 0 | 96 | 0 | 0 | 0 | 0 |
| **derecha** | 0 | 0 | 95 | 0 | 1 | 0 |
| **inicio** | 0 | 0 | 0 | 84 | 12 | 0 |
| **izquierda** | 0 | 0 | 0 | 0 | 96 | 0 |
| **retroceder** | 0 | 0 | 0 | 0 | 0 | 96 |

### GaussianNB

| true \ pred | alto | avanzar | derecha | inicio | izquierda | retroceder |
|---|---|---|---|---|---|---|
| **alto** | 91 | 4 | 0 | 0 | 0 | 1 |
| **avanzar** | 1 | 94 | 0 | 0 | 0 | 1 |
| **derecha** | 0 | 0 | 84 | 1 | 5 | 6 |
| **inicio** | 0 | 0 | 5 | 85 | 6 | 0 |
| **izquierda** | 0 | 0 | 13 | 6 | 77 | 0 |
| **retroceder** | 1 | 0 | 10 | 0 | 1 | 84 |

## 7. Safety-Critical Errors

Safety-critical: a stop command (`alto`, `stop`) predicted as a movement command.
Opposite-direction: `adelante↔atras`, `izquierda↔derecha`.

### KMeans
- Safety-critical errors : **0**  (rate: 0.0000)
- Opposite-direction errors: **1**  (rate: 0.0017)
- Recall for `alto`: **1.0000**

### GaussianNB
- Safety-critical errors : **5**  (rate: 0.0087)
- Opposite-direction errors: **19**  (rate: 0.0330)
- Recall for `alto`: **0.9479**

  Safety-critical misclassifications:
  - true=`alto` → pred=`avanzar`
  - true=`alto` → pred=`retroceder`
  - true=`alto` → pred=`avanzar`
  - true=`alto` → pred=`avanzar`
  - true=`alto` → pred=`avanzar`

## 8. Inference Time

| Model | Avg (ms) | Std (ms) | Min (ms) | Max (ms) |
|-------|----------|----------|----------|----------|
| KMeans | 0.53 | 0.17 | 0.39 | 1.49 |
| GaussianNB | 0.22 | 0.07 | 0.15 | 0.62 |

## 9. Model Artifact Size

| Model | Size (KB) |
|-------|-----------|
| KMeans | 15.59 |
| GaussianNB | 15.88 |

## 10. Recommendation for ROS 2 Integration

- **Accuracy:** KMeans leads (KMeans 0.9774 vs GaussianNB 0.8941).
- **Safety:** KMeans has fewer safety-critical errors (KMeans 0 vs GaussianNB 5).
- **Speed:** GaussianNB is faster (KMeans 0.53 ms vs GaussianNB 0.22 ms).
- **Size:** KMeans is smaller (KMeans 15.6 KB vs GaussianNB 15.9 KB).

**KMeans** achieves notably higher accuracy. If safety-critical recall for `alto` is acceptable (≥ 0.95), it is the preferred model for ROS 2 integration.

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
