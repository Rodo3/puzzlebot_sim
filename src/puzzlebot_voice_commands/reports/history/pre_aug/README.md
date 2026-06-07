# reports/

Evaluation reports are written here by the `evaluate_voice_models` script.

## Generated files

| File                          | Description                                           |
|-------------------------------|-------------------------------------------------------|
| `confusion_matrix_kmeans.csv` | Confusion matrix for the KMeans codebook model        |
| `confusion_matrix_gnb.csv`    | Confusion matrix for the Gaussian NB model            |
| `metrics_kmeans.json`         | Full per-class and aggregate metrics for KMeans       |
| `metrics_gnb.json`            | Full per-class and aggregate metrics for GNB          |
| `safety_metrics.json`         | Safety-critical and opposite-direction error counts   |
| `inference_time.json`         | Average inference time per model and per audio file   |
| `model_comparison.md`         | Markdown report comparing both models end-to-end      |

## Git

Report files are **not** committed to the repository.
Re-generate them by running `evaluate_voice_models` against your dataset and artifacts.
