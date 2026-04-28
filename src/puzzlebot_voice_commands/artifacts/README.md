# artifacts/

Trained model files and feature configurations are written here by the training pipeline.

## Generated files

| File                  | Created by              | Description                                      |
|-----------------------|-------------------------|--------------------------------------------------|
| `kmeans_model.pkl`    | `train_voice_models`    | Serialized KMeansCodebookClassifier              |
| `gnb_model.pkl`       | `train_voice_models`    | Serialized GaussianNaiveBayesClassifier          |
| `labels.json`         | `train_voice_models`    | Ordered list of class labels used during training|
| `feature_config.json` | `train_voice_models`    | MFCC extraction parameters used                 |
| `train_metadata.json` | `train_voice_models`    | Split sizes, random seed, per-class counts       |
| `features.json`       | `prepare_voice_dataset` | Pre-extracted MFCC features (optional cache)     |

## Git

Artifact files (`*.pkl`, `*.json`) are **not** committed to the repository.
Re-generate them by running `train_voice_models` against your dataset.
