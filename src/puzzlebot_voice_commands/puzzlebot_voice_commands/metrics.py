"""
All evaluation metrics implemented from scratch using NumPy only.
No sklearn.metrics allowed.

Metrics (implemented in Phase 5):
- Global accuracy
- Confusion matrix
- Precision, Recall, F1 per class
- Macro F1
- Per-command accuracy
- Safety-critical recall (alto, stop)
- Safety-critical error rate
- Opposite-direction error rate (adelante<->atras, izquierda<->derecha)
- Top-2 accuracy
- Confidence / decision margin statistics
- Average inference time
- Model artifact size in KB
"""
from typing import Dict, List, Optional, Tuple

import numpy as np


def accuracy(y_true: List[str], y_pred: List[str]) -> float:
    """Global classification accuracy. Implemented in Phase 5."""
    raise NotImplementedError("metrics.accuracy — implemented in Phase 5")


def confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> np.ndarray:
    """Confusion matrix of shape (n_classes, n_classes). Implemented in Phase 5."""
    raise NotImplementedError("metrics.confusion_matrix — implemented in Phase 5")


def precision_recall_f1(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> Dict[str, Dict[str, float]]:
    """Per-class precision, recall, and F1. Returns dict keyed by label.

    Implemented in Phase 5.
    """
    raise NotImplementedError("metrics.precision_recall_f1 — implemented in Phase 5")


def macro_f1(prf_dict: Dict[str, Dict[str, float]]) -> float:
    """Macro-averaged F1 score across all classes. Implemented in Phase 5."""
    raise NotImplementedError("metrics.macro_f1 — implemented in Phase 5")


def safety_critical_errors(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, int]:
    """Count safety-critical prediction errors (e.g., alto -> adelante).

    Safety-critical: stop/alto predicted as a movement command.
    Opposite-direction: adelante<->atras, izquierda<->derecha.
    Implemented in Phase 5.
    """
    raise NotImplementedError(
        "metrics.safety_critical_errors — implemented in Phase 5"
    )


def top2_accuracy(
    y_true: List[str],
    ranked_preds: List[List[str]],
) -> float:
    """Top-2 accuracy: true label in top-2 predictions. Implemented in Phase 5."""
    raise NotImplementedError("metrics.top2_accuracy — implemented in Phase 5")
