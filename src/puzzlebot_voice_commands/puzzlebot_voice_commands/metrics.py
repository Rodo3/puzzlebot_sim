"""
All evaluation metrics implemented from scratch using NumPy only.
No sklearn.metrics allowed.

Metrics provided:
- accuracy                  — global correct / total
- confusion_matrix          — (n_classes, n_classes) count matrix
- precision_recall_f1       — per-class precision, recall, F1
- macro_f1                  — unweighted mean F1 across classes
- per_command_accuracy      — per-class accuracy derived from confusion matrix
- safety_critical_errors    — alto/stop predicted as movement; opposite directions
- safety_critical_error_rate
- top2_accuracy             — true label appears in top-2 ranked predictions
- confidence_stats          — mean/std/min/max of decision margins or softmax scores
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Commands treated as safety-critical (stop / halt)
# ---------------------------------------------------------------------------
_STOP_COMMANDS = {'alto', 'stop', 'pare', 'halt'}

# Movement commands (a stop command predicted as one of these = safety error)
_MOVE_COMMANDS = {'avanzar', 'retroceder', 'izquierda', 'derecha',
                  'adelante', 'atras', 'forward', 'backward', 'left', 'right'}

# Opposite-direction pairs (either direction counts as a diagnostic error)
_OPPOSITE_PAIRS = [
    ('avanzar', 'retroceder'),
    ('retroceder', 'avanzar'),
    ('izquierda', 'derecha'),
    ('derecha', 'izquierda'),
    ('adelante', 'atras'),
    ('atras', 'adelante'),
    ('forward', 'backward'),
    ('backward', 'forward'),
    ('left', 'right'),
    ('right', 'left'),
]


def accuracy(y_true: List[str], y_pred: List[str]) -> float:
    """Global classification accuracy: correct predictions / total samples."""
    if len(y_true) == 0:
        return 0.0
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true)


def confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> np.ndarray:
    """Build a confusion matrix of shape (n_classes, n_classes).

    Row index = true class, column index = predicted class.
    Order follows the `labels` list.
    """
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    n = len(labels)
    matrix = np.zeros((n, n), dtype=np.int64)

    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            matrix[label_to_idx[t], label_to_idx[p]] += 1

    return matrix


def precision_recall_f1(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute per-class precision, recall, and F1-score.

    Definitions (for class c):
      TP = matrix[c, c]
      FP = sum(matrix[:, c]) - TP   (column sum minus diagonal)
      FN = sum(matrix[c, :]) - TP   (row sum minus diagonal)

      precision = TP / (TP + FP)   if (TP + FP) > 0 else 0
      recall    = TP / (TP + FN)   if (TP + FN) > 0 else 0
      f1        = 2 * P * R / (P + R) if (P + R) > 0 else 0

    Returns dict: {label: {'precision': float, 'recall': float, 'f1': float}}
    """
    matrix = confusion_matrix(y_true, y_pred, labels)
    result: Dict[str, Dict[str, float]] = {}

    for i, label in enumerate(labels):
        tp = int(matrix[i, i])
        fp = int(matrix[:, i].sum()) - tp
        fn = int(matrix[i, :].sum()) - tp

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * p * r / (p + r) if (p + r) > 0.0 else 0.0

        result[label] = {'precision': p, 'recall': r, 'f1': f1}

    return result


def macro_f1(prf_dict: Dict[str, Dict[str, float]]) -> float:
    """Macro-averaged F1: unweighted mean of per-class F1 scores."""
    if not prf_dict:
        return 0.0
    return float(np.mean([v['f1'] for v in prf_dict.values()]))


def per_command_accuracy(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> Dict[str, float]:
    """Per-class accuracy: TP / total_true for each class.

    Equivalent to the recall column, provided separately for clarity.
    """
    matrix = confusion_matrix(y_true, y_pred, labels)
    result: Dict[str, float] = {}

    for i, label in enumerate(labels):
        row_sum = int(matrix[i, :].sum())
        tp = int(matrix[i, i])
        result[label] = tp / row_sum if row_sum > 0 else 0.0

    return result


def safety_critical_errors(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, Any]:
    """Count and describe safety-critical and opposite-direction errors.

    Safety-critical error:
        True label is a stop command (alto, stop, …) but model predicts
        a movement command (adelante, atras, izquierda, derecha, …).

    Opposite-direction error:
        adelante → atras, atras → adelante,
        izquierda → derecha, derecha → izquierda.

    Returns a dict with:
        safety_critical_count   — int
        safety_critical_rate    — float (fraction of total samples)
        safety_critical_cases   — list of {true, pred} dicts
        opposite_direction_count
        opposite_direction_rate
        opposite_direction_cases
        stop_recall             — recall for each stop command present in y_true
    """
    n = len(y_true)
    safety_cases = []
    opposite_cases = []

    for t, p in zip(y_true, y_pred):
        t_low = t.lower()
        p_low = p.lower()

        if t_low in _STOP_COMMANDS and p_low in _MOVE_COMMANDS:
            safety_cases.append({'true': t, 'pred': p})

        if (t_low, p_low) in _OPPOSITE_PAIRS:
            opposite_cases.append({'true': t, 'pred': p})

    # Per-stop-command recall
    stop_recall: Dict[str, float] = {}
    present_stops = {t.lower() for t in y_true if t.lower() in _STOP_COMMANDS}
    for cmd in present_stops:
        total = sum(1 for t in y_true if t.lower() == cmd)
        correct = sum(1 for t, p in zip(y_true, y_pred)
                      if t.lower() == cmd and p.lower() == cmd)
        stop_recall[cmd] = correct / total if total > 0 else 0.0

    return {
        'safety_critical_count': len(safety_cases),
        'safety_critical_rate': len(safety_cases) / n if n > 0 else 0.0,
        'safety_critical_cases': safety_cases,
        'opposite_direction_count': len(opposite_cases),
        'opposite_direction_rate': len(opposite_cases) / n if n > 0 else 0.0,
        'opposite_direction_cases': opposite_cases,
        'stop_recall': stop_recall,
    }


def top2_accuracy(
    y_true: List[str],
    ranked_preds: List[List[str]],
) -> float:
    """Top-2 accuracy: fraction of samples where the true label appears in the
    top-2 ranked predictions.

    Args:
        y_true:        list of true string labels
        ranked_preds:  list of ranked prediction lists (best first),
                       one list per sample — only the first two entries are used.
    """
    if len(y_true) == 0:
        return 0.0
    correct = sum(
        1 for t, ranked in zip(y_true, ranked_preds)
        if t in ranked[:2]
    )
    return correct / len(y_true)


def confidence_stats(scores: List[float]) -> Dict[str, float]:
    """Summarise a list of confidence or margin values.

    Returns mean, std, min, max.
    """
    if not scores:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    arr = np.array(scores, dtype=np.float64)
    return {
        'mean': float(arr.mean()),
        'std':  float(arr.std()),
        'min':  float(arr.min()),
        'max':  float(arr.max()),
    }
