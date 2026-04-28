"""
Report generation for evaluation results.

Outputs (implemented in Phase 5):
- confusion_matrix_<model>.csv
- metrics_<model>.json
- safety_metrics.json
- inference_time.json
- model_comparison.md  (full Markdown report comparing both models)
"""
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def save_confusion_matrix_csv(
    matrix: np.ndarray,
    labels: List[str],
    path: Path,
) -> None:
    """Write confusion matrix to CSV. Implemented in Phase 5."""
    raise NotImplementedError(
        "reports.save_confusion_matrix_csv — implemented in Phase 5"
    )


def save_metrics_json(metrics: Dict[str, Any], path: Path) -> None:
    """Write metrics dict to JSON. Implemented in Phase 5."""
    raise NotImplementedError("reports.save_metrics_json — implemented in Phase 5")


def generate_comparison_report(
    kmeans_metrics: Dict[str, Any],
    gnb_metrics: Dict[str, Any],
    dataset_summary: Dict[str, Any],
    output_path: Path,
) -> None:
    """Write model_comparison.md with full evaluation narrative.

    Includes: dataset summary, MFCC config, model configs, metrics table,
    confusion matrix summary, safety errors, inference times, artifact sizes,
    recommendation, known limitations, and next steps.
    Implemented in Phase 5.
    """
    raise NotImplementedError(
        "reports.generate_comparison_report — implemented in Phase 5"
    )
