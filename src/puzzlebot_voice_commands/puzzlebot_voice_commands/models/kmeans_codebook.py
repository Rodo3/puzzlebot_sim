"""
KMeansCodebookClassifier — vector-quantization-style spoken word classifier.

Approach:
- One K-Means codebook trained per class on frame-level MFCC vectors.
- Inference: for each frame, find the nearest centroid in each class codebook.
  Predict the class with the lowest average minimum Euclidean distance.
- Decision margin = second_best_distance - best_distance (higher = more confident).

K-Means is implemented manually with NumPy (no sklearn).

Implemented in Phase 3.
"""
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import KMeansConfig


@dataclass
class KMeansCodebookClassifier:
    """Classifier that trains one K-Means codebook per class."""

    config: KMeansConfig = field(default_factory=KMeansConfig)
    labels_: List[str] = field(default_factory=list, init=False)
    codebooks_: Dict[str, np.ndarray] = field(default_factory=dict, init=False)
    is_fitted_: bool = field(default=False, init=False)

    def fit(
        self,
        frames_by_class: Dict[str, np.ndarray],
    ) -> 'KMeansCodebookClassifier':
        """Train one K-Means codebook per class.

        Args:
            frames_by_class: {label: (N_frames, n_mfcc)} array of frame vectors.

        Implemented in Phase 3.
        """
        raise NotImplementedError(
            "KMeansCodebookClassifier.fit — implemented in Phase 3"
        )

    def predict(self, frames: np.ndarray) -> Tuple[str, float]:
        """Predict class for a single audio sample.

        Args:
            frames: (N_frames, n_mfcc) array of frame-level MFCC vectors.

        Returns:
            (predicted_label, decision_margin)

        Implemented in Phase 3.
        """
        raise NotImplementedError(
            "KMeansCodebookClassifier.predict — implemented in Phase 3"
        )

    def predict_ranked(self, frames: np.ndarray) -> List[Tuple[str, float]]:
        """Return all classes ranked by average minimum distance (ascending).

        Implemented in Phase 3.
        """
        raise NotImplementedError(
            "KMeansCodebookClassifier.predict_ranked — implemented in Phase 3"
        )

    def save(self, path: Path) -> None:
        """Serialize model to pickle file. Implemented in Phase 3."""
        raise NotImplementedError(
            "KMeansCodebookClassifier.save — implemented in Phase 3"
        )

    @classmethod
    def load(cls, path: Path) -> 'KMeansCodebookClassifier':
        """Deserialize model from pickle file. Implemented in Phase 3."""
        raise NotImplementedError(
            "KMeansCodebookClassifier.load — implemented in Phase 3"
        )


def _kmeans(
    data: np.ndarray,
    n_clusters: int,
    max_iter: int,
    tol: float,
    random_state: int,
) -> np.ndarray:
    """Manual K-Means implementation. Returns centroids of shape (n_clusters, d).

    Implemented in Phase 3.
    """
    raise NotImplementedError("_kmeans — implemented in Phase 3")
