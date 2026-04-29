"""
KMeansCodebookClassifier — vector-quantization-style spoken word classifier.

Approach:
- One K-Means codebook trained per class on frame-level MFCC vectors.
- Inference: for each input frame, find its nearest centroid in each class
  codebook and compute the average minimum Euclidean distance across all frames.
  The class with the lowest average distance wins.
- Decision margin = second_best_distance - best_distance (higher = more confident).

K-Means is implemented manually with NumPy (no sklearn).
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..config import KMeansConfig
from ..serialization import load_pickle, save_pickle


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
            frames_by_class: {label: (N_frames, n_mfcc)} frame-level MFCC arrays.

        Returns:
            self (for chaining)
        """
        if not frames_by_class:
            raise ValueError("frames_by_class is empty — nothing to train on.")

        self.labels_ = sorted(frames_by_class.keys())
        self.codebooks_ = {}

        for label in self.labels_:
            data = frames_by_class[label]
            if data.ndim != 2:
                raise ValueError(
                    f"Class '{label}': expected 2-D frame array, got shape {data.shape}"
                )
            n_frames = data.shape[0]
            n_clusters = min(self.config.n_clusters, n_frames)
            if n_clusters < self.config.n_clusters:
                import warnings
                warnings.warn(
                    f"Class '{label}' has only {n_frames} frames — "
                    f"reducing n_clusters from {self.config.n_clusters} to {n_clusters}.",
                    UserWarning,
                    stacklevel=2,
                )
            centroids = _kmeans(
                data=data,
                n_clusters=n_clusters,
                max_iter=self.config.max_iter,
                tol=self.config.tolerance,
                random_state=self.config.random_state,
            )
            self.codebooks_[label] = centroids

        self.is_fitted_ = True
        return self

    def predict(self, frames: np.ndarray) -> Tuple[str, float]:
        """Predict the command class for one audio sample.

        Args:
            frames: (N_frames, n_mfcc) frame-level MFCC matrix.

        Returns:
            (predicted_label, decision_margin)
            decision_margin = second_best_avg_distance - best_avg_distance
            A higher margin means the model is more confident.
        """
        ranked = self.predict_ranked(frames)
        best_label, best_dist = ranked[0]
        margin = ranked[1][1] - best_dist if len(ranked) > 1 else 0.0
        return best_label, float(margin)

    def predict_ranked(self, frames: np.ndarray) -> List[Tuple[str, float]]:
        """Return all classes ranked by average minimum distance (ascending).

        Lower distance = better match.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if frames.ndim != 2:
            raise ValueError(
                f"Expected 2-D frame array, got shape {frames.shape}"
            )

        scores: List[Tuple[str, float]] = []
        for label in self.labels_:
            centroids = self.codebooks_[label]
            avg_dist = _avg_min_distance(frames, centroids)
            scores.append((label, avg_dist))

        scores.sort(key=lambda x: x[1])
        return scores

    def save(self, path: Path) -> None:
        """Serialize the fitted model to a pickle file."""
        if not self.is_fitted_:
            raise RuntimeError("Cannot save an unfitted model.")
        save_pickle(self, Path(path))

    @classmethod
    def load(cls, path: Path) -> 'KMeansCodebookClassifier':
        """Deserialize a model from a pickle file."""
        obj = load_pickle(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected KMeansCodebookClassifier, got {type(obj).__name__}"
            )
        return obj


# ---------------------------------------------------------------------------
# K-Means implementation (no sklearn)
# ---------------------------------------------------------------------------

def _kmeans(
    data: np.ndarray,
    n_clusters: int,
    max_iter: int,
    tol: float,
    random_state: int,
) -> np.ndarray:
    """Manual K-Means clustering. Returns centroids of shape (n_clusters, d).

    Algorithm:
      1. Initialise centroids by sampling n_clusters unique rows from data.
      2. Repeat until convergence or max_iter:
         a. Assign each point to the nearest centroid (Euclidean distance).
         b. Recompute each centroid as the mean of its assigned points.
         c. If a cluster is empty, reinitialise its centroid to a random data point.
         d. Check convergence: max shift across all centroids < tol.
    """
    rng = np.random.default_rng(random_state)
    n_points, d = data.shape

    # Step 1 — initialise by random sampling without replacement
    indices = rng.choice(n_points, size=n_clusters, replace=False)
    centroids = data[indices].copy().astype(np.float64)

    data64 = data.astype(np.float64)

    for _ in range(max_iter):
        # Step 2a — assign each point to the nearest centroid
        # distances shape: (n_points, n_clusters)
        distances = _pairwise_sq_distances(data64, centroids)
        assignments = np.argmin(distances, axis=1)  # (n_points,)

        # Step 2b/2c — recompute centroids, handle empty clusters
        new_centroids = np.empty_like(centroids)
        for k in range(n_clusters):
            mask = assignments == k
            if mask.sum() == 0:
                # Empty cluster — reinitialise to a random point
                new_centroids[k] = data64[rng.integers(n_points)]
            else:
                new_centroids[k] = data64[mask].mean(axis=0)

        # Step 2d — convergence check
        shifts = np.sqrt(((new_centroids - centroids) ** 2).sum(axis=1))
        centroids = new_centroids
        if shifts.max() < tol:
            break

    return centroids.astype(np.float32)


def _pairwise_sq_distances(
    X: np.ndarray,
    C: np.ndarray,
) -> np.ndarray:
    """Compute squared Euclidean distances between each row of X and each row of C.

    Returns shape (n_points, n_clusters).

    Uses the identity ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x·c^T
    which is much faster than a loop for large arrays.
    """
    # (n_points,) and (n_clusters,)
    X_sq = (X ** 2).sum(axis=1, keepdims=True)   # (n, 1)
    C_sq = (C ** 2).sum(axis=1, keepdims=True).T  # (1, k)
    cross = X @ C.T                               # (n, k)
    sq_dist = X_sq + C_sq - 2.0 * cross
    # Numerical errors can produce small negatives — clamp to zero
    np.clip(sq_dist, 0.0, None, out=sq_dist)
    return sq_dist


def _avg_min_distance(
    frames: np.ndarray,
    centroids: np.ndarray,
) -> float:
    """Average minimum Euclidean distance from each frame to the nearest centroid.

    This is the score used for classification: lower = closer match to this class.
    """
    frames64 = frames.astype(np.float64)
    centroids64 = centroids.astype(np.float64)
    sq_dist = _pairwise_sq_distances(frames64, centroids64)  # (n_frames, n_clusters)
    min_sq_dist = sq_dist.min(axis=1)                         # (n_frames,)
    return float(np.sqrt(min_sq_dist).mean())
