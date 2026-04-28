"""
GaussianNaiveBayesClassifier — lightweight supervised classifier.

Approach:
- Trains on fixed-length MFCC summary vectors (mean + std per frame sequence).
- Stores class priors, per-class feature means, and per-class feature variances.
- Inference uses Gaussian log-likelihood summed across features + log prior.
- Variance smoothing (epsilon) prevents numerical issues with near-zero variance.

All math implemented from scratch with NumPy (no sklearn).

Implemented in Phase 4.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import GNBConfig


@dataclass
class GaussianNaiveBayesClassifier:
    """Gaussian Naive Bayes classifier for fixed-length feature vectors."""

    config: GNBConfig = field(default_factory=GNBConfig)
    labels_: List[str] = field(default_factory=list, init=False)
    priors_: Dict[str, float] = field(default_factory=dict, init=False)
    means_: Dict[str, np.ndarray] = field(default_factory=dict, init=False)
    variances_: Dict[str, np.ndarray] = field(default_factory=dict, init=False)
    is_fitted_: bool = field(default=False, init=False)

    def fit(
        self,
        X: np.ndarray,
        y: List[str],
    ) -> 'GaussianNaiveBayesClassifier':
        """Train the classifier from fixed-length feature matrix X and labels y.

        Args:
            X: (N_samples, n_features) array of MFCC summary vectors.
            y: list of N_samples string labels.

        Implemented in Phase 4.
        """
        raise NotImplementedError(
            "GaussianNaiveBayesClassifier.fit — implemented in Phase 4"
        )

    def predict(self, x: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """Predict class for a single feature vector.

        Args:
            x: (n_features,) MFCC summary vector.

        Returns:
            (predicted_label, {label: normalized_log_prob})

        Implemented in Phase 4.
        """
        raise NotImplementedError(
            "GaussianNaiveBayesClassifier.predict — implemented in Phase 4"
        )

    def predict_ranked(self, x: np.ndarray) -> List[Tuple[str, float]]:
        """Return all classes ranked by log posterior probability (descending).

        Implemented in Phase 4.
        """
        raise NotImplementedError(
            "GaussianNaiveBayesClassifier.predict_ranked — implemented in Phase 4"
        )

    def save(self, path: Path) -> None:
        """Serialize model to pickle file. Implemented in Phase 4."""
        raise NotImplementedError(
            "GaussianNaiveBayesClassifier.save — implemented in Phase 4"
        )

    @classmethod
    def load(cls, path: Path) -> 'GaussianNaiveBayesClassifier':
        """Deserialize model from pickle file. Implemented in Phase 4."""
        raise NotImplementedError(
            "GaussianNaiveBayesClassifier.load — implemented in Phase 4"
        )
