"""
GaussianNaiveBayesClassifier — lightweight supervised classifier.

Approach:
- Trains on fixed-length MFCC summary vectors (mean + std of frame-level MFCCs).
- For each class stores: log prior, per-feature mean, per-feature variance.
- Inference: log posterior = log prior + sum of Gaussian log-likelihoods per feature.
- Variance smoothing (epsilon) prevents log(0) on near-zero variance features.

All math implemented from scratch with NumPy (no sklearn).

Math reference:
  log P(c | x) ∝ log P(c) + Σ_j log N(x_j | μ_cj, σ²_cj)

  log N(x | μ, σ²) = -0.5 * log(2π σ²) - (x - μ)² / (2 σ²)
                   = -0.5 * [log(2π) + log(σ²) + (x - μ)² / σ²]
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..config import GNBConfig
from ..serialization import load_pickle, save_pickle

# log(2π) precomputed once
_LOG_2PI = np.log(2.0 * np.pi)


@dataclass
class GaussianNaiveBayesClassifier:
    """Gaussian Naive Bayes classifier for fixed-length MFCC summary vectors."""

    config: GNBConfig = field(default_factory=GNBConfig)
    labels_: List[str] = field(default_factory=list, init=False)
    log_priors_: Dict[str, float] = field(default_factory=dict, init=False)
    means_: Dict[str, np.ndarray] = field(default_factory=dict, init=False)
    variances_: Dict[str, np.ndarray] = field(default_factory=dict, init=False)
    is_fitted_: bool = field(default=False, init=False)

    def fit(
        self,
        X: np.ndarray,
        y: List[str],
    ) -> 'GaussianNaiveBayesClassifier':
        """Estimate class priors, per-feature means and variances from training data.

        Args:
            X: (N_samples, n_features) float32 array of MFCC summary vectors.
            y: list of N_samples string class labels.

        Returns:
            self (for chaining)
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if len(y) != X.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} rows but y has {len(y)} labels."
            )
        if X.shape[0] == 0:
            raise ValueError("X is empty — nothing to train on.")

        X = X.astype(np.float64)
        self.labels_ = sorted(set(y))
        n_total = X.shape[0]

        for label in self.labels_:
            mask = np.array([yi == label for yi in y])
            X_c = X[mask]
            n_c = X_c.shape[0]

            # Log prior: log(n_c / n_total)
            self.log_priors_[label] = np.log(n_c / n_total)

            # Per-feature mean and variance with epsilon smoothing
            self.means_[label] = X_c.mean(axis=0)
            var = X_c.var(axis=0)
            self.variances_[label] = var + self.config.var_epsilon

        self.is_fitted_ = True
        return self

    def predict(self, x: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """Predict the class for a single MFCC summary vector.

        Args:
            x: (n_features,) float array.

        Returns:
            (predicted_label, {label: normalised_score})
            Scores are softmax-normalised log posteriors in [0, 1].
        """
        ranked = self.predict_ranked(x)
        best_label = ranked[0][0]

        # Softmax normalisation so scores sum to 1 (stable via log-sum-exp)
        log_posts = np.array([lp for _, lp in ranked])
        log_posts -= log_posts.max()          # shift for numerical stability
        exp_posts = np.exp(log_posts)
        norm = exp_posts.sum()
        scores = {label: float(exp_posts[i] / norm) for i, (label, _) in enumerate(ranked)}

        return best_label, scores

    def predict_ranked(self, x: np.ndarray) -> List[Tuple[str, float]]:
        """Return all classes ranked by log posterior (descending).

        Higher log posterior = better match.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if x.ndim != 1:
            raise ValueError(f"x must be 1-D, got shape {x.shape}")

        x64 = x.astype(np.float64)
        scores: List[Tuple[str, float]] = []

        for label in self.labels_:
            log_post = self.log_priors_[label] + _gaussian_log_likelihood(
                x64, self.means_[label], self.variances_[label]
            )
            scores.append((label, float(log_post)))

        scores.sort(key=lambda t: t[1], reverse=True)
        return scores

    def save(self, path: Path) -> None:
        """Serialize the fitted model to a pickle file."""
        if not self.is_fitted_:
            raise RuntimeError("Cannot save an unfitted model.")
        save_pickle(self, Path(path))

    @classmethod
    def load(cls, path: Path) -> 'GaussianNaiveBayesClassifier':
        """Deserialize a model from a pickle file."""
        obj = load_pickle(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected GaussianNaiveBayesClassifier, got {type(obj).__name__}"
            )
        return obj


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _gaussian_log_likelihood(
    x: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
) -> float:
    """Sum of per-feature Gaussian log-likelihoods.

    log N(x_j | μ_j, σ²_j) = -0.5 * [log(2π) + log(σ²_j) + (x_j - μ_j)² / σ²_j]

    Summing over all features gives the naive Bayes assumption
    (features are conditionally independent given the class).
    """
    diff_sq = (x - mean) ** 2
    log_likelihood = -0.5 * (_LOG_2PI + np.log(var) + diff_sq / var)
    return float(log_likelihood.sum())
