"""
HiddenMarkovModel — discrete-observation left-to-right HMM for speech commands.

Design:
- One HMM per command class; inference = argmax over per-class log-likelihoods.
- Observations are quantized with a global K-Means codebook (shared across all
  classes) that maps each MFCC frame vector -> integer symbol index.
- Topology: left-to-right (Bakis) — state i can only transition to i or i+1,
  matching the forward progression of speech.
- Training: Baum-Welch (EM) algorithm re-estimates A, B, and pi from sequences.
- Inference: Viterbi algorithm returns the log-likelihood of the best state path.

All maths done in log-space to avoid float underflow on long sequences.
NumPy only — no sklearn, scipy.special, or external ML libs.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import HMMConfig
from ..serialization import load_pickle, save_pickle


# ---------------------------------------------------------------------------
# Single HMM (one per class)
# ---------------------------------------------------------------------------

class _SingleHMM:
    """Discrete left-to-right HMM for one command class.

    Parameters
    ----------
    n_states  : number of hidden states
    n_symbols : codebook size (number of distinct observation symbols)
    log_zero  : value used in place of log(0) to avoid -inf
    """

    def __init__(self, n_states: int, n_symbols: int, log_zero: float) -> None:
        self.n_states  = n_states
        self.n_symbols = n_symbols
        self.log_zero  = log_zero

        # Parameters are stored in log-space for numerical stability
        self.log_pi: np.ndarray   # (n_states,)      initial state log-probs
        self.log_A:  np.ndarray   # (n_states, n_states)  transition log-probs
        self.log_B:  np.ndarray   # (n_states, n_symbols) emission log-probs
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_params(self, rng: np.random.Generator) -> None:
        S, M = self.n_states, self.n_symbols

        # pi: only the first state has non-zero probability (left-to-right)
        pi = np.zeros(S)
        pi[0] = 1.0
        self.log_pi = _safe_log(pi, self.log_zero)

        # A: left-to-right — state i -> i (self) or i+1 (forward), last state loops
        # Add small random noise so EM can break symmetry
        A = np.zeros((S, S))
        for i in range(S - 1):
            noise = rng.random(2)
            noise /= noise.sum()
            A[i, i]     = noise[0]
            A[i, i + 1] = noise[1]
        A[S - 1, S - 1] = 1.0
        self.log_A = _safe_log(A, self.log_zero)

        # B: uniform + small Dirichlet noise
        noise = rng.dirichlet(np.ones(M), size=S)  # (S, M)
        self.log_B = _safe_log(noise, self.log_zero)

    # ------------------------------------------------------------------
    # Baum-Welch training
    # ------------------------------------------------------------------

    def fit(
        self,
        sequences: List[np.ndarray],
        n_iter: int,
        rng: np.random.Generator,
    ) -> None:
        """Train on a list of observation sequences (each is a 1-D int array)."""
        self._init_params(rng)
        S, M = self.n_states, self.n_symbols

        for iteration in range(n_iter):
            # Accumulators for the new parameter estimates
            new_log_pi = np.full(S, self.log_zero)
            new_log_A  = np.full((S, S), self.log_zero)
            new_log_B  = np.full((S, M), self.log_zero)

            total_log_lik = 0.0

            for obs in sequences:
                T = len(obs)
                if T == 0:
                    continue

                log_alpha, log_scale = self._forward(obs)
                log_beta             = self._backward(obs, log_scale)

                log_lik = _logsumexp(log_alpha[-1])
                total_log_lik += log_lik

                # --- gamma: (T, S) — normalise each time step independently ---
                log_gamma = log_alpha + log_beta
                for t in range(T):
                    norm = _logsumexp(log_gamma[t])
                    log_gamma[t] -= norm

                # --- xi: (T-1, S, S) ---
                if T > 1:
                    log_xi = np.full((T - 1, S, S), self.log_zero)
                    for t in range(T - 1):
                        for i in range(S):
                            for j in range(S):
                                log_xi[t, i, j] = (
                                    log_alpha[t, i]
                                    + self.log_A[i, j]
                                    + self.log_B[j, obs[t + 1]]
                                    + log_beta[t + 1, j]
                                )
                        # Normalise over (i,j)
                        log_norm = _logsumexp_2d(log_xi[t])
                        log_xi[t] -= log_norm

                # Accumulate pi
                new_log_pi = _logaddexp_arrays(new_log_pi, log_gamma[0])

                # Accumulate A
                if T > 1:
                    log_xi_sum = _logsumexp_axis0(log_xi)   # (S, S)
                    new_log_A  = _logaddexp_arrays(new_log_A, log_xi_sum)

                # Accumulate B
                for t in range(T):
                    k = obs[t]
                    new_log_B[:, k] = _logaddexp_arrays(
                        new_log_B[:, k], log_gamma[t]
                    )

            # --- M-step: normalise ---
            self.log_pi = new_log_pi - _logsumexp(new_log_pi)

            for i in range(S):
                row_sum = _logsumexp(new_log_A[i])
                if row_sum <= self.log_zero + 1:
                    self.log_A[i] = self.log_A[i]   # keep previous if no data
                else:
                    self.log_A[i] = new_log_A[i] - row_sum

            for i in range(S):
                row_sum = _logsumexp(new_log_B[i])
                if row_sum <= self.log_zero + 1:
                    self.log_B[i] = self.log_B[i]
                else:
                    self.log_B[i] = new_log_B[i] - row_sum

        self._is_fitted = True

    # ------------------------------------------------------------------
    # Forward algorithm (scaled)
    # ------------------------------------------------------------------

    def _forward(
        self, obs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute log alpha and log scale factors.

        Returns
        -------
        log_alpha : (T, S)
        log_scale : (T,)  — each element is the log-normalisation constant used
                            at time t so that alpha sums to 1 in probability space.
        """
        T = len(obs)
        S = self.n_states
        log_alpha = np.full((T, S), self.log_zero)
        log_scale = np.zeros(T)

        log_alpha[0] = self.log_pi + self.log_B[:, obs[0]]
        log_scale[0] = _logsumexp(log_alpha[0])
        log_alpha[0] -= log_scale[0]

        for t in range(1, T):
            for j in range(S):
                log_alpha[t, j] = (
                    _logsumexp(log_alpha[t - 1] + self.log_A[:, j])
                    + self.log_B[j, obs[t]]
                )
            log_scale[t] = _logsumexp(log_alpha[t])
            log_alpha[t] -= log_scale[t]

        return log_alpha, log_scale

    # ------------------------------------------------------------------
    # Backward algorithm (scaled)
    # ------------------------------------------------------------------

    def _backward(
        self, obs: np.ndarray, log_scale: np.ndarray
    ) -> np.ndarray:
        """Compute log beta (T, S) using the same scale factors as forward."""
        T = len(obs)
        S = self.n_states
        log_beta = np.full((T, S), self.log_zero)
        log_beta[T - 1] = 0.0   # log(1)

        for t in range(T - 2, -1, -1):
            for i in range(S):
                log_beta[t, i] = _logsumexp(
                    self.log_A[i] + self.log_B[:, obs[t + 1]] + log_beta[t + 1]
                )
            log_beta[t] -= log_scale[t + 1]

        return log_beta

    # ------------------------------------------------------------------
    # Viterbi decoding
    # ------------------------------------------------------------------

    def log_likelihood(self, obs: np.ndarray) -> float:
        """Return log P(obs | model) via Viterbi (sum of log scale factors).

        This is faster than full forward and sufficient for argmax classification.
        """
        if not self._is_fitted:
            raise RuntimeError("HMM not fitted — call fit() first.")
        T = len(obs)
        S = self.n_states
        if T == 0:
            return self.log_zero

        log_delta = np.full((T, S), self.log_zero)
        log_delta[0] = self.log_pi + self.log_B[:, obs[0]]

        for t in range(1, T):
            for j in range(S):
                log_delta[t, j] = (
                    np.max(log_delta[t - 1] + self.log_A[:, j])
                    + self.log_B[j, obs[t]]
                )

        return float(np.max(log_delta[-1]))


# ---------------------------------------------------------------------------
# Classifier (one HMM per class)
# ---------------------------------------------------------------------------

@dataclass
class HiddenMarkovModelClassifier:
    """Classifier that trains one HMM per command class.

    Workflow:
      1. fit(sequences_by_class) — trains a shared K-Means codebook to quantize
         MFCC frames into discrete symbols, then fits one HMM per class.
      2. predict(frames) — quantizes frames with the codebook, then returns
         the class whose HMM assigns the highest Viterbi log-likelihood.
    """

    config: HMMConfig = field(default_factory=HMMConfig)
    labels_: List[str] = field(default_factory=list, init=False)
    codebook_: Optional[np.ndarray] = field(default=None, init=False)  # (n_symbols, n_mfcc)
    hmms_: Dict[str, _SingleHMM] = field(default_factory=dict, init=False)
    is_fitted_: bool = field(default=False, init=False)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        sequences_by_class: Dict[str, List[np.ndarray]],
    ) -> 'HiddenMarkovModelClassifier':
        """Train the classifier.

        Parameters
        ----------
        sequences_by_class : {label: [frames_1, frames_2, ...]}
            Each frames_i is an (n_frames_i, n_mfcc) array of MFCC frames
            for one audio clip.
        """
        if not sequences_by_class:
            raise ValueError("sequences_by_class is empty — nothing to train on.")

        self.labels_ = sorted(sequences_by_class.keys())
        cfg = self.config
        rng = np.random.default_rng(cfg.random_state)

        # --- 1. Build global codebook ---
        all_frames = np.concatenate(
            [f for seqs in sequences_by_class.values() for f in seqs],
            axis=0,
        ).astype(np.float64)
        self.codebook_ = _kmeans_codebook(
            all_frames, cfg.n_symbols, cfg.kmeans_max_iter, cfg.kmeans_tol, rng
        )

        # --- 2. Quantize each sequence ---
        quantized: Dict[str, List[np.ndarray]] = {}
        for label, seqs in sequences_by_class.items():
            quantized[label] = [
                _quantize(f.astype(np.float64), self.codebook_) for f in seqs
            ]

        # --- 3. Train one HMM per class ---
        for label in self.labels_:
            hmm = _SingleHMM(cfg.n_states, cfg.n_symbols, cfg.log_zero)
            hmm.fit(quantized[label], cfg.n_iter, np.random.default_rng(cfg.random_state))
            self.hmms_[label] = hmm

        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, frames: np.ndarray) -> Tuple[str, float]:
        """Classify one audio clip.

        Parameters
        ----------
        frames : (n_frames, n_mfcc) MFCC frame matrix

        Returns
        -------
        (predicted_label, log_likelihood_margin)
        margin = best_log_lik - second_best_log_lik  (higher = more confident)
        """
        ranked = self.predict_ranked(frames)
        best_label, best_ll = ranked[0]
        margin = best_ll - ranked[1][1] if len(ranked) > 1 else 0.0
        return best_label, float(margin)

    def predict_ranked(self, frames: np.ndarray) -> List[Tuple[str, float]]:
        """Return all classes ranked by Viterbi log-likelihood (descending).

        Higher log-likelihood = better match.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        obs = _quantize(frames.astype(np.float64), self.codebook_)
        scores = [
            (label, self.hmms_[label].log_likelihood(obs))
            for label in self.labels_
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Cannot save an unfitted model.")
        save_pickle(self, Path(path))

    @classmethod
    def load(cls, path: Path) -> 'HiddenMarkovModelClassifier':
        obj = load_pickle(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected HiddenMarkovModelClassifier, got {type(obj).__name__}"
            )
        return obj


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe_log(x: np.ndarray, log_zero: float) -> np.ndarray:
    """Element-wise log; replaces log(0) with log_zero."""
    with np.errstate(divide='ignore'):
        result = np.log(np.where(x > 0, x, np.finfo(float).tiny))
    result[x <= 0] = log_zero
    return result


def _logsumexp(log_x: np.ndarray) -> float:
    """Numerically stable log-sum-exp of a 1-D array."""
    m = log_x.max()
    if m <= -1e29:
        return float(m)
    return float(m + np.log(np.sum(np.exp(log_x - m))))


def _logsumexp_2d(log_x: np.ndarray) -> float:
    """log-sum-exp over all elements of a 2-D array."""
    m = log_x.max()
    if m <= -1e29:
        return float(m)
    return float(m + np.log(np.sum(np.exp(log_x - m))))


def _logsumexp_axis0(log_x: np.ndarray) -> np.ndarray:
    """log-sum-exp along axis 0; returns array with shape log_x.shape[1:]."""
    m = log_x.max(axis=0)
    diff = log_x - m
    return m + np.log(np.sum(np.exp(diff), axis=0))


def _logaddexp_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise log(exp(a) + exp(b)), numerically stable."""
    return np.logaddexp(a, b)


def _kmeans_codebook(
    data: np.ndarray,
    n_clusters: int,
    max_iter: int,
    tol: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """K-Means from scratch — returns centroids (n_clusters, d)."""
    n_points = data.shape[0]
    n_clusters = min(n_clusters, n_points)
    indices = rng.choice(n_points, size=n_clusters, replace=False)
    centroids = data[indices].copy()

    for _ in range(max_iter):
        # Assignment
        diff = data[:, np.newaxis, :] - centroids[np.newaxis, :, :]   # (N, K, d)
        sq_dist = (diff ** 2).sum(axis=2)                              # (N, K)
        assignments = np.argmin(sq_dist, axis=1)                       # (N,)

        # Update
        new_centroids = np.empty_like(centroids)
        for k in range(n_clusters):
            mask = assignments == k
            if mask.sum() == 0:
                new_centroids[k] = data[rng.integers(n_points)]
            else:
                new_centroids[k] = data[mask].mean(axis=0)

        shift = np.sqrt(((new_centroids - centroids) ** 2).sum(axis=1)).max()
        centroids = new_centroids
        if shift < tol:
            break

    return centroids.astype(np.float32)


def _quantize(frames: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Map each MFCC frame to the index of its nearest codebook centroid.

    Returns a 1-D int32 array of length n_frames.
    """
    cb64 = codebook.astype(np.float64)
    diff = frames[:, np.newaxis, :] - cb64[np.newaxis, :, :]   # (T, K, d)
    sq_dist = (diff ** 2).sum(axis=2)                          # (T, K)
    return np.argmin(sq_dist, axis=1).astype(np.int32)
