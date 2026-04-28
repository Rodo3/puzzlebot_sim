"""
Manual MFCC feature extraction pipeline using NumPy/SciPy only.

Pipeline (all steps implemented in Phase 2):
  1. Pre-emphasis filter
  2. Frame splitting with overlap
  3. Hamming window
  4. FFT + power spectrum
  5. Mel filterbank (built manually)
  6. Log of Mel energies
  7. DCT to get cepstral coefficients
  8. Aggregate frames -> fixed-length vector (mean + std over time)
  Optional: delta and delta-delta features (disabled by default)
"""
from typing import Tuple

import numpy as np

from .config import MFCCConfig


def extract_mfcc_frames(signal: np.ndarray, config: MFCCConfig) -> np.ndarray:
    """Return frame-level MFCC matrix of shape (n_frames, n_mfcc).

    Used by KMeansCodebookClassifier which works on individual frames.
    Implemented in Phase 2.
    """
    raise NotImplementedError("mfcc.extract_mfcc_frames — implemented in Phase 2")


def extract_mfcc_summary(signal: np.ndarray, config: MFCCConfig) -> np.ndarray:
    """Return fixed-length MFCC summary vector (mean + std across frames).

    Output shape: (n_mfcc * 2,) when deltas are disabled.
    Used by GaussianNaiveBayesClassifier.
    Implemented in Phase 2.
    """
    raise NotImplementedError("mfcc.extract_mfcc_summary — implemented in Phase 2")


def build_mel_filterbank(
    n_filters: int,
    n_fft: int,
    sample_rate: int,
    low_freq: float = 0.0,
    high_freq: float = None,
) -> np.ndarray:
    """Build a triangular Mel filterbank matrix of shape (n_filters, n_fft // 2 + 1).

    Implemented in Phase 2.
    """
    raise NotImplementedError("mfcc.build_mel_filterbank — implemented in Phase 2")
