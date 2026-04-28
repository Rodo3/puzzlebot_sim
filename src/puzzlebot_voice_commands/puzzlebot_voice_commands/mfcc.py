"""
Manual MFCC feature extraction pipeline — NumPy/SciPy only, no librosa.

Pipeline order:
  1.  Pre-emphasis:    y[t] = x[t] - alpha * x[t-1]
  2.  Framing:         split signal into overlapping frames
  3.  Hamming window:  reduce spectral leakage at frame edges
  4.  FFT:             convert each frame to frequency domain
  5.  Power spectrum:  |FFT|^2 / n_fft
  6.  Mel filterbank:  triangular filters on Mel scale (built manually)
  7.  Log energy:      log(filterbank energies + epsilon)
  8.  DCT:             decorrelate log-Mel energies -> cepstral coefficients
  9.  Summary:         mean + std over the time axis -> fixed-length vector

Optional delta and delta-delta features append velocity/acceleration cues.
"""
from typing import Optional

import numpy as np
from scipy.fft import dct as scipy_dct

from .config import MFCCConfig

# Small floor added before log to avoid log(0)
_LOG_FLOOR = 1e-10


def extract_mfcc_frames(signal: np.ndarray, config: MFCCConfig) -> np.ndarray:
    """Return frame-level MFCC matrix of shape (n_frames, n_mfcc).

    Used by KMeansCodebookClassifier, which operates on individual frames.
    """
    frames = _frame_signal(signal, config)           # (n_frames, frame_len)
    frames = frames * np.hamming(frames.shape[1])    # Hamming window

    power = _power_spectrum(frames, config.n_fft)    # (n_frames, n_fft//2+1)

    filterbank = build_mel_filterbank(
        config.n_filters, config.n_fft, config.sample_rate
    )                                                # (n_filters, n_fft//2+1)

    mel_energy = np.dot(power, filterbank.T)         # (n_frames, n_filters)
    log_mel = np.log(mel_energy + _LOG_FLOOR)        # (n_frames, n_filters)

    # DCT-II along the filter axis, keep first n_mfcc coefficients
    mfccs = scipy_dct(log_mel, type=2, axis=1, norm='ortho')
    mfccs = mfccs[:, :config.n_mfcc]                # (n_frames, n_mfcc)

    if config.include_delta:
        delta = _compute_delta(mfccs)
        mfccs = np.concatenate([mfccs, delta], axis=1)

    if config.include_delta_delta:
        if config.include_delta:
            delta2 = _compute_delta(mfccs[:, config.n_mfcc:config.n_mfcc * 2])
        else:
            delta2 = _compute_delta(_compute_delta(mfccs))
        mfccs = np.concatenate([mfccs, delta2], axis=1)

    return mfccs.astype(np.float32)


def extract_mfcc_summary(signal: np.ndarray, config: MFCCConfig) -> np.ndarray:
    """Return fixed-length MFCC summary vector (mean + std across frames).

    Output shape:
      - (n_mfcc * 2,)             when both deltas are disabled
      - (n_mfcc * 4,)             when include_delta only
      - (n_mfcc * 6,)             when both deltas enabled

    Used by GaussianNaiveBayesClassifier.
    """
    frames_mfcc = extract_mfcc_frames(signal, config)  # (n_frames, features)
    mean = frames_mfcc.mean(axis=0)
    std = frames_mfcc.std(axis=0)
    return np.concatenate([mean, std]).astype(np.float32)


def build_mel_filterbank(
    n_filters: int,
    n_fft: int,
    sample_rate: int,
    low_freq: float = 0.0,
    high_freq: Optional[float] = None,
) -> np.ndarray:
    """Build a triangular Mel filterbank matrix of shape (n_filters, n_fft//2+1).

    Each row is one triangular filter covering the Mel frequency scale.

    The conversion between Hz and Mel uses the formula:
      mel  = 2595 * log10(1 + hz / 700)
      hz   = 700 * (10^(mel / 2595) - 1)
    """
    if high_freq is None:
        high_freq = float(sample_rate) / 2.0

    n_fft_bins = n_fft // 2 + 1

    low_mel = _hz_to_mel(low_freq)
    high_mel = _hz_to_mel(high_freq)

    # n_filters + 2 equally spaced points on the Mel scale
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = _mel_to_hz(mel_points)

    # Map Hz frequencies to the nearest FFT bin index
    bin_indices = np.floor(
        (n_fft + 1) * hz_points / sample_rate
    ).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_fft_bins - 1)

    filterbank = np.zeros((n_filters, n_fft_bins), dtype=np.float32)

    for m in range(1, n_filters + 1):
        left = bin_indices[m - 1]
        center = bin_indices[m]
        right = bin_indices[m + 1]

        # Rising slope: left -> center
        if center > left:
            filterbank[m - 1, left:center] = (
                np.arange(left, center) - left
            ) / (center - left)

        # Falling slope: center -> right
        if right > center:
            filterbank[m - 1, center:right] = (
                right - np.arange(center, right)
            ) / (right - center)

        # Peak bin always gets weight 1.0
        if center < n_fft_bins:
            filterbank[m - 1, center] = 1.0

    return filterbank


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _preemphasis(signal: np.ndarray, alpha: float) -> np.ndarray:
    """Apply first-order high-pass pre-emphasis filter.

    y[t] = x[t] - alpha * x[t-1]
    Boosts high frequencies to balance the spectral tilt of speech.
    """
    return np.append(signal[0], signal[1:] - alpha * signal[:-1]).astype(np.float32)


def _frame_signal(signal: np.ndarray, config: MFCCConfig) -> np.ndarray:
    """Split signal into overlapping frames and return (n_frames, frame_len).

    1. Apply pre-emphasis.
    2. Compute frame and stride lengths in samples.
    3. Pad the signal so the last frame is complete.
    4. Use stride tricks for zero-copy frame extraction.
    """
    emphasized = _preemphasis(signal, config.pre_emphasis)

    frame_len = int(round(config.frame_size * config.sample_rate))
    frame_step = int(round(config.frame_stride * config.sample_rate))

    signal_len = len(emphasized)
    if signal_len <= frame_len:
        # Signal shorter than one frame — pad to exactly one frame
        pad_len = frame_len - signal_len
        emphasized = np.pad(emphasized, (0, pad_len))
        n_frames = 1
    else:
        n_frames = 1 + (signal_len - frame_len) // frame_step
        # Pad so the last frame is not truncated
        pad_len = (n_frames - 1) * frame_step + frame_len - signal_len
        if pad_len > 0:
            emphasized = np.pad(emphasized, (0, pad_len))

    # Efficient frame extraction via stride tricks (no data copy)
    shape = (n_frames, frame_len)
    strides = (emphasized.strides[0] * frame_step, emphasized.strides[0])
    frames = np.lib.stride_tricks.as_strided(emphasized, shape=shape, strides=strides)
    return frames.copy().astype(np.float32)


def _power_spectrum(frames: np.ndarray, n_fft: int) -> np.ndarray:
    """Compute power spectrum for each frame.

    Returns (n_frames, n_fft//2 + 1).
    Power = |FFT|^2 / n_fft
    """
    mag = np.abs(np.fft.rfft(frames, n=n_fft))  # (n_frames, n_fft//2+1)
    return (mag ** 2) / n_fft


def _compute_delta(features: np.ndarray, N: int = 2) -> np.ndarray:
    """Compute delta (first-order derivative) of feature matrix.

    Uses a regression window of half-width N:
      delta[t] = sum_{n=1}^{N} n * (c[t+n] - c[t-n]) / (2 * sum_{n=1}^{N} n^2)

    Frames near the edges are padded by replicating the boundary frame.
    """
    n_frames, n_feats = features.shape
    padded = np.pad(features, ((N, N), (0, 0)), mode='edge')
    denom = 2.0 * sum(n ** 2 for n in range(1, N + 1))
    delta = np.zeros_like(features)
    for t in range(n_frames):
        for n in range(1, N + 1):
            delta[t] += n * (padded[t + N + n] - padded[t + N - n])
        delta[t] /= denom
    return delta.astype(np.float32)


def _hz_to_mel(hz: float) -> float:
    """Convert Hz to Mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    """Convert Mel to Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
