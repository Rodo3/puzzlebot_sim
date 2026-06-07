"""
librosa_features.py — Frame-level feature extraction using librosa.

Produces the same (n_frames, n_features) shape as extract_mfcc_frames()
so HMM training/inference is backend-agnostic.

Features (all aligned to the same hop grid):
  - MFCC            : n_mfcc coefficients
  - Delta MFCC      : n_mfcc coefficients  (if include_delta)
  - ZCR             : 1 coefficient        (if include_zcr)
  - RMS energy      : 1 coefficient        (if include_rms)
  - Spectral contrast: 7 coefficients      (if include_contrast)
"""
import librosa
import numpy as np

from .config import MFCCConfig


def extract_librosa_frames(signal: np.ndarray, cfg: MFCCConfig) -> np.ndarray:
    """Extract frame-level features with librosa.

    Args:
        signal: float32 mono waveform, already normalized, at cfg.sample_rate.
        cfg:    MFCCConfig — uses n_mfcc, n_filters, n_fft, frame_size,
                frame_stride, cmvn, include_delta, include_zcr, include_rms,
                include_contrast.

    Returns:
        ndarray shape (n_frames, n_features).
    """
    sr          = cfg.sample_rate
    hop_length  = int(cfg.frame_stride * sr)   # 160 at 16 kHz
    win_length  = int(cfg.frame_size   * sr)   # 400 at 16 kHz
    n_fft       = max(cfg.n_fft, win_length)

    # ── MFCC ─────────────────────────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(
        y=signal, sr=sr,
        n_mfcc=cfg.n_mfcc,
        n_mels=cfg.n_filters,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=False,
    )  # (n_mfcc, T)

    parts = [mfcc]

    if cfg.include_delta:
        parts.append(librosa.feature.delta(mfcc, order=1))

    if cfg.include_delta_delta:
        parts.append(librosa.feature.delta(mfcc, order=2))

    # ── Extra features (align to MFCC frame count T) ─────────────────────────
    T = mfcc.shape[1]

    if cfg.include_zcr:
        zcr = librosa.feature.zero_crossing_rate(
            signal, frame_length=win_length, hop_length=hop_length, center=False,
        )  # (1, T')
        parts.append(zcr[:, :T])

    if cfg.include_rms:
        rms = librosa.feature.rms(
            y=signal, frame_length=win_length, hop_length=hop_length, center=False,
        )  # (1, T')
        parts.append(rms[:, :T])

    if cfg.include_contrast:
        contrast = librosa.feature.spectral_contrast(
            y=signal, sr=sr,
            n_fft=n_fft, hop_length=hop_length, center=False,
        )  # (7, T')
        parts.append(contrast[:, :T])

    # ── Stack → (n_features, T) → transpose → (T, n_features) ───────────────
    frames = np.vstack(parts).T.astype(np.float32)  # (T, n_features)

    # ── CMVN ─────────────────────────────────────────────────────────────────
    if cfg.cmvn and frames.shape[0] > 1:
        mean = frames.mean(axis=0, keepdims=True)
        std  = frames.std(axis=0, keepdims=True) + 1e-8
        frames = (frames - mean) / std

    return frames
