"""
Audio loading utilities.

Responsibilities:
- Load .wav files via scipy.io.wavfile
- Convert stereo to mono by averaging channels
- Normalize signal to [-1, 1]
- Resample to target_sr using scipy.signal if the file SR differs
"""
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from math import gcd


def load_wav(path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load a .wav file and return (float32 signal, sample_rate).

    Steps:
      1. Read with scipy.io.wavfile (returns int or float samples).
      2. Convert integer PCM to float32 in [-1, 1].
      3. Convert stereo to mono.
      4. Resample to target_sr if needed.

    Raises:
        FileNotFoundError: if path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    sr, data = wavfile.read(str(path))

    # Convert integer PCM formats to float32 in [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        # uint8 WAV is offset-binary: 128 = silence
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)

    # Stereo -> mono
    if data.ndim == 2:
        data = to_mono(data)

    # Resample if sample rate does not match target
    if sr != target_sr:
        warnings.warn(
            f"{path.name}: sample rate is {sr} Hz, expected {target_sr} Hz. "
            "Resampling with scipy.signal.resample_poly.",
            UserWarning,
            stacklevel=2,
        )
        data = _resample(data, sr, target_sr)
        sr = target_sr

    return data, sr


def to_mono(signal: np.ndarray) -> np.ndarray:
    """Convert stereo (N, 2) or multi-channel (N, C) array to mono (N,).

    Uses channel mean so no clipping occurs.
    """
    if signal.ndim == 1:
        return signal
    return signal.mean(axis=1).astype(np.float32)


def normalize(signal: np.ndarray) -> np.ndarray:
    """Normalize signal to [-1, 1] by dividing by its peak absolute value.

    If the signal is silent (all zeros), it is returned unchanged.
    """
    peak = np.max(np.abs(signal))
    if peak < 1e-10:
        return signal.astype(np.float32)
    return (signal / peak).astype(np.float32)


def _resample(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Rational resampling via scipy.signal.resample_poly.

    Reduces the up/down ratio by the GCD to keep computation efficient.
    """
    common = gcd(target_sr, orig_sr)
    up = target_sr // common
    down = orig_sr // common
    resampled = resample_poly(signal, up, down)
    return resampled.astype(np.float32)
