"""
Audio loading utilities.

Responsibilities:
- Load .wav files via scipy.io.wavfile
- Convert stereo to mono
- Normalize signal to [-1, 1]
- Warn or resample if sample rate != 16 kHz

Implemented in Phase 2.
"""
from pathlib import Path
from typing import Tuple

import numpy as np


def load_wav(path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load a .wav file and return (signal, sample_rate).

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError: if sample rate mismatch and resampling is not available.
    """
    raise NotImplementedError("audio_io.load_wav — implemented in Phase 2")


def to_mono(signal: np.ndarray) -> np.ndarray:
    """Convert stereo (N, 2) array to mono (N,) by averaging channels."""
    raise NotImplementedError("audio_io.to_mono — implemented in Phase 2")


def normalize(signal: np.ndarray) -> np.ndarray:
    """Normalize signal to [-1, 1] range."""
    raise NotImplementedError("audio_io.normalize — implemented in Phase 2")
