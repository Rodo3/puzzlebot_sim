"""
Central configuration dataclasses for the voice command recognition pipeline.
Implemented in Phase 2.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class MFCCConfig:
    """Parameters for MFCC feature extraction."""
    sample_rate: int = 16000
    pre_emphasis: float = 0.97
    frame_size: float = 0.025       # seconds
    frame_stride: float = 0.010     # seconds
    n_fft: int = 512
    n_filters: int = 26
    n_mfcc: int = 13
    include_delta: bool = False
    include_delta_delta: bool = False


@dataclass
class DatasetConfig:
    """Parameters for dataset loading and train/test split."""
    test_ratio: float = 0.3
    random_state: int = 42
    supported_commands: List[str] = field(default_factory=lambda: [
        'adelante', 'atras', 'izquierda', 'derecha', 'alto', 'inicio',
    ])


@dataclass
class KMeansConfig:
    """Parameters for KMeansCodebookClassifier."""
    n_clusters: int = 16
    max_iter: int = 300
    tolerance: float = 1e-4
    random_state: int = 42


@dataclass
class GNBConfig:
    """Parameters for GaussianNaiveBayesClassifier."""
    var_epsilon: float = 1e-9       # variance smoothing for numerical stability
