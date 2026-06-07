"""
Central configuration dataclasses for the voice command recognition pipeline.
"""
from dataclasses import dataclass, field
from typing import Dict
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
    cmvn: bool = False              # per-utterance cepstral mean-variance normalization
    include_min_max: bool = False   # append per-coefficient min and max to summary vector
    # librosa-backend flags (HMM only)
    use_librosa: bool = False
    include_zcr: bool = False       # zero-crossing rate (1 feature/frame)
    include_rms: bool = False       # root-mean-square energy (1 feature/frame)
    include_contrast: bool = False  # spectral contrast (7 features/frame)


@dataclass
class DatasetConfig:
    """Parameters for dataset loading and train/test split."""
    test_ratio: float = 0.3
    random_state: int = 42
    supported_commands: List[str] = field(default_factory=lambda: [
        'avanzar', 'retroceder', 'izquierda', 'derecha', 'alto', 'inicio',
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


@dataclass
class HMMConfig:
    """Parameters for HiddenMarkovModel classifier."""
    n_states: int = 5               # default hidden states per HMM
    n_symbols: int = 256            # codebook size for observation quantization
    n_iter: int = 20                # Baum-Welch EM iterations
    kmeans_max_iter: int = 300      # K-Means iterations for codebook training
    kmeans_tol: float = 1e-4
    random_state: int = 42
    log_zero: float = -1e30         # substitute for log(0)
    # Per-command n_states override — if set, each label gets its own state count
    n_states_per_class: Dict[str, int] = field(default_factory=dict)
