"""
Dataset discovery, loading, and train/test splitting.

Responsibilities:
- Scan a root folder and auto-discover class labels from subdirectory names
- Load .wav files per class
- Stratified manual train/test split (no sklearn)
- Return split metadata for reproducibility

Implemented in Phase 2.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from .config import DatasetConfig


@dataclass
class Sample:
    """Single audio sample with its file path and string label."""
    path: Path
    label: str


@dataclass
class DatasetSplit:
    """Result of a stratified train/test split."""
    train: List[Sample] = field(default_factory=list)
    test: List[Sample] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    samples_per_class: Dict[str, int] = field(default_factory=dict)
    train_per_class: Dict[str, int] = field(default_factory=dict)
    test_per_class: Dict[str, int] = field(default_factory=dict)


def discover_dataset(root: Path) -> Dict[str, List[Path]]:
    """Walk root directory and return {label: [wav_paths]} mapping.

    Labels are inferred from subdirectory names.
    Implemented in Phase 2.
    """
    raise NotImplementedError("dataset.discover_dataset — implemented in Phase 2")


def split_dataset(
    samples_by_class: Dict[str, List[Path]],
    config: DatasetConfig,
) -> DatasetSplit:
    """Stratified manual train/test split without sklearn.

    Implemented in Phase 2.
    """
    raise NotImplementedError("dataset.split_dataset — implemented in Phase 2")
