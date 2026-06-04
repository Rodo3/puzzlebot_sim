"""
Dataset discovery, loading, and stratified train/test splitting.

Responsibilities:
- Scan a root folder and auto-discover class labels from subdirectory names.
- Collect .wav file paths per class, warn on empty or small classes.
- Stratified manual train/test split (no sklearn):
    shuffle each class independently with a seeded RNG, then
    take the last floor(n * test_ratio) samples as the test set.
- Return a DatasetSplit with metadata for reproducibility.
"""
import json
import random
import warnings
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

    def summary(self) -> str:
        lines = [
            f"Classes : {len(self.labels)}  {self.labels}",
            f"Train   : {len(self.train)} samples",
            f"Test    : {len(self.test)} samples",
        ]
        for lbl in self.labels:
            lines.append(
                f"  {lbl:15s}  total={self.samples_per_class.get(lbl, 0)}"
                f"  train={self.train_per_class.get(lbl, 0)}"
                f"  test={self.test_per_class.get(lbl, 0)}"
            )
        return "\n".join(lines)

    def to_metadata_dict(self, config: DatasetConfig) -> dict:
        """Return a JSON-serialisable metadata dict for train_metadata.json."""
        return {
            "labels": self.labels,
            "n_train": len(self.train),
            "n_test": len(self.test),
            "test_ratio": config.test_ratio,
            "random_state": config.random_state,
            "samples_per_class": self.samples_per_class,
            "train_per_class": self.train_per_class,
            "test_per_class": self.test_per_class,
        }


def discover_dataset(root: Path) -> Dict[str, List[Path]]:
    """Walk root directory and return {label: [wav_paths]} mapping.

    Rules:
    - Each immediate subdirectory of root becomes one class label.
    - Only files with a .wav extension (case-insensitive) are collected.
    - Subdirectories with zero .wav files emit a warning and are excluded.
    - The returned dict is sorted alphabetically by label for reproducibility.

    Raises:
        FileNotFoundError: if root does not exist.
        ValueError: if no valid class subdirectories are found.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if not root.is_dir():
        raise ValueError(f"Dataset root is not a directory: {root}")

    result: Dict[str, List[Path]] = {}

    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not subdirs:
        raise ValueError(f"No class subdirectories found in: {root}")

    for subdir in subdirs:
        wav_files = sorted(
            p for p in subdir.iterdir()
            if p.is_file() and p.suffix.lower() == '.wav'
        )
        if not wav_files:
            warnings.warn(
                f"Class '{subdir.name}' has no .wav files — skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue
        result[subdir.name] = wav_files

    if not result:
        raise ValueError(
            f"No class with .wav files found under: {root}"
        )

    return dict(sorted(result.items()))


def _original_stem(path: Path) -> str:
    """Strip augmentation tags to recover the original clip stem.

    Augmented files are named  <original_stem>_aug_<tag>.wav.
    Stripping the known tags gives back the original stem so all
    variants of the same recording are grouped together.
    """
    s = path.stem
    for tag in ('_aug_ts_fast', '_aug_ts_slow', '_aug_ps_up',
                '_aug_noise', '_aug_vol_up', '_aug_vol_down',
                '_aug_pitch_up', '_aug_pitch_down'):
        s = s.replace(tag, '')
    return s


def split_dataset(
    samples_by_class: Dict[str, List[Path]],
    config: DatasetConfig,
) -> DatasetSplit:
    """Stratified, leakage-free train/test split.

    Algorithm (per class):
      1. Group all files by their original clip stem so that augmented
         variants of the same recording are never split across train/test.
      2. Shuffle the groups using a seeded RNG.
      3. Reserve the last floor(n_groups * test_ratio) groups for test.
      4. Expand groups back to individual file paths.

    This prevents data leakage where an augmented version of a test clip
    ends up in training, which would inflate test accuracy.
    """
    rng = random.Random(config.random_state)

    split = DatasetSplit()
    split.labels = sorted(samples_by_class.keys())

    for label in split.labels:
        paths = list(samples_by_class[label])
        split.samples_per_class[label] = len(paths)

        # Group files by original clip stem
        groups: Dict[str, List[Path]] = {}
        for p in paths:
            stem = _original_stem(p)
            groups.setdefault(stem, []).append(p)

        group_keys = sorted(groups.keys())
        n_groups = len(group_keys)

        if n_groups < 2:
            warnings.warn(
                f"Class '{label}' has only {n_groups} original clip(s) — "
                "cannot create a test split, assigning all to train.",
                UserWarning,
                stacklevel=2,
            )
            split.train.extend(Sample(p, label) for p in paths)
            split.train_per_class[label] = len(paths)
            split.test_per_class[label] = 0
            continue

        rng.shuffle(group_keys)

        n_test_groups = max(1, min(n_groups - 1, int(n_groups * config.test_ratio)))
        train_keys = group_keys[:n_groups - n_test_groups]
        test_keys  = group_keys[n_groups - n_test_groups:]

        train_paths = [p for k in train_keys for p in groups[k]]
        test_paths  = [p for k in test_keys  for p in groups[k]]

        split.train.extend(Sample(p, label) for p in train_paths)
        split.test.extend(Sample(p, label) for p in test_paths)
        split.train_per_class[label] = len(train_paths)
        split.test_per_class[label] = len(test_paths)

    return split
