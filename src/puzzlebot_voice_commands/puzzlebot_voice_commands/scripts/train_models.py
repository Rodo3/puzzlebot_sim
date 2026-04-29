"""
train_voice_models — train KMeans codebook and/or Gaussian NB classifier.

Usage (after building and sourcing):
  ros2 run puzzlebot_voice_commands train_voice_models \\
    --dataset    <path/to/voice_commands_dataset> \\
    --model      both \\
    --output-dir <path/to/artifacts>

Artifacts written to --output-dir:
  kmeans_model.pkl
  gnb_model.pkl
  labels.json
  feature_config.json
  train_metadata.json
"""
import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Dict

import numpy as np

from ..audio_io import load_wav, normalize
from ..config import DatasetConfig, GNBConfig, KMeansConfig, MFCCConfig
from ..dataset import DatasetSplit, discover_dataset, split_dataset
from ..mfcc import extract_mfcc_frames, extract_mfcc_summary
from ..models.gaussian_nb import GaussianNaiveBayesClassifier
from ..models.kmeans_codebook import KMeansCodebookClassifier
from ..serialization import artifact_size_kb, save_json, save_pickle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='train_voice_models',
        description=(
            'Train voice command recognition models on a folder-based .wav dataset. '
            'Supports KMeans codebook classifier, Gaussian Naive Bayes, or both.'
        ),
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to the root dataset folder (one subfolder per class).',
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['kmeans', 'gnb', 'both'],
        default='both',
        help='Which model(s) to train (default: both).',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to write trained model artifacts.',
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.3,
        help='Fraction of samples reserved for the test split (default: 0.3).',
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducible splits (default: 42).',
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=16,
        help='Number of K-Means clusters per class codebook (default: 16).',
    )
    parser.add_argument(
        '--n-mfcc',
        type=int,
        default=13,
        help='Number of MFCC coefficients (default: 13).',
    )
    parser.add_argument(
        '--n-filters',
        type=int,
        default=26,
        help='Number of Mel filterbank channels (default: 26).',
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=16000,
        help='Expected audio sample rate in Hz (default: 16000).',
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mfcc_cfg = MFCCConfig(
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        n_filters=args.n_filters,
    )
    dataset_cfg = DatasetConfig(
        test_ratio=args.test_ratio,
        random_state=args.random_state,
    )
    kmeans_cfg = KMeansConfig(
        n_clusters=args.n_clusters,
        random_state=args.random_state,
    )

    train_kmeans = args.model in ('kmeans', 'both')
    train_gnb = args.model in ('gnb', 'both')

    # ------------------------------------------------------------------
    # 1. Discover and split dataset
    # ------------------------------------------------------------------
    print(f"[train_voice_models] Dataset : {dataset_root}")
    try:
        samples_by_class = discover_dataset(dataset_root)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    split: DatasetSplit = split_dataset(samples_by_class, dataset_cfg)
    print(split.summary())

    # ------------------------------------------------------------------
    # 2. Save shared metadata artifacts
    # ------------------------------------------------------------------
    labels_path = output_dir / 'labels.json'
    save_json(split.labels, labels_path)

    feature_config_path = output_dir / 'feature_config.json'
    save_json({
        'sample_rate': mfcc_cfg.sample_rate,
        'pre_emphasis': mfcc_cfg.pre_emphasis,
        'frame_size': mfcc_cfg.frame_size,
        'frame_stride': mfcc_cfg.frame_stride,
        'n_fft': mfcc_cfg.n_fft,
        'n_filters': mfcc_cfg.n_filters,
        'n_mfcc': mfcc_cfg.n_mfcc,
        'include_delta': mfcc_cfg.include_delta,
        'include_delta_delta': mfcc_cfg.include_delta_delta,
    }, feature_config_path)

    train_metadata_path = output_dir / 'train_metadata.json'
    save_json(split.to_metadata_dict(dataset_cfg), train_metadata_path)

    print(f"\n[train_voice_models] Shared artifacts saved to: {output_dir}")

    # ------------------------------------------------------------------
    # 3. Train KMeans codebook classifier
    # ------------------------------------------------------------------
    if train_kmeans:
        print("\n[train_voice_models] Training KMeansCodebookClassifier ...")
        frames_by_class, load_errors = _load_frames(split, mfcc_cfg, dataset_root)

        if not frames_by_class:
            print("ERROR: No frames extracted — check dataset audio files.", file=sys.stderr)
            sys.exit(1)
        if load_errors:
            print(f"  WARNING: {load_errors} file(s) failed to load and were skipped.")

        # Log frame counts per class
        total_frames = 0
        for label in split.labels:
            n = frames_by_class.get(label, np.empty((0, mfcc_cfg.n_mfcc))).shape[0]
            total_frames += n
            print(f"  {label:15s}  {n} frames")
        print(f"  Total frames: {total_frames}")

        t0 = time.perf_counter()
        model = KMeansCodebookClassifier(config=kmeans_cfg)
        model.fit(frames_by_class)
        elapsed = time.perf_counter() - t0

        kmeans_path = output_dir / 'kmeans_model.pkl'
        model.save(kmeans_path)
        size_kb = artifact_size_kb(kmeans_path)

        print(f"\n  Training time : {elapsed:.2f}s")
        print(f"  n_clusters    : {kmeans_cfg.n_clusters} per class")
        print(f"  Artifact      : {kmeans_path}  ({size_kb:.1f} KB)")
        print("  KMeans: OK")

    # ------------------------------------------------------------------
    # 4. Train Gaussian Naive Bayes
    # ------------------------------------------------------------------
    if train_gnb:
        print("\n[train_voice_models] Training GaussianNaiveBayesClassifier ...")
        X_train, y_train, load_errors_gnb = _load_summaries(split, mfcc_cfg)

        if len(X_train) == 0:
            print("ERROR: No summary vectors extracted — check dataset.", file=sys.stderr)
            sys.exit(1)
        if load_errors_gnb:
            print(f"  WARNING: {load_errors_gnb} file(s) failed to load and were skipped.")

        # Log sample counts per class
        for label in split.labels:
            n = sum(1 for yi in y_train if yi == label)
            print(f"  {label:15s}  {n} samples")
        print(f"  Total samples : {len(X_train)}")
        print(f"  Feature dim   : {X_train.shape[1]}  (n_mfcc={mfcc_cfg.n_mfcc} × 2 mean+std)")

        gnb_cfg = GNBConfig()
        t0 = time.perf_counter()
        gnb_model = GaussianNaiveBayesClassifier(config=gnb_cfg)
        gnb_model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0

        gnb_path = output_dir / 'gnb_model.pkl'
        gnb_model.save(gnb_path)
        size_kb = artifact_size_kb(gnb_path)

        print(f"\n  Training time : {elapsed:.3f}s")
        print(f"  var_epsilon   : {gnb_cfg.var_epsilon}")
        print(f"  Artifact      : {gnb_path}  ({size_kb:.1f} KB)")
        print("  GNB: OK")

    # ------------------------------------------------------------------
    # 5. Final summary
    # ------------------------------------------------------------------
    print("\n--- Training summary ---")
    print(f"  Dataset       : {dataset_root}")
    print(f"  Labels        : {split.labels}")
    print(f"  Train samples : {len(split.train)}")
    print(f"  Test samples  : {len(split.test)}")
    print(f"  Output dir    : {output_dir}")
    if train_kmeans:
        print(f"  kmeans_model  : {output_dir / 'kmeans_model.pkl'}")
    if train_gnb:
        print(f"  gnb_model     : {output_dir / 'gnb_model.pkl'}")
    print("Done.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_frames(
    split: DatasetSplit,
    mfcc_cfg: MFCCConfig,
    dataset_root: Path,
) -> tuple:
    """Load all train samples and extract frame-level MFCCs per class.

    Returns:
        frames_by_class: Dict[label, (N_frames, n_mfcc) ndarray]
        n_errors: int — number of files that failed to load
    """
    accum: Dict[str, list] = {label: [] for label in split.labels}
    errors = 0

    for sample in split.train:
        try:
            signal, _ = load_wav(sample.path, target_sr=mfcc_cfg.sample_rate)
            signal = normalize(signal)
            frames = extract_mfcc_frames(signal, mfcc_cfg)  # (n_frames, n_mfcc)
            accum[sample.label].append(frames)
        except Exception as exc:
            warnings.warn(
                f"Skipping {sample.path.name}: {exc}",
                UserWarning,
                stacklevel=2,
            )
            errors += 1

    frames_by_class: Dict[str, np.ndarray] = {}
    for label, frame_list in accum.items():
        if frame_list:
            frames_by_class[label] = np.concatenate(frame_list, axis=0)

    return frames_by_class, errors


def _load_summaries(
    split: DatasetSplit,
    mfcc_cfg: MFCCConfig,
) -> tuple:
    """Load all train samples and extract fixed-length MFCC summary vectors.

    Returns:
        X:        (N_samples, n_features) float32 ndarray
        y:        list of N_samples string labels
        n_errors: int — number of files that failed to load
    """
    X_list = []
    y_list = []
    errors = 0

    for sample in split.train:
        try:
            signal, _ = load_wav(sample.path, target_sr=mfcc_cfg.sample_rate)
            signal = normalize(signal)
            vec = extract_mfcc_summary(signal, mfcc_cfg)   # (n_mfcc * 2,)
            X_list.append(vec)
            y_list.append(sample.label)
        except Exception as exc:
            warnings.warn(
                f"Skipping {sample.path.name}: {exc}",
                UserWarning,
                stacklevel=2,
            )
            errors += 1

    if X_list:
        X = np.stack(X_list, axis=0)   # (N, n_features)
    else:
        X = np.empty((0, mfcc_cfg.n_mfcc * 2), dtype=np.float32)

    return X, y_list, errors


if __name__ == '__main__':
    main()
