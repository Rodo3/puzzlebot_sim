"""
train_hmm.py — Train the HiddenMarkovModel classifier and save artifacts.

Usage (Windows, no ROS):
  $env:PYTHONPATH = "src\\puzzlebot_voice_commands"
  python -m puzzlebot_voice_commands.scripts.train_hmm `
    --dataset    datasets\\voice_commands_dataset `
    --output-dir artifacts

Usage (ROS 2 / WSL2):
  ros2 run puzzlebot_voice_commands train_hmm_models \\
    --dataset    src/puzzlebot_voice_commands/datasets/voice_commands_dataset \\
    --output-dir src/puzzlebot_voice_commands/artifacts

Artifacts written to --output-dir:
  hmm_model.pkl     — serialized HiddenMarkovModelClassifier
  hmm_config.json   — HMMConfig parameters used for training
"""
import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..audio_io import load_wav, normalize
from ..config import DatasetConfig, HMMConfig, MFCCConfig
from ..dataset import DatasetSplit, Sample, discover_dataset, split_dataset
from ..mfcc import extract_mfcc_frames
from ..models.hmm import HiddenMarkovModelClassifier
from ..serialization import save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='train_hmm_models',
        description='Train a Hidden Markov Model classifier for voice commands.',
    )
    parser.add_argument('--dataset',     required=True,
                        help='Path to dataset root folder.')
    parser.add_argument('--output-dir',  required=True,
                        help='Directory for model artifacts.')
    parser.add_argument('--test-ratio',  type=float, default=0.3)
    parser.add_argument('--random-state', type=int,  default=42)
    parser.add_argument('--sample-rate', type=int,   default=16000)
    parser.add_argument('--n-mfcc',      type=int,   default=13)
    parser.add_argument('--n-filters',   type=int,   default=26)
    parser.add_argument('--n-states',    type=int,   default=5,
                        help='Number of hidden states per HMM (default: 5).')
    parser.add_argument('--n-symbols',   type=int,   default=32,
                        help='Codebook size for observation quantization (default: 32).')
    parser.add_argument('--n-iter',      type=int,   default=20,
                        help='Baum-Welch EM iterations (default: 20).')
    return parser


def _load_frames(
    samples: List[Sample],
    mfcc_cfg: MFCCConfig,
) -> Dict[str, List[np.ndarray]]:
    """Load frame-level MFCCs grouped by class label."""
    sequences_by_class: Dict[str, List[np.ndarray]] = {}
    for sample in samples:
        try:
            signal, _ = load_wav(sample.path, target_sr=mfcc_cfg.sample_rate)
            signal = normalize(signal)
            frames = extract_mfcc_frames(signal, mfcc_cfg)
            sequences_by_class.setdefault(sample.label, []).append(frames)
        except Exception as exc:
            warnings.warn(f"Skipping {sample.path.name}: {exc}",
                          UserWarning, stacklevel=2)
    return sequences_by_class


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mfcc_cfg    = MFCCConfig(sample_rate=args.sample_rate,
                              n_mfcc=args.n_mfcc, n_filters=args.n_filters)
    dataset_cfg = DatasetConfig(test_ratio=args.test_ratio,
                                random_state=args.random_state)
    hmm_cfg     = HMMConfig(
        n_states=args.n_states,
        n_symbols=args.n_symbols,
        n_iter=args.n_iter,
        random_state=args.random_state,
    )

    print(f"[train_hmm] Dataset    : {dataset_root}")
    print(f"[train_hmm] n_states   : {hmm_cfg.n_states}")
    print(f"[train_hmm] n_symbols  : {hmm_cfg.n_symbols}")
    print(f"[train_hmm] n_iter     : {hmm_cfg.n_iter}")

    try:
        samples_by_class = discover_dataset(dataset_root)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    split: DatasetSplit = split_dataset(samples_by_class, dataset_cfg)
    print(split.summary())
    print(f"\n  Training on {len(split.train)} samples ...")

    sequences_by_class = _load_frames(split.train, mfcc_cfg)

    t0 = time.perf_counter()
    model = HiddenMarkovModelClassifier(config=hmm_cfg)
    model.fit(sequences_by_class)
    elapsed = time.perf_counter() - t0

    print(f"  Training done in {elapsed:.1f}s")
    print(f"  Classes : {model.labels_}")

    model_path = output_dir / 'hmm_model.pkl'
    model.save(model_path)
    print(f"  Model saved : {model_path}")

    cfg_dict = {
        'n_states':       hmm_cfg.n_states,
        'n_symbols':      hmm_cfg.n_symbols,
        'n_iter':         hmm_cfg.n_iter,
        'kmeans_max_iter': hmm_cfg.kmeans_max_iter,
        'kmeans_tol':     hmm_cfg.kmeans_tol,
        'random_state':   hmm_cfg.random_state,
        'log_zero':       hmm_cfg.log_zero,
    }
    save_json(cfg_dict, output_dir / 'hmm_config.json')
    print(f"  Config saved: {output_dir / 'hmm_config.json'}")

    # Quick sanity check on train set
    correct = 0
    total   = 0
    for label, seqs in sequences_by_class.items():
        for frames in seqs:
            pred, _ = model.predict(frames)
            correct += int(pred == label)
            total   += 1
    if total:
        print(f"\n  Train accuracy (sanity): {correct}/{total} = {correct/total:.3f}")


if __name__ == '__main__':
    main()
