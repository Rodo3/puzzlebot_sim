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


def _extract_frames(signal: np.ndarray, cfg: MFCCConfig) -> np.ndarray:
    if cfg.use_librosa:
        from ..librosa_features import extract_librosa_frames
        return extract_librosa_frames(signal, cfg)
    return extract_mfcc_frames(signal, cfg)


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
    parser.add_argument('--n-symbols',   type=int,   default=256,
                        help='Codebook size for observation quantization (default: 256).')
    parser.add_argument('--n-iter',      type=int,   default=20,
                        help='Baum-Welch EM iterations (default: 20).')
    parser.add_argument('--cmvn',        action='store_true', default=False,
                        help='Apply per-utterance cepstral mean-variance normalization.')
    parser.add_argument('--delta',       action='store_true', default=False,
                        help='Append delta coefficients to MFCC frames.')
    parser.add_argument('--delta-delta', action='store_true', default=False,
                        help='Append delta-delta coefficients (requires --delta).')
    parser.add_argument('--syllable-states', action='store_true', default=False,
                        help='Use per-command n_states based on syllable count '
                             '(alto=3, avanzar=4, derecha=4, inicio=4, izquierda=5, retroceder=6, '
                             'subir=3, bajar=3, tomar=3, soltar=3).')
    parser.add_argument('--librosa',     action='store_true', default=False,
                        help='Use librosa for feature extraction instead of manual MFCC.')
    parser.add_argument('--include-zcr', action='store_true', default=False,
                        help='Append zero-crossing rate (requires --librosa).')
    parser.add_argument('--include-rms', action='store_true', default=False,
                        help='Append RMS energy (requires --librosa).')
    parser.add_argument('--include-contrast', action='store_true', default=False,
                        help='Append spectral contrast — 7 features (requires --librosa).')
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
            frames = _extract_frames(signal, mfcc_cfg)
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

    SYLLABLE_STATES = {
        'alto': 4, 'avanzar': 6, 'derecha': 6,
        'inicio': 4, 'izquierda': 5, 'retroceder': 6,
        'subir': 6, 'bajar': 5, 'tomar': 6, 'soltar': 5,
    }  # gridsearch óptimo: n_symbols=64, acc=87.19%
    n_states_per_class = SYLLABLE_STATES if args.syllable_states else {}

    mfcc_cfg    = MFCCConfig(sample_rate=args.sample_rate,
                              n_mfcc=args.n_mfcc, n_filters=args.n_filters,
                              cmvn=args.cmvn,
                              include_delta=args.delta,
                              include_delta_delta=args.delta_delta,
                              use_librosa=args.librosa,
                              include_zcr=args.include_zcr,
                              include_rms=args.include_rms,
                              include_contrast=args.include_contrast)
    dataset_cfg = DatasetConfig(test_ratio=args.test_ratio,
                                random_state=args.random_state)
    hmm_cfg     = HMMConfig(
        n_states=args.n_states,
        n_symbols=args.n_symbols,
        n_iter=args.n_iter,
        random_state=args.random_state,
        n_states_per_class=n_states_per_class,
    )

    print(f"[train_hmm] Dataset    : {dataset_root}")
    print(f"[train_hmm] n_states   : {hmm_cfg.n_states}")
    print(f"[train_hmm] n_symbols  : {hmm_cfg.n_symbols}")
    print(f"[train_hmm] n_iter     : {hmm_cfg.n_iter}")
    if hmm_cfg.n_states_per_class:
        print(f"[train_hmm] n_states/class: {hmm_cfg.n_states_per_class}")

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
        'n_states':        hmm_cfg.n_states,
        'n_symbols':       hmm_cfg.n_symbols,
        'n_iter':          hmm_cfg.n_iter,
        'kmeans_max_iter': hmm_cfg.kmeans_max_iter,
        'kmeans_tol':      hmm_cfg.kmeans_tol,
        'random_state':    hmm_cfg.random_state,
        'log_zero':        hmm_cfg.log_zero,
        'n_states_per_class': hmm_cfg.n_states_per_class,
        'mfcc': {
            'sample_rate':         mfcc_cfg.sample_rate,
            'pre_emphasis':        mfcc_cfg.pre_emphasis,
            'frame_size':          mfcc_cfg.frame_size,
            'frame_stride':        mfcc_cfg.frame_stride,
            'n_fft':               mfcc_cfg.n_fft,
            'n_filters':           mfcc_cfg.n_filters,
            'n_mfcc':              mfcc_cfg.n_mfcc,
            'include_delta':       mfcc_cfg.include_delta,
            'include_delta_delta': mfcc_cfg.include_delta_delta,
            'cmvn':                mfcc_cfg.cmvn,
            'use_librosa':          mfcc_cfg.use_librosa,
            'include_zcr':         mfcc_cfg.include_zcr,
            'include_rms':         mfcc_cfg.include_rms,
            'include_contrast':    mfcc_cfg.include_contrast,
        },
    }
    save_json(cfg_dict, output_dir / 'hmm_config.json')
    print(f"  Config saved: {output_dir / 'hmm_config.json'}")

    # Quick sanity check on train set
    correct = 0
    total   = 0
    for label, seqs in sequences_by_class.items():
        for frames in seqs:
            pred, _ = model.predict(frames)  # frames already extracted above
            correct += int(pred == label)
            total   += 1
    if total:
        print(f"\n  Train accuracy (sanity): {correct}/{total} = {correct/total:.3f}")


if __name__ == '__main__':
    main()
