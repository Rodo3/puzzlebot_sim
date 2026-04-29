"""
cross_validate.py — K-fold cross-validation for KMeans and GaussianNB.

Splits the dataset into k folds, trains on k-1 and evaluates on 1,
repeating k times. Reports mean and std of accuracy, macro F1, macro
recall, and safety errors across folds.

Answers: "Is the 98% accuracy real or just a lucky split?"

Usage (Windows, no ROS):
  $env:PYTHONPATH = "src\\puzzlebot_voice_commands"
  python -m puzzlebot_voice_commands.scripts.cross_validate \\
    --dataset src\\puzzlebot_voice_commands\\datasets\\voice_commands_dataset \\
    --model   both \\
    --k       5
"""
import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..audio_io import load_wav, normalize
from ..config import DatasetConfig, GNBConfig, HMMConfig, KMeansConfig, MFCCConfig
from ..dataset import Sample, discover_dataset
from ..metrics import accuracy, macro_f1, precision_recall_f1, safety_critical_errors
from ..mfcc import extract_mfcc_frames, extract_mfcc_summary
from ..models.gaussian_nb import GaussianNaiveBayesClassifier
from ..models.hmm import HiddenMarkovModelClassifier
from ..models.kmeans_codebook import KMeansCodebookClassifier


def _load_all_samples(
    samples: List[Sample],
    mfcc_cfg: MFCCConfig,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """Load frames and summary vectors for every sample.

    Returns:
        frames_list:   one (n_frames, n_mfcc) array per sample
        summaries:     one (n_mfcc*2,) array per sample
        labels:        string label per sample
    """
    frames_list = []
    summaries = []
    labels = []

    for sample in samples:
        try:
            signal, _ = load_wav(sample.path, target_sr=mfcc_cfg.sample_rate)
            signal = normalize(signal)
            frames_list.append(extract_mfcc_frames(signal, mfcc_cfg))
            summaries.append(extract_mfcc_summary(signal, mfcc_cfg))
            labels.append(sample.label)
        except Exception as exc:
            warnings.warn(f"Skipping {sample.path.name}: {exc}", UserWarning, stacklevel=2)

    return frames_list, summaries, labels


def _kfold_indices(n: int, k: int, seed: int = 42):
    """Yield (train_indices, test_indices) for each fold."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    fold_sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]

    start = 0
    folds = []
    for size in fold_sizes:
        folds.append(indices[start:start + size])
        start += size

    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx.tolist(), test_idx.tolist()


def _run_cv(
    frames_list: List[np.ndarray],
    summaries: List[np.ndarray],
    labels: List[str],
    all_labels: List[str],
    k: int,
    run_kmeans: bool,
    run_gnb: bool,
    run_hmm: bool,
    mfcc_cfg: MFCCConfig,
) -> Dict:
    n = len(labels)
    results = {
        'kmeans': {'accuracy': [], 'macro_f1': [], 'macro_recall': [], 'safety_errors': []},
        'gnb':    {'accuracy': [], 'macro_f1': [], 'macro_recall': [], 'safety_errors': []},
        'hmm':    {'accuracy': [], 'macro_f1': [], 'macro_recall': [], 'safety_errors': []},
    }

    for fold_i, (train_idx, test_idx) in enumerate(_kfold_indices(n, k), 1):
        print(f"\n  Fold {fold_i}/{k}  "
              f"(train={len(train_idx)}, test={len(test_idx)})")

        y_test = [labels[i] for i in test_idx]

        if run_kmeans:
            # Build frames_by_class from train split
            frames_by_class: Dict[str, List[np.ndarray]] = {lbl: [] for lbl in all_labels}
            for i in train_idx:
                frames_by_class[labels[i]].append(frames_list[i])

            # Skip classes with no train frames (can happen in small datasets)
            frames_by_class = {
                lbl: np.concatenate(fl, axis=0)
                for lbl, fl in frames_by_class.items() if fl
            }

            model_km = KMeansCodebookClassifier(config=KMeansConfig())
            model_km.fit(frames_by_class)

            y_pred_km = [model_km.predict(frames_list[i])[0] for i in test_idx]
            prf = precision_recall_f1(y_test, y_pred_km, all_labels)
            sc  = safety_critical_errors(y_test, y_pred_km)
            mr  = sum(v['recall'] for v in prf.values()) / len(prf)

            results['kmeans']['accuracy'].append(accuracy(y_test, y_pred_km))
            results['kmeans']['macro_f1'].append(macro_f1(prf))
            results['kmeans']['macro_recall'].append(mr)
            results['kmeans']['safety_errors'].append(sc['safety_critical_count'])
            print(f"    KMeans   acc={accuracy(y_test, y_pred_km):.4f}  "
                  f"recall={mr:.4f}  safety_err={sc['safety_critical_count']}")

        if run_gnb:
            X_train = np.stack([summaries[i] for i in train_idx])
            y_train = [labels[i] for i in train_idx]

            model_gnb = GaussianNaiveBayesClassifier(config=GNBConfig())
            model_gnb.fit(X_train, y_train)

            y_pred_gnb = [model_gnb.predict(summaries[i])[0] for i in test_idx]
            prf = precision_recall_f1(y_test, y_pred_gnb, all_labels)
            sc  = safety_critical_errors(y_test, y_pred_gnb)
            mr  = sum(v['recall'] for v in prf.values()) / len(prf)

            results['gnb']['accuracy'].append(accuracy(y_test, y_pred_gnb))
            results['gnb']['macro_f1'].append(macro_f1(prf))
            results['gnb']['macro_recall'].append(mr)
            results['gnb']['safety_errors'].append(sc['safety_critical_count'])
            print(f"    GaussianNB acc={accuracy(y_test, y_pred_gnb):.4f}  "
                  f"recall={mr:.4f}  safety_err={sc['safety_critical_count']}")

        if run_hmm:
            seqs_by_class: Dict[str, List[np.ndarray]] = {lbl: [] for lbl in all_labels}
            for i in train_idx:
                seqs_by_class[labels[i]].append(frames_list[i])
            seqs_by_class = {lbl: seqs for lbl, seqs in seqs_by_class.items() if seqs}

            model_hmm = HiddenMarkovModelClassifier(config=HMMConfig())
            model_hmm.fit(seqs_by_class)

            y_pred_hmm = [model_hmm.predict(frames_list[i])[0] for i in test_idx]
            prf = precision_recall_f1(y_test, y_pred_hmm, all_labels)
            sc  = safety_critical_errors(y_test, y_pred_hmm)
            mr  = sum(v['recall'] for v in prf.values()) / len(prf)

            results['hmm']['accuracy'].append(accuracy(y_test, y_pred_hmm))
            results['hmm']['macro_f1'].append(macro_f1(prf))
            results['hmm']['macro_recall'].append(mr)
            results['hmm']['safety_errors'].append(sc['safety_critical_count'])
            print(f"    HMM        acc={accuracy(y_test, y_pred_hmm):.4f}  "
                  f"recall={mr:.4f}  safety_err={sc['safety_critical_count']}")

    return results


def _print_cv_summary(results: Dict, k: int) -> None:
    print(f"\n{'='*60}")
    print(f"  Cross-validation summary  (k={k})")
    print(f"{'='*60}")

    for model_name, key in [('KMeans', 'kmeans'), ('GaussianNB', 'gnb'), ('HMM', 'hmm')]:
        r = results[key]
        if not r['accuracy']:
            continue
        acc  = np.array(r['accuracy'])
        f1   = np.array(r['macro_f1'])
        rec  = np.array(r['macro_recall'])
        safe = np.array(r['safety_errors'])

        print(f"\n  {model_name}")
        print(f"  {'Metric':<22} {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
        print(f"  {'-'*58}")
        print(f"  {'Accuracy':<22} {acc.mean():8.4f}  {acc.std():8.4f}  "
              f"{acc.min():8.4f}  {acc.max():8.4f}")
        print(f"  {'Macro recall':<22} {rec.mean():8.4f}  {rec.std():8.4f}  "
              f"{rec.min():8.4f}  {rec.max():8.4f}")
        print(f"  {'Macro F1':<22} {f1.mean():8.4f}  {f1.std():8.4f}  "
              f"{f1.min():8.4f}  {f1.max():8.4f}")
        print(f"  {'Safety errors (total)':<22} {safe.sum():8.0f}  "
              f"{'(across all folds)':>28}")

        if acc.std() > 0.05:
            print(f"  WARNING: high variance (std={acc.std():.4f}) "
                  "— dataset may be too small for reliable estimates.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='cross_validate',
        description='K-fold cross-validation to verify model reliability.',
    )
    parser.add_argument('--dataset', required=True,
                        help='Path to dataset root folder.')
    parser.add_argument('--model', choices=['kmeans', 'gnb', 'hmm', 'both', 'all'],
                        default='both',
                        help='Models to cross-validate (both=kmeans+gnb, all=kmeans+gnb+hmm).')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of folds (default: 5).')
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--n-mfcc', type=int, default=13)
    parser.add_argument('--n-filters', type=int, default=26)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    mfcc_cfg = MFCCConfig(
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        n_filters=args.n_filters,
    )
    run_kmeans = args.model in ('kmeans', 'both', 'all')
    run_gnb    = args.model in ('gnb',    'both', 'all')
    run_hmm    = args.model in ('hmm',            'all')

    print(f"[cross_validate] Dataset : {dataset_root}")
    print(f"[cross_validate] k={args.k}  model={args.model}")

    try:
        samples_by_class = discover_dataset(dataset_root)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    all_labels = sorted(samples_by_class.keys())
    all_samples: List[Sample] = []
    for lbl, paths in samples_by_class.items():
        for p in paths:
            all_samples.append(Sample(path=p, label=lbl))

    print(f"  Classes : {all_labels}")
    print(f"  Samples : {len(all_samples)}")
    print(f"\n  Loading features ...")
    t0 = time.perf_counter()
    frames_list, summaries, labels = _load_all_samples(all_samples, mfcc_cfg)
    print(f"  Done in {time.perf_counter()-t0:.2f}s  ({len(labels)} loaded)")

    results = _run_cv(
        frames_list, summaries, labels, all_labels,
        k=args.k,
        run_kmeans=run_kmeans,
        run_gnb=run_gnb,
        run_hmm=run_hmm,
        mfcc_cfg=mfcc_cfg,
    )

    _print_cv_summary(results, args.k)


if __name__ == '__main__':
    main()
