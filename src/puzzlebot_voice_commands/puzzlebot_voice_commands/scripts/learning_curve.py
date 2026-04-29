"""
learning_curve.py — Train with increasing fractions of data and plot accuracy.

Answers: "Do we have enough recordings per person, or does accuracy keep
rising as we add more samples?"

If train accuracy stays near 1.0 while test accuracy is much lower and
keeps rising -> overfitting, need more data.
If both curves plateau -> enough data for the current model.

Usage (Windows, no ROS):
  $env:PYTHONPATH = "src\\puzzlebot_voice_commands"
  python -m puzzlebot_voice_commands.scripts.learning_curve \\
    --dataset    src\\puzzlebot_voice_commands\\datasets\\voice_commands_dataset \\
    --model      both \\
    --output-dir src\\puzzlebot_voice_commands\\reports
"""
import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..audio_io import load_wav, normalize
from ..config import DatasetConfig, GNBConfig, HMMConfig, KMeansConfig, MFCCConfig
from ..dataset import DatasetSplit, discover_dataset, split_dataset, Sample
from ..metrics import accuracy, macro_f1, precision_recall_f1
from ..mfcc import extract_mfcc_frames, extract_mfcc_summary
from ..models.gaussian_nb import GaussianNaiveBayesClassifier
from ..models.hmm import HiddenMarkovModelClassifier
from ..models.kmeans_codebook import KMeansCodebookClassifier
from ..serialization import save_json


# Training size fractions to evaluate
_FRACTIONS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]


def _load_samples(
    samples: List[Sample],
    mfcc_cfg: MFCCConfig,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    frames_list, summaries, labels = [], [], []
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


def _evaluate_at_fraction(
    train_frames: List[np.ndarray],
    train_summaries: List[np.ndarray],
    train_labels: List[str],
    test_frames: List[np.ndarray],
    test_summaries: List[np.ndarray],
    test_labels: List[str],
    all_labels: List[str],
    fraction: float,
    run_kmeans: bool,
    run_gnb: bool,
    run_hmm: bool,
    seed: int = 42,
) -> Dict:
    n_train_total = len(train_labels)
    n_use = max(len(all_labels), int(n_train_total * fraction))  # at least 1 per class
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_train_total, size=n_use, replace=False).tolist()

    result = {'fraction': fraction, 'n_train': n_use}

    if run_kmeans:
        frames_by_class: Dict[str, List] = {lbl: [] for lbl in all_labels}
        for i in indices:
            frames_by_class[train_labels[i]].append(train_frames[i])
        frames_by_class = {
            lbl: np.concatenate(fl, axis=0)
            for lbl, fl in frames_by_class.items() if fl
        }
        model = KMeansCodebookClassifier(config=KMeansConfig())
        model.fit(frames_by_class)

        # Train accuracy (subset used)
        y_train_sub = [train_labels[i] for i in indices]
        y_pred_train = [model.predict(train_frames[i])[0] for i in indices]
        # Test accuracy (full test set)
        y_pred_test = [model.predict(f)[0] for f in test_frames]

        result['kmeans_train_acc'] = round(accuracy(y_train_sub, y_pred_train), 4)
        result['kmeans_test_acc']  = round(accuracy(test_labels, y_pred_test), 4)
        prf = precision_recall_f1(test_labels, y_pred_test, all_labels)
        result['kmeans_test_f1']   = round(macro_f1(prf), 4)
        result['kmeans_test_recall'] = round(
            sum(v['recall'] for v in prf.values()) / len(prf), 4
        )

    if run_gnb:
        X_train = np.stack([train_summaries[i] for i in indices])
        y_train_sub = [train_labels[i] for i in indices]
        model = GaussianNaiveBayesClassifier(config=GNBConfig())
        model.fit(X_train, y_train_sub)

        y_pred_train = [model.predict(train_summaries[i])[0] for i in indices]
        X_test = np.stack(test_summaries)
        y_pred_test = [model.predict(x)[0] for x in X_test]

        result['gnb_train_acc'] = round(accuracy(y_train_sub, y_pred_train), 4)
        result['gnb_test_acc']  = round(accuracy(test_labels, y_pred_test), 4)
        prf = precision_recall_f1(test_labels, y_pred_test, all_labels)
        result['gnb_test_f1']   = round(macro_f1(prf), 4)
        result['gnb_test_recall'] = round(
            sum(v['recall'] for v in prf.values()) / len(prf), 4
        )

    if run_hmm:
        seqs_by_class: Dict[str, List[np.ndarray]] = {lbl: [] for lbl in all_labels}
        for i in indices:
            seqs_by_class[train_labels[i]].append(train_frames[i])
        seqs_by_class = {lbl: seqs for lbl, seqs in seqs_by_class.items() if seqs}

        model_hmm = HiddenMarkovModelClassifier(config=HMMConfig())
        model_hmm.fit(seqs_by_class)

        y_train_sub = [train_labels[i] for i in indices]
        y_pred_train = [model_hmm.predict(train_frames[i])[0] for i in indices]
        y_pred_test  = [model_hmm.predict(f)[0] for f in test_frames]

        result['hmm_train_acc'] = round(accuracy(y_train_sub, y_pred_train), 4)
        result['hmm_test_acc']  = round(accuracy(test_labels, y_pred_test), 4)
        prf = precision_recall_f1(test_labels, y_pred_test, all_labels)
        result['hmm_test_f1']   = round(macro_f1(prf), 4)
        result['hmm_test_recall'] = round(
            sum(v['recall'] for v in prf.values()) / len(prf), 4
        )

    return result


def _print_table(rows: List[Dict], run_kmeans: bool, run_gnb: bool, run_hmm: bool) -> None:
    print(f"\n{'='*72}")
    print("  Learning Curve")
    print(f"{'='*72}")

    if run_kmeans:
        print("\n  KMeans")
        print(f"  {'Frac':>6}  {'N train':>8}  {'Train acc':>10}  "
              f"{'Test acc':>10}  {'Test F1':>8}  {'Test Rec':>9}")
        print(f"  {'-'*60}")
        for r in rows:
            gap = r['kmeans_train_acc'] - r['kmeans_test_acc']
            flag = '  <-- overfit?' if gap > 0.10 else ''
            print(f"  {r['fraction']:6.0%}  {r['n_train']:8d}  "
                  f"{r['kmeans_train_acc']:10.4f}  {r['kmeans_test_acc']:10.4f}  "
                  f"{r['kmeans_test_f1']:8.4f}  {r['kmeans_test_recall']:9.4f}{flag}")

    if run_gnb:
        print("\n  GaussianNB")
        print(f"  {'Frac':>6}  {'N train':>8}  {'Train acc':>10}  "
              f"{'Test acc':>10}  {'Test F1':>8}  {'Test Rec':>9}")
        print(f"  {'-'*60}")
        for r in rows:
            gap = r['gnb_train_acc'] - r['gnb_test_acc']
            flag = '  <-- overfit?' if gap > 0.10 else ''
            print(f"  {r['fraction']:6.0%}  {r['n_train']:8d}  "
                  f"{r['gnb_train_acc']:10.4f}  {r['gnb_test_acc']:10.4f}  "
                  f"{r['gnb_test_f1']:8.4f}  {r['gnb_test_recall']:9.4f}{flag}")

    if run_hmm:
        print("\n  HMM")
        print(f"  {'Frac':>6}  {'N train':>8}  {'Train acc':>10}  "
              f"{'Test acc':>10}  {'Test F1':>8}  {'Test Rec':>9}")
        print(f"  {'-'*60}")
        for r in rows:
            gap = r['hmm_train_acc'] - r['hmm_test_acc']
            flag = '  <-- overfit?' if gap > 0.10 else ''
            print(f"  {r['fraction']:6.0%}  {r['n_train']:8d}  "
                  f"{r['hmm_train_acc']:10.4f}  {r['hmm_test_acc']:10.4f}  "
                  f"{r['hmm_test_f1']:8.4f}  {r['hmm_test_recall']:9.4f}{flag}")

    print(f"\n  Interpretation:")
    print("  - Train acc >> Test acc at 100% -> overfitting")
    print("  - Test acc still rising at 100% -> need more data")
    print("  - Test acc plateaus             -> sufficient data for this model")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='learning_curve',
        description='Plot accuracy vs training set size to detect overfitting.',
    )
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model', choices=['kmeans', 'gnb', 'hmm', 'both', 'all'],
                        default='both',
                        help='Models to evaluate (both=kmeans+gnb, all=kmeans+gnb+hmm).')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to save learning_curve.json.')
    parser.add_argument('--test-ratio', type=float, default=0.3)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--n-mfcc', type=int, default=13)
    parser.add_argument('--n-filters', type=int, default=26)
    return parser


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
    run_kmeans  = args.model in ('kmeans', 'both', 'all')
    run_gnb     = args.model in ('gnb',    'both', 'all')
    run_hmm     = args.model in ('hmm',            'all')

    print(f"[learning_curve] Dataset: {dataset_root}")
    try:
        samples_by_class = discover_dataset(dataset_root)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    split: DatasetSplit = split_dataset(samples_by_class, dataset_cfg)
    all_labels = split.labels
    print(split.summary())

    print("\n  Loading train features...")
    tr_frames, tr_sums, tr_labels = _load_samples(split.train, mfcc_cfg)
    print("  Loading test features...")
    te_frames, te_sums, te_labels = _load_samples(split.test, mfcc_cfg)

    rows = []
    for frac in _FRACTIONS:
        print(f"  Training at {frac:.0%}...", end=' ', flush=True)
        row = _evaluate_at_fraction(
            tr_frames, tr_sums, tr_labels,
            te_frames, te_sums, te_labels,
            all_labels, frac, run_kmeans, run_gnb, run_hmm,
            seed=args.random_state,
        )
        rows.append(row)
        print("done")

    _print_table(rows, run_kmeans, run_gnb, run_hmm)

    out_path = output_dir / 'learning_curve.json'
    save_json(rows, out_path)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
