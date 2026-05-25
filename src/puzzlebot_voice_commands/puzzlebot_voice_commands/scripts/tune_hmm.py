"""
tune_hmm.py — Grid search over HMM hyperparameters.

Trains one HMM per (n_states, n_symbols, n_iter) combination using a
stratified k-fold cross-validation and prints a results table sorted by
mean accuracy. Use this to find the best config before final training.

Usage (Windows, no ROS):
  $env:PYTHONPATH = "src\\puzzlebot_voice_commands"
  python -m puzzlebot_voice_commands.scripts.tune_hmm `
    --dataset   src\\puzzlebot_voice_commands\\datasets\\voice_commands_dataset `
    --n-states  5 8 10 `
    --n-symbols 32 64 `
    --n-iter    20 50 `
    --k         3

Default grid: n_states=[5,8], n_symbols=[32,64], n_iter=[20,50], k=3.
"""
import argparse
import sys
import time
import warnings
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..audio_io import load_wav, normalize
from ..config import HMMConfig, MFCCConfig
from ..dataset import Sample, discover_dataset
from ..metrics import accuracy, macro_f1, precision_recall_f1, safety_critical_errors
from ..mfcc import extract_mfcc_frames
from ..models.hmm import HiddenMarkovModelClassifier


def _load_frames(
    samples: List[Sample],
    mfcc_cfg: MFCCConfig,
) -> Tuple[List[np.ndarray], List[str]]:
    frames_list: List[np.ndarray] = []
    labels: List[str] = []
    for sample in samples:
        try:
            signal, _ = load_wav(sample.path, target_sr=mfcc_cfg.sample_rate)
            signal = normalize(signal)
            frames_list.append(extract_mfcc_frames(signal, mfcc_cfg))
            labels.append(sample.label)
        except Exception as exc:
            warnings.warn(f"Skipping {sample.path.name}: {exc}", UserWarning, stacklevel=2)
    return frames_list, labels


def _kfold_indices(n: int, k: int, seed: int = 42):
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


def _evaluate_combo(
    n_states: int,
    n_symbols: int,
    n_iter: int,
    frames_list: List[np.ndarray],
    labels: List[str],
    all_labels: List[str],
    k: int,
    random_state: int,
) -> Dict:
    accs, f1s, recalls, safety_totals, times = [], [], [], [], []

    for train_idx, test_idx in _kfold_indices(len(labels), k, seed=random_state):
        y_test = [labels[i] for i in test_idx]

        seqs_by_class: Dict[str, List[np.ndarray]] = {lbl: [] for lbl in all_labels}
        for i in train_idx:
            seqs_by_class[labels[i]].append(frames_list[i])
        seqs_by_class = {lbl: seqs for lbl, seqs in seqs_by_class.items() if seqs}

        cfg = HMMConfig(n_states=n_states, n_symbols=n_symbols, n_iter=n_iter,
                        random_state=random_state)
        model = HiddenMarkovModelClassifier(config=cfg)

        t0 = time.perf_counter()
        model.fit(seqs_by_class)
        times.append(time.perf_counter() - t0)

        y_pred = [model.predict(frames_list[i])[0] for i in test_idx]
        prf = precision_recall_f1(y_test, y_pred, all_labels)
        sc = safety_critical_errors(y_test, y_pred)
        mr = sum(v['recall'] for v in prf.values()) / len(prf)

        accs.append(accuracy(y_test, y_pred))
        f1s.append(macro_f1(prf))
        recalls.append(mr)
        safety_totals.append(sc['safety_critical_count'])

    return {
        'n_states':    n_states,
        'n_symbols':   n_symbols,
        'n_iter':      n_iter,
        'acc_mean':    float(np.mean(accs)),
        'acc_std':     float(np.std(accs)),
        'f1_mean':     float(np.mean(f1s)),
        'recall_mean': float(np.mean(recalls)),
        'safety_total': int(sum(safety_totals)),
        'train_time_s': float(np.sum(times)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='tune_hmm',
        description=(
            'Grid search over HMM hyperparameters using k-fold cross-validation. '
            'Prints a results table sorted by mean accuracy.'
        ),
    )
    parser.add_argument('--dataset', required=True,
                        help='Path to dataset root folder.')
    parser.add_argument('--n-states', nargs='+', type=int, default=[5, 8],
                        metavar='N',
                        help='Hidden state counts to try (default: 5 8).')
    parser.add_argument('--n-symbols', nargs='+', type=int, default=[32, 64],
                        metavar='N',
                        help='Codebook sizes to try (default: 32 64).')
    parser.add_argument('--n-iter', nargs='+', type=int, default=[20, 50],
                        metavar='N',
                        help='Baum-Welch iteration counts to try (default: 20 50).')
    parser.add_argument('--k', type=int, default=3,
                        help='Number of CV folds (default: 3).')
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--n-mfcc', type=int, default=13)
    parser.add_argument('--n-filters', type=int, default=26)
    parser.add_argument('--cmvn', action='store_true', default=False,
                        help='Apply CMVN normalization to MFCC frames.')
    parser.add_argument('--delta', action='store_true', default=False,
                        help='Append delta coefficients to MFCC frames.')
    parser.add_argument('--delta-delta', action='store_true', default=False,
                        help='Append delta-delta coefficients (requires --delta).')
    parser.add_argument('--random-state', type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    mfcc_cfg = MFCCConfig(
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        n_filters=args.n_filters,
        cmvn=args.cmvn,
        include_delta=args.delta,
        include_delta_delta=args.delta_delta,
    )

    grid = list(product(args.n_states, args.n_symbols, args.n_iter))
    n_combos = len(grid)

    print(f"[tune_hmm] Dataset  : {dataset_root}")
    print(f"[tune_hmm] Grid     : {n_combos} combinations  (k={args.k} folds each)")
    print(f"[tune_hmm] n_states : {args.n_states}")
    print(f"[tune_hmm] n_symbols: {args.n_symbols}")
    print(f"[tune_hmm] n_iter   : {args.n_iter}")
    if args.cmvn:
        print(f"[tune_hmm] CMVN     : enabled")
    if args.delta:
        print(f"[tune_hmm] Delta    : enabled")

    try:
        samples_by_class = discover_dataset(dataset_root)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    all_labels = sorted(samples_by_class.keys())
    all_samples: List[Sample] = [
        Sample(path=p, label=lbl)
        for lbl, paths in samples_by_class.items()
        for p in paths
    ]
    print(f"\n  Classes : {all_labels}")
    print(f"  Samples : {len(all_samples)}")
    print(f"\n  Loading features ...")
    t0 = time.perf_counter()
    frames_list, labels = _load_frames(all_samples, mfcc_cfg)
    print(f"  Done in {time.perf_counter()-t0:.2f}s  ({len(labels)} loaded)")

    results = []
    for combo_i, (ns, nsym, ni) in enumerate(grid, 1):
        print(f"\n  [{combo_i}/{n_combos}] n_states={ns}  n_symbols={nsym}  n_iter={ni}")
        r = _evaluate_combo(
            n_states=ns, n_symbols=nsym, n_iter=ni,
            frames_list=frames_list, labels=labels,
            all_labels=all_labels, k=args.k,
            random_state=args.random_state,
        )
        results.append(r)
        print(f"         acc={r['acc_mean']:.4f}±{r['acc_std']:.4f}  "
              f"recall={r['recall_mean']:.4f}  "
              f"safety={r['safety_total']}  "
              f"time={r['train_time_s']:.0f}s")

    results.sort(key=lambda x: (-x['acc_mean'], x['safety_total']))

    print(f"\n{'='*75}")
    print(f"  Grid search results  (k={args.k}, sorted by mean accuracy)")
    print(f"{'='*75}")
    header = (f"  {'n_states':>8}  {'n_symbols':>9}  {'n_iter':>6}  "
              f"{'acc_mean':>8}  {'acc_std':>7}  {'recall':>7}  "
              f"{'safety':>6}  {'time(s)':>7}")
    print(header)
    print(f"  {'-'*71}")
    for r in results:
        mark = '  <-- BEST' if r is results[0] else ''
        print(f"  {r['n_states']:>8}  {r['n_symbols']:>9}  {r['n_iter']:>6}  "
              f"  {r['acc_mean']:6.4f}  {r['acc_std']:6.4f}  "
              f"  {r['recall_mean']:6.4f}  {r['safety_total']:>6}  "
              f"  {r['train_time_s']:>6.0f}{mark}")

    best = results[0]
    print(f"\n  Best config: n_states={best['n_states']}  "
          f"n_symbols={best['n_symbols']}  n_iter={best['n_iter']}")
    print(f"  → acc={best['acc_mean']:.4f}±{best['acc_std']:.4f}  "
          f"recall={best['recall_mean']:.4f}  safety_errors={best['safety_total']}")
    print(f"\n  To train with best config:")
    print(f"    python -m puzzlebot_voice_commands.scripts.train_hmm \\")
    print(f"      --dataset    {dataset_root} \\")
    print(f"      --output-dir artifacts \\")
    print(f"      --n-states   {best['n_states']} \\")
    print(f"      --n-symbols  {best['n_symbols']} \\")
    print(f"      --n-iter     {best['n_iter']}")
    if args.cmvn:
        print(f"      --cmvn")
    if args.delta:
        print(f"      --delta")


if __name__ == '__main__':
    main()
