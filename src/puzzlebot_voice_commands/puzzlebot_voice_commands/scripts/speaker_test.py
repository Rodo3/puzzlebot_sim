"""
speaker_test.py — Per-speaker evaluation to verify each team member's voice
is reliably recognized.

Since the system is designed for exactly 4 known speakers, this test answers:
  "Does the model trained on all speakers correctly recognize EACH person?"

Two modes:
  --mode leave-one-out  Train on N-1 speakers, test on the left-out speaker.
                        Reveals how well the model generalizes within the team.
                        Low accuracy here means that speaker needs more recordings.

  --mode all-train      Train on ALL speakers, test each speaker individually.
                        Answers: "After full training, how well does the model
                        recognize each specific voice?"
                        This is the most relevant mode for the final system.

Dataset layout expected:
  <dataset-dir>/
  ├── jorge_avanzar_01.wav   (produced by merge_datasets.py)
  ├── jorge_alto_01.wav
  ├── valeria_avanzar_01.wav
  └── ...

The speaker name is extracted from the filename prefix (up to the first '_').

Usage (Windows, no ROS):
  $env:PYTHONPATH = "src\\puzzlebot_voice_commands"
  python -m puzzlebot_voice_commands.scripts.speaker_test \\
    --dataset    src\\puzzlebot_voice_commands\\datasets\\voice_commands_dataset \\
    --model      gnb \\
    --mode       all-train
"""
import argparse
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..audio_io import load_wav, normalize
from ..config import GNBConfig, HMMConfig, KMeansConfig, MFCCConfig
from ..dataset import Sample, discover_dataset
from ..metrics import accuracy, macro_f1, precision_recall_f1, safety_critical_errors
from ..mfcc import extract_mfcc_frames, extract_mfcc_summary
from ..models.gaussian_nb import GaussianNaiveBayesClassifier
from ..models.hmm import HiddenMarkovModelClassifier
from ..models.kmeans_codebook import KMeansCodebookClassifier
from ..serialization import save_json


def _speaker_from_filename(path: Path) -> str:
    """Extract speaker name from filename: 'jorge_avanzar_01.wav' -> 'jorge'."""
    return path.stem.split('_')[0]


def _load_dataset_with_speakers(
    dataset_root: Path,
    mfcc_cfg: MFCCConfig,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], List[str]]:
    """Load all samples and return (frames, summaries, labels, speakers)."""
    try:
        samples_by_class = discover_dataset(dataset_root)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    all_frames, all_sums, all_labels, all_speakers = [], [], [], []

    for label, paths in samples_by_class.items():
        for p in paths:
            speaker = _speaker_from_filename(p)
            try:
                signal, _ = load_wav(p, target_sr=mfcc_cfg.sample_rate)
                signal = normalize(signal)
                all_frames.append(extract_mfcc_frames(signal, mfcc_cfg))
                all_sums.append(extract_mfcc_summary(signal, mfcc_cfg))
                all_labels.append(label)
                all_speakers.append(speaker)
            except Exception as exc:
                warnings.warn(f"Skipping {p.name}: {exc}", UserWarning, stacklevel=2)

    return all_frames, all_sums, all_labels, all_speakers


def _fit_kmeans(frames_list, labels, all_labels):
    frames_by_class: Dict[str, List] = {lbl: [] for lbl in all_labels}
    for f, lbl in zip(frames_list, labels):
        frames_by_class[lbl].append(f)
    frames_by_class = {
        lbl: np.concatenate(fl, axis=0)
        for lbl, fl in frames_by_class.items() if fl
    }
    model = KMeansCodebookClassifier(config=KMeansConfig())
    model.fit(frames_by_class)
    return model


def _fit_gnb(summaries, labels):
    X = np.stack(summaries)
    model = GaussianNaiveBayesClassifier(config=GNBConfig())
    model.fit(X, labels)
    return model


def _fit_hmm(frames_list, labels, all_labels):
    seqs_by_class: Dict[str, List] = {lbl: [] for lbl in all_labels}
    for f, lbl in zip(frames_list, labels):
        seqs_by_class[lbl].append(f)
    seqs_by_class = {lbl: seqs for lbl, seqs in seqs_by_class.items() if seqs}
    model = HiddenMarkovModelClassifier(config=HMMConfig())
    model.fit(seqs_by_class)
    return model


def _evaluate_speaker(
    model_km, model_gnb, model_hmm,
    frames_list, summaries, labels,
    all_labels, speaker,
    run_kmeans, run_gnb, run_hmm,
) -> Dict:
    result = {'speaker': speaker, 'n_samples': len(labels)}

    if run_kmeans and model_km is not None:
        y_pred = [model_km.predict(f)[0] for f in frames_list]
        prf = precision_recall_f1(labels, y_pred, all_labels)
        sc  = safety_critical_errors(labels, y_pred)
        mr  = sum(v['recall'] for v in prf.values()) / len(prf)
        result['kmeans'] = {
            'accuracy':      round(accuracy(labels, y_pred), 4),
            'macro_recall':  round(mr, 4),
            'macro_f1':      round(macro_f1(prf), 4),
            'safety_errors': sc['safety_critical_count'],
            'per_class_recall': {lbl: round(prf[lbl]['recall'], 4) for lbl in all_labels},
        }

    if run_gnb and model_gnb is not None:
        y_pred = [model_gnb.predict(s)[0] for s in summaries]
        prf = precision_recall_f1(labels, y_pred, all_labels)
        sc  = safety_critical_errors(labels, y_pred)
        mr  = sum(v['recall'] for v in prf.values()) / len(prf)
        result['gnb'] = {
            'accuracy':      round(accuracy(labels, y_pred), 4),
            'macro_recall':  round(mr, 4),
            'macro_f1':      round(macro_f1(prf), 4),
            'safety_errors': sc['safety_critical_count'],
            'per_class_recall': {lbl: round(prf[lbl]['recall'], 4) for lbl in all_labels},
        }

    if run_hmm and model_hmm is not None:
        y_pred = [model_hmm.predict(f)[0] for f in frames_list]
        prf = precision_recall_f1(labels, y_pred, all_labels)
        sc  = safety_critical_errors(labels, y_pred)
        mr  = sum(v['recall'] for v in prf.values()) / len(prf)
        result['hmm'] = {
            'accuracy':      round(accuracy(labels, y_pred), 4),
            'macro_recall':  round(mr, 4),
            'macro_f1':      round(macro_f1(prf), 4),
            'safety_errors': sc['safety_critical_count'],
            'per_class_recall': {lbl: round(prf[lbl]['recall'], 4) for lbl in all_labels},
        }

    return result


def _print_results(results: List[Dict], all_labels: List[str],
                   run_kmeans: bool, run_gnb: bool, run_hmm: bool, mode: str) -> None:
    print(f"\n{'='*65}")
    print(f"  Speaker test  (mode={mode})")
    print(f"{'='*65}")

    model_list = (([('kmeans', 'KMeans')] if run_kmeans else []) +
                  ([('gnb', 'GaussianNB')] if run_gnb else []) +
                  ([('hmm', 'HMM')] if run_hmm else []))
    for model_key, model_name in model_list:
        print(f"\n  {model_name}")
        print(f"  {'Speaker':<14}  {'N':>4}  {'Acc':>7}  {'Recall':>7}  "
              f"{'F1':>7}  {'SafeErr':>7}")
        print(f"  {'-'*55}")

        for r in results:
            if model_key not in r:
                continue
            m = r[model_key]
            flag = '  <-- BAJO' if m['macro_recall'] < 0.90 else ''
            safe_flag = '  *** SAFETY ***' if m['safety_errors'] > 0 else ''
            print(f"  {r['speaker']:<14}  {r['n_samples']:4d}  "
                  f"{m['accuracy']:7.4f}  {m['macro_recall']:7.4f}  "
                  f"{m['macro_f1']:7.4f}  {m['safety_errors']:7d}"
                  f"{flag}{safe_flag}")

        # Per-class recall table
        print(f"\n  {model_name} — per-class recall per speaker")
        col_w = 9
        speakers = [r['speaker'] for r in results if model_key in r]
        header = f"  {'Class':<14}" + "".join(f"{s:>{col_w}}" for s in speakers)
        print(header)
        print(f"  {'-'*14}" + "-" * col_w * len(speakers))
        for lbl in all_labels:
            row = f"  {lbl:<14}"
            for r in results:
                if model_key not in r:
                    continue
                val = r[model_key]['per_class_recall'].get(lbl, 0.0)
                marker = '*' if val < 0.90 else ' '
                row += f"{marker}{val:>{col_w-1}.4f}"
            print(row)
        print("  (* = recall < 0.90)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='speaker_test',
        description='Per-speaker evaluation to verify each team member is recognized.',
    )
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model', choices=['kmeans', 'gnb', 'hmm', 'both', 'all'],
                        default='gnb',
                        help='Models to evaluate (both=kmeans+gnb, all=kmeans+gnb+hmm).')
    parser.add_argument('--mode', choices=['all-train', 'leave-one-out'],
                        default='all-train',
                        help=(
                            'all-train: train on all, test each speaker separately. '
                            'leave-one-out: train on N-1 speakers, test on the left-out one.'
                        ))
    parser.add_argument('--output-dir', default=None,
                        help='Optional directory to save speaker_test.json.')
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--n-mfcc', type=int, default=13)
    parser.add_argument('--n-filters', type=int, default=26)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    mfcc_cfg = MFCCConfig(sample_rate=args.sample_rate,
                          n_mfcc=args.n_mfcc, n_filters=args.n_filters)
    run_kmeans = args.model in ('kmeans', 'both', 'all')
    run_gnb    = args.model in ('gnb',    'both', 'all')
    run_hmm    = args.model in ('hmm',            'all')

    print(f"[speaker_test] Dataset : {dataset_root}")
    print(f"[speaker_test] Mode    : {args.mode}")

    all_frames, all_sums, all_labels_list, all_speakers = \
        _load_dataset_with_speakers(dataset_root, mfcc_cfg)

    class_labels = sorted(set(all_labels_list))
    speakers     = sorted(set(all_speakers))
    n_total      = len(all_labels_list)

    print(f"  Speakers : {speakers}")
    print(f"  Classes  : {class_labels}")
    print(f"  Samples  : {n_total}")

    # Count samples per speaker
    for sp in speakers:
        n = sum(1 for s in all_speakers if s == sp)
        print(f"    {sp:<14}  {n} samples")

    results = []

    if args.mode == 'all-train':
        # Train once on everything, evaluate per speaker
        print("\n  Training on ALL samples...")
        model_km  = _fit_kmeans(all_frames, all_labels_list, class_labels) if run_kmeans else None
        model_gnb = _fit_gnb(all_sums, all_labels_list) if run_gnb else None
        model_hmm = _fit_hmm(all_frames, all_labels_list, class_labels) if run_hmm else None

        for sp in speakers:
            idx = [i for i, s in enumerate(all_speakers) if s == sp]
            sp_frames = [all_frames[i] for i in idx]
            sp_sums   = [all_sums[i]   for i in idx]
            sp_labels = [all_labels_list[i] for i in idx]
            print(f"  Evaluating speaker '{sp}' ({len(idx)} samples)...", end=' ')
            r = _evaluate_speaker(
                model_km, model_gnb, model_hmm,
                sp_frames, sp_sums, sp_labels,
                class_labels, sp, run_kmeans, run_gnb, run_hmm,
            )
            results.append(r)
            print("done")

    else:  # leave-one-out
        for sp_test in speakers:
            train_idx = [i for i, s in enumerate(all_speakers) if s != sp_test]
            test_idx  = [i for i, s in enumerate(all_speakers) if s == sp_test]

            tr_frames = [all_frames[i] for i in train_idx]
            tr_sums   = [all_sums[i]   for i in train_idx]
            tr_labels = [all_labels_list[i] for i in train_idx]
            te_frames = [all_frames[i] for i in test_idx]
            te_sums   = [all_sums[i]   for i in test_idx]
            te_labels = [all_labels_list[i] for i in test_idx]

            print(f"\n  Leave out '{sp_test}' — "
                  f"train={len(train_idx)}, test={len(test_idx)}")
            model_km  = _fit_kmeans(tr_frames, tr_labels, class_labels) if run_kmeans else None
            model_gnb = _fit_gnb(tr_sums, tr_labels) if run_gnb else None
            model_hmm = _fit_hmm(tr_frames, tr_labels, class_labels) if run_hmm else None

            r = _evaluate_speaker(
                model_km, model_gnb, model_hmm,
                te_frames, te_sums, te_labels,
                class_labels, sp_test, run_kmeans, run_gnb, run_hmm,
            )
            results.append(r)

    _print_results(results, class_labels, run_kmeans, run_gnb, run_hmm, args.mode)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'speaker_test_{args.mode}.json'
        save_json(results, out_path)
        print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
