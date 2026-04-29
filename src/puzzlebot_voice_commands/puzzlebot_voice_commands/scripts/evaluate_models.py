"""
evaluate_voice_models — evaluate trained models on the test split and generate reports.

Usage (after building and sourcing):
  ros2 run puzzlebot_voice_commands evaluate_voice_models \\
    --dataset      <path/to/voice_commands_dataset> \\
    --artifact-dir <path/to/artifacts> \\
    --output-dir   <path/to/reports>

Reports written to --output-dir:
  confusion_matrix_kmeans.csv
  confusion_matrix_gnb.csv
  metrics_kmeans.json
  metrics_gnb.json
  safety_metrics.json
  inference_time.json
  model_comparison.md
"""
import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..audio_io import load_wav, normalize
from ..config import MFCCConfig
from ..dataset import DatasetSplit, Sample, discover_dataset, split_dataset
from ..config import DatasetConfig
from ..metrics import (
    accuracy,
    confidence_stats,
    confusion_matrix,
    macro_f1,
    per_command_accuracy,
    precision_recall_f1,
    safety_critical_errors,
    top2_accuracy,
)
from ..mfcc import extract_mfcc_frames, extract_mfcc_summary
from ..models.gaussian_nb import GaussianNaiveBayesClassifier
from ..models.kmeans_codebook import KMeansCodebookClassifier
from ..reports import (
    generate_comparison_report,
    save_confusion_matrix_csv,
    save_metrics_json,
)
from ..serialization import artifact_size_kb, load_json, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='evaluate_voice_models',
        description=(
            'Evaluate trained voice command models on the held-out test split '
            'and generate JSON, CSV, and Markdown report files.'
        ),
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to the root dataset folder.',
    )
    parser.add_argument(
        '--artifact-dir',
        type=str,
        required=True,
        help='Directory containing trained model artifacts (*.pkl, labels.json, etc.).',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to write evaluation reports.',
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['kmeans', 'gnb', 'both'],
        default='both',
        help='Which model(s) to evaluate (default: both).',
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.3,
        help='Test split ratio — must match the value used during training (default: 0.3).',
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed — must match the value used during training (default: 42).',
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_root  = Path(args.dataset)
    artifact_dir  = Path(args.artifact_dir)
    output_dir    = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_kmeans = args.model in ('kmeans', 'both')
    eval_gnb    = args.model in ('gnb',    'both')

    # ------------------------------------------------------------------
    # 1. Load MFCC config from artifact dir (falls back to defaults)
    # ------------------------------------------------------------------
    mfcc_cfg = _load_mfcc_config(artifact_dir)
    dataset_cfg = DatasetConfig(
        test_ratio=args.test_ratio,
        random_state=args.random_state,
    )

    # ------------------------------------------------------------------
    # 2. Discover dataset and reproduce the same train/test split
    # ------------------------------------------------------------------
    print(f"[evaluate_voice_models] Dataset: {dataset_root}")
    try:
        samples_by_class = discover_dataset(dataset_root)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    split: DatasetSplit = _load_or_rebuild_split(
        artifact_dir, samples_by_class, dataset_cfg
    )
    print(split.summary())
    print(f"Evaluating on {len(split.test)} test samples.\n")

    # ------------------------------------------------------------------
    # 3. Build dataset summary dict (shared across reports)
    # ------------------------------------------------------------------
    dataset_summary = {
        'dataset_root': str(dataset_root),
        'n_classes':    len(split.labels),
        'labels':       split.labels,
        'n_total':      sum(split.samples_per_class.values()),
        'n_train':      len(split.train),
        'n_test':       len(split.test),
        'test_ratio':   dataset_cfg.test_ratio,
        'random_state': dataset_cfg.random_state,
        'samples_per_class': split.samples_per_class,
        'train_per_class':   split.train_per_class,
        'test_per_class':    split.test_per_class,
        'mfcc_config': {
            'sample_rate':        mfcc_cfg.sample_rate,
            'pre_emphasis':       mfcc_cfg.pre_emphasis,
            'frame_size':         mfcc_cfg.frame_size,
            'frame_stride':       mfcc_cfg.frame_stride,
            'n_fft':              mfcc_cfg.n_fft,
            'n_filters':          mfcc_cfg.n_filters,
            'n_mfcc':             mfcc_cfg.n_mfcc,
            'include_delta':      mfcc_cfg.include_delta,
            'include_delta_delta': mfcc_cfg.include_delta_delta,
            'feature_dim':        mfcc_cfg.n_mfcc * 2,
        },
    }

    # ------------------------------------------------------------------
    # 4. Evaluate each model
    # ------------------------------------------------------------------
    kmeans_metrics_dict: Optional[Dict[str, Any]] = None
    gnb_metrics_dict:    Optional[Dict[str, Any]] = None

    if eval_kmeans:
        kmeans_path = artifact_dir / 'kmeans_model.pkl'
        if not kmeans_path.exists():
            print(f"WARNING: {kmeans_path} not found — skipping KMeans evaluation.",
                  file=sys.stderr)
        else:
            print("[evaluate_voice_models] Evaluating KMeansCodebookClassifier ...")
            model = KMeansCodebookClassifier.load(kmeans_path)
            kmeans_metrics_dict = _evaluate_kmeans(
                model, split, mfcc_cfg, split.labels
            )
            kmeans_metrics_dict['artifact_size_kb'] = round(
                artifact_size_kb(kmeans_path), 2
            )
            kmeans_metrics_dict['model_config'] = {
                'n_clusters':  model.config.n_clusters,
                'max_iter':    model.config.max_iter,
                'tolerance':   model.config.tolerance,
                'random_state': model.config.random_state,
            }
            _print_metrics_summary('KMeans', kmeans_metrics_dict)

            save_confusion_matrix_csv(
                np.array(kmeans_metrics_dict['confusion_matrix_list']),
                split.labels,
                output_dir / 'confusion_matrix_kmeans.csv',
            )
            save_metrics_json(kmeans_metrics_dict, output_dir / 'metrics_kmeans.json')
            print(f"  Reports saved to: {output_dir}")

    if eval_gnb:
        gnb_path = artifact_dir / 'gnb_model.pkl'
        if not gnb_path.exists():
            print(f"WARNING: {gnb_path} not found — skipping GNB evaluation.",
                  file=sys.stderr)
        else:
            print("\n[evaluate_voice_models] Evaluating GaussianNaiveBayesClassifier ...")
            model = GaussianNaiveBayesClassifier.load(gnb_path)
            gnb_metrics_dict = _evaluate_gnb(
                model, split, mfcc_cfg, split.labels
            )
            gnb_metrics_dict['artifact_size_kb'] = round(
                artifact_size_kb(gnb_path), 2
            )
            gnb_metrics_dict['model_config'] = {
                'var_epsilon': model.config.var_epsilon,
            }
            _print_metrics_summary('GaussianNB', gnb_metrics_dict)

            save_confusion_matrix_csv(
                np.array(gnb_metrics_dict['confusion_matrix_list']),
                split.labels,
                output_dir / 'confusion_matrix_gnb.csv',
            )
            save_metrics_json(gnb_metrics_dict, output_dir / 'metrics_gnb.json')
            print(f"  Reports saved to: {output_dir}")

    # ------------------------------------------------------------------
    # 5. Shared safety and inference time reports
    # ------------------------------------------------------------------
    if kmeans_metrics_dict or gnb_metrics_dict:
        safety_report: Dict[str, Any] = {}
        inf_time_report: Dict[str, Any] = {}

        for name, mdict in [('kmeans', kmeans_metrics_dict), ('gnb', gnb_metrics_dict)]:
            if mdict:
                safety_report[name] = mdict.get('safety', {})
                inf_time_report[name] = mdict.get('inference_time', {})

        save_json(safety_report,   output_dir / 'safety_metrics.json')
        save_json(inf_time_report, output_dir / 'inference_time.json')

        # Model comparison Markdown
        generate_comparison_report(
            kmeans_metrics=kmeans_metrics_dict,
            gnb_metrics=gnb_metrics_dict,
            dataset_summary=dataset_summary,
            output_path=output_dir / 'model_comparison.md',
        )
        print(f"\n[evaluate_voice_models] model_comparison.md written to: {output_dir}")

    # ------------------------------------------------------------------
    # 6. Final summary
    # ------------------------------------------------------------------
    print("\n--- Evaluation complete ---")
    print(f"  Output dir : {output_dir}")
    for fname in sorted(output_dir.iterdir()):
        print(f"    {fname.name}")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_kmeans(
    model: KMeansCodebookClassifier,
    split: DatasetSplit,
    mfcc_cfg: MFCCConfig,
    labels: List[str],
) -> Dict[str, Any]:
    """Run KMeans inference on the test split and collect all metrics."""
    y_true:       List[str]        = []
    y_pred:       List[str]        = []
    ranked_preds: List[List[str]]  = []
    margins:      List[float]      = []
    times_ms:     List[float]      = []

    for sample in split.test:
        try:
            signal, _ = load_wav(sample.path, target_sr=mfcc_cfg.sample_rate)
            signal = normalize(signal)
            frames = extract_mfcc_frames(signal, mfcc_cfg)

            t0 = time.perf_counter()
            ranked = model.predict_ranked(frames)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

            pred_label = ranked[0][0]
            margin = ranked[1][1] - ranked[0][1] if len(ranked) > 1 else 0.0

            y_true.append(sample.label)
            y_pred.append(pred_label)
            ranked_preds.append([lbl for lbl, _ in ranked])
            margins.append(margin)
        except Exception as exc:
            warnings.warn(f"Skipping {sample.path.name}: {exc}", UserWarning, stacklevel=2)

    return _build_metrics_dict(y_true, y_pred, ranked_preds, margins, times_ms, labels)


def _evaluate_gnb(
    model: GaussianNaiveBayesClassifier,
    split: DatasetSplit,
    mfcc_cfg: MFCCConfig,
    labels: List[str],
) -> Dict[str, Any]:
    """Run GNB inference on the test split and collect all metrics."""
    y_true:       List[str]        = []
    y_pred:       List[str]        = []
    ranked_preds: List[List[str]]  = []
    confidences:  List[float]      = []
    times_ms:     List[float]      = []

    for sample in split.test:
        try:
            signal, _ = load_wav(sample.path, target_sr=mfcc_cfg.sample_rate)
            signal = normalize(signal)
            vec = extract_mfcc_summary(signal, mfcc_cfg)

            t0 = time.perf_counter()
            pred_label, scores = model.predict(vec)
            ranked = model.predict_ranked(vec)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

            y_true.append(sample.label)
            y_pred.append(pred_label)
            ranked_preds.append([lbl for lbl, _ in ranked])
            confidences.append(scores.get(pred_label, 0.0))
        except Exception as exc:
            warnings.warn(f"Skipping {sample.path.name}: {exc}", UserWarning, stacklevel=2)

    return _build_metrics_dict(y_true, y_pred, ranked_preds, confidences, times_ms, labels)


def _build_metrics_dict(
    y_true:       List[str],
    y_pred:       List[str],
    ranked_preds: List[List[str]],
    conf_scores:  List[float],
    times_ms:     List[float],
    labels:       List[str],
) -> Dict[str, Any]:
    """Compute all metrics and pack them into a serialisable dict."""
    prf     = precision_recall_f1(y_true, y_pred, labels)
    cm      = confusion_matrix(y_true, y_pred, labels)
    safety  = safety_critical_errors(y_true, y_pred)

    macro_recall = sum(v['recall'] for v in prf.values()) / len(prf) if prf else 0.0

    return {
        'n_samples':              len(y_true),
        'labels':                 labels,
        'accuracy':               round(accuracy(y_true, y_pred), 6),
        'macro_f1':               round(macro_f1(prf), 6),
        'macro_recall':           round(macro_recall, 6),
        'top2_accuracy':          round(top2_accuracy(y_true, ranked_preds), 6),
        'per_class':              {
            lbl: {k: round(v, 6) for k, v in prf[lbl].items()}
            for lbl in labels
        },
        'per_command_accuracy':   {
            lbl: round(v, 6)
            for lbl, v in per_command_accuracy(y_true, y_pred, labels).items()
        },
        'confusion_matrix_list':  cm.tolist(),
        'safety': {
            'safety_critical_count':    safety['safety_critical_count'],
            'safety_critical_rate':     round(safety['safety_critical_rate'], 6),
            'safety_critical_cases':    safety['safety_critical_cases'],
            'opposite_direction_count': safety['opposite_direction_count'],
            'opposite_direction_rate':  round(safety['opposite_direction_rate'], 6),
            'opposite_direction_cases': safety['opposite_direction_cases'],
            'stop_recall':              {k: round(v, 6) for k, v in safety['stop_recall'].items()},
        },
        'safety_critical_count':  safety['safety_critical_count'],
        'safety_critical_rate':   round(safety['safety_critical_rate'], 6),
        'opposite_direction_count': safety['opposite_direction_count'],
        'confidence_stats':       confidence_stats(conf_scores),
        'inference_time':         confidence_stats(times_ms),
        'avg_inference_ms':       round(float(np.mean(times_ms)) if times_ms else 0.0, 4),
    }


def _print_metrics_summary(model_name: str, mdict: Dict[str, Any]) -> None:
    print(f"\n  === {model_name} results ===")
    print(f"  Accuracy   : {mdict['accuracy']:.4f}")
    print(f"  Macro F1   : {mdict['macro_f1']:.4f}")
    print(f"  Top-2 Acc  : {mdict['top2_accuracy']:.4f}")
    print(f"  Safety err : {mdict['safety_critical_count']}")
    print(f"  Opp-dir err: {mdict['opposite_direction_count']}")
    print(f"  Avg infer  : {mdict['avg_inference_ms']:.2f} ms")
    print(f"  Artifact   : {mdict.get('artifact_size_kb', 'N/A')} KB")
    print()
    print(f"  {'Class':15s}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    for lbl, prf in mdict['per_class'].items():
        print(f"  {lbl:15s}  {prf['precision']:6.3f}  "
              f"{prf['recall']:6.3f}  {prf['f1']:6.3f}")

    print(f"\n  {'Class':15s}  {'Recall':>8}")
    for lbl, prf in mdict['per_class'].items():
        flag = '  <-- BAJO' if prf['recall'] < 0.90 else ''
        print(f"  {lbl:15s}  {prf['recall']:8.4f}{flag}")
    macro_recall = sum(v['recall'] for v in mdict['per_class'].values()) / len(mdict['per_class'])
    print(f"  {'Macro recall':15s}  {macro_recall:8.4f}")


def _load_mfcc_config(artifact_dir: Path) -> MFCCConfig:
    """Load MFCC config from feature_config.json or return defaults."""
    cfg_path = artifact_dir / 'feature_config.json'
    if not cfg_path.exists():
        warnings.warn(
            f"feature_config.json not found in {artifact_dir}. Using default MFCCConfig.",
            UserWarning, stacklevel=2,
        )
        return MFCCConfig()
    data = load_json(cfg_path)
    return MFCCConfig(
        sample_rate=data.get('sample_rate', 16000),
        pre_emphasis=data.get('pre_emphasis', 0.97),
        frame_size=data.get('frame_size', 0.025),
        frame_stride=data.get('frame_stride', 0.010),
        n_fft=data.get('n_fft', 512),
        n_filters=data.get('n_filters', 26),
        n_mfcc=data.get('n_mfcc', 13),
        include_delta=data.get('include_delta', False),
        include_delta_delta=data.get('include_delta_delta', False),
    )


def _load_or_rebuild_split(
    artifact_dir: Path,
    samples_by_class: Dict,
    dataset_cfg: DatasetConfig,
) -> DatasetSplit:
    """Try to load train_metadata.json to verify split params, then rebuild split."""
    meta_path = artifact_dir / 'train_metadata.json'
    if meta_path.exists():
        meta = load_json(meta_path)
        stored_ratio = meta.get('test_ratio', dataset_cfg.test_ratio)
        stored_seed  = meta.get('random_state', dataset_cfg.random_state)
        if stored_ratio != dataset_cfg.test_ratio or stored_seed != dataset_cfg.random_state:
            warnings.warn(
                f"train_metadata.json has test_ratio={stored_ratio}, "
                f"random_state={stored_seed} but CLI args are "
                f"test_ratio={dataset_cfg.test_ratio}, "
                f"random_state={dataset_cfg.random_state}. "
                "Using values from train_metadata.json to reproduce the original split.",
                UserWarning, stacklevel=2,
            )
            dataset_cfg = DatasetConfig(
                test_ratio=stored_ratio,
                random_state=stored_seed,
            )

    from ..dataset import split_dataset
    return split_dataset(samples_by_class, dataset_cfg)


if __name__ == '__main__':
    main()
