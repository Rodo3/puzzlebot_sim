"""
Report generation for evaluation results.

Outputs:
- confusion_matrix_<model>.csv
- metrics_<model>.json
- safety_metrics.json
- inference_time.json
- model_comparison.md
"""
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .serialization import save_json


def save_confusion_matrix_csv(
    matrix: np.ndarray,
    labels: List[str],
    path: Path,
) -> None:
    """Write a confusion matrix to a CSV file.

    First row and first column are the class labels.
    Rows = true class, columns = predicted class.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['true \\ pred'] + labels)
        for i, label in enumerate(labels):
            writer.writerow([label] + [int(matrix[i, j]) for j in range(len(labels))])


def save_metrics_json(metrics: Dict[str, Any], path: Path) -> None:
    """Write a metrics dict to a pretty-printed JSON file."""
    save_json(metrics, Path(path))


def generate_comparison_report(
    kmeans_metrics: Optional[Dict[str, Any]],
    gnb_metrics: Optional[Dict[str, Any]],
    dataset_summary: Dict[str, Any],
    output_path: Path,
) -> None:
    """Write model_comparison.md — full Markdown evaluation report.

    Args:
        kmeans_metrics: output of _build_metrics_dict for KMeans, or None.
        gnb_metrics:    output of _build_metrics_dict for GNB, or None.
        dataset_summary: dict with dataset metadata.
        output_path:    path to write the .md file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    a = lines.append  # shorthand

    a('# Voice Command Recognition — Model Comparison Report')
    a('')

    # ------------------------------------------------------------------
    # 1. Dataset summary
    # ------------------------------------------------------------------
    a('## 1. Dataset Summary')
    a('')
    a(f"- **Root path:** `{dataset_summary.get('dataset_root', 'N/A')}`")
    a(f"- **Classes:** {dataset_summary.get('n_classes', 'N/A')}")
    a(f"- **Total samples:** {dataset_summary.get('n_total', 'N/A')}")
    a(f"- **Train samples:** {dataset_summary.get('n_train', 'N/A')}")
    a(f"- **Test samples:** {dataset_summary.get('n_test', 'N/A')}")
    a(f"- **Test ratio:** {dataset_summary.get('test_ratio', 'N/A')}")
    a(f"- **Random state:** {dataset_summary.get('random_state', 'N/A')}")
    a('')

    # Per-class sample counts
    a('### Samples per class')
    a('')
    a('| Class | Total | Train | Test |')
    a('|-------|-------|-------|------|')
    for lbl in dataset_summary.get('labels', []):
        total = dataset_summary.get('samples_per_class', {}).get(lbl, 'N/A')
        train = dataset_summary.get('train_per_class', {}).get(lbl, 'N/A')
        test  = dataset_summary.get('test_per_class', {}).get(lbl, 'N/A')
        a(f'| {lbl} | {total} | {train} | {test} |')
    a('')

    # ------------------------------------------------------------------
    # 2. MFCC configuration
    # ------------------------------------------------------------------
    a('## 2. MFCC Feature Configuration')
    a('')
    cfg = dataset_summary.get('mfcc_config', {})
    a(f"- sample_rate: {cfg.get('sample_rate', 16000)} Hz")
    a(f"- pre_emphasis: {cfg.get('pre_emphasis', 0.97)}")
    a(f"- frame_size: {cfg.get('frame_size', 0.025)} s  "
      f"({int(cfg.get('frame_size', 0.025) * cfg.get('sample_rate', 16000))} samples)")
    a(f"- frame_stride: {cfg.get('frame_stride', 0.010)} s  "
      f"({int(cfg.get('frame_stride', 0.010) * cfg.get('sample_rate', 16000))} samples)")
    a(f"- n_fft: {cfg.get('n_fft', 512)}")
    a(f"- n_filters: {cfg.get('n_filters', 26)}")
    a(f"- n_mfcc: {cfg.get('n_mfcc', 13)}")
    a(f"- include_delta: {cfg.get('include_delta', False)}")
    a(f"- include_delta_delta: {cfg.get('include_delta_delta', False)}")
    a(f"- **Feature vector size:** {cfg.get('feature_dim', 'N/A')} dimensions (mean + std)")
    a('')

    # ------------------------------------------------------------------
    # 3. Model configurations
    # ------------------------------------------------------------------
    a('## 3. Model Configurations')
    a('')
    if kmeans_metrics:
        km_cfg = kmeans_metrics.get('model_config', {})
        a('### KMeansCodebookClassifier')
        a(f"- n_clusters: {km_cfg.get('n_clusters', 16)} per class")
        a(f"- max_iter: {km_cfg.get('max_iter', 300)}")
        a(f"- tolerance: {km_cfg.get('tolerance', 1e-4)}")
        a(f"- random_state: {km_cfg.get('random_state', 42)}")
        a(f"- Feature input: frame-level MFCCs  `(n_frames × {cfg.get('n_mfcc', 13)})`")
        a('')
    if gnb_metrics:
        gnb_cfg = gnb_metrics.get('model_config', {})
        a('### GaussianNaiveBayesClassifier')
        a(f"- var_epsilon: {gnb_cfg.get('var_epsilon', 1e-9)}")
        a(f"- Feature input: MFCC summary vector  `({cfg.get('feature_dim', 'N/A')},)`")
        a('')

    # ------------------------------------------------------------------
    # 4. Metrics comparison table
    # ------------------------------------------------------------------
    a('## 4. Metrics Comparison')
    a('')
    a('| Metric | KMeans | GaussianNB |')
    a('|--------|--------|-----------|')

    def _fmt(d, key, fmt='.4f'):
        val = d.get(key, 'N/A') if d else 'N/A'
        return f'{val:{fmt}}' if isinstance(val, float) else str(val)

    rows = [
        ('Global accuracy',     'accuracy'),
        ('Macro recall',        'macro_recall'),
        ('Macro F1',            'macro_f1'),
        ('Top-2 accuracy',      'top2_accuracy'),
        ('Safety-critical errors', 'safety_critical_count'),
        ('Safety-critical rate',   'safety_critical_rate'),
        ('Opposite-dir errors',    'opposite_direction_count'),
        ('Avg inference time (ms)', 'avg_inference_ms'),
        ('Artifact size (KB)',      'artifact_size_kb'),
    ]
    for label, key in rows:
        km_val = _fmt(kmeans_metrics, key) if kmeans_metrics else 'N/A'
        gnb_val = _fmt(gnb_metrics, key) if gnb_metrics else 'N/A'
        a(f'| {label} | {km_val} | {gnb_val} |')
    a('')

    # ------------------------------------------------------------------
    # 5. Per-class F1 table
    # ------------------------------------------------------------------
    a('## 5. Per-class Precision / Recall / F1')
    a('')
    all_labels = dataset_summary.get('labels', [])
    if all_labels:
        a('| Class | KM-P | KM-R | KM-F1 | GNB-P | GNB-R | GNB-F1 |')
        a('|-------|------|------|-------|-------|-------|--------|')
        for lbl in all_labels:
            km_prf  = (kmeans_metrics or {}).get('per_class', {}).get(lbl, {})
            gnb_prf = (gnb_metrics or {}).get('per_class', {}).get(lbl, {})
            km_p  = f"{km_prf.get('precision', 0):.3f}"  if km_prf  else 'N/A'
            km_r  = f"{km_prf.get('recall',    0):.3f}"  if km_prf  else 'N/A'
            km_f  = f"{km_prf.get('f1',        0):.3f}"  if km_prf  else 'N/A'
            gnb_p = f"{gnb_prf.get('precision', 0):.3f}" if gnb_prf else 'N/A'
            gnb_r = f"{gnb_prf.get('recall',    0):.3f}" if gnb_prf else 'N/A'
            gnb_f = f"{gnb_prf.get('f1',        0):.3f}" if gnb_prf else 'N/A'
            a(f'| {lbl} | {km_p} | {km_r} | {km_f} | {gnb_p} | {gnb_r} | {gnb_f} |')
    a('')

    # ------------------------------------------------------------------
    # 6. Confusion matrix summary
    # ------------------------------------------------------------------
    a('## 6. Confusion Matrix Summary')
    a('')
    for model_name, mdict in [('KMeans', kmeans_metrics), ('GaussianNB', gnb_metrics)]:
        if not mdict:
            continue
        a(f'### {model_name}')
        a('')
        cm = mdict.get('confusion_matrix_list')
        labels_cm = mdict.get('labels', all_labels)
        if cm:
            matrix = np.array(cm)
            a('| true \\ pred | ' + ' | '.join(labels_cm) + ' |')
            a('|' + '---|' * (len(labels_cm) + 1))
            for i, lbl in enumerate(labels_cm):
                row_vals = ' | '.join(str(int(matrix[i, j])) for j in range(len(labels_cm)))
                a(f'| **{lbl}** | {row_vals} |')
        a('')

    # ------------------------------------------------------------------
    # 7. Safety-critical errors
    # ------------------------------------------------------------------
    a('## 7. Safety-Critical Errors')
    a('')
    a('Safety-critical: a stop command (`alto`, `stop`) predicted as a movement command.')
    a('Opposite-direction: `adelante↔atras`, `izquierda↔derecha`.')
    a('')
    for model_name, mdict in [('KMeans', kmeans_metrics), ('GaussianNB', gnb_metrics)]:
        if not mdict:
            continue
        sc = mdict.get('safety', {})
        a(f'### {model_name}')
        a(f"- Safety-critical errors : **{sc.get('safety_critical_count', 0)}**"
          f"  (rate: {sc.get('safety_critical_rate', 0):.4f})")
        a(f"- Opposite-direction errors: **{sc.get('opposite_direction_count', 0)}**"
          f"  (rate: {sc.get('opposite_direction_rate', 0):.4f})")
        stop_recall = sc.get('stop_recall', {})
        if stop_recall:
            for cmd, rec in stop_recall.items():
                a(f"- Recall for `{cmd}`: **{rec:.4f}**")
        cases = sc.get('safety_critical_cases', [])
        if cases:
            a('')
            a('  Safety-critical misclassifications:')
            for c in cases[:10]:   # cap at 10 to keep report readable
                a(f"  - true=`{c['true']}` → pred=`{c['pred']}`")
            if len(cases) > 10:
                a(f"  - … and {len(cases) - 10} more")
        a('')

    # ------------------------------------------------------------------
    # 8. Inference time comparison
    # ------------------------------------------------------------------
    a('## 8. Inference Time')
    a('')
    a('| Model | Avg (ms) | Std (ms) | Min (ms) | Max (ms) |')
    a('|-------|----------|----------|----------|----------|')
    for model_name, mdict in [('KMeans', kmeans_metrics), ('GaussianNB', gnb_metrics)]:
        if not mdict:
            a(f'| {model_name} | N/A | N/A | N/A | N/A |')
            continue
        it = mdict.get('inference_time', {})
        a(f"| {model_name} "
          f"| {it.get('mean', 0):.2f} "
          f"| {it.get('std',  0):.2f} "
          f"| {it.get('min',  0):.2f} "
          f"| {it.get('max',  0):.2f} |")
    a('')

    # ------------------------------------------------------------------
    # 9. Artifact size comparison
    # ------------------------------------------------------------------
    a('## 9. Model Artifact Size')
    a('')
    a('| Model | Size (KB) |')
    a('|-------|-----------|')
    for model_name, mdict in [('KMeans', kmeans_metrics), ('GaussianNB', gnb_metrics)]:
        val = (mdict or {}).get('artifact_size_kb', 'N/A')
        a(f'| {model_name} | {val} |')
    a('')

    # ------------------------------------------------------------------
    # 10. Recommendation
    # ------------------------------------------------------------------
    a('## 10. Recommendation for ROS 2 Integration')
    a('')
    rec = _recommendation(kmeans_metrics, gnb_metrics)
    a(rec)
    a('')

    # ------------------------------------------------------------------
    # 11. Known limitations
    # ------------------------------------------------------------------
    a('## 11. Known Limitations')
    a('')
    a('- Dataset size is small; metrics may not generalise well to unseen speakers.')
    a('- MFCC features capture spectral shape but not prosody or duration.')
    a('- KMeans codebook quality depends on having enough frames per class.')
    a('- GNB assumes feature independence given the class (Naive Bayes assumption).')
    a('- No data augmentation (noise, speed perturbation) was applied.')
    a('- Models were trained and tested on the same recording conditions.')
    a('')

    # ------------------------------------------------------------------
    # 12. Next steps before Puzzlebot integration
    # ------------------------------------------------------------------
    a('## 12. Next Steps Before Connecting to Puzzlebot')
    a('')
    a('1. Collect more diverse recordings (different speakers, microphones, distances).')
    a('2. Add noise augmentation to improve robustness in real environments.')
    a('3. Validate safety-critical recall ≥ 0.95 for `alto` before integration.')
    a('4. Implement a ROS 2 inference node (`voice_command_node.py`) in this package.')
    a('5. Publish recognised commands as `std_msgs/String` on `/voice_command` topic.')
    a('6. Add a confidence threshold: commands below threshold are ignored.')
    a('7. Wire the node into `puzzlebot_bringup` launch files.')
    a('')

    output_path.write_text('\n'.join(lines), encoding='utf-8')


def _recommendation(
    km: Optional[Dict[str, Any]],
    gnb: Optional[Dict[str, Any]],
) -> str:
    """Generate a text recommendation based on available metrics."""
    if km is None and gnb is None:
        return 'No models evaluated — run both models before generating this report.'

    if km is None:
        return (
            'Only GaussianNB was evaluated. '
            'Train and evaluate KMeans before making a final recommendation.'
        )
    if gnb is None:
        return (
            'Only KMeans was evaluated. '
            'Train and evaluate GaussianNB before making a final recommendation.'
        )

    km_acc  = km.get('accuracy', 0.0)
    gnb_acc = gnb.get('accuracy', 0.0)
    km_sc   = km.get('safety_critical_count', 999)
    gnb_sc  = gnb.get('safety_critical_count', 999)
    km_ms   = km.get('avg_inference_ms', 9999)
    gnb_ms  = gnb.get('avg_inference_ms', 9999)
    km_kb   = km.get('artifact_size_kb', 9999)
    gnb_kb  = gnb.get('artifact_size_kb', 9999)

    winner_acc = 'KMeans' if km_acc >= gnb_acc else 'GaussianNB'
    winner_safety = 'KMeans' if km_sc <= gnb_sc else 'GaussianNB'
    winner_speed = 'KMeans' if km_ms <= gnb_ms else 'GaussianNB'
    winner_size = 'KMeans' if km_kb <= gnb_kb else 'GaussianNB'

    lines = [
        f'- **Accuracy:** {winner_acc} leads '
        f'(KMeans {km_acc:.4f} vs GaussianNB {gnb_acc:.4f}).',
        f'- **Safety:** {winner_safety} has fewer safety-critical errors '
        f'(KMeans {km_sc} vs GaussianNB {gnb_sc}).',
        f'- **Speed:** {winner_speed} is faster '
        f'(KMeans {km_ms:.2f} ms vs GaussianNB {gnb_ms:.2f} ms).',
        f'- **Size:** {winner_size} is smaller '
        f'(KMeans {km_kb:.1f} KB vs GaussianNB {gnb_kb:.1f} KB).',
        '',
    ]

    # Overall recommendation prioritises safety, then accuracy
    if gnb_sc < km_sc and gnb_acc >= km_acc - 0.05:
        overall = (
            'Given fewer safety-critical errors and comparable accuracy, '
            '**GaussianNB** is recommended for the first ROS 2 integration attempt. '
            'Its fixed-length feature input also makes real-time inference simpler.'
        )
    elif km_acc > gnb_acc + 0.05:
        overall = (
            '**KMeans** achieves notably higher accuracy. '
            'If safety-critical recall for `alto` is acceptable (≥ 0.95), '
            'it is the preferred model for ROS 2 integration.'
        )
    else:
        overall = (
            'Both models show similar performance. '
            '**GaussianNB** is recommended for ROS 2 integration due to its '
            'simpler inference path (single matrix multiply vs. codebook search), '
            'smaller artifact size, and faster prediction time.'
        )

    lines.append(overall)
    return '\n'.join(lines)
