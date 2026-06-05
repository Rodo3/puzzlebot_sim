"""
generate_hmm_parameter_report.py — PDF report of HMM parameter evolution.

Trains a fresh HMM (does NOT touch artifacts_final/) and captures A and B
snapshots at three training stages for up to 3 selected words:
  - Initial : after linear-segmentation count initialisation
  - Mid     : after n_iter // 2 Baum-Welch iterations
  - Final   : after all iterations (B smoothed for display only)

Usage:
  python -m puzzlebot_voice_commands.scripts.generate_hmm_parameter_report \\
    --dataset datasets/voice_commands_dataset_aug \\
    --output-dir reports_hmm_parameters \\
    --words alto avanzar retroceder \\
    --n-symbols 256 --n-iter 20 \\
    --n-mfcc 20 --delta --cmvn --librosa \\
    --include-zcr --include-rms --include-contrast \\
    --syllable-states --smoothing-eps 1e-6
"""
import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ..audio_io import load_wav, normalize
from ..config import HMMConfig, MFCCConfig
from ..dataset import discover_dataset
from ..models.hmm import HiddenMarkovModelClassifier


SYLLABLE_STATES: Dict[str, int] = {
    'alto': 4, 'avanzar': 6, 'derecha': 6,
    'inicio': 4, 'izquierda': 5, 'retroceder': 6,
    'subir': 6, 'bajar': 5, 'tomar': 6, 'soltar': 5,
}


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='generate_hmm_parameter_report',
        description='Generate HMM A/B parameter evolution report (PDF).',
    )
    p.add_argument('--dataset',      required=True,
                   help='Path to dataset root (one subfolder per class).')
    p.add_argument('--output-dir',   required=True,
                   help='Directory for PDF, figures/, and metadata JSON.')
    p.add_argument('--words',        nargs='+',
                   default=['alto', 'avanzar', 'retroceder'],
                   help='Up to 3 words to include in the report.')
    p.add_argument('--n-symbols',    type=int, default=256,
                   help='Codebook size for quantisation (default: 256).')
    p.add_argument('--n-iter',       type=int, default=20,
                   help='Baum-Welch iterations (default: 20).')
    p.add_argument('--n-mfcc',       type=int, default=13)
    p.add_argument('--n-filters',    type=int, default=26)
    p.add_argument('--sample-rate',  type=int, default=16000)
    p.add_argument('--random-state', type=int, default=42)
    p.add_argument('--cmvn',         action='store_true')
    p.add_argument('--delta',        action='store_true')
    p.add_argument('--delta-delta',  action='store_true')
    p.add_argument('--librosa',      action='store_true')
    p.add_argument('--include-zcr',  action='store_true')
    p.add_argument('--include-rms',  action='store_true')
    p.add_argument('--include-contrast', action='store_true')
    p.add_argument('--syllable-states', action='store_true',
                   help='Use per-class n_states from the gridsearch optimum.')
    p.add_argument('--smoothing-eps', type=float, default=1e-6,
                   help='Epsilon added to B_final for display smoothing only.')
    return p


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _extract_frames(signal: np.ndarray, cfg: MFCCConfig) -> np.ndarray:
    if cfg.use_librosa:
        from ..librosa_features import extract_librosa_frames
        return extract_librosa_frames(signal, cfg)
    from ..mfcc import extract_mfcc_frames
    return extract_mfcc_frames(signal, cfg)


def _load_all_frames(
    samples_by_class: Dict[str, List[Path]],
    mfcc_cfg: MFCCConfig,
) -> Dict[str, List[np.ndarray]]:
    seqs: Dict[str, List[np.ndarray]] = {}
    total = sum(len(v) for v in samples_by_class.values())
    done = 0
    for label, paths in samples_by_class.items():
        class_seqs: List[np.ndarray] = []
        for path in paths:
            try:
                signal, _ = load_wav(path, target_sr=mfcc_cfg.sample_rate)
                signal = normalize(signal)
                frames = _extract_frames(signal, mfcc_cfg)
                class_seqs.append(frames)
            except Exception as exc:
                warnings.warn(f"Skipping {path.name}: {exc}",
                              UserWarning, stacklevel=2)
            done += 1
        if class_seqs:
            seqs[label] = class_seqs
        if done % 200 == 0 or done == total:
            print(f"  Features: {done}/{total} clips", end='\r', flush=True)
    print()
    return seqs


# ---------------------------------------------------------------------------
# Analysis phrase
# ---------------------------------------------------------------------------

def _concentration_phrase(B: np.ndarray) -> str:
    """Return a one-line analysis based on mean normalised row entropy of B."""
    n_sym = B.shape[1]
    B_safe = np.clip(B, 1e-12, None)
    B_norm = B_safe / B_safe.sum(axis=1, keepdims=True)
    H_rows = -(B_norm * np.log(B_norm)).sum(axis=1) / np.log(n_sym)
    if H_rows.mean() < 0.85:
        return ("La matriz B final conservó la especialización fonética "
                "porque presenta picos claros por estado.")
    return ("La matriz B final se dispersó porque las probabilidades "
            "quedaron distribuidas sin picos dominantes.")


# ---------------------------------------------------------------------------
# Heatmap helper
# ---------------------------------------------------------------------------

def _heatmap(ax: plt.Axes, matrix: np.ndarray, title: str, cmap: str) -> None:
    n_rows, n_cols = matrix.shape
    im = ax.imshow(matrix, aspect='auto', cmap=cmap,
                   interpolation='nearest', vmin=0.0)
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_ylabel('Estado', fontsize=7)
    # Y ticks — one per state
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([str(i) for i in range(n_rows)], fontsize=6)
    # X ticks — sparse for large B matrix
    if n_cols <= 10:
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels([str(i) for i in range(n_cols)], fontsize=6)
        ax.set_xlabel('Estado destino', fontsize=7)
    else:
        step = max(1, n_cols // 5)
        xt = sorted({0, step, 2 * step, 3 * step, 4 * step, n_cols - 1})
        ax.set_xticks(xt)
        ax.set_xticklabels([str(x) for x in xt], fontsize=6)
        ax.set_xlabel('Símbolo', fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ---------------------------------------------------------------------------
# Per-word figure
# ---------------------------------------------------------------------------

def _make_word_figure(
    word: str,
    snaps: Dict[str, Tuple[np.ndarray, np.ndarray]],
    smoothing_eps: float,
) -> plt.Figure:
    stages = ['initial', 'mid', 'final']
    labels = ['Inicial', 'Intermedia', 'Final']

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.suptitle(f'Evolución de parámetros HMM — "{word}"',
                 fontsize=12, fontweight='bold')

    for col, (stage, slabel) in enumerate(zip(stages, labels)):
        A, B = snaps[stage]
        # Display-only smoothing for final B
        if stage == 'final':
            B = B + smoothing_eps
            B = B / B.sum(axis=1, keepdims=True)
        _heatmap(axes[0, col], A, f'A — {slabel}',   cmap='YlOrRd')
        _heatmap(axes[1, col], B, f'B — {slabel}',   cmap='viridis')

    phrase = _concentration_phrase(snaps['final'][1])
    fig.text(0.5, 0.01, phrase, ha='center', va='bottom',
             fontsize=8, style='italic')
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    dataset_root = Path(args.dataset)
    words = args.words[:3]  # max 3 pages

    # Build configs
    mfcc_cfg = MFCCConfig(
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        n_filters=args.n_filters,
        cmvn=args.cmvn,
        include_delta=args.delta,
        include_delta_delta=args.delta_delta,
        use_librosa=args.librosa,
        include_zcr=args.include_zcr,
        include_rms=args.include_rms,
        include_contrast=args.include_contrast,
    )
    n_states_per_class = SYLLABLE_STATES if args.syllable_states else {}
    hmm_cfg = HMMConfig(
        n_states=5,
        n_symbols=args.n_symbols,
        n_iter=args.n_iter,
        random_state=args.random_state,
        n_states_per_class=n_states_per_class,
        kmeans_max_iter=50,  # enough for visualization; 300 is overkill
    )

    print(f"[report] Dataset  : {dataset_root}")
    print(f"[report] Words    : {words}")
    print(f"[report] n_symbols={args.n_symbols}  n_iter={args.n_iter}")

    # Discover dataset and validate requested words
    try:
        samples_by_class = discover_dataset(dataset_root)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    missing = [w for w in words if w not in samples_by_class]
    if missing:
        print(f"ERROR: words not found in dataset: {missing}", file=sys.stderr)
        print(f"  Available: {sorted(samples_by_class)}", file=sys.stderr)
        sys.exit(1)

    # Load features for the 3 report words only — codebook + BW only for these
    print(f"[report] Extracting features (3 words only) ...")
    report_samples = {w: sample_paths for w, sample_paths in samples_by_class.items()
                      if w in words}
    sequences_by_class = _load_all_frames(report_samples, mfcc_cfg)

    # Train report HMM with snapshot capture
    print(f"[report] Training HMM ...")
    model = HiddenMarkovModelClassifier(config=hmm_cfg)
    model.fit(sequences_by_class, snapshots_for=words)

    snapshots = model._snapshots_
    for w in words:
        stages_got = list(snapshots.get(w, {}).keys())
        print(f"[report]   {w}: snapshots captured = {stages_got}")

    # Generate PDF
    pdf_path = output_dir / 'hmm_parameter_report.pdf'
    print(f"[report] Writing PDF: {pdf_path}")

    with PdfPages(str(pdf_path)) as pdf:
        for word in words:
            if word not in snapshots or len(snapshots[word]) < 3:
                print(f"WARNING: incomplete snapshots for '{word}', skipping.",
                      file=sys.stderr)
                continue

            fig = _make_word_figure(word, snapshots[word], args.smoothing_eps)

            # PDF page
            pdf.savefig(fig, bbox_inches='tight')

            # Combined PNG (all 6 heatmaps)
            fig.savefig(str(figures_dir / f'{word}.png'),
                        dpi=150, bbox_inches='tight')

            # Individual heatmap PNGs
            for stage in ('initial', 'mid', 'final'):
                A, B = snapshots[word][stage]

                fig_a, ax_a = plt.subplots(figsize=(4, 3))
                _heatmap(ax_a, A, f'A {stage} — {word}', cmap='YlOrRd')
                fig_a.tight_layout()
                fig_a.savefig(str(figures_dir / f'{word}_A_{stage}.png'),
                              dpi=150, bbox_inches='tight')
                plt.close(fig_a)

                B_disp = B.copy()
                if stage == 'final':
                    B_disp += args.smoothing_eps
                    B_disp /= B_disp.sum(axis=1, keepdims=True)
                fig_b, ax_b = plt.subplots(figsize=(8, 3))
                _heatmap(ax_b, B_disp, f'B {stage} — {word}', cmap='viridis')
                fig_b.tight_layout()
                fig_b.savefig(str(figures_dir / f'{word}_B_{stage}.png'),
                              dpi=150, bbox_inches='tight')
                plt.close(fig_b)

            plt.close(fig)

    # Metadata JSON
    meta = {
        'generated': datetime.now().isoformat(),
        'dataset': str(dataset_root),
        'words': words,
        'n_symbols': args.n_symbols,
        'n_iter': args.n_iter,
        'n_states_per_class': {w: n_states_per_class.get(w, 5) for w in words},
        'smoothing_eps': args.smoothing_eps,
        'mfcc': {
            'n_mfcc': mfcc_cfg.n_mfcc,
            'cmvn': mfcc_cfg.cmvn,
            'include_delta': mfcc_cfg.include_delta,
            'use_librosa': mfcc_cfg.use_librosa,
            'include_zcr': mfcc_cfg.include_zcr,
            'include_rms': mfcc_cfg.include_rms,
            'include_contrast': mfcc_cfg.include_contrast,
        },
    }
    meta_path = output_dir / 'report_metadata.json'
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                         encoding='utf-8')
    print(f"[report] Metadata : {meta_path}")
    print(f"[report] Done.  PDF has {len(words)} page(s): {pdf_path}")


if __name__ == '__main__':
    main()
