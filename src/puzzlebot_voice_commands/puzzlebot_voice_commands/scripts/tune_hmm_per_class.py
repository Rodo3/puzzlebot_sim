"""
tune_hmm_per_class.py — Gridsearch de n_states por clase + n_symbols para HMM.

Busca el mejor número de estados HMM solo para las clases problemáticas,
manteniendo fijos los estados de las clases ya bien calibradas.
También varía n_symbols como dimensión del grid.
Usa features librosa (igual que el modelo de producción) y split sin leakage.

Uso:
  cd src/puzzlebot_voice_commands
  python -m puzzlebot_voice_commands.scripts.tune_hmm_per_class \
    --dataset   datasets/voice_commands_dataset_aug \
    --tune      avanzar subir tomar \
    --states    4 5 6 \
    --symbols   32 64 \
    --n-iter    10
"""
import argparse
import itertools
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..audio_io import load_wav, normalize
from ..config import DatasetConfig, HMMConfig, MFCCConfig
from ..dataset import discover_dataset, split_dataset
from ..librosa_features import extract_librosa_frames
from ..metrics import accuracy, macro_f1, precision_recall_f1
from ..models.hmm import HiddenMarkovModelClassifier

# Estados fijos para clases que ya funcionan bien
FIXED_STATES: Dict[str, int] = {
    'alto':       4,
    'avanzar':    6,
    'bajar':      5,
    'inicio':     4,
    'izquierda':  5,
    'retroceder': 6,
    'soltar':     5,
    'subir':      6,
    'tomar':      6,
}


def _extract_all(
    samples,
    mfcc_cfg: MFCCConfig,
) -> Tuple[List[np.ndarray], List[str]]:
    frames_list, labels = [], []
    for s in samples:
        try:
            signal, _ = load_wav(s.path, target_sr=mfcc_cfg.sample_rate)
            signal = normalize(signal)
            frames_list.append(extract_librosa_frames(signal, mfcc_cfg))
            labels.append(s.label)
        except Exception as exc:
            warnings.warn(f"Skipping {s.path.name}: {exc}", UserWarning, stacklevel=2)
    return frames_list, labels


def _train_eval(
    train_frames, train_labels,
    test_frames, test_labels,
    n_states_per_class: Dict[str, int],
    n_symbols: int,
    n_iter: int,
    random_state: int,
) -> Dict:
    all_labels = sorted(set(train_labels))

    seqs_by_class: Dict[str, List[np.ndarray]] = {lbl: [] for lbl in all_labels}
    for f, lbl in zip(train_frames, train_labels):
        seqs_by_class[lbl].append(f)

    cfg = HMMConfig(
        n_states=5,
        n_symbols=n_symbols,
        n_iter=n_iter,
        random_state=random_state,
        n_states_per_class=n_states_per_class,
    )
    model = HiddenMarkovModelClassifier(config=cfg)

    t0 = time.perf_counter()
    model.fit(seqs_by_class)
    elapsed = time.perf_counter() - t0

    y_pred = [model.predict(f)[0] for f in test_frames]
    prf = precision_recall_f1(test_labels, y_pred, all_labels)
    acc = accuracy(test_labels, y_pred)
    f1 = macro_f1(prf)
    per_class_recall = {lbl: prf[lbl]['recall'] for lbl in all_labels}

    return {
        'accuracy':         acc,
        'macro_f1':         f1,
        'per_class_recall': per_class_recall,
        'train_time_s':     elapsed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='tune_hmm_per_class',
        description='Gridsearch de n_states por clase + n_symbols para clases problemáticas del HMM.',
    )
    parser.add_argument('--dataset',     required=True, help='Dataset augmentado.')
    parser.add_argument('--tune',        nargs='+', default=['avanzar', 'subir', 'tomar'],
                        help='Clases a tunear.')
    parser.add_argument('--states',      nargs='+', type=int, default=[4, 5, 6],
                        help='Valores de n_states a probar (default: 4 5 6).')
    parser.add_argument('--symbols',     nargs='+', type=int, default=[32, 64],
                        help='Valores de n_symbols a probar (default: 32 64).')
    parser.add_argument('--n-iter',      type=int, default=10,
                        help='Iteraciones Baum-Welch por combinación (default: 10).')
    parser.add_argument('--n-mfcc',      type=int, default=20)
    parser.add_argument('--test-ratio',  type=float, default=0.3)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--output',      default=None,
                        help='Archivo JSON para guardar resultados (opcional).')
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    tune_classes = sorted(args.tune)

    mfcc_cfg = MFCCConfig(
        sample_rate=16000,
        n_mfcc=args.n_mfcc,
        n_filters=26,
        cmvn=True,
        include_delta=True,
        include_delta_delta=False,
        use_librosa=True,
        include_zcr=True,
        include_rms=True,
        include_contrast=True,
    )

    ds_cfg = DatasetConfig(test_ratio=args.test_ratio, random_state=args.random_state)

    # Producto cartesiano: estados por clase × n_symbols
    state_combos = list(itertools.product(args.states, repeat=len(tune_classes)))
    all_combos = [(sc, ns) for sc in state_combos for ns in args.symbols]
    total = len(all_combos)

    print(f"[tune_hmm_per_class] Dataset       : {dataset_root}")
    print(f"[tune_hmm_per_class] Clases a tunear: {tune_classes}")
    print(f"[tune_hmm_per_class] Estados        : {args.states}")
    print(f"[tune_hmm_per_class] n_symbols       : {args.symbols}")
    print(f"[tune_hmm_per_class] Combinaciones  : {total}  (n_iter={args.n_iter} cada una)")
    tiempo_est = total * args.n_iter * 1.5
    print(f"[tune_hmm_per_class] Tiempo estimado: ~{tiempo_est/60:.0f} min\n")

    samples_by_class = discover_dataset(dataset_root)
    split = split_dataset(samples_by_class, ds_cfg)

    print(f"  Extrayendo features de {len(split.train)} train + {len(split.test)} test ...")
    t0 = time.perf_counter()
    train_frames, train_labels = _extract_all(split.train, mfcc_cfg)
    test_frames,  test_labels  = _extract_all(split.test,  mfcc_cfg)
    print(f"  Listo en {time.perf_counter()-t0:.1f}s\n")

    results = []
    for i, (state_combo, n_symbols) in enumerate(all_combos, 1):
        states_map = dict(zip(tune_classes, state_combo))
        n_states_per_class = {**FIXED_STATES, **states_map}

        label_str = '  '.join(f'{cls}={s}' for cls, s in states_map.items())
        print(f"  [{i:02d}/{total:02d}]  {label_str}  sym={n_symbols}", end='  ', flush=True)

        r = _train_eval(
            train_frames, train_labels,
            test_frames,  test_labels,
            n_states_per_class=n_states_per_class,
            n_symbols=n_symbols,
            n_iter=args.n_iter,
            random_state=args.random_state,
        )

        tune_recalls = {cls: r['per_class_recall'].get(cls, 0.0) for cls in tune_classes}
        recall_str = '  '.join(f'{cls}={v:.2f}' for cls, v in tune_recalls.items())
        print(f"acc={r['accuracy']:.4f}  [{recall_str}]  {r['train_time_s']:.0f}s")

        results.append({
            'states':            states_map,
            'n_symbols':         n_symbols,
            'n_states_per_class': n_states_per_class,
            **r,
        })

    results.sort(key=lambda x: (
        -x['accuracy'],
        -sum(x['per_class_recall'].get(c, 0) for c in tune_classes),
    ))

    print(f"\n{'='*75}")
    print(f"  Resultados ordenados por accuracy")
    print(f"{'='*75}")
    header_parts = [f"{'acc':>6}", f"{'sym':>4}"] + [f"{c[:5]:>6}" for c in tune_classes]
    print("  " + "  ".join(header_parts))
    print(f"  {'-'*65}")
    for r in results[:10]:
        row = [f"{r['accuracy']:6.4f}", f"{r['n_symbols']:4d}"]
        for cls in tune_classes:
            s = r['states'][cls]
            rec = r['per_class_recall'].get(cls, 0)
            row.append(f" {s}={rec:.2f}")
        mark = "  <-- MEJOR" if r is results[0] else ""
        print("  " + "  ".join(row) + mark)

    best = results[0]
    print(f"\n  Mejor combinación:")
    print(f"    n_symbols: {best['n_symbols']}")
    for cls in tune_classes:
        print(f"    {cls}: {best['states'][cls]} estados  (recall={best['per_class_recall'].get(cls,0):.4f})")
    print(f"  Accuracy total: {best['accuracy']:.4f}")
    print(f"  Macro F1      : {best['macro_f1']:.4f}")

    print(f"\n  Config completa para train_hmm:")
    print(f"    n_symbols: {best['n_symbols']}")
    for cls, s in sorted(best['n_states_per_class'].items()):
        print(f"    {cls}: {s}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Resultados guardados en: {out}")


if __name__ == '__main__':
    main()
