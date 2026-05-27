"""
live_test.py — Prueba interactiva del modelo en tiempo real.

Graba 1.5 s desde el micrófono y muestra la predicción de GNB y KMeans.
Presiona Enter para grabar, 'q' + Enter para salir.

Usage (Windows, sin ROS):
  cd src/puzzlebot_voice_commands
  python -m puzzlebot_voice_commands.scripts.live_test --artifact-dir artifacts
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd


def _load_mfcc_cfg(cfg_path: Path):
    from ..config import MFCCConfig
    data = json.loads(cfg_path.read_text())
    return MFCCConfig(
        sample_rate=data.get('sample_rate', 16000),
        pre_emphasis=data.get('pre_emphasis', 0.97),
        frame_size=data.get('frame_size', 0.025),
        frame_stride=data.get('frame_stride', 0.01),
        n_fft=data.get('n_fft', 512),
        n_filters=data.get('n_filters', 26),
        n_mfcc=data.get('n_mfcc', 13),
        include_delta=data.get('include_delta', False),
        include_delta_delta=data.get('include_delta_delta', False),
        cmvn=data.get('cmvn', False),
        include_min_max=data.get('include_min_max', False),
    )


def _record(duration: float, fs: int) -> np.ndarray:
    frames = int(duration * fs)
    audio = sd.rec(frames, samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='live_test',
        description='Prueba interactiva: graba y predice en tiempo real.',
    )
    parser.add_argument('--artifact-dir', required=True,
                        help='Carpeta con los modelos entrenados.')
    parser.add_argument('--duration', type=float, default=1.5,
                        help='Duración de cada grabación en segundos (default: 1.5).')
    parser.add_argument('--models', nargs='+', choices=['gnb', 'kmeans', 'hmm'],
                        default=['hmm', 'gnb', 'kmeans'],
                        help='Modelos a usar (default: hmm gnb kmeans).')
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)

    from ..audio_io import normalize
    from ..mfcc import extract_mfcc_frames, extract_mfcc_summary

    # --- Cargar modelos -------------------------------------------------------
    models = {}

    if 'gnb' in args.models:
        from ..models.gaussian_nb import GaussianNaiveBayesClassifier
        gnb_path = artifact_dir / 'gnb_model.pkl'
        gnb_cfg_path = artifact_dir / 'feature_config.json'
        if not gnb_path.exists():
            print(f"ERROR: no se encontró {gnb_path}", file=sys.stderr)
            sys.exit(1)
        models['gnb'] = {
            'model': GaussianNaiveBayesClassifier.load(gnb_path),
            'cfg': _load_mfcc_cfg(gnb_cfg_path),
        }
        print(f"  GNB cargado     : {gnb_path.name}")

    if 'kmeans' in args.models:
        from ..models.kmeans_codebook import KMeansCodebookClassifier
        km_path = artifact_dir / 'kmeans_model.pkl'
        km_cfg_path = artifact_dir / 'kmeans_feature_config.json'
        if not km_path.exists():
            print(f"ERROR: no se encontró {km_path}", file=sys.stderr)
            sys.exit(1)
        models['kmeans'] = {
            'model': KMeansCodebookClassifier.load(km_path),
            'cfg': _load_mfcc_cfg(km_cfg_path),
        }
        print(f"  KMeans cargado  : {km_path.name}")

    if 'hmm' in args.models:
        from ..models.hmm import HiddenMarkovModelClassifier
        hmm_path = artifact_dir / 'hmm_model.pkl'
        hmm_cfg_path = artifact_dir / 'hmm_config.json'
        if not hmm_path.exists():
            print(f"ERROR: no se encontró {hmm_path}", file=sys.stderr)
            sys.exit(1)
        hmm_cfg_data = json.loads(hmm_cfg_path.read_text())
        from ..config import MFCCConfig
        hmm_mfcc = hmm_cfg_data.get('mfcc', {})
        hmm_cfg = MFCCConfig(
            sample_rate=hmm_mfcc.get('sample_rate', 16000),
            n_mfcc=hmm_mfcc.get('n_mfcc', 13),
            n_filters=hmm_mfcc.get('n_filters', 26),
            cmvn=hmm_mfcc.get('cmvn', False),
            include_delta=hmm_mfcc.get('include_delta', False),
            include_delta_delta=hmm_mfcc.get('include_delta_delta', False),
            use_librosa=hmm_mfcc.get('use_librosa', False),
            include_zcr=hmm_mfcc.get('include_zcr', False),
            include_rms=hmm_mfcc.get('include_rms', False),
            include_contrast=hmm_mfcc.get('include_contrast', False),
        )
        models['hmm'] = {
            'model': HiddenMarkovModelClassifier.load(hmm_path),
            'cfg': hmm_cfg,
        }
        print(f"  HMM cargado     : {hmm_path.name}")

    # --- Info del micrófono ---------------------------------------------------
    info = sd.query_devices(kind='input')
    fs = 16000

    print(f"\n  Microfono : {info['name']}")
    print(f"  Duracion  : {args.duration}s  |  Sample rate: {fs} Hz")
    print(f"  Modelos   : {', '.join(models.keys())}")
    print(f"\nPresiona Enter para grabar  |  'q' + Enter para salir\n")
    print("=" * 55)

    while True:
        try:
            cmd = input("\n  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd == 'q':
            break

        print("  Grabando...", end='', flush=True)
        audio = _record(args.duration, fs)
        print(f" listo  ({len(audio)/fs:.1f}s)")

        signal = normalize(audio)

        print()
        for name, entry in models.items():
            model = entry['model']
            cfg = entry['cfg']
            t0 = time.perf_counter()

            if name == 'gnb':
                summary = extract_mfcc_summary(signal, cfg)
                label, scores = model.predict(summary)
                ranked = model.predict_ranked(summary)
                elapsed = (time.perf_counter() - t0) * 1000
                conf = scores[label]
                print(f"  [GNB]    {label.upper():<12}  confianza={conf:.3f}  ({elapsed:.1f} ms)")
                for i, (lbl, _) in enumerate(ranked[:3], 1):
                    marker = "<--" if i == 1 else "   "
                    print(f"           {i}. {lbl:<12} score={scores.get(lbl,0):.3f} {marker}")

            elif name == 'kmeans':
                frames = extract_mfcc_frames(signal, cfg)
                ranked = model.predict_ranked(frames)
                elapsed = (time.perf_counter() - t0) * 1000
                label = ranked[0][0]
                margin = ranked[1][1] - ranked[0][1] if len(ranked) > 1 else 0.0
                print(f"  [KMeans] {label.upper():<12}  margen={margin:.4f}  ({elapsed:.1f} ms)")
                for i, (lbl, dist) in enumerate(ranked[:3], 1):
                    marker = "<--" if i == 1 else "   "
                    print(f"           {i}. {lbl:<12} dist={dist:.4f} {marker}")

            elif name == 'hmm':
                if cfg.use_librosa:
                    from ..librosa_features import extract_librosa_frames
                    frames = extract_librosa_frames(signal, cfg)
                else:
                    frames = extract_mfcc_frames(signal, cfg)
                ranked = model.predict_ranked(frames)
                elapsed = (time.perf_counter() - t0) * 1000
                label = ranked[0][0]
                margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 0.0
                print(f"  [HMM]    {label.upper():<12}  margen={margin:.2f}  ({elapsed:.1f} ms)")
                for i, (lbl, ll) in enumerate(ranked[:3], 1):
                    marker = "<--" if i == 1 else "   "
                    print(f"           {i}. {lbl:<12} log_lik={ll:.2f} {marker}")

        print()

    print("\nSaliendo.")


if __name__ == '__main__':
    main()
