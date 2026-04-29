"""
predict_voice_file — run inference on a single .wav file.

Usage (after building and sourcing):
  ros2 run puzzlebot_voice_commands predict_voice_file \\
    --model-type  kmeans \\
    --model-path  <path/to/artifacts/kmeans_model.pkl> \\
    --audio       <path/to/audio.wav>

  ros2 run puzzlebot_voice_commands predict_voice_file \\
    --model-type  gnb \\
    --model-path  <path/to/artifacts/gnb_model.pkl> \\
    --audio       <path/to/audio.wav>

Prints:
  Predicted command : <label>
  Decision margin   : <value>   (KMeans only — higher = more confident)
  Ranked predictions: <label> (<dist>), ...
"""
import argparse
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='predict_voice_file',
        description=(
            'Run voice command inference on a single .wav file '
            'using a trained KMeans or GNB model.'
        ),
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['kmeans', 'gnb'],
        required=True,
        help='Type of model to use for inference.',
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to the serialized model file (*.pkl).',
    )
    parser.add_argument(
        '--audio',
        type=str,
        required=True,
        help='Path to the .wav audio file to classify.',
    )
    parser.add_argument(
        '--feature-config',
        type=str,
        default=None,
        help='Optional path to feature_config.json (uses defaults if omitted).',
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    audio_path = Path(args.audio)
    model_path = Path(args.model_path)

    # --- Load MFCC config ---------------------------------------------------
    from ..config import MFCCConfig
    from ..serialization import load_json

    mfcc_cfg = MFCCConfig()
    if args.feature_config:
        cfg_data = load_json(Path(args.feature_config))
        mfcc_cfg = MFCCConfig(
            sample_rate=cfg_data.get('sample_rate', mfcc_cfg.sample_rate),
            pre_emphasis=cfg_data.get('pre_emphasis', mfcc_cfg.pre_emphasis),
            frame_size=cfg_data.get('frame_size', mfcc_cfg.frame_size),
            frame_stride=cfg_data.get('frame_stride', mfcc_cfg.frame_stride),
            n_fft=cfg_data.get('n_fft', mfcc_cfg.n_fft),
            n_filters=cfg_data.get('n_filters', mfcc_cfg.n_filters),
            n_mfcc=cfg_data.get('n_mfcc', mfcc_cfg.n_mfcc),
            include_delta=cfg_data.get('include_delta', mfcc_cfg.include_delta),
            include_delta_delta=cfg_data.get(
                'include_delta_delta', mfcc_cfg.include_delta_delta
            ),
        )

    # --- Load audio ---------------------------------------------------------
    from ..audio_io import load_wav, normalize
    from ..mfcc import extract_mfcc_frames, extract_mfcc_summary

    try:
        signal, _ = load_wav(audio_path, target_sr=mfcc_cfg.sample_rate)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    signal = normalize(signal)

    # --- Run inference ------------------------------------------------------
    if args.model_type == 'kmeans':
        from ..models.kmeans_codebook import KMeansCodebookClassifier

        try:
            model = KMeansCodebookClassifier.load(model_path)
        except (FileNotFoundError, TypeError) as exc:
            print(f"ERROR loading model: {exc}", file=sys.stderr)
            sys.exit(1)

        frames = extract_mfcc_frames(signal, mfcc_cfg)

        t0 = time.perf_counter()
        ranked = model.predict_ranked(frames)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        label, margin = ranked[0][0], (ranked[1][1] - ranked[0][1] if len(ranked) > 1 else 0.0)

        print(f"\nAudio             : {audio_path}")
        print(f"Model             : KMeans  ({model_path.name})")
        print(f"Predicted command : {label}")
        print(f"Decision margin   : {margin:.4f}  (higher = more confident)")
        print(f"Inference time    : {elapsed_ms:.1f} ms")
        print("\nRanked predictions:")
        for rank, (lbl, dist) in enumerate(ranked, 1):
            print(f"  {rank}. {lbl:15s}  avg_min_dist={dist:.4f}")

    elif args.model_type == 'gnb':
        from ..models.gaussian_nb import GaussianNaiveBayesClassifier

        try:
            model = GaussianNaiveBayesClassifier.load(model_path)
        except (FileNotFoundError, TypeError) as exc:
            print(f"ERROR loading model: {exc}", file=sys.stderr)
            sys.exit(1)

        summary = extract_mfcc_summary(signal, mfcc_cfg)

        t0 = time.perf_counter()
        label, scores = model.predict(summary)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        ranked = model.predict_ranked(summary)

        print(f"\nAudio             : {audio_path}")
        print(f"Model             : GaussianNB  ({model_path.name})")
        print(f"Predicted command : {label}")
        print(f"Confidence        : {scores[label]:.4f}  (softmax-normalised)")
        print(f"Inference time    : {elapsed_ms:.1f} ms")
        print("\nRanked predictions:")
        for rank, (lbl, log_post) in enumerate(ranked, 1):
            print(f"  {rank}. {lbl:15s}  log_posterior={log_post:.2f}  "
                  f"score={scores.get(lbl, 0.0):.4f}")


if __name__ == '__main__':
    main()
