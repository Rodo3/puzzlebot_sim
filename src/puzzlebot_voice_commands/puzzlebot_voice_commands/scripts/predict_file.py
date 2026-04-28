"""
predict_voice_file — run inference on a single .wav file.

Usage (after building and sourcing):
  ros2 run puzzlebot_voice_commands predict_voice_file \\
    --model-type  gnb \\
    --model-path  <path/to/artifacts/gnb_model.pkl> \\
    --audio       <path/to/audio.wav>

Prints the predicted command and confidence to stdout.
Full implementation in Phases 3/4.
"""
import argparse
import sys


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
        '--labels',
        type=str,
        default=None,
        help='Optional path to labels.json (for label validation).',
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    # Phases 3/4: implement model loading and per-file inference.
    print("[predict_voice_file] Phases 3/4 not yet implemented.")
    print(f"  model-type : {args.model_type}")
    print(f"  model-path : {args.model_path}")
    print(f"  audio      : {args.audio}")
    sys.exit(0)


if __name__ == '__main__':
    main()
