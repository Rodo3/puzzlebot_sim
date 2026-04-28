"""
train_voice_models — train KMeans codebook and/or Gaussian NB classifier.

Usage (after building and sourcing):
  ros2 run puzzlebot_voice_commands train_voice_models \\
    --dataset    <path/to/voice_commands_dataset> \\
    --model      both \\
    --output-dir <path/to/artifacts>

Artifacts written to output-dir:
  kmeans_model.pkl, gnb_model.pkl, labels.json,
  train_metadata.json, feature_config.json

Full implementation in Phases 3 and 4.
"""
import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='train_voice_models',
        description=(
            'Train voice command recognition models on a folder-based .wav dataset. '
            'Supports KMeans codebook classifier, Gaussian Naive Bayes, or both.'
        ),
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to the root dataset folder.',
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['kmeans', 'gnb', 'both'],
        default='both',
        help='Which model(s) to train (default: both).',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to write trained model artifacts.',
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.3,
        help='Fraction of samples reserved for the test split (default: 0.3).',
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducible splits (default: 42).',
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=16,
        help='Number of K-Means clusters per class codebook (default: 16).',
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    # Phases 3/4: implement training for each model type.
    print("[train_voice_models] Phases 3/4 not yet implemented.")
    print(f"  dataset     : {args.dataset}")
    print(f"  model       : {args.model}")
    print(f"  output-dir  : {args.output_dir}")
    sys.exit(0)


if __name__ == '__main__':
    main()
