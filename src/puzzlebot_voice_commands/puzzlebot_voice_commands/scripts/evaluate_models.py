"""
evaluate_voice_models — run evaluation on the test split and generate reports.

Usage (after building and sourcing):
  ros2 run puzzlebot_voice_commands evaluate_voice_models \\
    --dataset      <path/to/voice_commands_dataset> \\
    --artifact-dir <path/to/artifacts> \\
    --output-dir   <path/to/reports>

Reports written to output-dir:
  confusion_matrix_kmeans.csv, confusion_matrix_gnb.csv,
  metrics_kmeans.json, metrics_gnb.json,
  model_comparison.md, safety_metrics.json, inference_time.json

Full implementation in Phase 5.
"""
import argparse
import sys


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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    # Phase 5: implement full evaluation and report generation.
    print("[evaluate_voice_models] Phase 5 not yet implemented.")
    print(f"  dataset      : {args.dataset}")
    print(f"  artifact-dir : {args.artifact_dir}")
    print(f"  output-dir   : {args.output_dir}")
    sys.exit(0)


if __name__ == '__main__':
    main()
