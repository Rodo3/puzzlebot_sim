"""
prepare_voice_dataset — scan a folder-based dataset and extract MFCC features.

Usage (after building and sourcing):
  ros2 run puzzlebot_voice_commands prepare_voice_dataset \\
    --dataset <path/to/voice_commands_dataset> \\
    --output  <path/to/artifacts/features.json>

Full implementation in Phase 2.
"""
import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='prepare_voice_dataset',
        description=(
            'Scan a folder-based .wav dataset, extract MFCC features, '
            'and save them to a JSON artifact file.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Expected dataset layout:\n'
            '  <dataset>/\n'
            '  ├── adelante/\n'
            '  │   ├── adelante_01.wav\n'
            '  │   └── ...\n'
            '  ├── atras/\n'
            '  ├── izquierda/\n'
            '  ├── derecha/\n'
            '  ├── alto/\n'
            '  └── inicio/\n'
        ),
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to the root dataset folder (contains one subfolder per class).',
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to write the extracted features JSON file.',
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=16000,
        help='Expected sample rate in Hz (default: 16000).',
    )
    parser.add_argument(
        '--n-mfcc',
        type=int,
        default=13,
        help='Number of MFCC coefficients (default: 13).',
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    # Phase 2: implement dataset discovery, MFCC extraction, and JSON output.
    print("[prepare_voice_dataset] Phase 2 not yet implemented.")
    print(f"  dataset : {args.dataset}")
    print(f"  output  : {args.output}")
    sys.exit(0)


if __name__ == '__main__':
    main()
