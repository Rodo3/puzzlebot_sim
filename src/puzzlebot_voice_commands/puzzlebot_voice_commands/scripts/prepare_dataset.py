"""
prepare_voice_dataset — scan a folder-based dataset, extract MFCC features,
and save them to a JSON artifact file.

Usage (after building and sourcing):
  ros2 run puzzlebot_voice_commands prepare_voice_dataset \\
    --dataset <path/to/voice_commands_dataset> \\
    --output  <path/to/artifacts/features.json>

The output JSON has the structure:
  {
    "config": { <MFCCConfig fields> },
    "labels": ["adelante", "atras", ...],
    "samples": [
      {
        "path": "adelante/adelante_01.wav",
        "label": "adelante",
        "split": "train",
        "mfcc_mean": [...],
        "mfcc_std":  [...],
        "mfcc_frames": [[...], ...]   // only if --include-frames
      },
      ...
    ]
  }
"""
import argparse
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import List

from ..audio_io import load_wav, normalize
from ..config import DatasetConfig, MFCCConfig
from ..dataset import DatasetSplit, discover_dataset, split_dataset
from ..mfcc import extract_mfcc_frames, extract_mfcc_summary
from ..serialization import save_json


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
        help='Path to the root dataset folder (one subfolder per class).',
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
        help='Number of MFCC coefficients to keep (default: 13).',
    )
    parser.add_argument(
        '--n-filters',
        type=int,
        default=26,
        help='Number of Mel filterbank channels (default: 26).',
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.3,
        help='Fraction of samples reserved for test split (default: 0.3).',
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducible splits (default: 42).',
    )
    parser.add_argument(
        '--include-frames',
        action='store_true',
        default=False,
        help='Also store raw frame-level MFCCs in the JSON (increases file size).',
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    output_path = Path(args.output)

    mfcc_cfg = MFCCConfig(
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        n_filters=args.n_filters,
    )
    dataset_cfg = DatasetConfig(
        test_ratio=args.test_ratio,
        random_state=args.random_state,
    )

    # --- 1. Discover dataset ------------------------------------------------
    print(f"[prepare_voice_dataset] Scanning: {dataset_root}")
    try:
        samples_by_class = discover_dataset(dataset_root)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    total_files = sum(len(v) for v in samples_by_class.values())
    print(f"  Found {len(samples_by_class)} classes, {total_files} .wav files.")
    for label, paths in samples_by_class.items():
        print(f"    {label:15s}  {len(paths)} files")

    # --- 2. Stratified split ------------------------------------------------
    split: DatasetSplit = split_dataset(samples_by_class, dataset_cfg)
    print()
    print(split.summary())

    # --- 3. Extract MFCC features -------------------------------------------
    print(f"\n[prepare_voice_dataset] Extracting MFCCs "
          f"(n_mfcc={mfcc_cfg.n_mfcc}, n_filters={mfcc_cfg.n_filters}) ...")

    all_samples = (
        [(s, 'train') for s in split.train] +
        [(s, 'test')  for s in split.test]
    )

    records = []
    errors = 0
    t0 = time.perf_counter()

    for sample, split_tag in all_samples:
        try:
            signal, _ = load_wav(sample.path, target_sr=mfcc_cfg.sample_rate)
            signal = normalize(signal)
            summary = extract_mfcc_summary(signal, mfcc_cfg)
            record = {
                "path": str(sample.path.relative_to(dataset_root)),
                "label": sample.label,
                "split": split_tag,
                "mfcc_mean": summary[:mfcc_cfg.n_mfcc].tolist(),
                "mfcc_std":  summary[mfcc_cfg.n_mfcc:].tolist(),
            }
            if args.include_frames:
                frames = extract_mfcc_frames(signal, mfcc_cfg)
                record["mfcc_frames"] = frames.tolist()
            records.append(record)
        except Exception as exc:
            warnings.warn(
                f"Failed to process {sample.path}: {exc}",
                UserWarning,
                stacklevel=1,
            )
            errors += 1

    elapsed = time.perf_counter() - t0
    ok = len(records)
    print(f"  Processed {ok}/{len(all_samples)} files in {elapsed:.2f}s "
          f"({elapsed / max(ok, 1) * 1000:.1f} ms/file). "
          f"Errors: {errors}.")

    if ok == 0:
        print("ERROR: No features extracted. Check your dataset.", file=sys.stderr)
        sys.exit(1)

    # --- 4. Save JSON artifact -----------------------------------------------
    output_data = {
        "config": {
            "sample_rate": mfcc_cfg.sample_rate,
            "pre_emphasis": mfcc_cfg.pre_emphasis,
            "frame_size": mfcc_cfg.frame_size,
            "frame_stride": mfcc_cfg.frame_stride,
            "n_fft": mfcc_cfg.n_fft,
            "n_filters": mfcc_cfg.n_filters,
            "n_mfcc": mfcc_cfg.n_mfcc,
            "include_delta": mfcc_cfg.include_delta,
            "include_delta_delta": mfcc_cfg.include_delta_delta,
        },
        "labels": split.labels,
        "split_metadata": split.to_metadata_dict(dataset_cfg),
        "samples": records,
    }

    save_json(output_data, output_path)
    print(f"\n[prepare_voice_dataset] Features saved to: {output_path}")

    # --- 5. Validation summary ----------------------------------------------
    print("\n--- Validation summary ---")
    print(f"  Output file   : {output_path}")
    print(f"  File size     : {output_path.stat().st_size / 1024:.1f} KB")
    print(f"  Total records : {len(records)}")
    print(f"  Labels        : {split.labels}")
    print(f"  MFCC vector   : {mfcc_cfg.n_mfcc * 2}D  "
          f"(mean + std of {mfcc_cfg.n_mfcc} coefficients)")
    print("OK")


if __name__ == '__main__':
    main()
