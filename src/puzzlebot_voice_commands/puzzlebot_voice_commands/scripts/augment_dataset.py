"""
augment_dataset.py — Data augmentation for voice command dataset.

Generates synthetic variants of each .wav file using librosa:
  - Time stretching  : ±10% speed change
  - Pitch shifting   : ±1 semitone
  - Background noise : light Gaussian noise
  - Volume scaling   : ±20% amplitude

Reads from --input-dir and writes augmented files alongside originals in
--output-dir, naming them:  <original_stem>_aug_<tag>.wav

Usage (Windows):
  cd src/puzzlebot_voice_commands
  python -m puzzlebot_voice_commands.scripts.augment_dataset \
    --input-dir  datasets/voice_commands_dataset \
    --output-dir datasets/voice_commands_dataset_aug \
    --factor 3
"""
import argparse
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

EXPECTED_CLASSES = {'avanzar', 'retroceder', 'izquierda', 'derecha', 'alto', 'inicio',
                    'subir', 'bajar', 'tomar', 'soltar'}
SR = 16000


def _load(path: Path):
    import librosa
    signal, _ = librosa.load(str(path), sr=SR, mono=True)
    return signal.astype(np.float32)


def _save(signal: np.ndarray, path: Path):
    # Clip to [-1, 1] before saving
    signal = np.clip(signal, -1.0, 1.0)
    sf.write(str(path), signal, SR)


def _augmentations(signal: np.ndarray, rng: np.random.Generator):
    """Yield (tag, augmented_signal) for each transformation."""
    import librosa

    # Time stretch fast (+10%)
    yield 'ts_fast', librosa.effects.time_stretch(signal, rate=1.10).astype(np.float32)
    # Time stretch slow (-10%)
    yield 'ts_slow', librosa.effects.time_stretch(signal, rate=0.90).astype(np.float32)
    # Pitch shift up +1 semitone
    yield 'ps_up',   librosa.effects.pitch_shift(signal, sr=SR, n_steps=1).astype(np.float32)
    # Pitch shift down -1 semitone
    yield 'ps_down', librosa.effects.pitch_shift(signal, sr=SR, n_steps=-1).astype(np.float32)
    # Light Gaussian noise (SNR ~30 dB)
    noise = rng.normal(0, 0.005, len(signal)).astype(np.float32)
    yield 'noise',   (signal + noise).astype(np.float32)
    # Volume up +20%
    yield 'vol_up',  (signal * 1.20).astype(np.float32)
    # Volume down -20%
    yield 'vol_dn',  (signal * 0.80).astype(np.float32)


def augment(input_dir: Path, output_dir: Path, factor: int, dry_run: bool, seed: int):
    rng = np.random.default_rng(seed)
    all_tags = ['ts_fast', 'ts_slow', 'ps_up', 'ps_down', 'noise', 'vol_up', 'vol_dn']

    # factor controls how many augmentations per file (max = len(all_tags) = 7)
    n_aug = min(factor, len(all_tags))
    # Pick the first n_aug tags (deterministic selection)
    selected_tags = all_tags[:n_aug]

    total_orig = 0
    total_aug  = 0

    for cls in sorted(EXPECTED_CLASSES):
        cls_in  = input_dir  / cls
        cls_out = output_dir / cls
        if not cls_in.exists():
            warnings.warn(f"Class folder not found: {cls_in}", UserWarning)
            continue

        if not dry_run:
            cls_out.mkdir(parents=True, exist_ok=True)

        wav_files = sorted(cls_in.glob('*.wav'))
        total_orig += len(wav_files)

        for src in wav_files:
            # Copy original
            dest_orig = cls_out / src.name
            if not dry_run:
                shutil.copy2(src, dest_orig)

            # Generate augmented versions
            signal = None
            for tag in selected_tags:
                dest_aug = cls_out / f"{src.stem}_aug_{tag}.wav"
                if dry_run:
                    print(f"  [DRY] {src.name} -> {dest_aug.name}")
                    total_aug += 1
                    continue
                if signal is None:
                    signal = _load(src)
                for aug_tag, aug_signal in _augmentations(signal, rng):
                    if aug_tag == tag:
                        _save(aug_signal, dest_aug)
                        total_aug += 1
                        break

    return total_orig, total_aug


def build_parser():
    parser = argparse.ArgumentParser(
        prog='augment_voice_dataset',
        description='Augment voice command dataset with synthetic variants.',
    )
    parser.add_argument('--input-dir',  required=True,
                        help='Source dataset folder (one subfolder per class).')
    parser.add_argument('--output-dir', required=True,
                        help='Destination folder (created if needed).')
    parser.add_argument('--factor', type=int, default=3,
                        help='Augmentations per file: 1-7 (default: 3 = ts_fast, ts_slow, ps_up).')
    parser.add_argument('--dry-run', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    input_dir  = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists():
        print(f"ERROR: input dir not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    n_aug = min(args.factor, 7)
    tags  = ['ts_fast', 'ts_slow', 'ps_up', 'ps_down', 'noise', 'vol_up', 'vol_dn'][:n_aug]

    print(f"Input  : {input_dir}")
    print(f"Output : {output_dir}")
    print(f"Factor : {args.factor} augmentation(s) per file -> {tags}")
    if args.dry_run:
        print("Mode   : DRY RUN\n")

    total_orig, total_aug = augment(input_dir, output_dir, args.factor, args.dry_run, args.seed)

    print(f"\n--- Summary ---")
    print(f"  Original files  : {total_orig}")
    print(f"  Augmented files : {total_aug}")
    print(f"  Total output    : {total_orig + total_aug}")
    print(f"  Multiplier      : {(total_orig + total_aug) / max(total_orig, 1):.1f}x")
    if not args.dry_run:
        print(f"\n  Dataset ready at: {output_dir}")


if __name__ == '__main__':
    main()
