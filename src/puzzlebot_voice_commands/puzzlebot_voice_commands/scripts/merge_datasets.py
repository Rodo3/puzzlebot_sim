"""
merge_datasets.py — Combina múltiples carpetas de grabaciones en un solo dataset.

Cada carpeta de entrada representa las grabaciones de una persona:
  data_jorge/
  data_ana/
  data_luis/
  data_rodrigo/

Cada una tiene la misma estructura de subclases:
  <persona>/avanzar/*.wav
  <persona>/retroceder/*.wav
  ...

El script copia todos los .wav al dataset de salida renombrándolos para evitar
colisiones:
  <output>/avanzar/jorge_avanzar_01.wav
  <output>/avanzar/ana_avanzar_01.wav
  ...

Usage (desde el directorio del workspace, en WSL2):
  python3 src/puzzlebot_voice_commands/puzzlebot_voice_commands/scripts/merge_datasets.py \\
    --inputs  src/puzzlebot_voice_commands/datasets/data_jorge \\
              src/puzzlebot_voice_commands/datasets/data_ana   \\
              src/puzzlebot_voice_commands/datasets/data_luis  \\
              src/puzzlebot_voice_commands/datasets/data_rodrigo \\
    --output  src/puzzlebot_voice_commands/datasets/voice_commands_dataset

O también puede apuntar a una carpeta que contenga todas las subcarpetas de personas:
  python3 ... --input-dir src/puzzlebot_voice_commands/datasets \\
              --output    src/puzzlebot_voice_commands/datasets/voice_commands_dataset
"""
import argparse
import shutil
import sys
import warnings
from pathlib import Path
from typing import List

# Clases esperadas — solo se copian estas; el resto se ignora con advertencia.
EXPECTED_CLASSES = {'avanzar', 'retroceder', 'izquierda', 'derecha', 'alto', 'inicio'}


def _person_name(folder: Path) -> str:
    """Derive a short person identifier from a folder name.

    'data_jorge' -> 'jorge'
    'jorge'      -> 'jorge'
    """
    name = folder.name.lower()
    if name.startswith('data_'):
        name = name[len('data_'):]
    return name


def merge(input_dirs: List[Path], output_dir: Path, dry_run: bool = False) -> None:
    output_dir = Path(output_dir)

    # Validate all input dirs exist
    for d in input_dirs:
        if not d.is_dir():
            print(f"ERROR: input folder not found: {d}", file=sys.stderr)
            sys.exit(1)

    # Create output class subdirs
    if not dry_run:
        for cls in EXPECTED_CLASSES:
            (output_dir / cls).mkdir(parents=True, exist_ok=True)

    total_copied = 0
    total_skipped = 0
    summary: dict = {cls: {} for cls in EXPECTED_CLASSES}

    for person_dir in input_dirs:
        person = _person_name(person_dir)
        person_classes = sorted(p.name for p in person_dir.iterdir() if p.is_dir())

        # Warn about unexpected classes
        unexpected = set(person_classes) - EXPECTED_CLASSES
        if unexpected:
            warnings.warn(
                f"[{person}] Ignoring unexpected class folders: {sorted(unexpected)}",
                UserWarning,
            )

        # Warn about missing expected classes
        missing = EXPECTED_CLASSES - set(person_classes)
        if missing:
            warnings.warn(
                f"[{person}] Missing expected class folders: {sorted(missing)}",
                UserWarning,
            )

        for cls in EXPECTED_CLASSES:
            cls_dir = person_dir / cls
            if not cls_dir.exists():
                continue

            wav_files = sorted(cls_dir.glob('*.wav'))
            if not wav_files:
                warnings.warn(
                    f"[{person}/{cls}] No .wav files found — skipping.",
                    UserWarning,
                )
                continue

            summary[cls][person] = len(wav_files)

            for i, src in enumerate(wav_files, start=1):
                dest_name = f"{person}_{cls}_{i:02d}.wav"
                dest = output_dir / cls / dest_name

                if dest.exists():
                    warnings.warn(
                        f"Destination already exists, overwriting: {dest}",
                        UserWarning,
                    )

                if dry_run:
                    print(f"  [DRY RUN] {src} -> {dest}")
                else:
                    shutil.copy2(src, dest)
                total_copied += 1

    # Print summary table
    print(f"\n{'─'*60}")
    print(f"  Merge {'(DRY RUN) ' if dry_run else ''}summary")
    print(f"{'─'*60}")

    persons = sorted({p for cls_dict in summary.values() for p in cls_dict})
    col_w = max(len(p) for p in persons) + 2 if persons else 8

    header = f"  {'Class':<14}" + "".join(f"{p:>{col_w}}" for p in persons) + f"{'Total':>8}"
    print(header)
    print(f"  {'─'*12}" + "─" * (col_w * len(persons) + 8))

    for cls in sorted(EXPECTED_CLASSES):
        counts = summary[cls]
        row_total = sum(counts.values())
        row = f"  {cls:<14}" + "".join(f"{counts.get(p, 0):>{col_w}}" for p in persons) + f"{row_total:>8}"
        print(row)

    print(f"  {'─'*12}" + "─" * (col_w * len(persons) + 8))
    grand_total = sum(sum(d.values()) for d in summary.values())
    print(f"  {'TOTAL':<14}" + "".join(
        f"{sum(summary[cls].get(p, 0) for cls in EXPECTED_CLASSES):>{col_w}}"
        for p in persons
    ) + f"{grand_total:>8}")

    print(f"\n  Output : {output_dir}")
    print(f"  Files  : {total_copied} copied")
    if total_skipped:
        print(f"  Skipped: {total_skipped}")
    if not dry_run:
        print("\n  Ready to train:")
        print(f"    ros2 run puzzlebot_voice_commands train_voice_models \\")
        print(f"      --dataset {output_dir} \\")
        print(f"      --model both \\")
        print(f"      --output-dir <artifacts_dir>")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='merge_datasets',
        description=(
            'Merge per-person recording folders into a single dataset directory '
            'ready for train_voice_models.'
        ),
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        '--inputs',
        nargs='+',
        metavar='DIR',
        help='One or more per-person recording folders (e.g. data_jorge data_ana ...).',
    )
    source_group.add_argument(
        '--input-dir',
        metavar='DIR',
        help=(
            'Parent folder that contains all per-person subfolders. '
            'All immediate subdirectories that contain a known class subfolder '
            'are treated as person folders. The output folder is excluded automatically.'
        ),
    )
    parser.add_argument(
        '--output',
        required=True,
        metavar='DIR',
        help='Destination dataset folder (created if it does not exist).',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=False,
        help='Print what would be copied without actually copying anything.',
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()

    if args.inputs:
        input_dirs = [Path(d).resolve() for d in args.inputs]
    else:
        parent = Path(args.input_dir).resolve()
        if not parent.is_dir():
            print(f"ERROR: --input-dir not found: {parent}", file=sys.stderr)
            sys.exit(1)
        # Auto-discover: subdirs that have at least one expected-class subfolder
        input_dirs = []
        for candidate in sorted(parent.iterdir()):
            if not candidate.is_dir():
                continue
            if candidate.resolve() == output_dir:
                continue  # skip the output folder itself
            has_class = any((candidate / cls).is_dir() for cls in EXPECTED_CLASSES)
            if has_class:
                input_dirs.append(candidate)
        if not input_dirs:
            print(
                f"ERROR: No person folders found under {parent}. "
                "Each subfolder must contain at least one of: "
                f"{sorted(EXPECTED_CLASSES)}",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Input folders ({len(input_dirs)}):")
    for d in input_dirs:
        print(f"  {d}")
    print(f"Output : {output_dir}")
    if args.dry_run:
        print("Mode   : DRY RUN (nothing will be copied)\n")

    merge(input_dirs, output_dir, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
