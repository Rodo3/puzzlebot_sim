# datasets/

Place your voice command dataset here.

## Expected structure

```
datasets/
└── voice_commands_dataset/
    ├── adelante/
    │   ├── adelante_01.wav
    │   ├── adelante_02.wav
    │   └── ...
    ├── atras/
    ├── izquierda/
    ├── derecha/
    ├── alto/
    └── inicio/
```

## Audio requirements

| Property     | Required value     |
|--------------|--------------------|
| Format       | WAV                |
| Channels     | Mono preferred     |
| Sample rate  | 16 kHz preferred   |
| Duration     | ~1–2 seconds       |

If stereo audio is supplied, it will be automatically converted to mono.
If the sample rate differs from 16 kHz, a warning will be printed and
resampling will be attempted using SciPy.

## Class labels

Labels are inferred automatically from subdirectory names.
Default target commands: `adelante`, `atras`, `izquierda`, `derecha`, `alto`, `inicio`.

Extra subfolders will be discovered and used as additional classes.

## Minimum samples per class

At least 2 samples per class are required for a stratified train/test split.
Classes with fewer than 2 samples will be skipped with a warning.

## Git

`.wav` files are **not** committed to the repository.
Add your dataset locally before running the pipeline.
