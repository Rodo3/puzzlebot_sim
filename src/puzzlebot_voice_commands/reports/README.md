# reports/

Evaluation reports, gridsearch results, and parameter analysis for the
`puzzlebot_voice_commands` package.

## Structure

```
reports/
├── gridsearch/        — Per-class n_states × n_symbols gridsearch results
├── metrics/           — Confusion matrices, per-class metrics, safety analysis
├── hmm_parameters/    — PDF report + figures of A/B matrix evolution
└── history/           — Archived snapshots from earlier training runs
    ├── pre_aug/           — Before 4x data augmentation
    ├── hmm_manual/        — Manual HMM runs (post-aug)
    └── hmm_manual_pre_aug/— Manual HMM runs (pre-aug)
```

## Regenerate metrics

```powershell
python -m puzzlebot_voice_commands.scripts.evaluate_models `
  --dataset datasets\voice_commands_dataset_aug `
  --artifact-dir artifacts_final --output-dir reports\metrics --model hmm
```

## Regenerate HMM parameter report

```powershell
python -m puzzlebot_voice_commands.scripts.generate_hmm_parameter_report `
  --dataset datasets\voice_commands_dataset_aug `
  --output-dir reports\hmm_parameters `
  --words alto avanzar retroceder `
  --n-symbols 64 --n-iter 20 `
  --n-mfcc 20 --delta --cmvn --librosa `
  --include-zcr --include-rms --include-contrast `
  --syllable-states --smoothing-eps 1e-6
```

## Gridsearch files

| File | Classes tuned | Best result |
|------|--------------|-------------|
| `tune_alto_bajar_tomar.json` | alto, bajar, tomar | alto=4, bajar=4, tomar=4 |
| `tune_avanzar_derecha_soltar.json` | avanzar, derecha, soltar | avanzar=4, derecha=5, soltar=5 |
| `tune_avanzar_subir_tomar.json` | avanzar, subir, tomar (n_symbols=64) | avanzar=6, subir=6, tomar=6 |
| `tune_bajar.json` | bajar (n_symbols=64) | bajar=5 |
| `tune_soltar_tomar.json` | soltar, tomar | soltar=5, tomar=6 |
| `tune_tomar_78.json` | tomar states 7-8 | tomar=7 (worse globally) |
| `tune_derecha_64.json` | derecha (n_symbols=64) | derecha=6 |
