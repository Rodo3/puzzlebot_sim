# puzzlebot_voice_commands — CLAUDE.md

## Estado actual
**Fase 10 completa:** Expansión a 10 palabras + optimización HMM via gridsearch n_states×n_symbols.
**Modelo activo:** Solo HMM (KMeans descartado en producción).
**Accuracy final: 87.40%** con split leakage-free.

### Fixes aplicados
- `voice_inference.py`: resamplea el audio del browser (44100→16000 Hz) antes de extraer features.
- `bridge_node.py`: broadcast WebSocket directo desde `_handle_audio_bytes`, sin depender del roundtrip DDS.
- `dataset.py`: split leakage-free via `_original_stem()` — variantes augmentadas del mismo clip nunca cruzan train/test.

### Nota importante sobre el micrófono
Los modelos se entrenaron con un dispositivo de audio específico. Usar el mismo micrófono/audífonos
del entrenamiento garantiza mejor accuracy. Con un micrófono diferente la predicción puede degradarse.

### Dependencias Python requeridas en Linux
```bash
pip install "numpy>=1.25" scipy librosa "coverage>=7.2" fastapi "uvicorn[standard]" websockets
```

---

## Modelo de producción (`artifacts_final/`)

| Modelo | Accuracy (test) | Archivos |
|---|---|---|
| HMM librosa + syllable-states | **87.40%** | `hmm_model.pkl`, `hmm_config.json` |

Dataset: 4 hablantes × 20 clips/cmd × 10 cmds × 4x augmentation = **3200 clips**.
Comandos: `avanzar`, `retroceder`, `izquierda`, `derecha`, `alto`, `inicio`, `subir`, `bajar`, `tomar`, `soltar`.

Config HMM (`hmm_config.json`):
- `n_mfcc=20`, `delta=True`, `cmvn=True`, `use_librosa=True`
- `include_zcr=True`, `include_rms=True`, `include_contrast=True`
- `n_symbols=64`, `n_iter=5`
- syllable-states: `alto=4, avanzar=6, bajar=5, derecha=6, inicio=4, izquierda=5, retroceder=6, soltar=5, subir=6, tomar=6`

---

## Archivos clave

```
puzzlebot_voice_commands/
├── config.py               — MFCCConfig, HMMConfig (con n_states_per_class)
├── audio_io.py             — load_wav, normalize
├── librosa_features.py     — extract_librosa_frames (MFCC+ZCR+RMS+contrast) ← HMM usa este
├── dataset.py              — split leakage-free via _original_stem
├── voice_inference.py      — VoiceInferenceEngine: carga HMM, infer(pcm) → resultado
├── voice_commands_node.py  — Nodo ROS 2 con mic local (futuro uso en robot)
├── models/
│   └── hmm.py              — HiddenMarkovModelClassifier + _SingleHMM
└── scripts/
    ├── grabar.py               — Grabación interactiva (10 palabras)
    ├── augment_dataset.py      — Augmentation 4x
    ├── train_hmm.py            — Entrenamiento HMM
    ├── tune_hmm_per_class.py   — Gridsearch n_states×n_symbols por clase
    ├── evaluate_models.py      — Evaluación
    └── live_test.py            — Test en vivo (mic → predicción)
```

---

## Cómo cargar el modelo HMM

```python
import json
from pathlib import Path
from puzzlebot_voice_commands.models.hmm import HiddenMarkovModelClassifier
from puzzlebot_voice_commands.config import MFCCConfig

artifact_dir = Path('artifacts_final')

hmm_cfg_data = json.loads((artifact_dir / 'hmm_config.json').read_text())
hmm_mfcc = hmm_cfg_data['mfcc']
hmm_cfg = MFCCConfig(
    sample_rate=hmm_mfcc.get('sample_rate', 16000),
    n_mfcc=hmm_mfcc.get('n_mfcc', 20),
    n_filters=hmm_mfcc.get('n_filters', 26),
    cmvn=hmm_mfcc.get('cmvn', True),
    include_delta=hmm_mfcc.get('include_delta', True),
    include_delta_delta=hmm_mfcc.get('include_delta_delta', False),
    use_librosa=hmm_mfcc.get('use_librosa', True),
    include_zcr=hmm_mfcc.get('include_zcr', True),
    include_rms=hmm_mfcc.get('include_rms', True),
    include_contrast=hmm_mfcc.get('include_contrast', True),
)
hmm = HiddenMarkovModelClassifier.load(artifact_dir / 'hmm_model.pkl')
```

## Inferencia

```python
from puzzlebot_voice_commands.audio_io import normalize
from puzzlebot_voice_commands.librosa_features import extract_librosa_frames

signal = normalize(audio.flatten())
frames = extract_librosa_frames(signal, hmm_cfg)
ranked = hmm.predict_ranked(frames)  # [(label, log_lik), ...] — mayor ll = mejor
label  = ranked[0][0]
margin = ranked[0][1] - ranked[1][1]  # confianza
```

---

## Comandos útiles (Windows)

```powershell
cd src\puzzlebot_voice_commands

# Entrenamiento HMM (config final)
python -m puzzlebot_voice_commands.scripts.train_hmm `
  --dataset datasets\voice_commands_dataset_aug `
  --output-dir artifacts_final --n-iter 5 --syllable-states `
  --n-mfcc 20 --n-symbols 64 --cmvn --delta --librosa `
  --include-zcr --include-rms --include-contrast

# Evaluación
python -m puzzlebot_voice_commands.scripts.evaluate_models `
  --dataset datasets\voice_commands_dataset_aug `
  --artifact-dir artifacts_final --output-dir reports --model hmm

# Test en vivo
python -m puzzlebot_voice_commands.scripts.live_test `
  --artifact-dir artifacts_final --models hmm

# Gridsearch (ejemplo)
python -m puzzlebot_voice_commands.scripts.tune_hmm_per_class `
  --dataset datasets\voice_commands_dataset_aug `
  --tune tomar bajar --states 4 5 6 --symbols 64 --n-iter 5 `
  --output reports\tune_tomar_bajar.json
```

```bash
# WSL2 — construir y correr el nodo
colcon build --packages-select puzzlebot_voice_commands
source install/setup.bash
ros2 run puzzlebot_voice_commands voice_commands_node \
  --ros-args -p artifact_dir:=src/puzzlebot_voice_commands/artifacts_final

# Disparar grabación
ros2 topic pub --once /voice/trigger std_msgs/String "data: 'record'"

# Monitorear
ros2 topic echo /voice/command
ros2 topic echo /voice/status
```

---

## Pendiente (futuro)

- `launch/voice_commands.launch.py` — solo necesario con mic local en el robot
- Integrar `voice_commands_node` con `puzzlebot_bringup` cuando los nodos de control estén listos
