# puzzlebot_voice_commands — CLAUDE.md

## Estado actual
**Fase 8b completa:** 4 hablantes, augmentation 4x, modelos finales entrenados y guardados.
**Fase 9 en progreso:** Nodo ROS 2 de inferencia pendiente de implementar.

---

## Modelos de producción (`artifacts_final/`)

| Modelo | Accuracy (test) | Archivos |
|---|---|---|
| KMeans | **97.74%** | `kmeans_model.pkl`, `kmeans_feature_config.json` |
| HMM librosa + syllable-states | **92.01%** | `hmm_model.pkl`, `hmm_config.json` |

Dataset: 4 hablantes × 20 clips/cmd × 4x augmentation = **1920 clips**.
Comandos: `avanzar`, `retroceder`, `izquierda`, `derecha`, `alto`, `inicio`.

Config HMM (`hmm_config.json`):
- `n_mfcc=20`, `delta=True`, `cmvn=True`, `use_librosa=True`
- `include_zcr=True`, `include_rms=True`, `include_contrast=True`
- syllable-states: `alto=3, avanzar=4, derecha=4, inicio=4, izquierda=5, retroceder=6`
- `n_symbols=32`, `n_iter=20`

---

## Lo que falta — Fase 9

### 1. `voice_commands_node.py` (PENDIENTE)

Nodo ROS 2 que:
- **Suscribe** `/voice/trigger` (`std_msgs/String`) → dispara una grabación
- **Graba** audio con `sounddevice` (~1.5–2 s)
- **Infiere** con KMeans + HMM desde `artifacts_final/`
- **Publica** en los tópicos de abajo
- Aplica umbral de confianza (no publica si el margen es muy bajo)

Referencia de implementación: `scripts/live_test.py` tiene toda la lógica de carga e inferencia.

**Flujo:**
```
/voice/trigger recibido
  → status = "listening"
  → sounddevice.rec(frames, sr=16000)
  → status = "processing"
  → normalize → extract_features → model.predict_ranked
  → publicar /voice/command, /voice/confidence, /voice/ranked_predictions, /voice/inference_time_ms
  → status = "idle"
```

**Tópicos ROS:**
| Tópico | Tipo | Dirección |
|---|---|---|
| `/voice/trigger` | `std_msgs/String` | entrada |
| `/voice/command` | `std_msgs/String` | salida |
| `/voice/confidence` | `std_msgs/Float32` | salida |
| `/voice/status` | `std_msgs/String` | salida (`idle`/`listening`/`processing`) |
| `/voice/ranked_predictions` | `std_msgs/String` | salida (JSON con top-3) |
| `/voice/inference_time_ms` | `std_msgs/Float32` | salida |

### 2. Launch file `launch/voice_commands.launch.py` (PENDIENTE)

### 3. Integrar con `puzzlebot_bringup` (PENDIENTE)

---

## Archivos clave

```
puzzlebot_voice_commands/
├── config.py               — MFCCConfig, HMMConfig (con n_states_per_class)
├── audio_io.py             — load_wav, normalize
├── librosa_features.py     — extract_librosa_frames (MFCC+ZCR+RMS+contrast)
├── mfcc.py                 — extract_mfcc_frames manual (NumPy, para KMeans)
├── voice_commands_node.py  — PENDIENTE DE CREAR
├── models/
│   ├── hmm.py              — HiddenMarkovModelClassifier + _SingleHMM
│   └── kmeans_codebook.py  — KMeansCodebookClassifier
└── scripts/
    └── live_test.py        — referencia: carga modelos + inferencia por mic
```

---

## Cómo cargar los modelos (de `live_test.py`)

```python
import json
from pathlib import Path
from puzzlebot_voice_commands.models.hmm import HiddenMarkovModelClassifier
from puzzlebot_voice_commands.models.kmeans_codebook import KMeansCodebookClassifier
from puzzlebot_voice_commands.config import MFCCConfig

artifact_dir = Path('artifacts_final')

# KMeans
km_cfg_data = json.loads((artifact_dir / 'kmeans_feature_config.json').read_text())
km_cfg = MFCCConfig(
    sample_rate=km_cfg_data.get('sample_rate', 16000),
    n_mfcc=km_cfg_data.get('n_mfcc', 13),
    include_delta=km_cfg_data.get('include_delta', False),
    cmvn=km_cfg_data.get('cmvn', False),
)
kmeans = KMeansCodebookClassifier.load(artifact_dir / 'kmeans_model.pkl')

# HMM librosa
hmm_cfg_data = json.loads((artifact_dir / 'hmm_config.json').read_text())
hmm_mfcc = hmm_cfg_data['mfcc']
hmm_cfg = MFCCConfig(
    sample_rate=hmm_mfcc.get('sample_rate', 16000),
    n_mfcc=hmm_mfcc.get('n_mfcc', 13),
    cmvn=hmm_mfcc.get('cmvn', False),
    include_delta=hmm_mfcc.get('include_delta', False),
    use_librosa=hmm_mfcc.get('use_librosa', False),
    include_zcr=hmm_mfcc.get('include_zcr', False),
    include_rms=hmm_mfcc.get('include_rms', False),
    include_contrast=hmm_mfcc.get('include_contrast', False),
)
hmm = HiddenMarkovModelClassifier.load(artifact_dir / 'hmm_model.pkl')
```

## Inferencia

```python
from puzzlebot_voice_commands.audio_io import normalize
from puzzlebot_voice_commands.mfcc import extract_mfcc_frames
from puzzlebot_voice_commands.librosa_features import extract_librosa_frames

signal = normalize(audio)

# KMeans
frames_km = extract_mfcc_frames(signal, km_cfg)
ranked_km = kmeans.predict_ranked(frames_km)   # [(label, dist), ...]

# HMM
frames_hmm = extract_librosa_frames(signal, hmm_cfg)
ranked_hmm = hmm.predict_ranked(frames_hmm)    # [(label, log_lik), ...]
```

---

## Comandos útiles (Windows)

```powershell
cd src\puzzlebot_voice_commands

# Prueba en vivo (referencia de comportamiento)
python -m puzzlebot_voice_commands.scripts.live_test `
  --artifact-dir artifacts_final --models kmeans hmm

# Evaluar modelos
python -m puzzlebot_voice_commands.scripts.evaluate_models `
  --dataset datasets\voice_commands_dataset_aug `
  --artifact-dir artifacts_final --output-dir reports --model all
```
