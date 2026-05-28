# puzzlebot_voice_commands — CLAUDE.md

## Estado actual
**Fase 8b completa:** 4 hablantes, augmentation 4x, modelos finales entrenados y guardados.
**Fase 9 completa:** Nodo ROS 2 de inferencia implementado. Integración con dashboard vía bridge.

### Archivos nuevos (Fase 9)
- `voice_inference.py` — Motor de inferencia puro (sin ROS, sin sounddevice). Usado por el bridge para inferencia remota vía POST /audio desde el dashboard.
- `voice_commands_node.py` — Nodo ROS 2 completo para uso futuro con micrófono local en el robot (sounddevice). Listo pero no integrado al launch principal hasta tener los nodos de control.

### Dependencias Python requeridas en Linux
```bash
pip install "numpy>=1.25" scipy librosa "coverage>=7.2" fastapi "uvicorn[standard]" websockets
```

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

Config KMeans (`kmeans_feature_config.json`):
- `n_mfcc=20`, `include_delta=True`, `cmvn=True`

---

## Lo que falta — Fase 9

### 1. `puzzlebot_voice_commands/voice_commands_node.py` (PENDIENTE)

El entry point ya está declarado en `setup.py:49`:
```python
'voice_commands_node = puzzlebot_voice_commands.voice_commands_node:main',
```

**Clase:** `VoiceCommandsNode(rclpy.node.Node)`

**`__init__`:**
- Declara parámetro `artifact_dir` (default: `'artifacts_final'`)
- Declara parámetro `duration` (default: `1.5` segundos)
- Declara parámetro `confidence_threshold` (default: `0.0` — margen mínimo KMeans para publicar)
- Carga ambos modelos con la lógica de `scripts/live_test.py`
- Crea 5 publishers y 1 subscriber

**Grabación en hilo separado** (importante: no bloquear el spin de ROS):
```python
import threading
def _on_trigger(self, msg):
    if self._busy:
        return
    self._busy = True
    threading.Thread(target=self._record_and_infer, daemon=True).start()
```

**Flujo `_record_and_infer()`:**
```
status = "listening"
→ sounddevice.rec(frames, samplerate=16000, channels=1, dtype='float32') + sd.wait()
status = "processing"
→ normalize(audio)
→ extract_mfcc_frames(signal, km_cfg)  → kmeans.predict_ranked(frames_km)
→ extract_librosa_frames(signal, hmm_cfg) → hmm.predict_ranked(frames_hmm)
→ ensemble: tomar etiqueta de KMeans (más preciso), margen = ranked_km[1][1] - ranked_km[0][1]
→ si margen < confidence_threshold: no publicar /voice/command
→ publicar /voice/command, /voice/confidence, /voice/ranked_predictions, /voice/inference_time_ms
status = "idle"
self._busy = False
```

**Tópicos ROS:**
| Tópico | Tipo | Dirección |
|---|---|---|
| `/voice/trigger` | `std_msgs/String` | entrada |
| `/voice/command` | `std_msgs/String` | salida |
| `/voice/confidence` | `std_msgs/Float32` | salida (margen KMeans) |
| `/voice/status` | `std_msgs/String` | salida (`idle`/`listening`/`processing`) |
| `/voice/ranked_predictions` | `std_msgs/String` | salida (JSON top-3 ambos modelos) |
| `/voice/inference_time_ms` | `std_msgs/Float32` | salida |

**Formato `/voice/ranked_predictions` (JSON):**
```json
{
  "kmeans": [["avanzar", 0.0312], ["alto", 0.0841], ["derecha", 0.1205]],
  "hmm":    [["avanzar", -45.2],  ["derecha", -51.8], ["alto", -60.1]]
}
```

**Entry point estándar del repo:**
```python
def main(args=None):
    rclpy.init(args=args)
    node = VoiceCommandsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

---

### 2. `launch/voice_commands.launch.py` — PENDIENTE (futuro)

Solo necesario cuando se use `voice_commands_node` con mic local en el robot.
El bridge ya integra la inferencia remota vía POST /audio.

### 3. Integrar con `puzzlebot_bringup` — PENDIENTE (futuro)

Cuando los nodos de control estén listos y se defina qué acción ejecuta cada comando de voz.

---

## Archivos clave

```
puzzlebot_voice_commands/
├── config.py               — MFCCConfig, HMMConfig (con n_states_per_class)
├── audio_io.py             — load_wav, normalize
├── librosa_features.py     — extract_librosa_frames (MFCC+ZCR+RMS+contrast) ← HMM usa este
├── mfcc.py                 — extract_mfcc_frames manual (NumPy) ← KMeans usa este
├── voice_inference.py      — VoiceInferenceEngine: carga ambos modelos, infer(pcm) → resultado
├── voice_commands_node.py  — Nodo ROS 2 con mic local (futuro uso en robot)
├── models/
│   ├── hmm.py              — HiddenMarkovModelClassifier + _SingleHMM
│   └── kmeans_codebook.py  — KMeansCodebookClassifier
└── scripts/
    └── live_test.py        — Referencia: carga + grabación + inferencia interactiva
```

---

## Cómo cargar los modelos (extraído de `live_test.py`)

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
    n_filters=hmm_mfcc.get('n_filters', 26),
    cmvn=hmm_mfcc.get('cmvn', False),
    include_delta=hmm_mfcc.get('include_delta', False),
    include_delta_delta=hmm_mfcc.get('include_delta_delta', False),
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

signal = normalize(audio.flatten())

# KMeans — usa extract_mfcc_frames (manual, NumPy)
frames_km = extract_mfcc_frames(signal, km_cfg)
ranked_km = kmeans.predict_ranked(frames_km)   # [(label, dist), ...] — menor dist = mejor

# HMM — usa extract_librosa_frames (porque use_librosa=True en config)
frames_hmm = extract_librosa_frames(signal, hmm_cfg)
ranked_hmm = hmm.predict_ranked(frames_hmm)    # [(label, log_lik), ...] — mayor ll = mejor

# Decisión final (KMeans es más preciso)
label = ranked_km[0][0]
margin_km = ranked_km[1][1] - ranked_km[0][1]  # diferencia con segundo lugar (positivo = confianza)
```

---

## Comandos útiles (Windows)

```powershell
cd src\puzzlebot_voice_commands

# Prueba en vivo (referencia de comportamiento esperado del nodo)
python -m puzzlebot_voice_commands.scripts.live_test `
  --artifact-dir artifacts_final --models kmeans hmm

# Evaluar modelos
python -m puzzlebot_voice_commands.scripts.evaluate_models `
  --dataset datasets\voice_commands_dataset_aug `
  --artifact-dir artifacts_final --output-dir reports --model all
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
