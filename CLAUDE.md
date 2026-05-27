# puzzlebot_sim — CLAUDE.md

## Resumen del repositorio
Workspace ROS 2 Humble para el robot diferencial Puzzlebot (Jetson + LiDAR).
Contiene: simulación Gazebo, SLAM, localización, planificación, percepción, reconocimiento de voz y un dashboard web en tiempo real.

**IMPORTANTE — reglas de seguridad:**
- Nunca publiques comandos de control desde el bridge o el frontend: `/cmd_vel`, `/goal_pose`, `/initialpose`.
- Nunca hagas `git push` o `git commit` automáticamente desde Claude Code.
- Nunca borres archivos sin confirmar con el usuario.

---

## Partes del repositorio

### ROS 2 (`src/`)
| Paquete | Tipo | Rol |
|---|---|---|
| `puzzlebot_bringup` | Python | Launch files para simulación y robot físico |
| `puzzlebot_control` | Python | State machine de misión |
| `puzzlebot_controller` | C++ | Pure-pursuit steering |
| `puzzlebot_description` | CMake | URDF, SDF, meshes, RViz |
| `puzzlebot_localization` | C++ | Odometría + Kalman filter |
| `puzzlebot_msgs` | CMake/rosidl | Mensajes custom |
| `puzzlebot_perception` | Python | Cámara, ArUco, YOLO |
| `puzzlebot_planning` | Python | A* planner + obstacle avoidance |
| `puzzlebot_slam` | Python | SLAM log-odds + MCL |
| `puzzlebot_voice_commands` | Python | Reconocimiento de voz offline (MFCC + HMM) |
| `puzzlebot_web_bridge` | Python | Bridge ROS 2 → WebSocket |
| `shared_utils` | Python | Utilidades compartidas |

### Dashboard web (`web_dashboard/`)
Frontend React + Vite. Solo visualización. Se conecta al bridge vía WebSocket.

---

## Estado actual por rama

### `feat/voice-command-recognition` ← RAMA ACTIVA para voz
**Fase completada (8b):** Dataset 4 hablantes + augmentation + modelos finales entrenados.
**Fase en progreso (9):** Nodo ROS 2 de inferencia (`voice_commands_node.py`) — pendiente de implementar.

**Modelos finales (en `artifacts_final/`):**
| Modelo | Accuracy | Archivo |
|---|---|---|
| KMeans | 97.74% | `kmeans_model.pkl` + `kmeans_feature_config.json` |
| HMM librosa + syllable-states | 92.01% | `hmm_model.pkl` + `hmm_config.json` |

**Dataset:** 4 hablantes (rodo, jorge, valeria, jesus) × 20 clips/cmd × 4x aug = 1920 clips.
**Comandos:** `avanzar`, `retroceder`, `izquierda`, `derecha`, `alto`, `inicio`.

**Lo que falta en esta rama:**
- [ ] Implementar `voice_commands_node.py` — nodo ROS 2 que:
  - Suscribe a `/voice/trigger` (`std_msgs/String`) para iniciar grabación
  - Graba audio con `sounddevice` (1.5–2 s)
  - Corre inferencia con KMeans + HMM desde `artifacts_final/`
  - Publica `/voice/command`, `/voice/confidence`, `/voice/status`, `/voice/ranked_predictions`, `/voice/inference_time_ms`
  - Aplica umbral de confianza (comandos de baja confianza no se publican)
- [ ] Crear launch file `launch/voice_commands.launch.py`
- [ ] Integrar con `puzzlebot_bringup` launch

**Tópicos ROS de voz:**
| Tópico | Tipo | Dirección |
|---|---|---|
| `/voice/trigger` | `std_msgs/String` | entrada (dispara grabación) |
| `/voice/command` | `std_msgs/String` | salida |
| `/voice/confidence` | `std_msgs/Float32` | salida |
| `/voice/status` | `std_msgs/String` | salida (`idle`/`listening`/`processing`) |
| `/voice/ranked_predictions` | `std_msgs/String` | salida (JSON top-3) |
| `/voice/inference_time_ms` | `std_msgs/Float32` | salida |

**Archivos clave de voz:**
```
src/puzzlebot_voice_commands/
├── artifacts_final/              ← modelos de producción (pkl)
├── puzzlebot_voice_commands/
│   ├── config.py                 ← MFCCConfig, HMMConfig (con n_states_per_class)
│   ├── librosa_features.py       ← extracción librosa: MFCC+ZCR+RMS+contrast
│   ├── mfcc.py                   ← extracción manual NumPy (para KMeans)
│   ├── audio_io.py               ← load_wav, normalize
│   ├── voice_commands_node.py    ← PENDIENTE DE CREAR
│   ├── models/
│   │   ├── hmm.py                ← HiddenMarkovModelClassifier
│   │   └── kmeans_codebook.py    ← KMeansCodebookClassifier
│   └── scripts/
│       ├── live_test.py          ← prueba interactiva (referencia para el nodo)
│       └── train_hmm.py          ← entrenamiento HMM
```

### `feat/web-dashboard`
**Estado:** Completo. Bridge WebSocket + frontend React funcionando.
Dashboard muestra: odometría, scan LiDAR, mapa, cámara, cmd_vel, tópicos de voz.

---

## Comandos comunes

### Voice commands (Windows, sin ROS)
```powershell
cd src\puzzlebot_voice_commands

# Live test (mic → predicción)
python -m puzzlebot_voice_commands.scripts.live_test `
  --artifact-dir artifacts_final --models kmeans hmm

# Reentrenar HMM (si se agregan hablantes)
python -m puzzlebot_voice_commands.scripts.train_hmm `
  --dataset datasets\voice_commands_dataset_aug `
  --output-dir artifacts_final `
  --n-mfcc 20 --n-states 5 --n-symbols 32 --n-iter 20 `
  --cmvn --delta --librosa --include-zcr --include-rms --include-contrast `
  --syllable-states
```

### Build ROS 2 (WSL2)
```bash
colcon build --packages-select puzzlebot_voice_commands
source install/setup.bash

# Ejecutar nodo (cuando esté implementado)
ros2 run puzzlebot_voice_commands voice_commands_node \
  --ros-args -p artifact_dir:=src/puzzlebot_voice_commands/artifacts_final
```

### Build web dashboard
```bash
cd web_dashboard
npm install
npm run dev -- --host 0.0.0.0
# Acceso: http://localhost:5173
```

### Bridge WebSocket
```bash
ros2 run puzzlebot_web_bridge bridge_node
```

### Simulación Gazebo
```bash
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze mode:=mapping
```

---

## Contexto técnico del nodo de voz (para implementación)

El `live_test.py` ya tiene la lógica completa de carga + inferencia. El nodo ROS 2
debe replicar ese flujo pero reemplazando el `input()` por un subscriber de trigger.

**Carga de modelos (ver `live_test.py` líneas 86–125):**
```python
# KMeans: lee kmeans_feature_config.json → MFCCConfig
# HMM:    lee hmm_config.json → MFCCConfig con use_librosa=True
```

**Flujo de inferencia por grabación:**
1. Recibe mensaje en `/voice/trigger`
2. Publica `status = "listening"`
3. Graba N segundos con `sounddevice.rec()`
4. Publica `status = "processing"`
5. `normalize(audio)` → extrae features → `model.predict_ranked(frames)`
6. Publica resultados en `/voice/*`
7. Publica `status = "idle"`

**Config HMM (de `artifacts_final/hmm_config.json`):**
- n_mfcc=20, delta=True, cmvn=True, use_librosa=True
- include_zcr=True, include_rms=True, include_contrast=True
- syllable-states: alto=3, avanzar=4, derecha=4, inicio=4, izquierda=5, retroceder=6

---

## Reglas para Claude Code
1. No ejecutar `git push` ni `git commit` automáticamente.
2. No borrar archivos sin confirmación.
3. No publicar a tópicos de control desde ningún componente nuevo.
4. El bridge es solo lectura de tópicos.
5. El frontend es solo visualización.
6. Al agregar dependencias Python, actualizar `package.xml` y `setup.py`.
7. Al agregar dependencias npm, usar solo lo estrictamente necesario.
8. Antes de cambiar de rama, verificar con `git status` que no haya cambios sin commitear.
