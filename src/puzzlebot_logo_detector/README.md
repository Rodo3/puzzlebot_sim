# puzzlebot_logo_detector

Detección de logos de clientes en los tráileres del almacén (**Pepsi / Amazon /
Walmart**) con **YOLO11n** exportado a ONNX, corriendo sobre **ONNX Runtime**.

Es un nodo de percepción puro: publica qué logo ve y dónde (bbox), sin tomar
decisiones. El `state_machine_node` (`puzzlebot_control`) consume `/logo_detection/result`
durante la fase `SCANNING_LOGOS` para identificar el tráiler correcto comparándolo
con el string leído del QR.

> Para correr el detector **en la Jetson Nano sin ROS workspace**, ver `ros2_deploy/CLAUDE.md`.

---

## Nodo: `yolo_node` (en `puzzlebot_perception`)

El nodo de inferencia principal está en `puzzlebot_perception` para estandarizar
los tópicos del proyecto. Este paquete (`puzzlebot_logo_detector`) provee los modelos.

```bash
# Standalone (siempre infiere)
ros2 run puzzlebot_perception yolo_node

# En el robot (headless + gateado por la misión)
ros2 run puzzlebot_perception yolo_node --ros-args \
  -p show_window:=false \
  -p gate_by_mission:=true \
  -p inference_hz:=3.0
```

### Suscribe
| Tópico | Tipo | Notas |
|---|---|---|
| `/camera/image/compressed` | sensor_msgs/CompressedImage | Entrada de cámara (BEST_EFFORT QoS) |
| `/mission_state` | std_msgs/String | Solo si `gate_by_mission:=true` |

### Publica
| Tópico | Tipo | Contenido |
|---|---|---|
| `/detections` | vision_msgs/Detection2DArray | Detecciones con clase y confianza (bbox normalizado [0,1]) |
| `/yolo/debug_image` | sensor_msgs/Image | Frame anotado para debug (rqt) |

### Formato de salida (`/detections`)

```
detection.results[0].hypothesis.class_id  → "Pepsi" / "Amazon" / "Walmart"
detection.results[0].hypothesis.score     → confianza [0, 1]
detection.bbox.center.position.x/y        → centro normalizado [0, 1]
detection.bbox.size_x/y                   → tamaño normalizado [0, 1]
```

Clases del modelo: `0=Pepsi`, `1=Amazon`, `2=Walmart`.

### Parámetros

| Parámetro | Default | Descripción |
|---|---|---|
| `weights_path` | auto-detect en `puzzlebot_logo_detector/models/best.onnx` | Ruta al modelo ONNX |
| `confidence` | `0.70` | Umbral mínimo de confianza |
| `nms_thresh` | `0.45` | IoU threshold para NMS |
| `imgsz` | `640` | Lado de inferencia (letterbox cuadrado) |
| `camera_topic` | `/camera/image/compressed` | Fuente de imagen |
| `show_window` | `false` | Ventana OpenCV — poner `false` en robot headless |
| `inference_hz` | `5.0` | Máximo de inferencias por segundo |
| `gate_by_mission` | `false` | Si `true`, solo infiere en `active_states` |
| `mission_state_topic` | `/mission_state` | Estado de misión a escuchar (si gateado) |
| `active_states` | `["SCANNING_LOGOS"]` | Estados en los que SÍ infiere |

---

## Gating por estado de misión (`gate_by_mission`)

YOLO es la parte pesada del stack de percepción. Con `gate_by_mission:=true` el nodo
**solo corre la inferencia mientras `/mission_state ∈ active_states`** (por default
`SCANNING_LOGOS`). Fuera de esa fase descarta el frame pendiente sin inferir.

- Default **`false`** para uso standalone (pruebas) sin depender del `state_machine_node`.
- Requiere que `state_machine_node` esté publicando `/mission_state`.

---

## Modelos

| Archivo | Descripción |
|---|---|
| `models/best.onnx` | YOLO11n opset 20 — para PC/laptop |
| `models/best.pt` | Pesos PyTorch (requiere Ultralytics) |

> Para Jetson Nano usar `best_opset19.onnx` (ver `ros2_deploy/CLAUDE.md`).

---

## Rendimiento en Jetson Nano 2GB

| Estrategia | Efecto |
|---|---|
| `gate_by_mission:=true` | YOLO solo corre en la fase de detección → 0 costo en navegación |
| `inference_hz:=3.0` | Limita a 3 inferencias/s para no saturar RAM |
| `imgsz:=640` | Requerido por el modelo actual |

---

## Build

```bash
colcon build --packages-select puzzlebot_logo_detector puzzlebot_perception
source install/setup.bash
```

Requiere `onnxruntime` (`pip install onnxruntime`).

## Herramienta auxiliar: `record_logos`

```bash
ros2 run puzzlebot_logo_detector record_logos
```
Graba video de la cámara para construir/ampliar el dataset de entrenamiento de logos.
