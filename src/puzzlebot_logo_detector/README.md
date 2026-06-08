# puzzlebot_logo_detector

Detección de logos de clientes en los tráileres del almacén (**Pepsi / Amazon /
Walmart**) con **YOLO11n** exportado a ONNX, corriendo sobre **ONNX Runtime**.

Es un nodo de percepción puro: publica qué logo ve y dónde (bbox), sin tomar
decisiones. El `state_machine_node` (`puzzlebot_control`) consume `/logo_detection/result`
durante la fase `SCANNING_LOGOS` para identificar el tráiler correcto comparándolo
con el string leído del QR.

---

## Nodo: `logo_detector_node`

```bash
# Standalone (siempre infiere)
ros2 run puzzlebot_logo_detector logo_detector_node

# En el robot (headless + gateado por la misión — ver abajo)
ros2 run puzzlebot_logo_detector logo_detector_node --ros-args \
  -p show_window:=false -p gate_by_mission:=true
```

### Suscribe
| Tópico | Tipo | Notas |
|---|---|---|
| `/camera/image/compressed` | sensor_msgs/CompressedImage | Entrada de cámara (SensorDataQoS) |
| `/mission_state` | std_msgs/String | **Solo** si `gate_by_mission:=true` |

### Publica
| Tópico | Tipo | Contenido |
|---|---|---|
| `/logo_detection/result` | std_msgs/String | JSON array de detecciones (ver abajo) |
| `/logo_detection/image` | sensor_msgs/Image | Frame anotado para debug (rqt) |

### Formato de salida (`/logo_detection/result`)

```json
[{"class_id": 2,
  "class_name": "Walmart",
  "confidence": 0.91,
  "bbox": {"x1": 120.0, "y1": 80.0, "x2": 210.0, "y2": 170.0}}]
```

Las coordenadas `bbox` están en **píxeles del frame original** (ya des-letterboxed),
así que el dashboard puede dibujar el overlay directamente sobre el stream crudo.
Clases del modelo: `0=Pepsi`, `1=Amazon`, `2=Walmart`.

### Parámetros

| Parámetro | Default | Descripción |
|---|---|---|
| `weights_path` | `models/best.onnx` | Ruta al modelo ONNX |
| `confidence` | `0.70` | Umbral mínimo de confianza |
| `imgsz` | `640` | Lado de inferencia (letterbox cuadrado) |
| `camera_topic` | `/camera/image/compressed` | Fuente de imagen |
| `show_window` | `true` | Ventana OpenCV — **poner `false` en robot headless** |
| `inference_hz` | `5.0` | Máximo de inferencias por segundo |
| `gate_by_mission` | `false` | Si `true`, solo infiere en `active_states` |
| `mission_state_topic` | `/mission_state` | Estado de misión a escuchar (si gateado) |
| `active_states` | `["SCANNING_LOGOS"]` | Estados en los que SÍ infiere |

---

## Gating por estado de misión (`gate_by_mission`)

YOLO es la parte pesada del stack de percepción. Con `gate_by_mission:=true` el nodo
**solo corre la inferencia mientras `/mission_state ∈ active_states`** (por default
`SCANNING_LOGOS`). Fuera de esa fase descarta el frame pendiente sin inferir → cero
costo de YOLO durante navegación/idle.

- Default **`false`** para que el nodo funcione standalone (pruebas) sin depender del
  `state_machine_node`.
- Requiere que `state_machine_node` esté publicando `/mission_state`.
- El `qr_node` tiene el mismo mecanismo (gateado a `SCANNING_QR`).

---

## Rendimiento en Jetson Orin

Orden de impacto para aligerar, de mayor a menor:

1. **Execution provider:** hoy usa `CPUExecutionProvider`. Cambiar a CUDA/TensorRT EP
   en la Jetson es el mayor salto de rendimiento.
2. **Gating por misión:** YOLO solo corre en el dock (fase breve) → ya no es costo continuo.
   Con esto suele bastar para mantener `imgsz=640`.
3. **`imgsz`:** bajar a 320 es ~4× más rápido, a costa de detalle (peor para logos
   chicos/lejanos). Para los logos grandes del dock 320 suele alcanzar.
4. **`inference_hz`:** 2-3 Hz es suficiente para una escena estática.

---

## Build

```bash
colcon build --packages-select puzzlebot_logo_detector
source install/setup.bash
```

Requiere `onnxruntime` (`pip install onnxruntime`, o `onnxruntime-gpu` en Jetson).

## Herramienta auxiliar: `record_logos`

```bash
ros2 run puzzlebot_logo_detector record_logos
```
Graba video de la cámara para construir/ampliar el dataset de entrenamiento de logos.
