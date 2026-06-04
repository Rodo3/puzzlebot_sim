# Logo Detector — ROS 2 Deploy

Nodo ROS 2 para detección de logos de trailers con YOLO11n, destinado al robot Puzzlebot.
El modelo fue entrenado en Windows con PyTorch + Ultralytics. Esta carpeta contiene todo lo necesario para correrlo en ROS 2.

---

## Contexto del proyecto

El robot Puzzlebot lee un QR en un pallet, detecta el logo del trailer con su cámara, valida que el logo coincida con el cliente esperado y deposita el pallet.

**Clases detectadas:**
| ID | Clase | Logo real |
|----|-------|-----------|
| 0  | Pepsi | Sticker circular azul/rojo |
| 1  | Amazon | Sticker con letra "e" azul |
| 2  | Walmart | Sticker estrella azul/naranja |

**Cámara del Puzzlebot:** 240p (320×240 px). Por eso el nodo usa `--imgsz 320` por defecto.

---

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `best.pt` | Pesos YOLO11n para usar con Ultralytics (PyTorch) |
| `best.onnx` | Pesos YOLO11n para usar con ONNX Runtime (sin PyTorch) |
| `data.yaml` | Definición de clases y rutas del dataset |
| `logo_detector_node.py` | Nodo ROS 2 de inferencia |

---

## Dependencias

### Opción A — con Ultralytics (usa best.pt)
```bash
pip install ultralytics
# ROS 2 packages: rclpy, cv_bridge, sensor_msgs
```

### Opción B — con ONNX Runtime (usa best.onnx, más ligero, sin PyTorch)
```bash
pip install onnxruntime
# ROS 2 packages: rclpy, cv_bridge, sensor_msgs
# IMPORTANTE: el nodo actual NO soporta ONNX Runtime todavía.
# Si no hay PyTorch en el robot, pedir al agente que adapte el nodo.
```

---

## Correr el nodo

Desde la carpeta `ros2_deploy/`:

```bash
# Opción básica (usa best.pt, conf=0.90, topic por defecto)
python logo_detector_node.py

# Con parámetros explícitos
python logo_detector_node.py \
  --weights best.pt \
  --conf 0.90 \
  --imgsz 320 \
  --camera-topic /camera/image_raw
```

### Topics

| Topic | Tipo | Dirección | Descripción |
|-------|------|-----------|-------------|
| `/camera/image_raw` | `sensor_msgs/Image` | Entrada | Frame de la cámara del Puzzlebot |
| `/logo_detection/result` | `std_msgs/String` | Salida | JSON con detecciones |
| `/logo_detection/image` | `sensor_msgs/Image` | Salida | Frame anotado para debug en RViz |

### Formato del JSON publicado en /logo_detection/result

```json
[
  {
    "class_id": 2,
    "class_name": "Walmart",
    "confidence": 0.9823,
    "bbox": {"x1": 120.0, "y1": 45.0, "x2": 200.0, "y2": 130.0}
  }
]
```

Lista vacía `[]` cuando no se detecta ningún logo.

---

## Métricas del modelo (test set, v4 — 2026-06-03)

| Clase | mAP50 | mAP50-95 | Precision | Recall |
|-------|-------|----------|-----------|--------|
| Pepsi (circle) | 0.995 | 0.971 | — | — |
| Amazon (e) | 0.995 | 0.976 | — | — |
| Walmart (star) | 0.984 | 0.921 | — | — |
| **Global** | **0.991** | **0.956** | **0.993** | **0.987** |

Entrenado con 4450 imágenes sintéticas + 128 fondos reales COCO128.
GPU: RTX 4060 Laptop, torch-2.11.0+cu128, YOLO11n (2.6M parámetros).

---

## Si necesitas adaptar el nodo (instrucciones para el agente)

- **Cambiar a ONNX Runtime:** El nodo actual usa Ultralytics (`best.pt`). Si el robot no tiene PyTorch, reescribir la inferencia usando `onnxruntime` con `best.onnx`. El preprocesado es: resize a `imgsz×imgsz`, normalizar a [0,1], transponer a BCHW, pasar por `session.run()`.
- **Cambiar topic de cámara:** Modificar `--camera-topic` o el default `CAMERA_TOPIC_DEFAULT` en el script.
- **Cambiar umbral de confianza:** `--conf` (default 0.90). Bajar si el modelo no detecta; subir si hay falsos positivos.
- **Publicar tipo de mensaje distinto:** El resultado sale como JSON en `std_msgs/String`. Si se necesita un mensaje custom, crear el tipo en el paquete ROS 2 y adaptar `_image_callback`.

---

## Modelo original y reentrenamiento

El pipeline completo de entrenamiento está en:
```
c:\Users\rpzda\Documents\Data_Final\logo_detection_training\
```

Para reentrenar ver `CLAUDE.md` en esa carpeta. Los pesos originales están en:
```
runs\detect\runs\logo_yolo11n\weights\best.pt
runs\detect\runs\logo_yolo11n\weights\best.onnx
```
