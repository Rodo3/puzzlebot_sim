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

## Métricas del modelo (v9 — 2026-06-04)

Evaluado en **dominio REAL** (frames del Puzzlebot held-out + % de frames con detección
en el video completo @ conf 0.70). Es el benchmark que refleja el desempeño en el robot.

| Métrica | v4 (solo sintético) | **v9 (real+hue)** |
|---|---|---|
| mAP50 test real | 0.982 | **0.991** |
| % frames con detección (video) | 33% | **60%** |
| Falsos positivos | — | ninguno (verificado) |

**mAP50 por clase (test real):** Pepsi=0.995 · Amazon=0.983 · Walmart=0.995

Entrenado con sintético (4450 + COCO128) + **frames reales** de videos (Puzzlebot + celular)
sobre-muestreados, con **augmentación de hue fuerte (hsv_h=0.40)** para el tinte magenta de
la cámara 240p. YOLO11n (2.6M params). GPU: RTX 4060 Laptop, torch-2.11.0+cu128.

**Conf de despliegue recomendado: 0.60** (verificado sin falsos positivos; capta logos
lejanos con margen). El default del nodo ya es 0.60.

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
