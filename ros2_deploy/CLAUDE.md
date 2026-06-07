# Logo Detector — ROS 2 Deploy (Jetson Nano)

Script standalone para detección de logos en la Jetson Nano. No requiere colcon build.
Usa **ONNX Runtime** con `best_opset19.onnx` (opset 19, compatible con onnxruntime 1.16.3).

---

## Clases detectadas

| ID | Clase   | Logo real                    |
|----|---------|------------------------------|
| 0  | Pepsi   | Sticker circular azul/rojo   |
| 1  | Amazon  | Sticker con letra "e" azul   |
| 2  | Walmart | Sticker estrella azul/naranja|

---

## Archivos

| Archivo              | Descripción                                          |
|----------------------|------------------------------------------------------|
| `best.onnx`          | Modelo YOLO11n opset 20 (no compatible con Nano)     |
| `best_opset19.onnx`  | Modelo YOLO11n opset 19 — **usar este en la Jetson** |
| `best.pt`            | Pesos PyTorch (requiere Ultralytics, no usar en Nano)|
| `data.yaml`          | Definición de clases del dataset                     |
| `logo_detector_node.py` | Nodo ROS 2 standalone con ONNX Runtime            |

---

## Setup en Jetson Nano (una sola vez)

### 1. Instalar onnxruntime (sin internet en la Jetson)

En tu máquina con internet:
```bash
pip3 download onnxruntime --platform manylinux2014_aarch64 --python-version 38 --only-binary=:all: -d ~/wheels/
scp ~/wheels/*.whl puzzlebot@10.42.0.1:~/
```

En la Jetson:
```bash
pip3 install --no-index --find-links=~/ onnxruntime
```

### 2. Copiar archivos a la Jetson

Desde tu máquina (dentro de `puzzlebot_sim/`):
```bash
scp ros2_deploy/best_opset19.onnx ros2_deploy/data.yaml ros2_deploy/logo_detector_node.py puzzlebot@10.42.0.1:~/yolo_test/
```

### 3. Convertir modelo a opset 19 (si se re-exporta el modelo)

```bash
pip3 install onnx
python3 -c "
import onnx
from onnx import version_converter
model = onnx.load('ros2_deploy/best.onnx')
converted = version_converter.convert_version(model, 19)
onnx.save(converted, 'ros2_deploy/best_opset19.onnx')
"
```

---

## Correr el nodo en la Jetson

```bash
cd ~/yolo_test
python3 logo_detector_node.py --weights best_opset19.onnx --conf 0.60 --imgsz 640
```

### Tópicos

| Tópico                    | Tipo                    | Dirección | Descripción                        |
|---------------------------|-------------------------|-----------|------------------------------------|
| `/camera/image/compressed`| sensor_msgs/CompressedImage | Entrada | Cámara del Puzzlebot (QoS BEST_EFFORT) |
| `/logo_detection/result`  | std_msgs/String         | Salida    | JSON array con detecciones         |
| `/logo_detection/image`   | sensor_msgs/Image       | Salida    | Frame anotado para debug (rqt)     |

### Formato JSON de `/logo_detection/result`

```json
[
  {
    "class_id": 1,
    "class_name": "Amazon",
    "confidence": 0.8731,
    "bbox": {"x1": 120.0, "y1": 45.0, "x2": 310.0, "y2": 220.0}
  }
]
```

Lista vacía `[]` cuando no hay detecciones.

### Parámetros CLI

| Parámetro        | Default                      | Descripción                        |
|------------------|------------------------------|------------------------------------|
| `--weights`      | `best.onnx`                  | Ruta al modelo — usar `best_opset19.onnx` en Jetson |
| `--conf`         | `0.60`                       | Umbral de confianza                |
| `--imgsz`        | `320`                        | Tamaño de inferencia — usar `640` (requerido por el modelo actual) |
| `--camera-topic` | `/camera/image/compressed`   | Tópico de entrada de la cámara     |

---

## Dependencias en Jetson Nano

| Paquete       | Cómo instalar                          | Estado en Nano 2GB |
|---------------|----------------------------------------|--------------------|
| `numpy`       | Ya incluido en JetPack                 | ✅ 1.17.4          |
| `cv2`         | Ya incluido en JetPack                 | ✅ 4.2.0           |
| `onnxruntime` | Via wheel aarch64 (ver Setup arriba)   | ✅ 1.16.3          |
| `rclpy`       | Ya incluido en ROS 2 Humble            | ✅                 |

---

## Métricas del modelo (v9 — 2026-06-04)

| Métrica                        | v9 (real+hue) |
|--------------------------------|---------------|
| mAP50 test real                | **0.991**     |
| % frames con detección (video) | **60%**       |
| Falsos positivos               | ninguno       |

**mAP50 por clase:** Pepsi=0.995 · Amazon=0.983 · Walmart=0.995

Conf recomendada de despliegue: **0.60**
