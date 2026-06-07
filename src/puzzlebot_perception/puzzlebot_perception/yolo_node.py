"""
yolo_node.py — Detección de logos de clientes con YOLOv8/v11 (Ultralytics).

FUNCIÓN:
  Corre inferencia YOLO sobre la imagen de la cámara y publica las detecciones
  en formato vision_msgs/Detection2DArray. El Mission Manager consume este
  tópico en SEARCH_TRAILER_LOGO / MATCH_TRAILER_CLIENT para identificar
  el tráiler correcto comparando el class_id (nombre del logo) con el cliente
  leído del QR.

MODELO:
  - En desarrollo/PC: modelo .pt de PyTorch (CPU)
  - En Jetson Orin:   modelo .engine de TensorRT (GPU, use_trt:=true)
  El modelo debe estar entrenado con los logos de los clientes como clases.
  Puedes usar un modelo YOLOv8n pre-entrenado en COCO como punto de partida
  y hacer fine-tuning con tus logos (transfer learning).

TÓPICOS SUSCRITOS:
  /camera/image/compressed  (sensor_msgs/CompressedImage)

TÓPICOS PUBLICADOS:
  /detections      (vision_msgs/Detection2DArray) — detecciones con clase y confianza
  /yolo/debug_image (sensor_msgs/Image)           — imagen con bounding boxes (si enabled)

PARÁMETROS:
  image_topic         [/camera/image/compressed]
  model_path          ['']           ruta al .pt o .engine; vacío → usa yolov8n.pt
  use_trt             [false]        True en Jetson con TensorRT
  confidence_thresh   [0.45]         umbral mínimo de confianza para publicar
  nms_thresh          [0.50]         NMS IoU threshold
  detection_rate_hz   [10.0]         límite de inferencia
  publish_debug       [false]        publicar imagen con bboxes
  camera_width        [640]          resolución de entrada al modelo
  camera_height       [480]
"""

import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

# Ultralytics se importa lazy para no bloquear el nodo si no está instalado
_ultralytics_ok = False
try:
    from ultralytics import YOLO as _YOLO
    _ultralytics_ok = True
except ImportError:
    pass


class YoloNode(Node):

    def __init__(self):
        super().__init__('yolo_node')

        self.declare_parameter('image_topic',       '/camera/image/compressed')
        self.declare_parameter('model_path',        '')
        self.declare_parameter('use_trt',           False)
        self.declare_parameter('confidence_thresh', 0.45)
        self.declare_parameter('nms_thresh',        0.50)
        self.declare_parameter('detection_rate_hz', 10.0)
        self.declare_parameter('publish_debug',     False)
        self.declare_parameter('camera_width',      640)
        self.declare_parameter('camera_height',     480)

        image_topic   = self.get_parameter('image_topic').value
        model_path    = self.get_parameter('model_path').value
        self._use_trt = self.get_parameter('use_trt').value
        self._conf    = self.get_parameter('confidence_thresh').value
        self._iou     = self.get_parameter('nms_thresh').value
        max_hz        = self.get_parameter('detection_rate_hz').value
        self._debug   = self.get_parameter('publish_debug').value
        self._w       = self.get_parameter('camera_width').value
        self._h       = self.get_parameter('camera_height').value

        self._min_period = 1.0 / max(max_hz, 0.1)
        self._last_proc  = 0.0
        self._bridge     = CvBridge()
        self._model      = None

        if not _ultralytics_ok:
            self.get_logger().error(
                'ultralytics no está instalado. '
                'Instala con: pip install ultralytics\n'
                'El nodo seguirá corriendo pero no publicará detecciones.')
        else:
            self._load_model(model_path)

        # Suscripción
        self.create_subscription(
            CompressedImage, image_topic,
            self._image_cb, qos_profile_sensor_data)

        # Publicadores
        self._pub_det   = self.create_publisher(Detection2DArray, '/detections',       10)
        if self._debug:
            self._pub_dbg = self.create_publisher(Image, '/yolo/debug_image', 10)

        self.get_logger().info(
            f'yolo_node iniciado — modelo={model_path or "yolov8n.pt (default)"} '
            f'conf={self._conf} trt={self._use_trt}')

    # ── Carga del modelo ──────────────────────────────────────────────────────

    def _load_model(self, path: str):
        try:
            if path and path.endswith('.engine'):
                self._model = _YOLO(path, task='detect')
                self.get_logger().info(f'Modelo TensorRT cargado: {path}')
            elif path:
                self._model = _YOLO(path)
                self.get_logger().info(f'Modelo PyTorch cargado: {path}')
            else:
                # Fallback: yolov8n.pt (se descarga automáticamente la primera vez)
                self._model = _YOLO('yolov8n.pt')
                self.get_logger().warn(
                    'model_path vacío — usando yolov8n.pt (COCO). '
                    'Para producción usa un modelo entrenado con logos de clientes.')
        except Exception as e:
            self.get_logger().error(f'Error cargando modelo YOLO: {e}')
            self._model = None

    # ── Callback de imagen ────────────────────────────────────────────────────

    def _image_cb(self, msg: CompressedImage):
        if self._model is None:
            return

        now = time.monotonic()
        if now - self._last_proc < self._min_period:
            return
        self._last_proc = now

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            import cv2
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return
            frame = cv2.resize(frame, (self._w, self._h))
        except Exception as e:
            self.get_logger().warn(f'Error decodificando imagen: {e}', throttle_duration_sec=2.0)
            return

        # Inferencia
        try:
            results = self._model.predict(
                frame,
                conf=self._conf,
                iou=self._iou,
                verbose=False,
                device='0' if self._use_trt else 'cpu',
            )
        except Exception as e:
            self.get_logger().warn(f'Error en inferencia YOLO: {e}', throttle_duration_sec=2.0)
            return

        det_array = Detection2DArray()
        det_array.header.stamp    = self.get_clock().now().to_msg()
        det_array.header.frame_id = 'camera_link'

        if results and len(results) > 0:
            result = results[0]
            names  = result.names   # dict {int: str}

            for box in result.boxes:
                xyxy  = box.xyxy[0].cpu().numpy()
                conf  = float(box.conf[0].cpu().numpy())
                cls   = int(box.cls[0].cpu().numpy())
                label = names.get(cls, str(cls))

                det = Detection2D()
                det.header = det_array.header

                # BBox en píxeles → normalizado [0,1]
                x1, y1, x2, y2 = xyxy
                det.bbox.center.position.x = float((x1 + x2) / 2.0 / self._w)
                det.bbox.center.position.y = float((y1 + y2) / 2.0 / self._h)
                det.bbox.size_x            = float((x2 - x1) / self._w)
                det.bbox.size_y            = float((y2 - y1) / self._h)

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = label
                hyp.hypothesis.score    = conf
                det.results.append(hyp)

                det_array.detections.append(det)

        self._pub_det.publish(det_array)

        if det_array.detections:
            labels = [(d.results[0].hypothesis.class_id,
                       d.results[0].hypothesis.score)
                      for d in det_array.detections]
            self.get_logger().info(
                f'Detecciones: {labels}',
                throttle_duration_sec=1.0)

        if self._debug and det_array.detections:
            import cv2
            debug = frame.copy()
            for d in det_array.detections:
                cx = int(d.bbox.center.position.x * self._w)
                cy = int(d.bbox.center.position.y * self._h)
                sw = int(d.bbox.size_x * self._w)
                sh = int(d.bbox.size_y * self._h)
                x1 = cx - sw // 2
                y1 = cy - sh // 2
                label = d.results[0].hypothesis.class_id
                score = d.results[0].hypothesis.score
                cv2.rectangle(debug, (x1, y1), (x1+sw, y1+sh), (0, 200, 255), 2)
                cv2.putText(debug, f'{label} {score:.2f}',
                            (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
            self._pub_dbg.publish(
                self._bridge.cv2_to_imgmsg(debug, encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
