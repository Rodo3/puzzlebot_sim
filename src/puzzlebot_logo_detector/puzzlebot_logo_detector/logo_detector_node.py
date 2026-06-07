"""
Nodo ROS 2 para detección de logos de trailers con YOLO11n (ONNX Runtime).

Suscribe:
  /camera/image/compressed  (sensor_msgs/CompressedImage)
  /mission_state            (std_msgs/String)  — solo si gate_by_mission=True

Publica:
  /logo_detection/result  (std_msgs/String)  — JSON con clase, confianza y bbox
  /logo_detection/image   (sensor_msgs/Image) — frame anotado para debug

Parámetros:
  weights_path   (str)   ruta a best.onnx
  confidence     (float) umbral de confianza (default 0.70)
  imgsz          (int)   tamaño de inferencia cuadrado (default 640)
  camera_topic   (str)   topic de entrada
  show_window    (bool)  mostrar ventana OpenCV (FALSE en robot headless)
  inference_hz   (float) máximo de inferencias por segundo (default 5.0)
  display_scale  (float) factor de escala de la ventana (default 3.0)
  gate_by_mission (bool) si True, solo infiere cuando /mission_state ∈ active_states
                         (ahorra cómputo: no corre YOLO fuera de SCANNING_LOGOS).
                         Default False (corre siempre, uso standalone).
  mission_state_topic (str)  topic del estado de misión (default /mission_state)
  active_states  (str[]) estados en los que SÍ infiere (default ["SCANNING_LOGOS"])
"""

import json
import os

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

_PKG_SHARE = get_package_share_directory("puzzlebot_logo_detector")
_DEFAULT_WEIGHTS = os.path.join(_PKG_SHARE, "models", "best.onnx")

# Nombres de clase según data.yaml del entrenamiento
_CLASS_NAMES = {0: "Pepsi", 1: "Amazon", 2: "Walmart"}

_WINDOW = "Logo Detector — Puzzlebot"

# Colores BGR por clase
_COLORS = {0: (0, 0, 220), 1: (220, 140, 0), 2: (0, 180, 0)}


def _letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Redimensiona manteniendo aspecto con padding gris. Retorna (img, scale, pad_w, pad_h)."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = (size - new_w) // 2
    pad_h = (size - new_h) // 2
    img = cv2.copyMakeBorder(
        img, pad_h, size - new_h - pad_h, pad_w, size - new_w - pad_w,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return img, scale, pad_w, pad_h


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.45) -> list[int]:
    """NMS simple sobre boxes xyxy."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]
    return keep


def _draw(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    out = frame.copy()
    for d in detections:
        b = d["bbox"]
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        color = _COLORS.get(d["class_id"], (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d['class_name']} {d['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(out, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


class LogoDetectorNode(Node):

    def __init__(self):
        super().__init__("logo_detector")

        self.declare_parameter("weights_path", _DEFAULT_WEIGHTS)
        self.declare_parameter("confidence", 0.70)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("camera_topic", "/camera/image/compressed")
        self.declare_parameter("show_window", True)
        self.declare_parameter("inference_hz", 5.0)
        self.declare_parameter("display_scale", 3.0)
        # Gating por estado de misión: si gate_by_mission=True, solo corre la
        # inferencia cuando /mission_state ∈ active_states (ahorra CPU/GPU; no
        # corre YOLO durante navegación/idle). Default False → corre siempre
        # (uso standalone). El launch de la misión debe activarlo.
        self.declare_parameter("gate_by_mission", False)
        self.declare_parameter("mission_state_topic", "/mission_state")
        self.declare_parameter("active_states", ["SCANNING_LOGOS"])

        weights = self.get_parameter("weights_path").value
        self.conf = self.get_parameter("confidence").value
        self.imgsz = self.get_parameter("imgsz").value
        camera_topic = self.get_parameter("camera_topic").value
        self.show_window = self.get_parameter("show_window").value
        inference_hz = self.get_parameter("inference_hz").value
        self.display_scale = self.get_parameter("display_scale").value
        self.gate_by_mission = self.get_parameter("gate_by_mission").value
        mission_state_topic = self.get_parameter("mission_state_topic").value
        self.active_states = list(self.get_parameter("active_states").value)
        self._mission_state = None

        try:
            import onnxruntime as ort
        except ImportError:
            self.get_logger().fatal("onnxruntime no instalado: pip install onnxruntime")
            raise SystemExit(1)

        weights_abs = os.path.realpath(weights)
        if not os.path.isfile(weights_abs):
            self.get_logger().fatal(f"No se encuentra el modelo: {weights_abs}")
            raise SystemExit(1)

        self.session = ort.InferenceSession(
            weights_abs, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.bridge = CvBridge()
        self.pending = None

        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.sub = self.create_subscription(
            CompressedImage, camera_topic, self._image_callback, camera_qos
        )
        self.pub_result = self.create_publisher(String, "/logo_detection/result", 10)
        self.pub_image = self.create_publisher(Image, "/logo_detection/image", 10)

        if self.gate_by_mission:
            self.create_subscription(String, mission_state_topic, self._mission_state_cb, 10)
            self.get_logger().info(
                f"Gating activo — inferencia solo en estados {self.active_states} "
                f"(escuchando {mission_state_topic})")

        self.create_timer(1.0 / inference_hz, self._detect)

        if self.show_window:
            cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(_WINDOW, int(320 * self.display_scale), int(240 * self.display_scale))

        self.get_logger().info(f"Logo detector listo (ONNX) — pesos: {weights_abs}")
        self.get_logger().info(f"conf={self.conf}  imgsz={self.imgsz}  hz={inference_hz}")
        self.get_logger().info(f"Escuchando: {camera_topic}")

    def _image_callback(self, msg: CompressedImage):
        self.pending = msg

    def _mission_state_cb(self, msg: String):
        self._mission_state = msg.data.strip()

    def _gate_open(self) -> bool:
        """True si se permite correr la inferencia ahora."""
        if not self.gate_by_mission:
            return True
        return self._mission_state in self.active_states

    def _detect(self):
        if not self._gate_open():
            self.pending = None   # descarta el frame pendiente, no acumula
            return
        if self.pending is None:
            return
        msg = self.pending
        self.pending = None

        buf = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return

        orig_h, orig_w = frame.shape[:2]
        inp, scale, pad_w, pad_h = _letterbox(frame, self.imgsz)
        inp_rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = inp_rgb.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

        raw = self.session.run(None, {self.input_name: tensor})[0]  # (1, 7, 8400)
        preds = raw[0].T  # (8400, 7)

        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        mask = confidences >= self.conf
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # Convertir xywh (en espacio letterbox) a xyxy en espacio original
        cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = (cx - bw / 2 - pad_w) / scale
        y1 = (cy - bh / 2 - pad_h) / scale
        x2 = (cx + bw / 2 - pad_w) / scale
        y2 = (cy + bh / 2 - pad_h) / scale
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        keep = _nms(boxes_xyxy, confidences)

        detections = []
        for i in keep:
            detections.append({
                "class_id": int(class_ids[i]),
                "class_name": _CLASS_NAMES.get(int(class_ids[i]), str(class_ids[i])),
                "confidence": round(float(confidences[i]), 4),
                "bbox": {
                    "x1": float(boxes_xyxy[i, 0]),
                    "y1": float(boxes_xyxy[i, 1]),
                    "x2": float(boxes_xyxy[i, 2]),
                    "y2": float(boxes_xyxy[i, 3]),
                },
            })

        result_msg = String()
        result_msg.data = json.dumps(detections)
        self.pub_result.publish(result_msg)

        annotated = _draw(frame, detections)
        img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        img_msg.header.stamp = msg.header.stamp
        img_msg.header.frame_id = msg.header.frame_id
        self.pub_image.publish(img_msg)

        if detections:
            labels = ", ".join(f"{d['class_name']} {d['confidence']:.2f}" for d in detections)
            self.get_logger().info(f"Detectado: {labels}")

        if self.show_window:
            cv2.imshow(_WINDOW, annotated)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = LogoDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
