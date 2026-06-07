"""
yolo_node.py — Detección de logos con YOLO11n (ONNX Runtime).

TÓPICOS SUSCRITOS:
  /camera/image/compressed  (sensor_msgs/CompressedImage)
  /mission_state            (std_msgs/String)  — solo si gate_by_mission=True

TÓPICOS PUBLICADOS:
  /detections        (vision_msgs/Detection2DArray) — detecciones con clase y confianza
  /yolo/debug_image  (sensor_msgs/Image)            — frame anotado con bboxes

PARÁMETROS:
  weights_path        (str)    ruta a best.onnx; vacío → busca en puzzlebot_logo_detector/models
  confidence          (float)  umbral de confianza (default 0.70)
  nms_thresh          (float)  IoU threshold para NMS (default 0.45)
  imgsz               (int)    tamaño de entrada cuadrado al modelo (default 640)
  camera_topic        (str)    tópico de imagen comprimida
  inference_hz        (float)  máximo de inferencias por segundo (default 5.0)
  show_window         (bool)   mostrar ventana OpenCV — False en robot headless
  gate_by_mission     (bool)   si True, solo infiere cuando /mission_state ∈ active_states
  mission_state_topic (str)    tópico de estado de misión (default /mission_state)
  active_states       (str[])  estados habilitados (default ["SCANNING_LOGOS"])
"""

import os

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

_CLASS_NAMES = {0: "Popsi", 1: "Emezon", 2: "Wolmar"}
_COLORS      = {0: (0, 0, 220), 1: (220, 140, 0), 2: (0, 180, 0)}
_WINDOW      = "YOLO — Puzzlebot"


def _default_weights() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(
            get_package_share_directory("puzzlebot_logo_detector"), "models", "best.onnx"
        )
    except Exception:
        return ""


def _letterbox(img, size):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pw, ph = (size - nw) // 2, (size - nh) // 2
    img = cv2.copyMakeBorder(
        img, ph, size - nh - ph, pw, size - nw - pw,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return img, scale, pw, ph


def _nms(boxes, scores, iou_thresh):
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


def _draw(frame, detections):
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


class YoloNode(Node):

    def __init__(self):
        super().__init__('yolo_node')

        self.declare_parameter('weights_path',        _default_weights())
        self.declare_parameter('confidence',           0.70)
        self.declare_parameter('nms_thresh',           0.45)
        self.declare_parameter('imgsz',                640)
        self.declare_parameter('camera_topic',         '/camera/image/compressed')
        self.declare_parameter('inference_hz',         5.0)
        self.declare_parameter('show_window',          False)
        self.declare_parameter('gate_by_mission',      False)
        self.declare_parameter('mission_state_topic',  '/mission_state')
        self.declare_parameter('active_states',        ['SCANNING_LOGOS'])

        weights         = self.get_parameter('weights_path').value
        self.conf       = self.get_parameter('confidence').value
        self.iou        = self.get_parameter('nms_thresh').value
        self.imgsz      = self.get_parameter('imgsz').value
        camera_topic    = self.get_parameter('camera_topic').value
        inference_hz    = self.get_parameter('inference_hz').value
        self.show_win   = self.get_parameter('show_window').value
        self.gate       = self.get_parameter('gate_by_mission').value
        mission_topic   = self.get_parameter('mission_state_topic').value
        self.active_states = list(self.get_parameter('active_states').value)
        self._mission_state = None
        self.pending = None

        try:
            import onnxruntime as ort
        except ImportError:
            self.get_logger().fatal('onnxruntime no instalado: pip install onnxruntime')
            raise SystemExit(1)

        weights_abs = os.path.realpath(weights)
        if not os.path.isfile(weights_abs):
            self.get_logger().fatal(f'Modelo no encontrado: {weights_abs}')
            raise SystemExit(1)

        self.session    = ort.InferenceSession(weights_abs, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.bridge     = CvBridge()

        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(CompressedImage, camera_topic, self._img_cb, cam_qos)
        self._pub_det = self.create_publisher(Detection2DArray, '/detections',       10)
        self._pub_dbg = self.create_publisher(Image,            '/yolo/debug_image', 10)

        if self.gate:
            self.create_subscription(String, mission_topic, self._mission_cb, 10)
            self.get_logger().info(
                f'Gating activo — estados habilitados: {self.active_states}')

        self.create_timer(1.0 / max(inference_hz, 0.1), self._detect)

        if self.show_win:
            cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)

        self.get_logger().info(
            f'yolo_node listo (ONNX) — modelo={weights_abs} '
            f'conf={self.conf} iou={self.iou} hz={inference_hz}')

    def _img_cb(self, msg: CompressedImage):
        self.pending = msg

    def _mission_cb(self, msg: String):
        self._mission_state = msg.data.strip()

    def _gate_open(self):
        if not self.gate:
            return True
        return self._mission_state in self.active_states

    def _detect(self):
        if not self._gate_open():
            self.pending = None
            return
        if self.pending is None:
            return
        msg = self.pending
        self.pending = None

        buf   = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return

        orig_h, orig_w = frame.shape[:2]
        inp, scale, pw, ph = _letterbox(frame, self.imgsz)
        tensor = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)[np.newaxis]

        raw   = self.session.run(None, {self.input_name: tensor})[0]
        preds = raw[0].T  # (8400, 4+classes)

        boxes_xywh  = preds[:, :4]
        cls_scores  = preds[:, 4:]
        cls_ids     = np.argmax(cls_scores, axis=1)
        confs       = cls_scores[np.arange(len(cls_scores)), cls_ids]

        mask        = confs >= self.conf
        boxes_xywh  = boxes_xywh[mask]
        confs       = confs[mask]
        cls_ids     = cls_ids[mask]

        cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = np.clip((cx - bw / 2 - pw) / scale, 0, orig_w)
        y1 = np.clip((cy - bh / 2 - ph) / scale, 0, orig_h)
        x2 = np.clip((cx + bw / 2 - pw) / scale, 0, orig_w)
        y2 = np.clip((cy + bh / 2 - ph) / scale, 0, orig_h)
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        keep = _nms(boxes_xyxy, confs, self.iou)

        det_array             = Detection2DArray()
        det_array.header.stamp    = self.get_clock().now().to_msg()
        det_array.header.frame_id = 'camera_link'
        raw_dets = []

        for i in keep:
            bx1, by1, bx2, by2 = boxes_xyxy[i]
            cls_id   = int(cls_ids[i])
            cls_name = _CLASS_NAMES.get(cls_id, str(cls_id))
            conf     = float(confs[i])

            det = Detection2D()
            det.header = det_array.header
            det.bbox.center.position.x = float((bx1 + bx2) / 2.0 / orig_w)
            det.bbox.center.position.y = float((by1 + by2) / 2.0 / orig_h)
            det.bbox.size_x            = float((bx2 - bx1) / orig_w)
            det.bbox.size_y            = float((by2 - by1) / orig_h)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = cls_name
            hyp.hypothesis.score    = conf
            det.results.append(hyp)
            det_array.detections.append(det)
            raw_dets.append({
                'class_id': cls_id, 'class_name': cls_name, 'confidence': conf,
                'bbox': {'x1': float(bx1), 'y1': float(by1), 'x2': float(bx2), 'y2': float(by2)},
            })

        self._pub_det.publish(det_array)

        annotated = _draw(frame, raw_dets)
        dbg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        dbg.header.stamp    = msg.header.stamp
        dbg.header.frame_id = msg.header.frame_id
        self._pub_dbg.publish(dbg)

        if raw_dets:
            msg_log = ' | '.join(f"{d['class_name']} {d['confidence']:.2f}" for d in raw_dets)
            self.get_logger().info(f'[YOLO] Detectado: {msg_log}')
        else:
            self.get_logger().info('[YOLO] Sin detecciones', throttle_duration_sec=2.0)

        if self.show_win:
            cv2.imshow(_WINDOW, annotated)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
