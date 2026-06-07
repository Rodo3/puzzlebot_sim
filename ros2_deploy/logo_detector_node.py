"""
logo_detector_node.py
Nodo ROS 2 standalone para detección de logos con YOLO11n (ONNX Runtime).
Publica en los mismos tópicos que yolo_node del workspace ROS.

Suscribe:
  /camera/image/compressed  (sensor_msgs/CompressedImage, BEST_EFFORT QoS)

Publica:
  /detections        (vision_msgs/Detection2DArray) — bbox normalizado [0,1]
  /yolo/debug_image  (sensor_msgs/Image)            — frame anotado para debug

Uso:
  python3 logo_detector_node.py --weights best_opset19.onnx --conf 0.30 --imgsz 640
"""

import argparse

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

WEIGHTS_DEFAULT      = "best_opset19.onnx"
CAMERA_TOPIC_DEFAULT = "/camera/image/compressed"
CLASS_NAMES = {0: "Popsi", 1: "Emezon", 2: "Wolmar"}
COLORS      = {0: (0, 0, 220), 1: (220, 140, 0), 2: (0, 180, 0)}


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


def _nms(boxes, scores, iou_thresh=0.45):
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
        color = COLORS.get(d["class_id"], (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d['class_name']} {d['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(out, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


class LogoDetectorNode(Node):

    def __init__(self, weights: str, conf: float, imgsz: int, camera_topic: str):
        super().__init__("logo_detector")

        try:
            import onnxruntime as ort
        except ImportError:
            self.get_logger().fatal("onnxruntime no instalado: pip3 install onnxruntime")
            raise SystemExit(1)

        self.session    = ort.InferenceSession(weights, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.conf       = conf
        self.imgsz      = imgsz
        self.bridge     = CvBridge()
        self.pending    = None

        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.sub      = self.create_subscription(CompressedImage, camera_topic, self._image_cb, cam_qos)
        self.pub_det  = self.create_publisher(Detection2DArray, "/detections",       10)
        self.pub_dbg  = self.create_publisher(Image,            "/yolo/debug_image", 10)

        self.get_logger().info(f"Logo detector listo (ONNX) — pesos: {weights}, conf: {conf}")
        self.get_logger().info(f"Escuchando: {camera_topic}")

    def _image_cb(self, msg: CompressedImage):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        orig_h, orig_w = frame.shape[:2]

        inp, scale, pw, ph = _letterbox(frame, self.imgsz)
        tensor = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)[np.newaxis]

        raw   = self.session.run(None, {self.input_name: tensor})[0]
        preds = raw[0].T  # (8400, 4+classes)

        boxes_xywh = preds[:, :4]
        cls_scores = preds[:, 4:]
        cls_ids    = np.argmax(cls_scores, axis=1)
        confs      = cls_scores[np.arange(len(cls_scores)), cls_ids]

        mask       = confs >= self.conf
        boxes_xywh = boxes_xywh[mask]
        confs      = confs[mask]
        cls_ids    = cls_ids[mask]

        cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = np.clip((cx - bw / 2 - pw) / scale, 0, orig_w)
        y1 = np.clip((cy - bh / 2 - ph) / scale, 0, orig_h)
        x2 = np.clip((cx + bw / 2 - pw) / scale, 0, orig_w)
        y2 = np.clip((cy + bh / 2 - ph) / scale, 0, orig_h)
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        keep = _nms(boxes_xyxy, confs)

        det_array             = Detection2DArray()
        det_array.header.stamp    = self.get_clock().now().to_msg()
        det_array.header.frame_id = "camera_link"
        raw_dets = []

        for i in keep:
            bx1, by1, bx2, by2 = boxes_xyxy[i]
            cls_id   = int(cls_ids[i])
            cls_name = CLASS_NAMES.get(cls_id, str(cls_id))
            conf     = round(float(confs[i]), 4)

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
                "class_id": cls_id, "class_name": cls_name, "confidence": conf,
                "bbox": {"x1": float(bx1), "y1": float(by1), "x2": float(bx2), "y2": float(by2)},
            })

        self.pub_det.publish(det_array)

        annotated = _draw(frame, raw_dets)
        dbg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        dbg.header = msg.header
        self.pub_dbg.publish(dbg)

        if raw_dets:
            labels = ", ".join(f"{d['class_name']} {d['confidence']:.2f}" for d in raw_dets)
            self.get_logger().info(f"[YOLO] Detectado: {labels}")
        else:
            self.get_logger().info("[YOLO] Sin detecciones", throttle_duration_sec=2.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",      default=WEIGHTS_DEFAULT)
    parser.add_argument("--conf",         type=float, default=0.60)
    parser.add_argument("--imgsz",        type=int,   default=640)
    parser.add_argument("--camera-topic", default=CAMERA_TOPIC_DEFAULT)
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = LogoDetectorNode(
        weights=args.weights,
        conf=args.conf,
        imgsz=args.imgsz,
        camera_topic=args.camera_topic,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
