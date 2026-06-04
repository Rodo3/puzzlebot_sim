"""
logo_detector_node.py
Nodo ROS 2 para detección de logos con YOLO11n.

Suscribe:
  /camera/image_raw  (sensor_msgs/Image)

Publica:
  /logo_detection/result  (std_msgs/String)  — JSON con clase, confianza y bbox
  /logo_detection/image   (sensor_msgs/Image) — frame anotado para debug

Uso:
  python logo_detector_node.py
  python logo_detector_node.py --weights best.pt --conf 0.90 --camera-topic /camera/image_raw
"""

import argparse
import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge


WEIGHTS_DEFAULT = "best.pt"
CAMERA_TOPIC_DEFAULT = "/camera/image_raw"


class LogoDetectorNode(Node):

    def __init__(self, weights: str, conf: float, imgsz: int, camera_topic: str):
        super().__init__("logo_detector")

        try:
            from ultralytics import YOLO
        except ImportError:
            self.get_logger().fatal("Ultralytics no instalado: pip install ultralytics")
            raise SystemExit(1)

        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image, camera_topic, self._image_callback, 10
        )
        self.pub_result = self.create_publisher(String, "/logo_detection/result", 10)
        self.pub_image = self.create_publisher(Image, "/logo_detection/image", 10)

        self.get_logger().info(f"Logo detector listo — pesos: {weights}, conf: {conf}")
        self.get_logger().info(f"Escuchando: {camera_topic}")

    def _image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        results = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)

        detections = []
        if results and results[0].boxes:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                names = results[0].names
                detections.append({
                    "class_id": cls_id,
                    "class_name": names.get(cls_id, str(cls_id)),
                    "confidence": round(conf_val, 4),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                })

        result_msg = String()
        result_msg.data = json.dumps(detections)
        self.pub_result.publish(result_msg)

        # Publicar imagen anotada para debug
        annotated = results[0].plot() if results else frame
        img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        img_msg.header = msg.header
        self.pub_image.publish(img_msg)


def parse_args():
    parser = argparse.ArgumentParser(description="Nodo ROS 2 de detección de logos YOLO11.")
    parser.add_argument("--weights",      default=WEIGHTS_DEFAULT)
    parser.add_argument("--conf",         type=float, default=0.90)
    parser.add_argument("--imgsz",        type=int,   default=320,
                        help="Tamaño de inferencia (320 para cámara 240p del Puzzlebot)")
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
