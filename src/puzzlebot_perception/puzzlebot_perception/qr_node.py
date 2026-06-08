"""QR code detection node — pure sensor.

Detects QR codes in the camera stream using ``cv2.QRCodeDetector`` (built into
OpenCV, no extra system dependencies) and publishes the decoded strings. This
node makes **no** decisions — it just reports what it sees. Downstream nodes
(e.g. ``state_machine_node`` in ``puzzlebot_control``) decide what to do with
the decoded content.

Published topics
----------------
/qr/detections   std_msgs/String     JSON array: [{"data": "wolmar",
                                       "corners": [[x0,y0],...,[x3,y3]]}].
                                       Empty array "[]" when no QR is visible.
/qr/debug_image  sensor_msgs/Image   Annotated frame (bounding boxes), only
                                       when publish_debug_image is true.

Expected QR strings: ``wolmar``, ``popsi``, ``emezon`` (internal client names
that the state machine maps to the trailer logos Walmart / Pepsi / Amazon).
"""

import json
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String


class QrNode(Node):
    def __init__(self):
        super().__init__('qr_node')

        self.declare_parameter('image_topic', '/camera/image/compressed')
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('max_processing_hz', 10.0)
        self.declare_parameter('min_qr_area_px', 400.0)
        # Si no se decodifica nada al tamaño nativo, reintentar una vez sobre el
        # frame escalado 2x. Ayuda al QR chico (4.5 cm) a distancia, sin costo
        # cuando ya hay detección al tamaño nativo.
        self.declare_parameter('upscale_retry', True)
        # Gating por estado de misión: si gate_by_mission=True, solo procesa
        # frames cuando /mission_state ∈ active_states (no busca QR fuera de
        # SCANNING_QR). Default False → procesa siempre (uso standalone).
        self.declare_parameter('gate_by_mission', False)
        self.declare_parameter('mission_state_topic', '/mission_state')
        self.declare_parameter('active_states', ['SCANNING_QR'])

        self._image_topic   = self.get_parameter('image_topic').get_parameter_value().string_value
        self._debug         = self.get_parameter('publish_debug_image').value
        max_hz              = float(self.get_parameter('max_processing_hz').value)
        self._min_area      = float(self.get_parameter('min_qr_area_px').value)
        self._upscale_retry = bool(self.get_parameter('upscale_retry').value)
        self._gate_by_mission = bool(self.get_parameter('gate_by_mission').value)
        self._active_states   = list(self.get_parameter('active_states').value)
        self._mission_state   = None
        self._min_proc_interval = (1.0 / max_hz) if max_hz > 0.0 else 0.0
        self._last_proc_time    = 0.0

        self._bridge   = CvBridge()
        self._detector = cv2.QRCodeDetector()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Auto-detect CompressedImage vs raw Image by topic name suffix.
        if self._image_topic.endswith('compressed'):
            self.create_subscription(
                CompressedImage, self._image_topic, self._compressed_cb, sensor_qos)
        else:
            self.create_subscription(
                Image, self._image_topic, self._image_cb, sensor_qos)

        self._pub_det = self.create_publisher(String, '/qr/detections', 10)
        self._pub_dbg = (
            self.create_publisher(Image, '/qr/debug_image', 10) if self._debug else None
        )

        if self._gate_by_mission:
            mission_topic = self.get_parameter('mission_state_topic').get_parameter_value().string_value
            self.create_subscription(String, mission_topic, self._mission_state_cb, 10)
            self.get_logger().info(
                f'Gating activo — procesa solo en estados {self._active_states} '
                f'(escuchando {mission_topic})')

        self.get_logger().info(
            f'qr_node started — image_topic={self._image_topic}, '
            f'max_hz={max_hz}, min_area={self._min_area}px, debug={self._debug}')

    # ------------------------------------------------------------------ #
    #  Image callbacks
    # ------------------------------------------------------------------ #

    def _compressed_cb(self, msg: CompressedImage):
        if not self._gate_open() or not self._rate_ok():
            return
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        self._process(frame)

    def _image_cb(self, msg: Image):
        if not self._gate_open() or not self._rate_ok():
            return
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'cv_bridge failed: {exc}')
            return
        self._process(frame)

    # ------------------------------------------------------------------ #
    #  Gating por estado de misión
    # ------------------------------------------------------------------ #

    def _gate_open(self) -> bool:
        if not self._gate_by_mission:
            return True
        return self._mission_state in self._active_states

    def _mission_state_cb(self, msg: String):
        prev_open = self._gate_open()
        self._mission_state = msg.data.strip()
        # Al cerrarse el gate, publica una lista vacía para limpiar overlays/consumidores.
        if prev_open and not self._gate_open():
            self._pub_det.publish(String(data='[]'))

    def _rate_ok(self) -> bool:
        if self._min_proc_interval <= 0.0:
            return True
        now = time.monotonic()
        if now - self._last_proc_time < self._min_proc_interval:
            return False
        self._last_proc_time = now
        return True

    # ------------------------------------------------------------------ #
    #  Detection
    # ------------------------------------------------------------------ #

    def _detect(self, gray, coord_scale: float):
        """Decodifica QRs sobre ``gray`` y devuelve las esquinas mapeadas al
        frame original multiplicando por ``coord_scale`` (1.0 = nativo,
        0.5 = se detectó sobre un upscale 2x)."""
        ok, decoded_list, points_array, _ = self._detector.detectAndDecodeMulti(gray)
        if not ok or points_array is None:
            return []
        raw = []
        for data, corners in zip(decoded_list, points_array):
            if not data:
                continue
            pts = np.asarray(corners, dtype=np.float32).reshape(4, 2) * coord_scale
            raw.append((data, pts))
        return raw

    def _process(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        raw = self._detect(gray, 1.0)
        if not raw and self._upscale_retry:
            big = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            raw = self._detect(big, 0.5)

        detections = []
        half_w, half_h = w / 2.0, h / 2.0
        for data, pts in raw:
            area = float(cv2.contourArea(pts))
            if area < self._min_area:
                continue
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())
            detections.append({
                'data': data,
                'corners': pts.tolist(),
                'area_px': round(area, 1),
                'center': {
                    'x': round(cx, 1),
                    'y': round(cy, 1),
                    # Normalizado a [-1, 1] desde el centro del frame.
                    # Útil para encuadrar/centrar el robot frente al QR.
                    'nx': round((cx - half_w) / half_w, 4),
                    'ny': round((cy - half_h) / half_h, 4),
                },
            })

        out = String()
        out.data = json.dumps(detections)
        self._pub_det.publish(out)

        if self._pub_dbg is not None:
            annotated = self._annotate(frame, detections)
            img_msg = self._bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            self._pub_dbg.publish(img_msg)

        if detections:
            names = ', '.join(d['data'] for d in detections)
            self.get_logger().info(f'QR: {names}')

    @staticmethod
    def _annotate(frame, detections):
        annotated = frame.copy()
        for det in detections:
            pts = np.asarray(det['corners'], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(annotated, [pts], True, (0, 220, 0), 2)
            x, y = pts[0, 0]
            cv2.putText(annotated, det['data'], (int(x), int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
        return annotated


def main(args=None):
    rclpy.init(args=args)
    node = QrNode()
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
