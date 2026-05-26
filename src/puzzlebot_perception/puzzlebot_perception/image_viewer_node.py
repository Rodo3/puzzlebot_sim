#!/usr/bin/env python3
"""
Image Viewer Node — PC del operador
-------------------------------------
Se suscribe a un tópico de cámara y muestra en OpenCV con overlay de FPS.

Si se proporciona un YAML de calibración (calib_yaml), aplica la corrección
de distorsión directamente en el viewer — sin necesidad de calib_apply_node.

Uso:
  # Sin corrección (default)
  ros2 run puzzlebot_perception image_viewer_node

  # Con corrección de distorsión integrada
  ros2 run puzzlebot_perception image_viewer_node \
      --ros-args -p rectify:=true

  # Tópico custom o YAML alternativo
  ros2 run puzzlebot_perception image_viewer_node \
      --ros-args -p topic:=/camera/image/compressed \
                 -p rectify:=true \
                 -p calib_yaml:=/ruta/al/camera_calibration.yaml
"""

import os
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

import cv2
import numpy as np
import time


class ImageViewerNode(Node):

    def __init__(self):
        super().__init__('image_viewer_node')

        _default_yaml = os.path.join(
            get_package_share_directory('puzzlebot_bringup'),
            'config', 'camera_calibration.yaml')

        self.declare_parameter('topic',         '/camera/image/compressed')
        self.declare_parameter('window_title',  'Puzzlebot Camera')
        self.declare_parameter('show_fps',      True)
        self.declare_parameter('window_width',  960)
        self.declare_parameter('window_height', 480)
        self.declare_parameter('rectify',       False)
        self.declare_parameter('calib_yaml',    _default_yaml)

        topic         = self.get_parameter('topic').value
        self.title    = self.get_parameter('window_title').value
        self.show_fps = self.get_parameter('show_fps').value
        win_w         = self.get_parameter('window_width').value
        win_h         = self.get_parameter('window_height').value
        rectify       = self.get_parameter('rectify').value
        calib_yaml    = self.get_parameter('calib_yaml').value

        self.bridge   = CvBridge()
        self.map1     = None
        self.map2     = None

        if rectify:
            self._load_calibration(calib_yaml)

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        if 'compressed' in topic.lower():
            self.sub = self.create_subscription(
                CompressedImage, topic, self._cb_compressed, qos)
        else:
            self.sub = self.create_subscription(
                Image, topic, self._cb_raw, qos)

        self.last_time   = time.time()
        self.fps_display = 0.0
        self.frame_count = 0

        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title, win_w, win_h)

        self.create_timer(0.033, self._poll_window)  # ~30 Hz

        mode = 'CON corrección de distorsión' if self.map1 is not None else 'sin corrección'
        self.get_logger().info(
            f'ImageViewerNode — {topic} [{mode}]\n'
            f'  Q / ESC para salir.'
        )

    # ------------------------------------------------------------------
    def _load_calibration(self, path: str):
        if not os.path.exists(path):
            self.get_logger().warn(
                f'calib_yaml no encontrado: {path}\n'
                'Arrancando sin corrección de distorsión.')
            return

        with open(path) as f:
            calib = yaml.safe_load(f)

        w = calib['image_width']
        h = calib['image_height']
        K = np.array(calib['camera_matrix']['data'],           dtype=np.float64).reshape(3, 3)
        D = np.array(calib['distortion_coefficients']['data'], dtype=np.float64)

        # Mapas precalculados: cv2.remap es mucho más rápido que cv2.undistort por frame
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            K, D, None, K, (w, h), cv2.CV_16SC2)

        self.get_logger().info(
            f'Calibración cargada — {w}x{h}  '
            f'fx={K[0,0]:.1f} fy={K[1,1]:.1f}  '
            f'RMS={calib.get("rms_error", "N/A")} px'
        )

    # ------------------------------------------------------------------
    def _cb_compressed(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn('No se pudo decodificar el frame',
                                   throttle_duration_sec=5.0)
            return
        self._display(frame, msg.header.stamp, len(msg.data))

    def _cb_raw(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}', throttle_duration_sec=5.0)
            return
        self._display(frame, msg.header.stamp, msg.step * msg.height)

    # ------------------------------------------------------------------
    def _display(self, frame: np.ndarray, stamp, size_bytes: int):
        self.frame_count += 1

        now = time.time()
        dt  = now - self.last_time
        if dt >= 1.0:
            self.fps_display = self.frame_count / dt
            self.frame_count = 0
            self.last_time   = now

        # Aplicar corrección de distorsión si está cargada
        if self.map1 is not None:
            frame = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

        if self.show_fps:
            ts_str  = f'{stamp.sec % 10_000}.{stamp.nanosec // 1_000_000:03d}'
            size_kb = size_bytes / 1024
            rect_tag = ' [rect]' if self.map1 is not None else ''

            cv2.putText(frame, f'FPS: {self.fps_display:.1f}{rect_tag}',
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 80), 2, cv2.LINE_AA)
            cv2.putText(frame, f'{size_kb:.1f} KB',
                        (8, 48), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f't={ts_str}',
                        (8, 68), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(self.title, frame)

    # ------------------------------------------------------------------
    def _poll_window(self):
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            self.get_logger().info('Cerrando viewer...')
            cv2.destroyAllWindows()
            rclpy.shutdown()

    # ------------------------------------------------------------------
    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


# ======================================================================
def main(args=None):
    rclpy.init(args=args)
    node = ImageViewerNode()
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
