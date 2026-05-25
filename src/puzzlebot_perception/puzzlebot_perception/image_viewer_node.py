#!/usr/bin/env python3
"""
Image Viewer Node — PC del operador
-------------------------------------
Se suscribe a /camera/image/compressed (sensor_msgs/CompressedImage, JPEG)
que publica camera_publisher.py en la Jetson, descomprime y muestra en OpenCV
con overlay de FPS, tamaño y timestamp.

Uso:
  ros2 run puzzlebot_perception image_viewer_node
  ros2 run puzzlebot_perception image_viewer_node \
      --ros-args -p topic:=/camera/image/compressed
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import cv2
import numpy as np
import time


class ImageViewerNode(Node):

    def __init__(self):
        super().__init__('image_viewer_node')

        self.declare_parameter('topic',         '/camera/image/compressed')
        self.declare_parameter('window_title',  'Puzzlebot Camera')
        self.declare_parameter('show_fps',      True)
        self.declare_parameter('window_width',  960)
        self.declare_parameter('window_height', 480)

        topic         = self.get_parameter('topic').value
        self.title    = self.get_parameter('window_title').value
        self.show_fps = self.get_parameter('show_fps').value
        win_w         = self.get_parameter('window_width').value
        win_h         = self.get_parameter('window_height').value

        # Best-effort QoS: descarta frames viejos si llegan tarde
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub = self.create_subscription(
            CompressedImage, topic, self.image_callback, qos
        )

        # Estado FPS
        self.last_time   = time.time()
        self.fps_display = 0.0
        self.frame_count = 0

        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title, win_w, win_h)

        # Timer separado para cv2.waitKey: OpenCV necesita procesar eventos
        # de ventana fuera del callback de ROS para no crashear en Wayland/X11.
        self.create_timer(0.033, self._poll_window)  # ~30 Hz

        self.get_logger().info(
            f'ImageViewerNode iniciado  —  escuchando: {topic}\n'
            f'  Presiona  Q  o  ESC  en la ventana para salir.'
        )

    # ------------------------------------------------------------------
    def image_callback(self, msg: CompressedImage):
        # Decodificar JPEG → array numpy → imagen BGR
        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warn('No se pudo decodificar el frame',
                                   throttle_duration_sec=5.0)
            return

        self.frame_count += 1

        # FPS con ventana deslizante de 1 s
        now = time.time()
        dt  = now - self.last_time
        if dt >= 1.0:
            self.fps_display = self.frame_count / dt
            self.frame_count = 0
            self.last_time   = now

        if self.show_fps:
            stamp   = msg.header.stamp
            ts_str  = f'{stamp.sec % 10_000}.{stamp.nanosec // 1_000_000:03d}'
            size_kb = len(msg.data) / 1024

            cv2.putText(frame, f'FPS: {self.fps_display:.1f}',
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
        """Procesa eventos de ventana OpenCV en el hilo del executor."""
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # Q o ESC
            self.get_logger().info('Cerrando viewer…')
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
