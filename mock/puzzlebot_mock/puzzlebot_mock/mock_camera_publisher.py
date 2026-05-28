"""
mock_camera_publisher.py — Simula el feed de cámara del Puzzlebot.

Genera frames sintéticos (gradiente de color animado) y los publica como
sensor_msgs/CompressedImage en /camera/image/compressed, igual que la
cámara real de la Jetson.

Requiere: Pillow (pip install Pillow)

Uso:
  ros2 run puzzlebot_mock mock_camera
"""
import io
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

WIDTH  = 320
HEIGHT = 240
FPS    = 5   # Hz


class MockCameraPublisher(Node):
    def __init__(self):
        super().__init__('mock_camera_publisher')
        self._pub = self.create_publisher(CompressedImage, '/camera/image/compressed', 1)
        self._t0  = time.time()
        self.create_timer(1.0 / FPS, self._publish_frame)
        self.get_logger().info(
            f'mock_camera_publisher started — {WIDTH}x{HEIGHT} @ {FPS} Hz'
        )

    def _make_frame(self, t: float) -> np.ndarray:
        """Generate an animated RGB frame with a color gradient + timestamp text."""
        img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        # Animated color gradient that shifts over time
        phase = t * 0.5
        for col in range(WIDTH):
            r = int(127 + 127 * math.sin(2 * math.pi * col / WIDTH + phase))
            g = int(127 + 127 * math.sin(2 * math.pi * col / WIDTH + phase + 2.09))
            b = int(127 + 127 * math.sin(2 * math.pi * col / WIDTH + phase + 4.19))
            img[:, col] = [r, g, b]

        # Horizontal scan line that moves up and down
        scan_y = int(HEIGHT / 2 + HEIGHT / 3 * math.sin(t * 1.5))
        scan_y = max(1, min(HEIGHT - 2, scan_y))
        img[scan_y - 1:scan_y + 2, :] = [255, 255, 255]

        # Center crosshair
        cy, cx = HEIGHT // 2, WIDTH // 2
        img[cy - 10:cy + 10, cx - 1:cx + 2] = [0, 255, 0]
        img[cy - 1:cy + 2, cx - 10:cx + 10] = [0, 255, 0]

        return img

    def _encode_jpeg(self, img: np.ndarray) -> bytes:
        try:
            from PIL import Image
            pil = Image.fromarray(img, 'RGB')
            buf = io.BytesIO()
            pil.save(buf, format='JPEG', quality=70)
            return buf.getvalue()
        except ImportError:
            self.get_logger().warn('Pillow not installed — publishing empty camera frame.')
            return b''

    def _publish_frame(self):
        t = time.time() - self._t0
        img = self._make_frame(t)
        jpeg = self._encode_jpeg(img)

        if not jpeg:
            return

        msg = CompressedImage()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        msg.format          = 'jpeg'
        msg.data            = list(jpeg)

        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MockCameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
