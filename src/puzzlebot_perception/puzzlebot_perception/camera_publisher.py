#!/usr/bin/env python3
"""
Camera Publisher Node — Jetson side
-------------------------------------
Captures frames from the CSI/USB camera, compresses them as JPEG, and
publishes on /camera/image/compressed (sensor_msgs/CompressedImage).

Uso:
  ros2 run puzzlebot_perception camera_publisher
  ros2 run puzzlebot_perception camera_publisher \
      --ros-args -p camera_index:=0 -p jpeg_quality:=80 -p frame_rate:=15.0
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import cv2
import numpy as np


class CameraPublisherNode(Node):

    def __init__(self):
        super().__init__('camera_publisher')

        self.declare_parameter('camera_index',  0)
        self.declare_parameter('camera_width',  640)
        self.declare_parameter('camera_height', 480)
        self.declare_parameter('frame_rate',    15.0)
        self.declare_parameter('jpeg_quality',  80)
        self.declare_parameter('topic',         '/camera/image/compressed')

        idx     = self.get_parameter('camera_index').value
        w       = self.get_parameter('camera_width').value
        h       = self.get_parameter('camera_height').value
        rate    = self.get_parameter('frame_rate').value
        self.quality = self.get_parameter('jpeg_quality').value
        topic   = self.get_parameter('topic').value

        self.cap = cv2.VideoCapture(idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        if not self.cap.isOpened():
            self.get_logger().error(f'Could not open camera at index {idx}')

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub = self.create_publisher(CompressedImage, topic, qos)
        self.create_timer(1.0 / rate, self._capture)

        self.get_logger().info(
            f'camera_publisher started — {w}x{h} @ {rate} Hz, '
            f'JPEG quality {self.quality}, topic: {topic}'
        )

    def _capture(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Camera read failed', throttle_duration_sec=5.0)
            return

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        ok, buf = cv2.imencode('.jpg', frame, encode_params)
        if not ok:
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        msg.format = 'jpeg'
        msg.data = buf.tobytes()
        self.pub.publish(msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisherNode()
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
