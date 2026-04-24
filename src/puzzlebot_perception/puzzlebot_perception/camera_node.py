import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.declare_parameter('camera_width',  640)
        self.declare_parameter('camera_height', 480)
        self.declare_parameter('camera_index',  0)
        self.declare_parameter('frame_rate',    30.0)

        w     = self.get_parameter('camera_width').value
        h     = self.get_parameter('camera_height').value
        idx   = self.get_parameter('camera_index').value
        rate  = self.get_parameter('frame_rate').value

        self.bridge = CvBridge()
        self.cap    = cv2.VideoCapture(idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        self.pub = self.create_publisher(Image, '/cam_img', 10)
        self.create_timer(1.0 / rate, self.capture)
        self.get_logger().info(f'camera_node started ({w}x{h} @ {rate} Hz, device {idx})')

    def capture(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Camera read failed', throttle_duration_sec=5.0)
            return
        self.pub.publish(self.bridge.cv2_to_imgmsg(frame, encoding='bgr8'))

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
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
