import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge
import cv2
import numpy as np


class ArucoNode(Node):
    def __init__(self):
        super().__init__('aruco_node')
        self.declare_parameter('marker_size',   0.10)    # metres
        self.declare_parameter('aruco_dict',    'DICT_6X6_250')

        self.marker_size = self.get_parameter('marker_size').value
        dict_name = self.get_parameter('aruco_dict').value
        aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, dict_name))
        self.detector = cv2.aruco.ArucoDetector(
            aruco_dict, cv2.aruco.DetectorParameters())

        self.bridge  = CvBridge()
        self.cam_mat = None
        self.dist    = None

        self.sub_img  = self.create_subscription(Image,      '/cam_img',     self.img_cb,  10)
        self.sub_info = self.create_subscription(CameraInfo, '/cam_info',    self.info_cb, 10)
        self.pub      = self.create_publisher(PoseArray, '/aruco/poses', 10)

        self.get_logger().info('aruco_node started')

    def info_cb(self, msg: CameraInfo):
        self.cam_mat = np.array(msg.k).reshape(3, 3)
        self.dist    = np.array(msg.d)

    def img_cb(self, msg: Image):
        if self.cam_mat is None:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return

        pose_array = PoseArray()
        pose_array.header = msg.header
        pose_array.header.frame_id = 'camera_frame'

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_size, self.cam_mat, self.dist)

        for rvec, tvec in zip(rvecs, tvecs):
            pose = Pose()
            pose.position.x = float(tvec[0][0])
            pose.position.y = float(tvec[0][1])
            pose.position.z = float(tvec[0][2])
            # Convert rotation vector to quaternion
            rot_mat, _ = cv2.Rodrigues(rvec)
            pose.orientation = self._rotmat_to_quat(rot_mat)
            pose_array.poses.append(pose)

        self.pub.publish(pose_array)

    @staticmethod
    def _rotmat_to_quat(R):
        from geometry_msgs.msg import Quaternion
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            x, y, z, w = 0.0, 0.0, 0.0, 1.0
        q = Quaternion()
        q.x, q.y, q.z, q.w = x, y, z, w
        return q


def main(args=None):
    rclpy.init(args=args)
    node = ArucoNode()
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
