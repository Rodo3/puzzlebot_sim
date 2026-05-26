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
        self.declare_parameter('marker_size',    0.10)
        self.declare_parameter('aruco_dict',     'DICT_6X6_250')
        self.declare_parameter('image_topic',    '/camera/image_raw')
        self.declare_parameter('cam_info_topic', '/camera/camera_info')

        self.marker_size = self.get_parameter('marker_size').value
        dict_name  = self.get_parameter('aruco_dict').value
        img_topic  = self.get_parameter('image_topic').value
        info_topic = self.get_parameter('cam_info_topic').value

        aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, dict_name))
        self.detector = cv2.aruco.ArucoDetector(
            aruco_dict, cv2.aruco.DetectorParameters())

        self.bridge  = CvBridge()
        self.cam_mat = None
        self.dist    = None

        # Fallback intrinsics for simulation (640x480, 80deg FOV)
        self._sim_cam_mat = np.array([
            [554.26,   0.0, 320.0],
            [  0.0, 554.26, 240.0],
            [  0.0,   0.0,   1.0],
        ], dtype=np.float64)
        self._sim_dist = np.zeros(5)

        self.sub_img  = self.create_subscription(
            Image, img_topic, self.img_cb, 10)
        self.sub_info = self.create_subscription(
            CameraInfo, info_topic, self.info_cb, 10)
        self.pub = self.create_publisher(PoseArray, '/aruco/poses', 10)

        self.get_logger().info(
            f'aruco_node started  image={img_topic}  cam_info={info_topic}')

    def info_cb(self, msg: CameraInfo):
        self.cam_mat = np.array(msg.k).reshape(3, 3)
        self.dist    = np.array(msg.d)

    def img_cb(self, msg: Image):
        cam_mat = self.cam_mat if self.cam_mat is not None else self._sim_cam_mat
        dist    = self.dist    if self.dist    is not None else self._sim_dist

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return

        pose_array = PoseArray()
        pose_array.header = msg.header
        pose_array.header.frame_id = 'camera_link'

        # API nueva OpenCV 4.8+ — usar solvePnP por cada marker
        obj_pts = np.array([
            [-self.marker_size / 2,  self.marker_size / 2, 0],
            [ self.marker_size / 2,  self.marker_size / 2, 0],
            [ self.marker_size / 2, -self.marker_size / 2, 0],
            [-self.marker_size / 2, -self.marker_size / 2, 0],
        ], dtype=np.float32)

        for corner in corners:
            img_pts = corner[0].astype(np.float32)
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, cam_mat, dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue

            pose = Pose()
            pose.position.x = float(tvec[0])
            pose.position.y = float(tvec[1])
            pose.position.z = float(tvec[2])
            rot_mat, _ = cv2.Rodrigues(rvec)
            pose.orientation = self._rotmat_to_quat(rot_mat)
            pose_array.poses.append(pose)

        self.get_logger().info(
            f'Detected {len(ids)} ArUco marker(s): ids={ids.flatten().tolist()}',
            throttle_duration_sec=1.0)
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
