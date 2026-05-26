"""
Nodo 3/3 — Aplicación de la calibración intrínseca en tiempo real.

Carga el YAML generado por calib_compute_node, suscribe la imagen
comprimida de la cámara, rectifica cada frame y publica:
  /cam_img_rect  — imagen sin distorsión (sensor_msgs/Image)
  /cam_info      — parámetros intrínsecos (sensor_msgs/CameraInfo)

El tópico /cam_info permite que aruco_node y otros nodos reciban la
calibración sin necesidad de leer el YAML directamente.

Uso:
  ros2 run puzzlebot_perception calib_apply_node
  ros2 run puzzlebot_perception calib_apply_node \\
    --ros-args -p calib_yaml:=src/puzzlebot_bringup/config/camera_calibration.yaml \\
               -p show_preview:=true
"""

import os
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np


class CalibApplyNode(Node):
    def __init__(self):
        super().__init__('calib_apply_node')

        self.declare_parameter('image_topic',
                               '/camera/image/compressed')
        self.declare_parameter('calib_yaml',
                               os.path.expanduser('~/calib_images/camera_calibration.yaml'))
        self.declare_parameter('frame_id',      'camera_link')
        self.declare_parameter('show_preview',  False)

        self.image_topic  = self.get_parameter('image_topic').value
        calib_path        = self.get_parameter('calib_yaml').value
        self.frame_id     = self.get_parameter('frame_id').value
        self.show_preview = self.get_parameter('show_preview').value

        if not os.path.exists(calib_path):
            self.get_logger().error(
                f'Archivo de calibración no encontrado: {calib_path}\n'
                'Ejecuta calib_compute_node primero para generarlo.'
            )
            raise FileNotFoundError(calib_path)

        self._load_calibration(calib_path)

        # Mapas precalculados para rectificación eficiente (evita cv2.undistort por frame)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K, self.D, None, self.K, (self.w, self.h), cv2.CV_16SC2)

        self.bridge = CvBridge()

        self.pub_rect = self.create_publisher(Image,      '/cam_img_rect', 10)
        self.pub_info = self.create_publisher(CameraInfo, '/cam_info',     10)

        if 'compressed' in self.image_topic.lower():
            self.sub = self.create_subscription(
                CompressedImage, self.image_topic, self._cb_compressed,
                qos_profile_sensor_data)
        else:
            self.sub = self.create_subscription(
                Image, self.image_topic, self._cb_raw,
                qos_profile_sensor_data)

    def _load_calibration(self, path: str):
        with open(path) as f:
            calib = yaml.safe_load(f)

        self.w = calib['image_width']
        self.h = calib['image_height']
        self.K = np.array(calib['camera_matrix']['data'],           dtype=np.float64).reshape(3, 3)
        self.D = np.array(calib['distortion_coefficients']['data'], dtype=np.float64)
        rms    = calib.get('rms_error', 'N/A')

        self.get_logger().info(
            f'Calibración cargada desde: {path}\n'
            f'  Resolución:   {self.w}x{self.h}\n'
            f'  fx={self.K[0,0]:.2f}  fy={self.K[1,1]:.2f}  '
            f'cx={self.K[0,2]:.2f}  cy={self.K[1,2]:.2f}\n'
            f'  D={self.D.ravel().tolist()}\n'
            f'  RMS={rms} px'
        )

    def _make_cam_info(self, stamp) -> CameraInfo:
        info                  = CameraInfo()
        info.header.stamp     = stamp
        info.header.frame_id  = self.frame_id
        info.width            = self.w
        info.height           = self.h
        info.distortion_model = 'plumb_bob'
        info.k                = self.K.ravel().tolist()
        info.d                = self.D.ravel().tolist()
        info.r                = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        P                     = np.hstack([self.K, np.zeros((3, 1))])
        info.p                = P.ravel().tolist()
        return info

    def _process(self, frame: np.ndarray, stamp):
        # Advertir si la resolución real no coincide con la calibración
        actual_h, actual_w = frame.shape[:2]
        if actual_w != self.w or actual_h != self.h:
            self.get_logger().warn(
                f'Resolución actual {actual_w}x{actual_h} ≠ '
                f'calibración {self.w}x{self.h}. '
                'La rectificación puede ser incorrecta. Recalibra en la misma resolución.',
                throttle_duration_sec=10.0,
            )

        rectified           = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
        rect_msg            = self.bridge.cv2_to_imgmsg(rectified, encoding='bgr8')
        rect_msg.header.stamp    = stamp
        rect_msg.header.frame_id = self.frame_id

        self.pub_rect.publish(rect_msg)
        self.pub_info.publish(self._make_cam_info(stamp))

        if self.show_preview:
            combined = np.hstack([frame, rectified])
            cv2.putText(combined, 'Original',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(combined, 'Rectificada',
                        (self.w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('Calibracion aplicada', combined)
            cv2.waitKey(1)

    def _cb_compressed(self, msg: CompressedImage):
        buf   = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is not None:
            self._process(frame, msg.header.stamp)

    def _cb_raw(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self._process(frame, msg.header.stamp)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CalibApplyNode()
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
