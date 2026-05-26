"""
Nodo de detección ArUco con estimación de pose absoluta del robot.

Compatible con OpenCV 4.5.x (API legada) y OpenCV 4.7+ (nueva API).

Pipeline:
  1. Suscribe /camera/image/compressed (CompressedImage)
  2. Carga intrínsecos K, D desde camera_calibration.yaml
  3. Carga extrínsecos base_link → camera_link desde camera_extrinsics.yaml
  4. Carga mapa de marcadores desde aruco_map.yaml
  5. Detecta marcadores y estima T_camera_marker con solvePnP
  6. Calcula pose del robot:
       T_map_base = T_map_marker · inv(T_camera_marker) · inv(T_base_camera)
  7. Filtra detecciones malas
  8. Fusiona múltiples marcadores con pesos 1/dist²
  9. Publica /aruco/pose  (PoseWithCovarianceStamped, frame=map)
 10. Publica /aruco/debug_image

Uso:
  ros2 run puzzlebot_perception aruco_node \\
    --ros-args \\
    -p camera_info_file:=src/puzzlebot_bringup/config/camera_calibration.yaml \\
    -p extrinsics_file:=src/puzzlebot_bringup/config/camera_extrinsics.yaml \\
    -p marker_map_file:=src/puzzlebot_bringup/config/aruco_map.yaml \\
    -p publish_debug_image:=true
"""

import os
import math
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np

_LEGACY = not hasattr(cv2.aruco, 'ArucoDetector')


# ---------------------------------------------------------------------------
# Utilidades de transformación
# ---------------------------------------------------------------------------

def _euler_to_rot(roll, pitch, yaw):
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _make_transform(x, y, z, roll, pitch, yaw):
    T = np.eye(4)
    T[:3, :3] = _euler_to_rot(roll, pitch, yaw)
    T[:3,  3] = [x, y, z]
    return T


def _rvec_tvec_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = tvec.ravel()
    return T


def _yaw_from_rot(R):
    return math.atan2(R[1, 0], R[0, 0])


def _rot_to_quat(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s;                 z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return x, y, z, w


def _normalize_angle(a):
    while a >  math.pi: a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return a


def _marker_obj_points(length):
    h = length / 2.0
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]],
                    dtype=np.float32)


# ---------------------------------------------------------------------------
# Nodo principal
# ---------------------------------------------------------------------------
class ArucoNode(Node):
    def __init__(self):
        super().__init__('aruco_node')
        self._declare_parameters()
        self._read_parameters()
        self._load_camera_calibration()
        self._load_extrinsics()
        self._load_marker_map()
        self._setup_detector()
        self._setup_comms()

        self.bridge         = CvBridge()
        self.last_pose      = None
        self._obj_pts       = _marker_obj_points(self.marker_length)
        self._solvepnp_flag = getattr(cv2, 'SOLVEPNP_IPPE_SQUARE',
                                      cv2.SOLVEPNP_ITERATIVE)
        self.get_logger().info(
            f'aruco_node listo  |  OpenCV {cv2.__version__} '
            f'({"legacy" if _LEGACY else "nueva"} API)'
        )

    # ------------------------------------------------------------------
    def _declare_parameters(self):
        self.declare_parameter('image_topic',
                               '/camera/image/compressed')
        self.declare_parameter('camera_info_file',
                               os.path.expanduser('~/calib_images/camera_calibration.yaml'))
        self.declare_parameter('extrinsics_file',
                               os.path.expanduser('~/calib_images/camera_extrinsics.yaml'))
        self.declare_parameter('marker_map_file',
                               os.path.expanduser('~/calib_images/aruco_map.yaml'))
        self.declare_parameter('dictionary',          'DICT_4X4_50')
        self.declare_parameter('marker_length',       0.08)
        self.declare_parameter('camera_frame',        'camera_link')
        self.declare_parameter('base_frame',          'base_link')
        self.declare_parameter('map_frame',           'map')
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('min_marker_area_px',  800.0)
        self.declare_parameter('max_detection_distance', 2.0)
        self.declare_parameter('max_position_jump',   0.5)
        self.declare_parameter('max_yaw_jump',        0.6)
        self.declare_parameter('reject_unknown_ids',  True)
        self.declare_parameter('near_marker_position_std', 0.03)
        self.declare_parameter('far_marker_position_std',  0.15)
        self.declare_parameter('near_marker_yaw_std',      0.05)
        self.declare_parameter('far_marker_yaw_std',       0.25)

    def _read_parameters(self):
        g = self.get_parameter
        self.image_topic      = g('image_topic').value
        self.camera_info_file = g('camera_info_file').value
        self.extrinsics_file  = g('extrinsics_file').value
        self.marker_map_file  = g('marker_map_file').value
        self.dict_name        = g('dictionary').value
        self.marker_length    = g('marker_length').value
        self.camera_frame     = g('camera_frame').value
        self.base_frame       = g('base_frame').value
        self.map_frame        = g('map_frame').value
        self._pub_debug_en    = g('publish_debug_image').value
        self.min_area         = g('min_marker_area_px').value
        self.max_dist         = g('max_detection_distance').value
        self.max_pos_jump     = g('max_position_jump').value
        self.max_yaw_jump     = g('max_yaw_jump').value
        self.reject_unknown   = g('reject_unknown_ids').value
        self.near_pos_std     = g('near_marker_position_std').value
        self.far_pos_std      = g('far_marker_position_std').value
        self.near_yaw_std     = g('near_marker_yaw_std').value
        self.far_yaw_std      = g('far_marker_yaw_std').value

    def _load_camera_calibration(self):
        path = self.camera_info_file
        if not os.path.exists(path):
            self.get_logger().error(
                f'[CALIB] No encontrado: {path}\n'
                'Ejecuta calib_compute_node primero.')
            raise FileNotFoundError(path)
        with open(path) as f:
            calib = yaml.safe_load(f)
        self.calib_w = calib['image_width']
        self.calib_h = calib['image_height']
        self.K = np.array(calib['camera_matrix']['data'],           dtype=np.float64).reshape(3, 3)
        self.D = np.array(calib['distortion_coefficients']['data'], dtype=np.float64)
        rms    = calib.get('rms_error', 'N/A')
        self.get_logger().info(
            f'[CALIB] {path}\n'
            f'  Resolución: {self.calib_w}x{self.calib_h}  RMS={rms} px\n'
            f'  fx={self.K[0,0]:.2f}  fy={self.K[1,1]:.2f}  '
            f'cx={self.K[0,2]:.2f}  cy={self.K[1,2]:.2f}\n'
            f'  D={self.D.ravel().tolist()}'
        )

    def _load_extrinsics(self):
        path = self.extrinsics_file
        if not os.path.exists(path):
            self.get_logger().warn(
                f'[EXTR] No encontrado: {path}. Usando identidad.')
            self.T_base_camera = np.eye(4)
            return
        with open(path) as f:
            data = yaml.safe_load(f)
        ext = data.get('camera_extrinsics', data)
        x, y, z   = ext['x'], ext['y'], ext['z']
        ro, pi, ya = ext['roll'], ext['pitch'], ext['yaw']
        self.T_base_camera = _make_transform(x, y, z, ro, pi, ya)
        self.get_logger().info(
            f'[EXTR] {ext.get("parent_frame","base_link")} → '
            f'{ext.get("child_frame","camera_link")}  '
            f'x={x:.3f} y={y:.3f} z={z:.3f} '
            f'roll={ro:.3f} pitch={pi:.3f} yaw={ya:.3f}'
        )

    def _load_marker_map(self):
        path = self.marker_map_file
        if not os.path.exists(path):
            self.get_logger().warn(f'[MAP] No encontrado: {path}. Sin mapa.')
            self.marker_map = {}
            return
        with open(path) as f:
            data = yaml.safe_load(f)
        self.marker_map = {}
        for mid, pose in data.get('aruco_markers', {}).items():
            self.marker_map[int(mid)] = _make_transform(
                pose['x'], pose['y'], pose['z'],
                pose['roll'], pose['pitch'], pose['yaw'])
        self.get_logger().info(
            f'[MAP] {len(self.marker_map)} marcadores conocidos: '
            f'{sorted(self.marker_map.keys())}'
        )

    def _setup_detector(self):
        dict_id    = getattr(cv2.aruco, self.dict_name)
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self.aruco_dict = aruco_dict

        if _LEGACY:
            # API legada: guardar dict + params para cv2.aruco.detectMarkers
            try:
                self.aruco_params = cv2.aruco.DetectorParameters_create()
            except AttributeError:
                self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = None
        else:
            # Nueva API: ArucoDetector
            params = cv2.aruco.DetectorParameters()
            self.detector     = cv2.aruco.ArucoDetector(aruco_dict, params)
            self.aruco_params = None

        self.get_logger().info(
            f'[DET] {self.dict_name}  marker_length={self.marker_length} m')

    def _setup_comms(self):
        self.pub_pose  = self.create_publisher(
            PoseWithCovarianceStamped, '/aruco/pose', 10)
        # PoseArray para compatibilidad con kalman_filter_node (C++)
        self.pub_poses = self.create_publisher(
            PoseArray, '/aruco/poses', 10)
        self.pub_ids   = self.create_publisher(
            Int32MultiArray, '/aruco/detected_ids', 10)
        self.pub_debug = self.create_publisher(
            Image, '/aruco/debug_image', 10) if self._pub_debug_en else None

        if 'compressed' in self.image_topic.lower():
            self.sub = self.create_subscription(
                CompressedImage, self.image_topic, self._cb_compressed,
                qos_profile_sensor_data)
        else:
            self.sub = self.create_subscription(
                Image, self.image_topic, self._cb_raw,
                qos_profile_sensor_data)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _cb_compressed(self, msg: CompressedImage):
        buf   = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is not None:
            self._process(frame, msg.header.stamp)

    def _cb_raw(self, msg: Image):
        self._process(
            self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8'),
            msg.header.stamp)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def _process(self, frame, stamp):
        actual_h, actual_w = frame.shape[:2]
        if actual_w != self.calib_w or actual_h != self.calib_h:
            self.get_logger().warn(
                f'Resolución {actual_w}x{actual_h} ≠ calibración '
                f'{self.calib_w}x{self.calib_h}',
                throttle_duration_sec=10.0)

        gray           = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detect_markers(gray)

        if ids is None:
            self.pub_ids.publish(Int32MultiArray(data=[]))
            if self.pub_debug:
                self._publish_debug_image(frame, [], [], None, stamp)
            return

        # Publicar todos los IDs visibles (sin filtrar) para identificación física
        raw_ids = ids.flatten().tolist()
        self.pub_ids.publish(Int32MultiArray(data=[int(i) for i in raw_ids]))

        candidates = []
        for i, mid in enumerate(ids.flatten().tolist()):
            corner = corners[i]
            area   = float(cv2.contourArea(corner.reshape(4, 2)))
            ok, T_cm, dist, rvec, tvec = self.estimate_marker_pose(corner)
            if not ok:
                continue
            robot_pose = self.transform_marker_to_robot_pose(mid, T_cm)
            candidates.append(dict(id=mid, area=area, dist=dist,
                                   T_cm=T_cm, rvec=rvec, tvec=tvec,
                                   corner=corner, robot_pose=robot_pose))

        valid = self.filter_detections(candidates)

        if self.pub_debug:
            valid_ids = [c['id'] for c in valid]
            self._publish_debug_image(
                frame, candidates, valid_ids,
                valid[0]['robot_pose'] if valid else None, stamp)

        if not valid:
            return

        x, y, yaw, cov = self.fuse_multiple_marker_poses(valid)
        self.publish_pose(x, y, yaw, cov, stamp)
        self.publish_pose_array(x, y, yaw, stamp)
        self.last_pose = (x, y, yaw)

    # ------------------------------------------------------------------
    # Detección (compatible con legacy y nueva API)
    # ------------------------------------------------------------------
    def detect_markers(self, gray):
        if _LEGACY:
            return cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params)
        else:
            return self.detector.detectMarkers(gray)

    # ------------------------------------------------------------------
    # Estimación de pose con solvePnP
    # ------------------------------------------------------------------
    def estimate_marker_pose(self, corner):
        img_pts = corner.reshape(4, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            self._obj_pts, img_pts, self.K, self.D,
            flags=self._solvepnp_flag)
        if not ok:
            return False, None, 0.0, None, None
        dist = float(np.linalg.norm(tvec))
        T_cm = _rvec_tvec_to_T(rvec, tvec)
        return True, T_cm, dist, rvec, tvec

    # ------------------------------------------------------------------
    # T_map_base = T_map_marker · inv(T_camera_marker) · inv(T_base_camera)
    # ------------------------------------------------------------------
    def transform_marker_to_robot_pose(self, marker_id, T_camera_marker):
        if marker_id not in self.marker_map:
            return None
        T_map_marker  = self.marker_map[marker_id]
        T_map_camera  = T_map_marker @ np.linalg.inv(T_camera_marker)
        T_map_base    = T_map_camera @ np.linalg.inv(self.T_base_camera)
        x   = float(T_map_base[0, 3])
        y   = float(T_map_base[1, 3])
        yaw = float(_yaw_from_rot(T_map_base[:3, :3]))
        return (x, y, yaw)

    # ------------------------------------------------------------------
    # Filtrado
    # ------------------------------------------------------------------
    def filter_detections(self, candidates):
        valid = []
        for c in candidates:
            mid, area, dist, pose = c['id'], c['area'], c['dist'], c['robot_pose']
            reason = None
            if self.reject_unknown and mid not in self.marker_map:
                reason = f'ID {mid} no está en el mapa'
            elif area < self.min_area:
                reason = f'ID {mid}: área {area:.0f} < {self.min_area:.0f} px'
            elif dist > self.max_dist:
                reason = f'ID {mid}: dist {dist:.2f} > {self.max_dist:.2f} m'
            elif pose is None:
                reason = f'ID {mid}: pose global no calculable'
            elif self.last_pose is not None:
                dx   = abs(pose[0] - self.last_pose[0])
                dy   = abs(pose[1] - self.last_pose[1])
                dyaw = abs(_normalize_angle(pose[2] - self.last_pose[2]))
                if max(dx, dy) > self.max_pos_jump:
                    reason = f'ID {mid}: salto posición ({dx:.2f},{dy:.2f}) m'
                elif dyaw > self.max_yaw_jump:
                    reason = (f'ID {mid}: salto yaw '
                              f'{math.degrees(dyaw):.1f}°')
            if reason:
                self.get_logger().debug(f'Rechazado: {reason}')
            else:
                valid.append(c)
                self.get_logger().info(
                    f'[OK] ID={mid}  dist={dist:.2f}m  area={area:.0f}px  '
                    f'pose=({pose[0]:.3f},{pose[1]:.3f},'
                    f'{math.degrees(pose[2]):.1f}°)')
        return valid

    # ------------------------------------------------------------------
    # Fusión con pesos 1/dist²
    # ------------------------------------------------------------------
    def fuse_multiple_marker_poses(self, valid):
        weights    = [1.0 / max(c['dist'] ** 2, 1e-6) for c in valid]
        W          = sum(weights)
        x_est      = sum(c['robot_pose'][0] * w for c, w in zip(valid, weights)) / W
        y_est      = sum(c['robot_pose'][1] * w for c, w in zip(valid, weights)) / W
        sin_sum    = sum(math.sin(c['robot_pose'][2]) * w for c, w in zip(valid, weights))
        cos_sum    = sum(math.cos(c['robot_pose'][2]) * w for c, w in zip(valid, weights))
        yaw_est    = math.atan2(sin_sum / W, cos_sum / W)
        avg_dist   = sum(c['dist'] * w for c, w in zip(valid, weights)) / W
        cov        = self._compute_covariance(avg_dist, n=len(valid))
        self.get_logger().info(
            f'Robot: x={x_est:.3f}  y={y_est:.3f}  '
            f'yaw={math.degrees(yaw_est):.1f}°  '
            f'({len(valid)} marker(s)  avg_dist={avg_dist:.2f}m  '
            f'pos_std={math.sqrt(cov[0]):.3f}m)')
        return x_est, y_est, yaw_est, cov

    def _compute_covariance(self, avg_dist, n=1):
        NEAR, FAR = 0.3, self.max_dist
        t     = float(np.clip((avg_dist - NEAR) / max(FAR - NEAR, 1e-6), 0.0, 1.0))
        pos_s = self.near_pos_std + t * (self.far_pos_std - self.near_pos_std)
        yaw_s = self.near_yaw_std + t * (self.far_yaw_std - self.near_yaw_std)
        scale = 1.0 / math.sqrt(min(n, 4))
        pos_s *= scale;  yaw_s *= scale
        cov = [0.0] * 36
        cov[0]  = pos_s ** 2   # x
        cov[7]  = pos_s ** 2   # y
        cov[14] = 0.01          # z
        cov[21] = 0.1 ** 2      # roll
        cov[28] = 0.1 ** 2      # pitch
        cov[35] = yaw_s ** 2   # yaw
        return cov

    # ------------------------------------------------------------------
    # Publicación de pose
    # ------------------------------------------------------------------
    def publish_pose(self, x, y, yaw, cov, stamp):
        msg                 = PoseWithCovarianceStamped()
        msg.header.stamp    = stamp
        msg.header.frame_id = self.map_frame
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0
        qx, qy, qz, qw = _rot_to_quat(_euler_to_rot(0.0, 0.0, yaw))
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        msg.pose.covariance = cov
        self.pub_pose.publish(msg)

    def publish_pose_array(self, x, y, yaw, stamp):
        p = Pose()
        p.position.x = x
        p.position.y = y
        qx, qy, qz, qw = _rot_to_quat(_euler_to_rot(0.0, 0.0, yaw))
        p.orientation.x = qx
        p.orientation.y = qy
        p.orientation.z = qz
        p.orientation.w = qw
        arr = PoseArray()
        arr.header.stamp    = stamp
        arr.header.frame_id = self.map_frame
        arr.poses           = [p]
        self.pub_poses.publish(arr)

    # ------------------------------------------------------------------
    # Debug image
    # ------------------------------------------------------------------
    def _publish_debug_image(self, frame, candidates, valid_ids, robot_pose, stamp):
        vis = frame.copy()
        for c in candidates:
            mid    = c['id']
            corner = c['corner'].reshape(4, 2).astype(int)
            is_ok  = mid in valid_ids
            color  = (0, 255, 0) if is_ok else (0, 100, 255)
            cv2.polylines(vis, [corner], True, color, 2)
            cx, cy = int(corner[:, 0].mean()), int(corner[:, 1].mean())
            cv2.putText(vis, f'ID {mid}',      (cx - 20, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(vis, f'{c["dist"]:.2f}m', (cx - 20, cy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if is_ok and c['rvec'] is not None:
                axis_len = self.marker_length * 0.8
                try:
                    cv2.drawFrameAxes(vis, self.K, self.D,
                                      c['rvec'], c['tvec'], axis_len)
                except AttributeError:
                    cv2.aruco.drawAxis(vis, self.K, self.D,
                                       c['rvec'], c['tvec'], axis_len)

        if robot_pose:
            x, y, yaw = robot_pose
            cv2.putText(vis,
                        f'Robot: x={x:.2f} y={y:.2f} yaw={math.degrees(yaw):.1f}deg',
                        (10, vis.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        n = len(valid_ids)
        cv2.putText(vis, f'Validos: {n}/{len(candidates)}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 0) if n > 0 else (0, 100, 255), 2)

        dbg                  = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
        dbg.header.stamp     = stamp
        dbg.header.frame_id  = self.camera_frame
        self.pub_debug.publish(dbg)


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
