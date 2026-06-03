"""
Mapping SLAM node.

This ROS wrapper wires together the mapping components:

  OdometryBuffer      /odom timestamp synchronization
  LocalScanMatcher   future scan-to-map pose correction hook
  KeyframeManager    optional scan integration gate
  OccupancyGridMap   log-odds grid + Bresenham ray integration

The current default behavior is intentionally equivalent to the validated
Gazebo mapper: scans are integrated using the odometry pose, and map->odom is an
identity transform.  The component boundaries are ready for a future physical
robot scan matcher and dynamic map->odom correction.

--- TF ownership protocol ---

map→odom puede tener tres dueños posibles (mutuamente excluyentes):
  1. slam_node (este nodo):  publish_map_odom_tf = true
  2. aruco_map_odom:         publica TF + /map_to_odom cuando aruco:=true
  3. mcl:                    publica TF cuando mcl:=true

Cuando el dueño es aruco_map_odom (caso más común en robot real):
  - slam_node recibe la corrección via /map_to_odom → _map_to_odom_cb
  - slam_node USA esa corrección para proyectar odom_pose al frame map
  - slam_node NO debe sobreescribir self._map_odom_* con scan matching
    (param scan_match_updates_map_odom: false → default cuando publish_map_odom_tf: false)

Esto evita el feedback loop: scan_matcher → sobreescribe corrección de ArUco
→ siguiente scan usa pose incorrecta → matcher aun más confundido → divergencia.
"""

import math
import os
from datetime import datetime

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from std_msgs.msg import Bool, String
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import LaserScan
import tf2_ros

from .keyframe_manager import KeyframeManager
from .occupancy_grid_map import OccupancyGridMap
from .odometry_buffer import OdometryBuffer
from .scan_matcher import LocalScanMatcher
from .slam_types import Pose2D


class SlamNode(Node):
    def __init__(self):
        super().__init__('slam_node')

        self._declare_parameters()
        self._map_frame = self.get_parameter('map_frame').value
        self._odom_frame = self.get_parameter('odom_frame').value
        self._publish_map_odom_tf = bool(
            self.get_parameter('publish_map_odom_tf').value)
        # scan_match_updates_map_odom: controla si el scan matcher puede
        # sobreescribir la corrección interna map→odom.
        # Por defecto: igual a publish_map_odom_tf (solo cuando SLAM es dueño).
        # Cuando ArUco o MCL son dueños (publish_map_odom_tf: false),
        # este parámetro debe ser False para no borrar la corrección externa.
        self._scan_match_updates_map_odom = bool(
            self.get_parameter('scan_match_updates_map_odom').value)
        map_resolution = float(self.get_parameter('map_resolution').value)
        map_width_meters = float(self.get_parameter('map_width_meters').value)
        map_height_meters = float(self.get_parameter('map_height_meters').value)
        legacy_size_pixels = int(self.get_parameter('map_size_pixels').value)
        legacy_size_meters = float(self.get_parameter('map_size_meters').value)
        if map_resolution > 0.0 and map_width_meters > 0.0 and map_height_meters > 0.0:
            width_pixels = int(math.ceil(map_width_meters / map_resolution))
            height_pixels = int(math.ceil(map_height_meters / map_resolution))
            size_pixels = max(width_pixels, height_pixels)
            size_meters = size_pixels * map_resolution
        else:
            map_resolution = legacy_size_meters / legacy_size_pixels
            width_pixels = None
            height_pixels = None
            size_pixels = legacy_size_pixels
            size_meters = legacy_size_meters

        self._grid_map = OccupancyGridMap(
            size_pixels=size_pixels,
            size_meters=size_meters,
            origin_x=float(self.get_parameter('map_origin_x').value),
            origin_y=float(self.get_parameter('map_origin_y').value),
            p_occ=float(self.get_parameter('p_occ').value),
            p_free=float(self.get_parameter('p_free').value),
            l_clamp=float(self.get_parameter('l_clamp').value),
            scan_step=int(self.get_parameter('scan_step').value),
            max_range_factor=float(self.get_parameter('max_range_factor').value),
            min_useful_range=float(self.get_parameter('min_useful_range').value),
            max_mapping_range=float(self.get_parameter('max_mapping_range').value),
            lidar_x=float(self.get_parameter('lidar_x').value),
            lidar_y=float(self.get_parameter('lidar_y').value),
            lidar_yaw=float(self.get_parameter('lidar_yaw').value),
            width_pixels=width_pixels,
            height_pixels=height_pixels,
            resolution=map_resolution,
        )
        self._odom_buffer = OdometryBuffer(
            buffer_sec=float(self.get_parameter('pose_buffer_sec').value),
            max_lookup_age=float(self.get_parameter('max_scan_pose_age').value),
        )
        self._keyframes = KeyframeManager(
            enabled=bool(self.get_parameter('use_keyframes').value),
            min_translation=float(self.get_parameter('keyframe_min_translation').value),
            min_rotation=float(self.get_parameter('keyframe_min_rotation').value),
        )
        self._localization_only = bool(self.get_parameter('localization_only').value)
        loc_map_path = self.get_parameter('localization_map_path').value

        # En modo localization_only, forzar scan_matching_enabled=true y
        # cargar el mapa PNG para que el matcher tenga referencias desde el inicio.
        if self._localization_only:
            if not loc_map_path:
                self.get_logger().error(
                    'localization_only=true pero localization_map_path está vacío')
            else:
                self._grid_map.load_from_png(loc_map_path)
                self.get_logger().info(
                    f'[localization_only] Mapa cargado desde: {loc_map_path}')

        scan_matching_param = (
            True if self._localization_only
            else bool(self.get_parameter('scan_matching_enabled').value)
        )
        self._scan_matcher = LocalScanMatcher(enabled=scan_matching_param)

        # Parámetros para scan match → EKF feedback
        self._scan_match_min_score      = float(self.get_parameter('scan_match_min_score').value)
        self._scan_match_cov_xy         = float(self.get_parameter('scan_match_cov_xy').value)
        self._scan_match_cov_theta      = float(self.get_parameter('scan_match_cov_theta').value)
        self._scan_match_cov_by_score   = bool(self.get_parameter('scan_match_cov_scale_by_score').value)
        self._scan_match_max_cov_scale  = float(self.get_parameter('scan_match_max_cov_scale').value)

        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._pub_map        = self.create_publisher(OccupancyGrid, '/map', map_qos)
        self._pub_scan_match = self.create_publisher(
            PoseWithCovarianceStamped, '/scan_match/pose', 10)
        self._tf = tf2_ros.TransformBroadcaster(self)

        # El callback de scan corre en su propio thread para que el scan_matcher
        # (NumPy puro — libera el GIL) no bloquee los timers de TF ni los demás
        # callbacks mientras busca la corrección de pose.
        self._scan_cb_group = MutuallyExclusiveCallbackGroup()

        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(
            LaserScan, '/scan', self._scan_cb, qos_profile_sensor_data,
            callback_group=self._scan_cb_group)
        self.create_subscription(
            TransformStamped, '/map_to_odom', self._map_to_odom_cb, 10)
        self.create_subscription(Bool,   '/slam/reset',    self._slam_reset_cb, 10)
        self.create_subscription(String, '/slam/load_map', self._load_map_cb,   10)

        self.create_timer(0.1, self._broadcast_tf)
        self.create_timer(1.0, self._publish_map)

        self._map_odom_x   = 0.0
        self._map_odom_y   = 0.0
        self._map_odom_yaw = 0.0

        mode = 'LOCALIZATION-ONLY' if self._localization_only else 'MAPPING'
        self.get_logger().info(
            f'slam_node ready [{mode}] — '
            f'{self._grid_map.width_pixels}×{self._grid_map.height_pixels} px '
            f'@ {self._grid_map.resolution:.3f} m/px, '
            f'size=({self._grid_map.width_meters:.2f}, {self._grid_map.height_meters:.2f}) m, '
            f'origin=({self._grid_map.origin_x:.2f}, {self._grid_map.origin_y:.2f}), '
            f'scan_matching={self._scan_matcher.enabled}, '
            f'publish_map_odom_tf={self._publish_map_odom_tf}, '
            f'scan_match_updates_map_odom={self._scan_match_updates_map_odom}')

    def _declare_parameters(self) -> None:
        self.declare_parameter('map_size_pixels', 500)
        self.declare_parameter('map_size_meters', 25.0)
        self.declare_parameter('map_width_meters', 0.0)
        self.declare_parameter('map_height_meters', 0.0)
        self.declare_parameter('map_resolution', 0.0)
        self.declare_parameter('map_origin_x', -12.5)
        self.declare_parameter('map_origin_y', -12.5)

        self.declare_parameter('p_occ', 0.75)
        self.declare_parameter('p_free', 0.45)
        self.declare_parameter('l_clamp', 5.0)

        self.declare_parameter('scan_step', 1)
        self.declare_parameter('max_range_factor', 0.95)
        self.declare_parameter('min_useful_range', 0.20)
        self.declare_parameter('max_mapping_range', 0.0)
        self.declare_parameter('lidar_x', 0.0)
        self.declare_parameter('lidar_y', 0.0)
        self.declare_parameter('lidar_yaw', 0.0)

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('publish_map_odom_tf', True)

        self.declare_parameter('pose_buffer_sec', 3.0)
        self.declare_parameter('max_scan_pose_age', 0.20)

        self.declare_parameter('use_keyframes', False)
        self.declare_parameter('keyframe_min_translation', 0.10)
        self.declare_parameter('keyframe_min_rotation', math.radians(5.0))

        self.declare_parameter('scan_matching_enabled', False)
        self.declare_parameter('scan_match_min_score',  15.0)
        self.declare_parameter('scan_match_cov_xy',     0.04)
        self.declare_parameter('scan_match_cov_theta',  0.03)
        # Fase 3: covarianza adaptativa según score del matcher.
        # Cuando el score es bajo (match marginal), se infla la covarianza
        # para que el EKF le dé menos peso. Cuando el score es alto (match
        # limpio), la covarianza baja al valor base → mayor influencia.
        # cov_scale = (scan_match_min_score / score).clamp(1/max_scale, 1.0)
        # Con max_scale=4.0: score=min_score → cov×4, score=4×min → cov×1.
        self.declare_parameter('scan_match_cov_scale_by_score', True)
        self.declare_parameter('scan_match_max_cov_scale',      4.0)

        # localization_only: carga el mapa PNG y hace scan matching contra él
        # para alimentar al EKF con /scan_match/pose, pero NO integra nuevos
        # scans (no modifica el mapa). Usar con use_map:=true.
        self.declare_parameter('localization_only',     False)
        self.declare_parameter('localization_map_path', '')

        # scan_match_updates_map_odom:
        #   True  → el scan matcher puede actualizar la corrección map→odom interna.
        #           Apropiado cuando SLAM es el único dueño del TF (publish_map_odom_tf: true).
        #   False → la corrección interna solo viene de /map_to_odom (ArUco/MCL).
        #           OBLIGATORIO cuando ArUco o MCL publican map→odom externamente,
        #           para evitar que el scan matcher sobreescriba correcciones absolutas.
        # Nota: el valor por defecto es True para compatibilidad con Gazebo (modo solo SLAM).
        # En real_robot.launch.py con aruco:=true, publish_map_odom_tf se pone en false
        # automáticamente, y se recomienda pasar scan_match_updates_map_odom:=false también.
        self.declare_parameter('scan_match_updates_map_odom', True)

    def _load_map_cb(self, msg: String) -> None:
        path = msg.data.strip()
        if not path or not os.path.isfile(path):
            self.get_logger().warn(f'load_map: file not found: {path}')
            return
        try:
            self._grid_map.load_from_png(path)
            self._localization_only = True
            self.get_logger().info(f'Static map loaded: {path}')
            self._publish_map()
        except Exception as exc:
            self.get_logger().error(f'load_map error: {exc}')

    def _slam_reset_cb(self, msg: Bool) -> None:
        if not msg.data:
            return
        self._grid_map.reset()
        self._map_odom_x   = 0.0
        self._map_odom_y   = 0.0
        self._map_odom_yaw = 0.0
        self.get_logger().info('SLAM map reset via /slam/reset')

    def _odom_cb(self, msg: Odometry) -> None:
        self._odom_buffer.add(msg)

    def _map_to_odom_cb(self, msg: TransformStamped) -> None:
        if msg.header.frame_id != self._map_frame or msg.child_frame_id != self._odom_frame:
            return
        self._map_odom_x = msg.transform.translation.x
        self._map_odom_y = msg.transform.translation.y
        self._map_odom_yaw = math.atan2(
            2.0 * msg.transform.rotation.w * msg.transform.rotation.z,
            1.0 - 2.0 * msg.transform.rotation.z * msg.transform.rotation.z,
        )

    def _scan_cb(self, scan: LaserScan) -> None:
        odom_pose = self._odom_buffer.lookup(scan.header.stamp)
        if odom_pose is None:
            self.get_logger().warn(
                'Skipping scan: no odom pose close enough to scan timestamp',
                throttle_duration_sec=2.0)
            return

        # CRÍTICO: transformar odom_pose al frame map antes de pasarla al scan_matcher.
        # El matcher trabaja en coordenadas del mapa; si recibe odom_pose busca en
        # el área incorrecta del grid → matches basura → corrección divergente.
        map_init = self._odom_to_map_pose(odom_pose)
        map_pose = self._scan_matcher.match(scan, map_init, self._grid_map)

        # Sanity check: si la corrección del matcher supera umbrales físicos,
        # el match es inválido (mapa vacío, pose inicial muy mala). En ese caso
        # usamos la pose derivada de odometría transformada al mapa, sin corregir.
        correction_dist = math.hypot(map_pose.x - map_init.x, map_pose.y - map_init.y)
        correction_yaw  = abs(math.atan2(math.sin(map_pose.yaw - map_init.yaw),
                                          math.cos(map_pose.yaw - map_init.yaw)))
        if correction_dist > 0.18 or correction_yaw > math.radians(10.0):
            self.get_logger().warn(
                f'Scan match rejected (dist={correction_dist:.2f}m '
                f'yaw={math.degrees(correction_yaw):.1f}°) — usando odom',
                throttle_duration_sec=2.0)
            map_pose = map_init

        # True si el matcher encontró una corrección confiable Y la sanity check
        # no la revirtió. map_pose is not map_init: durante warmup/disabled el
        # matcher devuelve el mismo objeto initial_pose → descarta silenciosamente.
        scan_matched = (
            self._scan_matcher.enabled
            and self._scan_matcher.last_score >= self._scan_match_min_score
            and map_pose is not map_init
        )

        # Actualiza map→odom solo cuando SLAM es el dueño del TF
        # (scan_match_updates_map_odom: true) Y el scan matching está activo.
        #
        # Cuando ArUco o MCL son dueños de map→odom:
        #   - _map_to_odom_cb() actualiza self._map_odom_* desde /map_to_odom
        #   - NO debe sobrescribirse aquí, o se pierde la corrección absoluta
        #
        # Cuando SLAM es dueño (publish_map_odom_tf: true, sin ArUco):
        #   - Actualizar permite que el scan matcher refine la pose del mapa
        if self._scan_match_updates_map_odom:
            self._update_map_odom_tf(map_pose, odom_pose)

        # Publicar corrección de pose al EKF para loop closure liviano.
        # El Kalman la fusiona como medición adicional junto con ArUco.
        if scan_matched:
            self._publish_scan_match(map_pose, scan.header.stamp)

        # localization_only: no modificar el mapa, solo usar scan matching para EKF.
        if self._localization_only:
            return

        if not self._keyframes.should_integrate(map_pose):
            return

        if not self._grid_map.integrate_scan(scan, map_pose):
            self.get_logger().warn(
                'Skipping scan: robot pose is outside the map bounds',
                throttle_duration_sec=2.0)

    def _publish_scan_match(self, map_pose: Pose2D, stamp) -> None:
        # Fase 3: escalar covarianza inversamente al score del matcher.
        # score alto (match limpio)  → cov_scale cercano a 1 → más influencia en EKF
        # score bajo (match marginal) → cov_scale cercano a max_scale → menos influencia
        if self._scan_match_cov_by_score:
            score = max(self._scan_matcher.last_score, self._scan_match_min_score)
            # cov_scale = min_score / score → 1.0 cuando score == min_score
            #                               → < 1.0 cuando score > min_score (mejor match)
            # Invertimos para que mejor score = cov más pequeña:
            raw_scale = self._scan_match_min_score / score
            # Clamp: [1/max_scale, 1.0] — nunca reduce más allá de lo configurado
            cov_scale = max(1.0 / self._scan_match_max_cov_scale, min(raw_scale, 1.0))
        else:
            cov_scale = 1.0

        cov_xy    = self._scan_match_cov_xy    * cov_scale
        cov_theta = self._scan_match_cov_theta * cov_scale

        msg = PoseWithCovarianceStamped()
        msg.header.stamp    = stamp
        msg.header.frame_id = self._map_frame
        msg.pose.pose.position.x    = map_pose.x
        msg.pose.pose.position.y    = map_pose.y
        msg.pose.pose.orientation.z = math.sin(map_pose.yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(map_pose.yaw / 2.0)
        cov = [0.0] * 36
        cov[0]  = cov_xy
        cov[7]  = cov_xy
        cov[35] = cov_theta
        msg.pose.covariance = cov
        self._pub_scan_match.publish(msg)
        self.get_logger().debug(
            f'scan_match pub: score={self._scan_matcher.last_score:.1f} '
            f'cov_scale={cov_scale:.3f} '
            f'cov_xy={cov_xy:.4f} cov_th={cov_theta:.4f}')

    def _odom_to_map_pose(self, odom_pose: Pose2D) -> Pose2D:
        """Aplica la corrección map→odom almacenada para obtener la pose en frame map.

        T_map_odom · odom_pose:
          mx   = x_tf + odom_x·cos(yaw_tf) − odom_y·sin(yaw_tf)
          my   = y_tf + odom_x·sin(yaw_tf) + odom_y·cos(yaw_tf)
          myaw = yaw_tf + odom_yaw
        """
        c = math.cos(self._map_odom_yaw)
        s = math.sin(self._map_odom_yaw)
        return Pose2D(
            self._map_odom_x + odom_pose.x * c - odom_pose.y * s,
            self._map_odom_y + odom_pose.x * s + odom_pose.y * c,
            math.atan2(
                math.sin(self._map_odom_yaw + odom_pose.yaw),
                math.cos(self._map_odom_yaw + odom_pose.yaw)),
        )

    def _update_map_odom_tf(self, map_pose: Pose2D, odom_pose: Pose2D) -> None:
        """Calcula la corrección map→odom a partir del par (pose_en_mapa, pose_en_odom).

        Derivación 2D (T_map_odom · odom_pose = map_pose):
          yaw_tf = yaw_m − yaw_o
          x_tf   = x_m − (x_o · cos(yaw_tf) − y_o · sin(yaw_tf))
          y_tf   = y_m − (x_o · sin(yaw_tf) + y_o · cos(yaw_tf))
        """
        yaw_tf = math.atan2(
            math.sin(map_pose.yaw - odom_pose.yaw),
            math.cos(map_pose.yaw - odom_pose.yaw))
        c = math.cos(yaw_tf)
        s = math.sin(yaw_tf)
        self._map_odom_x   = map_pose.x - (odom_pose.x * c - odom_pose.y * s)
        self._map_odom_y   = map_pose.y - (odom_pose.x * s + odom_pose.y * c)
        self._map_odom_yaw = yaw_tf

    def _publish_map(self) -> None:
        msg = self._grid_map.to_msg(
            stamp=self.get_clock().now().to_msg(),
            frame_id=self._map_frame,
        )
        self._pub_map.publish(msg)

    def _broadcast_tf(self) -> None:
        if not self._publish_map_odom_tf:
            return
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self._map_frame
        tf.child_frame_id = self._odom_frame
        tf.transform.translation.x = self._map_odom_x
        tf.transform.translation.y = self._map_odom_y
        tf.transform.rotation.z = math.sin(self._map_odom_yaw / 2.0)
        tf.transform.rotation.w = math.cos(self._map_odom_yaw / 2.0)
        self._tf.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = SlamNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        _save_map_on_exit(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _save_map_on_exit(node: SlamNode) -> None:
    if node._localization_only:
        return   # mapa pre-cargado, no sobreescribir
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'slam_map_{timestamp}.png'
    save_dir = os.environ.get('SLAM_MAP_DIR', os.getcwd())
    path = os.path.join(save_dir, filename)
    try:
        node._grid_map.to_png(path)
        node.get_logger().info(f'Map saved to: {path}')
    except Exception as e:
        node.get_logger().error(f'Failed to save map: {e}')


if __name__ == '__main__':
    main()
