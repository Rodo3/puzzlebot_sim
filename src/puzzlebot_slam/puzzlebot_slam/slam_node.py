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
"""

import math

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
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


class SlamNode(Node):
    def __init__(self):
        super().__init__('slam_node')

        self._declare_parameters()
        self._map_frame = self.get_parameter('map_frame').value
        self._odom_frame = self.get_parameter('odom_frame').value

        self._grid_map = OccupancyGridMap(
            size_pixels=int(self.get_parameter('map_size_pixels').value),
            size_meters=float(self.get_parameter('map_size_meters').value),
            origin_x=float(self.get_parameter('map_origin_x').value),
            origin_y=float(self.get_parameter('map_origin_y').value),
            p_occ=float(self.get_parameter('p_occ').value),
            p_free=float(self.get_parameter('p_free').value),
            l_clamp=float(self.get_parameter('l_clamp').value),
            scan_step=int(self.get_parameter('scan_step').value),
            max_range_factor=float(self.get_parameter('max_range_factor').value),
            min_useful_range=float(self.get_parameter('min_useful_range').value),
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
        self._scan_matcher = LocalScanMatcher(
            enabled=bool(self.get_parameter('scan_matching_enabled').value),
        )

        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._pub_map = self.create_publisher(OccupancyGrid, '/map', map_qos)
        self._tf = tf2_ros.TransformBroadcaster(self)

        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(
            LaserScan, '/scan', self._scan_cb, qos_profile_sensor_data)

        self.create_timer(1.0 / 30.0, self._broadcast_tf)
        self.create_timer(0.5, self._publish_map)

        self.get_logger().info(
            'slam_node ready — '
            f'{self._grid_map.size_pixels}×{self._grid_map.size_pixels} px '
            f'@ {self._grid_map.resolution:.3f} m/px, '
            f'origin=({self._grid_map.origin_x:.2f}, {self._grid_map.origin_y:.2f}), '
            f'l_occ={self._grid_map.l_occ:+.2f} '
            f'l_free={self._grid_map.l_free:+.2f}, '
            f'scan_matching={self._scan_matcher.enabled}')

    def _declare_parameters(self) -> None:
        self.declare_parameter('map_size_pixels', 500)
        self.declare_parameter('map_size_meters', 25.0)
        self.declare_parameter('map_origin_x', -12.5)
        self.declare_parameter('map_origin_y', -12.5)

        self.declare_parameter('p_occ', 0.75)
        self.declare_parameter('p_free', 0.45)
        self.declare_parameter('l_clamp', 5.0)

        self.declare_parameter('scan_step', 1)
        self.declare_parameter('max_range_factor', 0.95)
        self.declare_parameter('min_useful_range', 0.20)

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')

        self.declare_parameter('pose_buffer_sec', 3.0)
        self.declare_parameter('max_scan_pose_age', 0.20)

        self.declare_parameter('use_keyframes', False)
        self.declare_parameter('keyframe_min_translation', 0.10)
        self.declare_parameter('keyframe_min_rotation', math.radians(5.0))

        self.declare_parameter('scan_matching_enabled', False)

    def _odom_cb(self, msg: Odometry) -> None:
        self._odom_buffer.add(msg)

    def _scan_cb(self, scan: LaserScan) -> None:
        odom_pose = self._odom_buffer.lookup(scan.header.stamp)
        if odom_pose is None:
            self.get_logger().warn(
                'Skipping scan: no odom pose close enough to scan timestamp',
                throttle_duration_sec=2.0)
            return

        map_pose = self._scan_matcher.match(scan, odom_pose, self._grid_map)
        if not self._keyframes.should_integrate(map_pose):
            return

        if not self._grid_map.integrate_scan(scan, map_pose):
            self.get_logger().warn(
                'Skipping scan: robot pose is outside the map bounds',
                throttle_duration_sec=2.0)

    def _publish_map(self) -> None:
        msg = self._grid_map.to_msg(
            stamp=self.get_clock().now().to_msg(),
            frame_id=self._map_frame,
        )
        self._pub_map.publish(msg)

    def _broadcast_tf(self) -> None:
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self._map_frame
        tf.child_frame_id = self._odom_frame
        tf.transform.rotation.w = 1.0
        self._tf.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = SlamNode()
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
