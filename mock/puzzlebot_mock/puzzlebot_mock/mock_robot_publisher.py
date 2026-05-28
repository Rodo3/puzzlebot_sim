"""
mock_robot_publisher.py — Simula todos los tópicos core del Puzzlebot.

Publica:
  /odom          nav_msgs/Odometry       10 Hz  — robot girando en círculo
  /scan          sensor_msgs/LaserScan   10 Hz  — LIDAR en cuarto 4x4m
  /cmd_vel       geometry_msgs/Twist     10 Hz  — velocidades del movimiento circular
  /cmd_vel_in    geometry_msgs/Twist     10 Hz  — igual con pequeño ruido (simula pre-avoidance)
  /map           nav_msgs/OccupancyGrid   1 Hz  — cuarto rectangular con paredes

Uso:
  ros2 run puzzlebot_mock mock_robot
"""
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Quaternion, Twist
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header

# Room dimensions (half-size from center, in meters)
ROOM_HALF = 2.0
# Circle the robot follows
CIRCLE_RADIUS = 0.8   # meters
ANGULAR_VEL   = 0.4   # rad/s  →  full circle in ~15.7 s
LINEAR_VEL    = CIRCLE_RADIUS * ANGULAR_VEL   # m/s

# Map parameters
MAP_RES    = 0.05  # m/cell → 5 cm resolution
MAP_SIZE   = int(ROOM_HALF * 2 / MAP_RES)  # 80 cells per side


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw / 2.0)
    q.z = math.sin(yaw / 2.0)
    return q


def ray_distance(rx: float, ry: float, angle: float) -> float:
    """Distance from robot at (rx, ry) to the nearest room wall at given world angle."""
    dx, dy = math.cos(angle), math.sin(angle)
    candidates = []

    if dx > 1e-9:
        candidates.append(( ROOM_HALF - rx) / dx)
    elif dx < -1e-9:
        candidates.append((-ROOM_HALF - rx) / dx)

    if dy > 1e-9:
        candidates.append(( ROOM_HALF - ry) / dy)
    elif dy < -1e-9:
        candidates.append((-ROOM_HALF - ry) / dy)

    positives = [d for d in candidates if d > 0]
    return min(positives) if positives else 3.5


def build_map() -> OccupancyGrid:
    """Static occupancy grid: free center, walls on the perimeter."""
    grid = OccupancyGrid()
    grid.info = MapMetaData()
    grid.info.resolution = MAP_RES
    grid.info.width  = MAP_SIZE
    grid.info.height = MAP_SIZE
    grid.info.origin.position.x = -ROOM_HALF
    grid.info.origin.position.y = -ROOM_HALF
    grid.info.origin.orientation.w = 1.0

    wall_cells = 2  # wall thickness in cells

    data = [0] * (MAP_SIZE * MAP_SIZE)
    for row in range(MAP_SIZE):
        for col in range(MAP_SIZE):
            is_wall = (
                row < wall_cells or row >= MAP_SIZE - wall_cells or
                col < wall_cells or col >= MAP_SIZE - wall_cells
            )
            data[row * MAP_SIZE + col] = 100 if is_wall else 0

    grid.data = data
    return grid


class MockRobotPublisher(Node):
    def __init__(self):
        super().__init__('mock_robot_publisher')

        self._pub_odom   = self.create_publisher(Odometry,      '/odom',        10)
        self._pub_scan   = self.create_publisher(LaserScan,      '/scan',        10)
        self._pub_vel    = self.create_publisher(Twist,          '/cmd_vel',     10)
        self._pub_vel_in = self.create_publisher(Twist,          '/cmd_vel_in',  10)
        self._pub_map    = self.create_publisher(OccupancyGrid,  '/map',          1)

        self._static_map = build_map()
        self._t0 = time.time()

        # 10 Hz main loop
        self.create_timer(0.1,  self._publish_robot_state)
        # 1 Hz map (static, but bridge expects it)
        self.create_timer(1.0,  self._publish_map)

        self.get_logger().info(
            'mock_robot_publisher started — robot moves in a %.1f m radius circle', CIRCLE_RADIUS
        )

    def _now(self) -> float:
        return time.time() - self._t0

    def _robot_pose(self, t: float):
        """Returns (x, y, yaw) for time t."""
        yaw = ANGULAR_VEL * t
        x   = CIRCLE_RADIUS * math.cos(yaw)
        y   = CIRCLE_RADIUS * math.sin(yaw)
        # Robot heading is tangent to the circle (90° offset from radial direction)
        heading = yaw + math.pi / 2.0
        return x, y, heading

    def _ros_stamp(self):
        return self.get_clock().now().to_msg()

    def _publish_robot_state(self):
        t = self._now()
        x, y, yaw = self._robot_pose(t)
        stamp = self._ros_stamp()

        # ── Odometry ──────────────────────────────────────────────────────
        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base_footprint'

        odom.pose.pose.position.x    = x
        odom.pose.pose.position.y    = y
        odom.pose.pose.orientation   = yaw_to_quaternion(yaw)
        odom.twist.twist.linear.x    = LINEAR_VEL
        odom.twist.twist.angular.z   = ANGULAR_VEL

        self._pub_odom.publish(odom)

        # ── cmd_vel ───────────────────────────────────────────────────────
        vel = Twist()
        vel.linear.x  = LINEAR_VEL
        vel.angular.z = ANGULAR_VEL
        self._pub_vel.publish(vel)

        # ── cmd_vel_in (add small noise to simulate pre-avoidance command) ─
        vel_in = Twist()
        vel_in.linear.x  = LINEAR_VEL  + np.random.uniform(-0.02, 0.02)
        vel_in.angular.z = ANGULAR_VEL + np.random.uniform(-0.05, 0.05)
        self._pub_vel_in.publish(vel_in)

        # ── LaserScan ─────────────────────────────────────────────────────
        n_rays      = 360
        angle_min   = -math.pi
        angle_max   =  math.pi
        angle_step  = (angle_max - angle_min) / n_rays

        ranges = []
        for i in range(n_rays):
            world_angle = yaw + angle_min + i * angle_step
            d = ray_distance(x, y, world_angle)
            # Add small noise to simulate real LIDAR
            d += np.random.uniform(-0.02, 0.02)
            ranges.append(float(np.clip(d, 0.12, 3.5)))

        scan = LaserScan()
        scan.header.stamp    = stamp
        scan.header.frame_id = 'base_scan'
        scan.angle_min       = angle_min
        scan.angle_max       = angle_max
        scan.angle_increment = angle_step
        scan.time_increment  = 0.0
        scan.range_min       = 0.12
        scan.range_max       = 3.5
        scan.ranges          = ranges

        self._pub_scan.publish(scan)

    def _publish_map(self):
        self._static_map.header.stamp    = self._ros_stamp()
        self._static_map.header.frame_id = 'map'
        self._static_map.info.map_load_time = self._ros_stamp()
        self._pub_map.publish(self._static_map)


def main(args=None):
    rclpy.init(args=args)
    node = MockRobotPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
