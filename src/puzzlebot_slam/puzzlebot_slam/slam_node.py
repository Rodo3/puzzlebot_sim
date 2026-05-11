"""
Mapping SLAM node — log-odds occupancy grid from LiDAR + odometry.

Implements the standard occupancy grid mapping algorithm from
Thrun, Burgard, Fox (Probabilistic Robotics, ch. 9):

  For each beam z_t of the scan:
    Let z_t^max be the beam's max valid range.
    Walk Bresenham from the robot cell toward the beam endpoint.
    For each cell c on the line:
      if c is the endpoint AND z_t < z_t^max:
        l(c) += l_occ        # inverse_sensor_model: occupied
      else:
        l(c) += l_free       # inverse_sensor_model: free
    Clamp l(c) to [-l_max, +l_max].

Inputs:
  /scan   sensor_msgs/LaserScan
  /odom   nav_msgs/Odometry

Outputs:
  /map    nav_msgs/OccupancyGrid   (latched QoS)
  TF      map → odom               (identity)

OccupancyGrid convention (REP-105 / nav_msgs):
  - data is row-major: data[row * width + col]
  - row=0 corresponds to the BOTTOM of the map (lowest y)
  - cell(col, row) covers world coords:
      x in [origin.x + col*res,     origin.x + (col+1)*res]
      y in [origin.y + row*res,     origin.y + (row+1)*res]
"""

import math
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from sensor_msgs.msg import LaserScan
import tf2_ros


# ── Bresenham line ────────────────────────────────────────────────────────────

def _bresenham(x0: int, y0: int, x1: int, y1: int):
    """Yield integer cells from (x0,y0) to (x1,y1) inclusive."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


# ── Node ──────────────────────────────────────────────────────────────────────

class SlamNode(Node):

    def __init__(self):
        super().__init__('slam_node')

        # ── Map geometry ──
        self.declare_parameter('map_size_pixels', 500)
        self.declare_parameter('map_size_meters', 25.0)
        self.declare_parameter('map_origin_x',   -12.5)
        self.declare_parameter('map_origin_y',   -12.5)

        # ── Sensor model (log-odds) ──
        # Probabilities p_occ and p_free determine increments:
        #   l_occ  = log(p_occ  / (1 - p_occ))
        #   l_free = log(p_free / (1 - p_free))
        # Slightly conservative free-space updates keep small pose errors from
        # erasing walls that were already observed as occupied.
        self.declare_parameter('p_occ',  0.75)
        self.declare_parameter('p_free', 0.45)
        # Clamp keeps the filter responsive instead of saturating forever.
        self.declare_parameter('l_clamp', 5.0)

        # ── Beam processing ──
        self.declare_parameter('scan_step',       1)
        self.declare_parameter('max_range_factor', 0.95)
        # Ignore very-close beams (likely robot self-hits)
        self.declare_parameter('min_useful_range', 0.20)

        # ── Frames ──
        self.declare_parameter('map_frame',  'map')
        self.declare_parameter('odom_frame', 'odom')

        # ── Pose/scan timing ──
        self.declare_parameter('pose_buffer_sec',   3.0)
        self.declare_parameter('max_scan_pose_age', 0.20)

        # Resolve parameters
        self._px      = int(self.get_parameter('map_size_pixels').value)
        meters        = float(self.get_parameter('map_size_meters').value)
        self._orig_x  = float(self.get_parameter('map_origin_x').value)
        self._orig_y  = float(self.get_parameter('map_origin_y').value)
        self._res     = meters / self._px

        p_occ  = float(self.get_parameter('p_occ').value)
        p_free = float(self.get_parameter('p_free').value)
        self._l_occ  = math.log(p_occ  / (1.0 - p_occ))     # ~+0.85
        self._l_free = math.log(p_free / (1.0 - p_free))    # ~-0.40
        self._l_clamp = float(self.get_parameter('l_clamp').value)

        self._step      = int(self.get_parameter('scan_step').value)
        self._max_rangef = float(self.get_parameter('max_range_factor').value)
        self._min_range = float(self.get_parameter('min_useful_range').value)

        self._map_frame  = self.get_parameter('map_frame').value
        self._odom_frame = self.get_parameter('odom_frame').value
        self._pose_buffer_sec = float(self.get_parameter('pose_buffer_sec').value)
        self._max_pose_age    = float(self.get_parameter('max_scan_pose_age').value)

        # Log-odds grid, indexed [row, col]
        self._grid = np.zeros((self._px, self._px), dtype=np.float32)

        # Latest robot pose in odom frame, plus a short time-indexed history.
        self._have_pose = False
        self._rx = 0.0
        self._ry = 0.0
        self._ryaw = 0.0
        self._odom_buffer = deque()

        # ── Publishers / subscriptions ──
        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._pub_map = self.create_publisher(OccupancyGrid, '/map', map_qos)
        self._tf = tf2_ros.TransformBroadcaster(self)

        self.create_subscription(Odometry,  '/odom', self._odom_cb, 10)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)

        # Static map→odom TF at 30 Hz (high rate avoids RViz dropping odom msgs)
        self.create_timer(1.0 / 30.0, self._broadcast_tf)
        # Periodic map publish at 2 Hz (the data updates per-scan internally)
        self.create_timer(0.5, self._publish_map)

        self.get_logger().info(
            f'slam_node ready — {self._px}×{self._px} px @ {self._res:.3f} m/px, '
            f'origin=({self._orig_x:.2f}, {self._orig_y:.2f}), '
            f'l_occ={self._l_occ:+.2f} l_free={self._l_free:+.2f}')

    # ── World ↔ cell ────────────────────────────────────────────────────────

    def _world_to_cell(self, wx: float, wy: float):
        """REP-105: col grows with +x, row grows with +y. row 0 = bottom."""
        col = int(math.floor((wx - self._orig_x) / self._res))
        row = int(math.floor((wy - self._orig_y) / self._res))
        return col, row

    def _in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self._px and 0 <= row < self._px

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        return stamp.sec + stamp.nanosec * 1e-9

    @staticmethod
    def _angle_lerp(a0: float, a1: float, ratio: float) -> float:
        delta = math.atan2(math.sin(a1 - a0), math.cos(a1 - a0))
        return math.atan2(math.sin(a0 + ratio * delta),
                          math.cos(a0 + ratio * delta))

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        stamp = self._stamp_to_sec(msg.header.stamp)
        self._rx = msg.pose.pose.position.x
        self._ry = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._ryaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._have_pose = True
        self._odom_buffer.append((stamp, self._rx, self._ry, self._ryaw))

        cutoff = stamp - self._pose_buffer_sec
        while self._odom_buffer and self._odom_buffer[0][0] < cutoff:
            self._odom_buffer.popleft()

    def _lookup_pose(self, stamp) -> tuple | None:
        if not self._odom_buffer:
            return None

        t = self._stamp_to_sec(stamp)
        if t <= 0.0:
            return self._odom_buffer[-1][1:]

        first = self._odom_buffer[0]
        last = self._odom_buffer[-1]
        if t <= first[0]:
            return first[1:] if abs(first[0] - t) <= self._max_pose_age else None
        if t >= last[0]:
            return last[1:] if abs(t - last[0]) <= self._max_pose_age else None

        for i in range(1, len(self._odom_buffer)):
            before = self._odom_buffer[i - 1]
            after = self._odom_buffer[i]
            if before[0] <= t <= after[0]:
                dt = after[0] - before[0]
                if dt <= 1e-9:
                    return after[1:]
                ratio = (t - before[0]) / dt
                x = before[1] + ratio * (after[1] - before[1])
                y = before[2] + ratio * (after[2] - before[2])
                yaw = self._angle_lerp(before[3], after[3], ratio)
                return x, y, yaw
        return None

    def _scan_cb(self, scan: LaserScan):
        """Integrate a single scan into the log-odds grid."""
        if not self._have_pose:
            return
        pose = self._lookup_pose(scan.header.stamp)
        if pose is None:
            self.get_logger().warn(
                'Skipping scan: no odom pose close enough to scan timestamp',
                throttle_duration_sec=2.0)
            return
        self._integrate_scan(scan, pose)

    def _integrate_scan(self, scan: LaserScan, pose):
        rx, ry, ryaw = pose
        r_col, r_row = self._world_to_cell(rx, ry)
        if not self._in_bounds(r_col, r_row):
            return  # robot left the map — nothing useful to do

        rmin = max(self._min_range, scan.range_min)
        rmax = scan.range_max
        hit_threshold = rmax * self._max_rangef

        for i in range(0, len(scan.ranges), self._step):
            r = scan.ranges[i]
            if not math.isfinite(r) or r < rmin or r > rmax:
                continue

            is_hit = r < hit_threshold

            # Clip ray length to the map edge so Bresenham stays in bounds.
            # Endpoint in world frame:
            ang = scan.angle_min + i * scan.angle_increment + ryaw
            cos_a, sin_a = math.cos(ang), math.sin(ang)
            end_x = rx + r * cos_a
            end_y = ry + r * sin_a
            e_col, e_row = self._world_to_cell(end_x, end_y)

            # If endpoint is OOB, walk along the ray until we hit a border cell.
            if not self._in_bounds(e_col, e_row):
                e_col = max(0, min(self._px - 1, e_col))
                e_row = max(0, min(self._px - 1, e_row))
                is_hit = False  # clipped endpoint is NOT a real obstacle

            # Walk Bresenham — every cell except the last gets l_free.
            # If the ray is a real hit, the last cell gets l_occ instead.
            cells = list(_bresenham(r_col, r_row, e_col, e_row))
            if not cells:
                continue

            for col, row in cells[:-1]:
                if self._in_bounds(col, row):
                    v = self._grid[row, col] + self._l_free
                    if v < -self._l_clamp:
                        v = -self._l_clamp
                    self._grid[row, col] = v

            ec, er = cells[-1]
            if self._in_bounds(ec, er):
                if is_hit:
                    v = self._grid[er, ec] + self._l_occ
                    if v > self._l_clamp:
                        v = self._l_clamp
                    self._grid[er, ec] = v
                else:
                    v = self._grid[er, ec] + self._l_free
                    if v < -self._l_clamp:
                        v = -self._l_clamp
                    self._grid[er, ec] = v

    # ── Publishers ───────────────────────────────────────────────────────────

    def _publish_map(self):
        msg = OccupancyGrid()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self._map_frame

        msg.info = MapMetaData()
        msg.info.resolution = self._res
        msg.info.width      = self._px
        msg.info.height     = self._px
        msg.info.origin.position.x = self._orig_x
        msg.info.origin.position.y = self._orig_y
        msg.info.origin.orientation.w = 1.0

        # Convert log-odds → OccupancyGrid encoding.
        # Probability p = 1 - 1/(1+exp(l)).  We use thresholds on log-odds
        # directly so untouched cells (l=0 → p=0.5) become "unknown" (-1).
        flat = self._grid.flatten()
        data = np.full(flat.shape, -1, dtype=np.int8)
        data[flat >  0.5] = 100   # ≈ p > 0.62
        data[flat < -0.5] = 0     # ≈ p < 0.38
        msg.data = data.tolist()

        self._pub_map.publish(msg)

    def _broadcast_tf(self):
        tf = TransformStamped()
        tf.header.stamp    = self.get_clock().now().to_msg()
        tf.header.frame_id = self._map_frame
        tf.child_frame_id  = self._odom_frame
        tf.transform.rotation.w = 1.0
        self._tf.sendTransform(tf)


# ── Entry point ───────────────────────────────────────────────────────────────

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
