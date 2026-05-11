"""
Monte Carlo Localisation (MCL) node.

Uses a pre-built PNG map (white=free, black=obstacle) to localise the robot
against laser scan data, producing a map→odom TF correction.

Inputs:
  /scan   sensor_msgs/LaserScan
  /odom   nav_msgs/Odometry

Outputs:
  /mcl/particles  geometry_msgs/PoseArray    (visualisation)
  /mcl/pose       geometry_msgs/PoseStamped  (best estimate)
  /mcl/map        nav_msgs/OccupancyGrid     (latched)
  TF  map → odom  (correction transform)

Parameters:
  map_path        str    path to grayscale PNG (white=free, black=wall)
  map_resolution  float  0.05  metres per pixel
  map_origin_x    float  world x at pixel col=0, row=height-1 (bottom-left)
  map_origin_y    float  world y at pixel col=0, row=height-1
  num_particles   int    500
  top_k           int    150   survivors kept per filter step
  noise_xy        float  0.05  [m]  motion noise std-dev
  noise_theta     float  0.05  [rad]
  score_rays      int    36    laser rays sampled per particle
  ray_step        float  0.025 [m]  ray-marching step
  hit_sigma       float  0.20  [m]  scan-likelihood Gaussian sigma
  map_frame       str    'map'
  odom_frame      str    'odom'
  laser_offset_x  float  0.0
  laser_offset_y  float  0.0
"""

import math
import os
import random

from PIL import Image
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, PoseArray, PoseStamped, Quaternion, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
import tf2_ros


# ── PNG loader ────────────────────────────────────────────────────────────────

def _load_png_grayscale(path):
    img = Image.open(path).convert('L')
    w, h = img.size
    return w, h, bytearray(img.tobytes())


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _yaw_from_quaternion(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def _quaternion_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _rotate_2d(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return x * c - y * s, x * s + y * c


# ── MCL node ──────────────────────────────────────────────────────────────────

class MCLNode(Node):

    def __init__(self):
        super().__init__('mcl')

        self.declare_parameter('map_path',
            os.path.join(os.path.dirname(__file__), 'maze_map.png'))
        self.declare_parameter('map_resolution', 0.05)
        self.declare_parameter('map_origin_x',  -5.54)
        self.declare_parameter('map_origin_y',  -8.10)
        self.declare_parameter('num_particles',  500)
        self.declare_parameter('top_k',          150)
        self.declare_parameter('noise_xy',        0.05)
        self.declare_parameter('noise_theta',     0.05)
        self.declare_parameter('score_rays',      36)
        self.declare_parameter('ray_step',         0.025)
        self.declare_parameter('hit_sigma',        0.20)
        self.declare_parameter('map_frame',       'map')
        self.declare_parameter('odom_frame',      'odom')
        self.declare_parameter('laser_offset_x',   0.0)
        self.declare_parameter('laser_offset_y',   0.0)

        self._res         = self.get_parameter('map_resolution').value
        self._orig_x      = self.get_parameter('map_origin_x').value
        self._orig_y      = self.get_parameter('map_origin_y').value
        self._n           = self.get_parameter('num_particles').value
        self._k           = self.get_parameter('top_k').value
        self._noise_xy    = self.get_parameter('noise_xy').value
        self._noise_th    = self.get_parameter('noise_theta').value
        self._n_rays      = self.get_parameter('score_rays').value
        self._ray_step    = self.get_parameter('ray_step').value
        self._hit_sigma   = self.get_parameter('hit_sigma').value
        self._map_frame   = self.get_parameter('map_frame').value
        self._odom_frame  = self.get_parameter('odom_frame').value
        self._lx          = self.get_parameter('laser_offset_x').value
        self._ly          = self.get_parameter('laser_offset_y').value

        map_path = self.get_parameter('map_path').value
        self._map_w, self._map_h, self._map = _load_png_grayscale(map_path)
        self.get_logger().info(
            f'MCL: loaded map {self._map_w}×{self._map_h} px from {map_path}')

        self._free_cells = []
        for row in range(self._map_h):
            for col in range(self._map_w):
                if self._map[row * self._map_w + col] > 127:
                    wx = self._orig_x + (col + 0.5) * self._res
                    wy = self._orig_y + (self._map_h - 1 - row + 0.5) * self._res
                    self._free_cells.append((wx, wy))

        self._particles: list = []
        self._prev_odom = None
        self._initialise_particles()

        qos_latched = rclpy.qos.QoSProfile(
            depth=1,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
        )
        self._pub_array = self.create_publisher(PoseArray,     '/mcl/particles', 10)
        self._pub_pose  = self.create_publisher(PoseStamped,   '/mcl/pose',      10)
        self._pub_map   = self.create_publisher(OccupancyGrid, '/mcl/map', qos_latched)
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self._map_timer_count = 0
        self._map_timer = self.create_timer(3.0, self._map_timer_cb)

        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)
        self.create_subscription(Odometry,  '/odom', self._odom_cb, 10)

        self.get_logger().info(
            f'MCL: N={self._n}, top_k={self._k}, '
            f'{len(self._free_cells)} free cells')

    # ── Map republisher ───────────────────────────────────────────────────

    def _map_timer_cb(self):
        self._publish_map()
        self._map_timer_count += 1
        if self._map_timer_count >= 10:
            self._map_timer.cancel()

    def _publish_map(self):
        grid = OccupancyGrid()
        grid.header.stamp.sec = 0
        grid.header.stamp.nanosec = 0
        grid.header.frame_id = self._map_frame
        grid.info.resolution = self._res
        grid.info.width      = self._map_w
        grid.info.height     = self._map_h
        grid.info.origin.position.x = self._orig_x
        grid.info.origin.position.y = self._orig_y
        grid.info.origin.orientation.w = 1.0
        data = []
        for row in range(self._map_h):
            png_row = self._map_h - 1 - row
            for col in range(self._map_w):
                data.append(0 if self._map[png_row * self._map_w + col] > 127 else 100)
        grid.data = data
        self._pub_map.publish(grid)

    # ── Particle management ───────────────────────────────────────────────

    def _initialise_particles(self):
        self._particles = []
        for wx, wy in random.choices(self._free_cells, k=self._n):
            self._particles.append([wx, wy, random.uniform(-math.pi, math.pi)])

    def _resample(self, survivors):
        new_p = list(survivors)
        while len(new_p) < self._n:
            b = random.choice(survivors)
            new_p.append([
                b[0] + random.gauss(0.0, self._noise_xy),
                b[1] + random.gauss(0.0, self._noise_xy),
                _wrap(b[2] + random.gauss(0.0, self._noise_th)),
            ])
        self._particles = new_p

    # ── Map helpers ───────────────────────────────────────────────────────

    def _map_value(self, wx, wy) -> int:
        col = int((wx - self._orig_x) / self._res)
        row = self._map_h - 1 - int((wy - self._orig_y) / self._res)
        if 0 <= col < self._map_w and 0 <= row < self._map_h:
            return self._map[row * self._map_w + col]
        return 0

    def _expected_range(self, sx, sy, angle, rmin, rmax) -> float:
        d = max(rmin, self._ray_step)
        while d <= rmax:
            if self._map_value(sx + d * math.cos(angle), sy + d * math.sin(angle)) <= 127:
                return d
            d += self._ray_step
        return rmax

    # ── Scoring ───────────────────────────────────────────────────────────

    def _score_particle(self, px, py, pth, scan: LaserScan) -> float:
        total = len(scan.ranges)
        if total == 0:
            return 0.0
        step = max(1, total // self._n_rays)
        sdx, sdy = _rotate_2d(self._lx, self._ly, pth)
        sx, sy = px + sdx, py + sdy
        score = 0.0
        for i in range(0, total, step):
            r = scan.ranges[i]
            if not math.isfinite(r) or r < scan.range_min or r > scan.range_max:
                continue
            angle = scan.angle_min + i * scan.angle_increment + pth
            expected = self._expected_range(sx, sy, angle, scan.range_min, scan.range_max)
            err = r - expected
            score += math.exp(-0.5 * (err / self._hit_sigma) ** 2)
        return score

    # ── Motion delta ──────────────────────────────────────────────────────

    def _compute_delta(self, prev: Odometry, curr: Odometry):
        x0 = prev.pose.pose.position.x
        y0 = prev.pose.pose.position.y
        h0 = _yaw_from_quaternion(prev.pose.pose.orientation)
        x1 = curr.pose.pose.position.x
        y1 = curr.pose.pose.position.y
        h1 = _yaw_from_quaternion(curr.pose.pose.orientation)
        dx_w, dy_w = x1 - x0, y1 - y0
        c0, s0 = math.cos(-h0), math.sin(-h0)
        return dx_w * c0 - dy_w * s0, dx_w * s0 + dy_w * c0, _wrap(h1 - h0)

    def _move_particles(self, dx_r, dy_r, dth):
        for p in self._particles:
            cp, sp = math.cos(p[2]), math.sin(p[2])
            p[0] += dx_r * cp - dy_r * sp + random.gauss(0.0, self._noise_xy)
            p[1] += dx_r * sp + dy_r * cp + random.gauss(0.0, self._noise_xy)
            p[2]  = _wrap(p[2] + dth + random.gauss(0.0, self._noise_th))

    def _broadcast_map_to_odom(self, best, stamp):
        if self._prev_odom is None:
            return
        ox = self._prev_odom.pose.pose.position.x
        oy = self._prev_odom.pose.pose.position.y
        oyaw = _yaw_from_quaternion(self._prev_odom.pose.pose.orientation)
        corr_yaw = _wrap(best[2] - oyaw)
        rx, ry = _rotate_2d(ox, oy, corr_yaw)
        tf = TransformStamped()
        tf.header.stamp    = stamp
        tf.header.frame_id = self._map_frame
        tf.child_frame_id  = self._odom_frame
        tf.transform.translation.x = best[0] - rx
        tf.transform.translation.y = best[1] - ry
        tf.transform.rotation = _quaternion_from_yaw(corr_yaw)
        self._tf_broadcaster.sendTransform(tf)

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        if self._prev_odom is None:
            self._prev_odom = msg
            return
        dx_r, dy_r, dth = self._compute_delta(self._prev_odom, msg)
        self._prev_odom = msg
        if self._particles:
            self._move_particles(dx_r, dy_r, dth)

    def _scan_cb(self, scan: LaserScan):
        if not self._particles:
            return
        scored = [(self._score_particle(p[0], p[1], p[2], scan), p)
                  for p in self._particles]
        scored.sort(key=lambda t: t[0], reverse=True)
        survivors = [p for _, p in scored[:self._k]]
        self._resample(survivors)
        stamp = scan.header.stamp
        self._publish_particles(stamp)
        best = self._publish_best_pose(survivors, stamp)
        if best is not None:
            self._broadcast_map_to_odom(best, stamp)

    # ── Publishing ────────────────────────────────────────────────────────

    def _publish_particles(self, stamp):
        msg = PoseArray()
        msg.header.stamp    = stamp
        msg.header.frame_id = self._map_frame
        for p in self._particles:
            pose = Pose()
            pose.position.x  = p[0]
            pose.position.y  = p[1]
            pose.orientation = _quaternion_from_yaw(p[2])
            msg.poses.append(pose)
        self._pub_array.publish(msg)

    def _publish_best_pose(self, survivors, stamp):
        if not survivors:
            return None
        mx = sum(p[0] for p in survivors) / len(survivors)
        my = sum(p[1] for p in survivors) / len(survivors)
        mth = math.atan2(
            sum(math.sin(p[2]) for p in survivors),
            sum(math.cos(p[2]) for p in survivors))
        msg = PoseStamped()
        msg.header.stamp    = stamp
        msg.header.frame_id = self._map_frame
        msg.pose.position.x  = mx
        msg.pose.position.y  = my
        msg.pose.orientation = _quaternion_from_yaw(mth)
        self._pub_pose.publish(msg)
        return [mx, my, mth]


def main(args=None):
    rclpy.init(args=args)
    node = MCLNode()
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
