import math
import random

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from puzzlebot_msgs.msg import ParticleArray


# ---------------------------------------------------------------------------
# Particle filter helpers
# ---------------------------------------------------------------------------

def _systematic_resample(particles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return resampled particle array using systematic resampling."""
    N = len(weights)
    positions = (np.arange(N) + random.random()) / N
    cumsum = np.cumsum(weights)
    indices = np.searchsorted(cumsum, positions)
    return particles[indices].copy()


def _score_particle(pose: np.ndarray, scan: LaserScan,
                    log_odds: np.ndarray, map_pixels: int,
                    resolution: float) -> float:
    """Rate a particle against the current occupancy map."""
    score = 1e-300
    step = max(1, len(scan.ranges) // 36)
    for i in range(0, len(scan.ranges), step):
        r = scan.ranges[i]
        if not math.isfinite(r) or r < scan.range_min or r > scan.range_max:
            continue
        angle = scan.angle_min + i * scan.angle_increment
        wx = pose[0] + r * math.cos(pose[2] + angle)
        wy = pose[1] + r * math.sin(pose[2] + angle)
        cx = int(wx / resolution)
        cy = int(wy / resolution)
        if 0 <= cx < map_pixels and 0 <= cy < map_pixels:
            lo = log_odds[cy, cx]
            if lo > 0:
                score += float(lo)
    return score


def _bresenham(x0, y0, x1, y1):
    """Yield (x, y) cells along the line from (x0,y0) to (x1,y1)."""
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class SlamNode(Node):
    def __init__(self):
        super().__init__('slam_node')

        self.declare_parameter('map_size_pixels',  800)
        self.declare_parameter('map_size_meters',  16.0)
        self.declare_parameter('n_particles',      200)
        self.declare_parameter('particle_noise_x', 0.02)
        self.declare_parameter('particle_noise_y', 0.02)
        self.declare_parameter('particle_noise_t', 0.01)
        self.declare_parameter('update_frequency', 5.0)

        self.map_pixels  = self.get_parameter('map_size_pixels').as_parameter_event().value \
                           if False else self.get_parameter('map_size_pixels').value
        self.map_pixels  = self.get_parameter('map_size_pixels').value
        self.map_meters  = self.get_parameter('map_size_meters').value
        self.n_particles = self.get_parameter('n_particles').value
        self.noise       = np.array([
            self.get_parameter('particle_noise_x').value,
            self.get_parameter('particle_noise_y').value,
            self.get_parameter('particle_noise_t').value,
        ])
        self.resolution  = self.map_meters / self.map_pixels

        # Particles: [x, y, theta] array, separate weights array
        self.particles = np.zeros((self.n_particles, 3))
        self.particles[:, 0] = np.random.uniform(0, self.map_meters, self.n_particles)
        self.particles[:, 1] = np.random.uniform(0, self.map_meters, self.n_particles)
        self.particles[:, 2] = np.random.uniform(-math.pi, math.pi, self.n_particles)
        self.weights   = np.ones(self.n_particles) / self.n_particles

        # Log-odds map
        self.log_odds = np.zeros((self.map_pixels, self.map_pixels), dtype=np.float32)

        # Pending motion from odometry
        self.pending_dd  = 0.0
        self.pending_dth = 0.0
        self.last_odom_t = None
        self.latest_scan = None

        # Subscriptions
        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self._scan_cb, 10)
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self._odom_cb, 10)

        # Publishers
        self.pub_map        = self.create_publisher(OccupancyGrid, '/map', 1)
        self.pub_particles  = self.create_publisher(ParticleArray, '/particles', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        hz = self.get_parameter('update_frequency').value
        self.create_timer(1.0 / hz, self._update)

        self.get_logger().info(
            f'slam_node started ({self.n_particles} particles, '
            f'{self.map_pixels}x{self.map_pixels} map @ {self.resolution:.3f} m/px)')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _scan_cb(self, msg: LaserScan):
        self.latest_scan = msg

    def _odom_cb(self, msg: Odometry):
        t = rclpy.time.Time.from_msg(msg.header.stamp)
        if self.last_odom_t is None:
            self.last_odom_t = t
            return
        dt = (t - self.last_odom_t).nanoseconds * 1e-9
        self.last_odom_t = t
        if dt <= 0.0 or dt > 1.0:
            return
        self.pending_dd  += msg.twist.twist.linear.x  * dt
        self.pending_dth += msg.twist.twist.angular.z * dt

    # ------------------------------------------------------------------
    # MCL update cycle
    # ------------------------------------------------------------------

    def _update(self):
        if self.latest_scan is None:
            return

        self._predict(self.pending_dd, self.pending_dth)
        self.pending_dd = self.pending_dth = 0.0

        self._weight(self.latest_scan)
        self.particles = _systematic_resample(self.particles, self.weights)
        self.weights[:] = 1.0 / self.n_particles

        self._update_map(self.latest_scan)
        self._publish_map()
        self._publish_particles()
        self._broadcast_map_tf()

    def _predict(self, dd: float, dth: float):
        noise = np.random.randn(self.n_particles, 3) * self.noise
        self.particles[:, 0] += dd * np.cos(self.particles[:, 2]) + noise[:, 0]
        self.particles[:, 1] += dd * np.sin(self.particles[:, 2]) + noise[:, 1]
        self.particles[:, 2] += dth + noise[:, 2]

    def _weight(self, scan: LaserScan):
        for i, p in enumerate(self.particles):
            self.weights[i] = _score_particle(
                p, scan, self.log_odds, self.map_pixels, self.resolution)
        self.weights += 1e-300
        self.weights /= self.weights.sum()

    def _update_map(self, scan: LaserScan):
        best_idx = int(np.argmax(self.weights))
        pose = self.particles[best_idx]
        L_OCC, L_FREE = 0.85, -0.40

        rx = int(pose[0] / self.resolution)
        ry = int(pose[1] / self.resolution)

        for i in range(0, len(scan.ranges), 2):
            r = scan.ranges[i]
            if not math.isfinite(r) or r < scan.range_min or r > scan.range_max:
                continue
            angle = scan.angle_min + i * scan.angle_increment
            ex = int((pose[0] + r * math.cos(pose[2] + angle)) / self.resolution)
            ey = int((pose[1] + r * math.sin(pose[2] + angle)) / self.resolution)

            for cx, cy in _bresenham(rx, ry, ex, ey):
                if 0 <= cx < self.map_pixels and 0 <= cy < self.map_pixels:
                    self.log_odds[cy, cx] = np.clip(
                        self.log_odds[cy, cx] + L_FREE, -5.0, 5.0)

            if 0 <= ex < self.map_pixels and 0 <= ey < self.map_pixels:
                self.log_odds[ey, ex] = np.clip(
                    self.log_odds[ey, ex] + L_OCC, -5.0, 5.0)

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------

    def _publish_map(self):
        msg = OccupancyGrid()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info = MapMetaData()
        msg.info.resolution = self.resolution
        msg.info.width      = self.map_pixels
        msg.info.height     = self.map_pixels

        flat = self.log_odds.flatten()
        data = np.where(flat > 0.1, 100, np.where(flat < -0.1, 0, -1)).astype(np.int8)
        msg.data = data.tolist()
        self.pub_map.publish(msg)

    def _publish_particles(self):
        msg = ParticleArray()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.num_particles   = self.n_particles
        flat = np.column_stack([
            self.particles,
            self.weights.reshape(-1, 1)
        ]).astype(np.float32).flatten()
        msg.data = flat.tolist()
        self.pub_particles.publish(msg)

    def _broadcast_map_tf(self):
        tf = TransformStamped()
        tf.header.stamp    = self.get_clock().now().to_msg()
        tf.header.frame_id = 'map'
        tf.child_frame_id  = 'odom'
        tf.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(tf)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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

