import math

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class ObstacleAvoidanceNode(Node):
    """
    Reactive obstacle avoidance + localization-uncertainty safety layer.

    Receives /cmd_vel_in from the steering controller and /scan from the LiDAR.
    Reads /odom covariance to scale velocity when localization is uncertain.

    Priority (highest to lowest):
      1. EMERGENCY  — obstacle within stop_distance → full stop (ignores all else)
      2. UNCERTAIN  — LiDAR slow zone OR trace(P_xy) > cov_slow_threshold → scale speed
      3. DEGRADED   — trace(P_xy) > cov_stop_threshold → full stop (lost localization)
      4. NORMAL     — pass /cmd_vel_in through unchanged
    """

    def __init__(self):
        super().__init__('obstacle_avoidance_node')

        # ── LiDAR safety parameters ──────────────────────────────────────────
        self.declare_parameter('stop_distance',   0.30)   # metres — full stop
        self.declare_parameter('slow_distance',   0.60)   # metres — scale speed
        self.declare_parameter('front_angle_deg', 30.0)   # half-cone ahead

        # ── Localization uncertainty parameters (Fase 5) ─────────────────────
        # trace(P_xy) = odom.pose.covariance[0] + odom.pose.covariance[7]
        #
        # cov_slow_threshold: trace(P_xy) above this → start reducing speed.
        #   Typical: 0.15 m² (σ ≈ 38 cm combined) — pose getting uncertain.
        #   Start conservative (0.50) and lower after tuning.
        #
        # cov_stop_threshold: trace(P_xy) above this → full stop.
        #   Typical: 0.80 m² (σ ≈ 63 cm combined) — localization lost.
        #   Must be > cov_slow_threshold.
        #
        # cov_timeout_sec: if no /odom message received for this long, treat as
        #   degraded (localization source died).
        self.declare_parameter('cov_slow_threshold', 0.15)   # [m²] 
        self.declare_parameter('cov_stop_threshold', 0.8)#²] 80     
        self.declare_parameter('cov_timeout_sec',    2.0)    # [s]

        self.declare_parameter('stuck_timeout_sec',  3.0)   # [s] bloqueado → permitir retroceso
        self.declare_parameter('reverse_speed',      0.07)  # [m/s] velocidad de retroceso
        self.declare_parameter('reverse_duration_sec', 2.0) # [s] duración del retroceso

        self.stop_d          = self.get_parameter('stop_distance').value
        self.slow_d          = self.get_parameter('slow_distance').value
        self.front_a         = math.radians(self.get_parameter('front_angle_deg').value)
        self.cov_slow        = self.get_parameter('cov_slow_threshold').value
        self.cov_stop        = self.get_parameter('cov_stop_threshold').value
        self.cov_timeout     = self.get_parameter('cov_timeout_sec').value
        self.stuck_timeout   = self.get_parameter('stuck_timeout_sec').value
        self.reverse_speed   = self.get_parameter('reverse_speed').value
        self.reverse_dur     = self.get_parameter('reverse_duration_sec').value

        self.min_front       = float('inf')
        self.trace_p_xy      = 0.0
        self._last_odom_t    = None

        # Seguimiento de tiempo bloqueado para activar retroceso
        self._blocked_since  = None   # monotonic timestamp cuando empezó emergency stop
        self._reversing_since = None  # cuando empezó el retroceso activo

        # ── Subscriptions ────────────────────────────────────────────────────
        # Teleop priority: if a recent message arrives on /cmd_vel_teleop, navigation
        # commands are suppressed so the bridge's direct cmd_vel wins uncontested.
        self.declare_parameter('teleop_timeout_sec', 1.5)
        self._teleop_timeout = self.get_parameter('teleop_timeout_sec').value
        self._last_teleop_t  = None

        self.sub_scan_   = self.create_subscription(
            LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.sub_cmd_    = self.create_subscription(
            Twist, '/cmd_vel_in', self.cmd_cb, 10)
        self.sub_odom_   = self.create_subscription(
            Odometry, '/odom', self.odom_cb, 10)
        self.sub_teleop_ = self.create_subscription(
            Twist, '/cmd_vel_teleop', self._teleop_cb, 10)

        self.pub_cmd_  = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info(
            f'obstacle_avoidance_node started '
            f'(stop={self.stop_d} m, slow={self.slow_d} m, '
            f'cov_slow={self.cov_slow} m², cov_stop={self.cov_stop} m²)')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _teleop_cb(self, msg: Twist):
        import time as _time
        self._last_teleop_t = _time.monotonic()

    def _teleop_active(self):
        if self._last_teleop_t is None:
            return False
        import time as _time
        return (_time.monotonic() - self._last_teleop_t) < self._teleop_timeout

    def odom_cb(self, msg: Odometry):
        self.trace_p_xy  = msg.pose.covariance[0] + msg.pose.covariance[7]
        self._last_odom_t = self.get_clock().now()

    def scan_cb(self, msg: LaserScan):
        angles = np.arange(len(msg.ranges)) * msg.angle_increment + msg.angle_min
        ranges = np.array(msg.ranges)
        valid  = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
        front  = np.abs(angles) <= self.front_a
        mask   = valid & front
        self.min_front = float(np.min(ranges[mask])) if mask.any() else float('inf')

    def cmd_cb(self, msg: Twist):
        # Teleop has priority: suppress navigation commands while user is driving
        if self._teleop_active():
            return
        out = Twist()
        import time as _time
        now_mono = _time.monotonic()

        # ── Priority 1: LiDAR emergency stop con retroceso automático ────────
        if self.min_front <= self.stop_d:
            # Registrar cuándo empezó el bloqueo
            if self._blocked_since is None:
                self._blocked_since = now_mono

            blocked_secs = now_mono - self._blocked_since

            # Si llevamos mucho tiempo bloqueados → activar retroceso
            if blocked_secs >= self.stuck_timeout:
                if self._reversing_since is None:
                    self._reversing_since = now_mono
                    self.get_logger().warn(
                        f'STUCK {blocked_secs:.1f}s → iniciando retroceso',
                        throttle_duration_sec=1.0)

                reverse_elapsed = now_mono - self._reversing_since
                if reverse_elapsed < self.reverse_dur:
                    # Retroceder con giro suave
                    out.linear.x  = -self.reverse_speed
                    out.angular.z = 0.20   # giro suave mientras retrocede
                    self.pub_cmd_.publish(out)
                    self.get_logger().info(
                        f'REVERSING {reverse_elapsed:.1f}s/{self.reverse_dur}s',
                        throttle_duration_sec=0.5)
                    return
                else:
                    # Retroceso completado — resetear para permitir navegación normal
                    self.get_logger().info('Retroceso completado → reanudando navegación')
                    self._blocked_since   = None
                    self._reversing_since = None
                    # Publicar Twist vacío un ciclo y dejar que el siguiente cmd pase
                    self.pub_cmd_.publish(out)
                    return
            else:
                # Aún no cumple el timeout → parada normal
                self._reversing_since = None
                self.pub_cmd_.publish(out)
                self.get_logger().warn(
                    f'EMERGENCY STOP — obstacle at {self.min_front:.2f} m '
                    f'(bloqueado {blocked_secs:.1f}s/{self.stuck_timeout}s)',
                    throttle_duration_sec=1.0)
                return
        else:
            # Frontal libre — resetear contadores de bloqueo
            self._blocked_since   = None
            self._reversing_since = None

        # ── Priority 2: Localization timeout (odom source dead) ───────────────
        odom_age = self._odom_age_sec()
        if odom_age is not None and odom_age > self.cov_timeout:
            self.pub_cmd_.publish(out)
            self.get_logger().warn(
                f'LOCALIZATION TIMEOUT — no /odom for {odom_age:.1f} s → parado',
                throttle_duration_sec=2.0)
            return

        # ── Priority 3: Localization uncertainty stop ─────────────────────────
        if self.trace_p_xy > self.cov_stop:
            self.pub_cmd_.publish(out)
            self.get_logger().warn(
                f'LOCALIZATION LOST — trace(P_xy)={self.trace_p_xy:.3f} m² '
                f'> {self.cov_stop} m² → parado',
                throttle_duration_sec=2.0)
            return

        # ── Compute combined scale factor ─────────────────────────────────────
        # Two independent scale sources; take the more restrictive one.

        # Source A: LiDAR slow zone
        if self.min_front <= self.slow_d and msg.linear.x > 0:
            lidar_scale = (self.min_front - self.stop_d) / (self.slow_d - self.stop_d)
            lidar_scale = max(0.0, min(lidar_scale, 1.0))
        else:
            lidar_scale = 1.0

        # Source B: EKF covariance slow zone
        if self.trace_p_xy > self.cov_slow:
            span = max(self.cov_stop - self.cov_slow, 1e-6)
            cov_scale = 1.0 - (self.trace_p_xy - self.cov_slow) / span
            cov_scale = max(0.0, min(cov_scale, 1.0))
            self.get_logger().info(
                f'UNCERTAIN — trace(P_xy)={self.trace_p_xy:.3f} m² '
                f'→ vel_scale={cov_scale:.2f}',
                throttle_duration_sec=2.0)
        else:
            cov_scale = 1.0

        scale = min(lidar_scale, cov_scale)

        # ── Apply scale and publish ───────────────────────────────────────────
        if scale < 1.0:
            out.linear.x  = msg.linear.x  * scale
            out.angular.z = msg.angular.z  * scale
        else:
            out = msg
        self.pub_cmd_.publish(out)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _odom_age_sec(self):
        if self._last_odom_t is None:
            return None
        return (self.get_clock().now() - self._last_odom_t).nanoseconds * 1e-9


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidanceNode()
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
