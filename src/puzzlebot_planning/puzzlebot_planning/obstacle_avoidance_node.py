"""
obstacle_avoidance_node.py — Capa de seguridad reactiva ante obstáculos y localización.

POSICIÓN EN EL PIPELINE:
  steering_controller_node  →  /cmd_vel_in  →  [ESTE NODO]  →  /cmd_vel  →  robot/Gazebo

FUNCIÓN:
  Intercepta el comando de velocidad del controlador antes de enviarlo al robot.
  Combina dos fuentes de información para decidir si el comando es seguro:

    1. LiDAR (/scan): obstáculos físicos detectados en tiempo real.
    2. EKF covarianza (/odom): incertidumbre de localización del robot.

  Prioridades (de mayor a menor):
    1. EMERGENCY   — obstáculo < stop_distance → freno total + retroceso si lleva mucho tiempo bloqueado
    2. LOC_TIMEOUT — no llega /odom en > cov_timeout_sec → freno total
    3. LOC_LOST    — trace(P_xy) > cov_stop_threshold → freno total (localización perdida)
    4. SLOW        — zona LiDAR entre stop/slow O trace(P_xy) > cov_slow_threshold → escala velocidad
    5. NORMAL      — pasa el comando sin modificar

TOPICS SUSCRITOS:
  /cmd_vel_in  (geometry_msgs/Twist)   — velocidad calculada por el controlador
  /scan        (sensor_msgs/LaserScan) — datos del LiDAR
  /odom        (nav_msgs/Odometry)     — covarianza del EKF (para monitoreo de localización)

TOPICS PUBLICADOS:
  /cmd_vel     (geometry_msgs/Twist)   — velocidad final hacia el robot

PARÁMETROS:
  stop_distance        [0.30 m]   — distancia mínima al obstáculo; por debajo → freno total
  slow_distance        [0.60 m]   — inicio del frenado gradual por LiDAR
  front_angle_deg      [30.0°]    — semi-ángulo del cono frontal monitoreado
  cov_slow_threshold   [0.15 m²]  — trace(P_xy) arriba de este → frenado gradual
  cov_stop_threshold   [0.80 m²]  — trace(P_xy) arriba de este → freno total
  cov_timeout_sec      [2.0 s]    — sin /odom por este tiempo → freno total
  stuck_timeout_sec    [3.0 s]    — bloqueado más de esto → activa retroceso automático
  reverse_speed        [0.07 m/s] — velocidad de retroceso
  reverse_duration_sec [2.0 s]    — duración del retroceso
"""

import math
import time as _time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool


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

        # ── Localization uncertainty parameters ──────────────────────────────
        # trace(P_xy) = odom.pose.covariance[0] + odom.pose.covariance[7]
        # cov_slow_threshold: por encima → inicio de frenado gradual (σ ≈ 38 cm)
        # cov_stop_threshold: por encima → freno total (localización perdida, σ ≈ 63 cm)
        self.declare_parameter('cov_slow_threshold', 0.15)
        self.declare_parameter('cov_stop_threshold', 0.80)
        self.declare_parameter('cov_timeout_sec',    2.0)

        # ── Anti-stuck: retroceso automático cuando lleva mucho tiempo bloqueado
        self.declare_parameter('stuck_timeout_sec',    3.0)
        self.declare_parameter('reverse_speed',        0.07)
        self.declare_parameter('reverse_duration_sec', 2.0)

        # ── Emergency stop: rotación permitida ──────────────────────────────────
        # En EMERGENCY STOP se frena SIEMPRE el avance (linear.x=0), pero si
        # emergency_allow_rotation=True se deja pasar el giro del comando entrante
        # (acotado a emergency_max_angular). Así el robot puede pivotear en su lugar
        # para orientarse hacia un hueco libre en vez de quedarse congelado contra
        # la pared. Poner en False para el comportamiento clásico (congelar todo).
        self.declare_parameter('emergency_allow_rotation', True)
        self.declare_parameter('emergency_max_angular',     1.0)  # [rad/s] clamp del giro

        self.stop_d          = self.get_parameter('stop_distance').value
        self.slow_d          = self.get_parameter('slow_distance').value
        self.front_a         = math.radians(self.get_parameter('front_angle_deg').value)
        self.cov_slow        = self.get_parameter('cov_slow_threshold').value
        self.cov_stop        = self.get_parameter('cov_stop_threshold').value
        self.cov_timeout     = self.get_parameter('cov_timeout_sec').value
        self.stuck_timeout   = self.get_parameter('stuck_timeout_sec').value
        self.reverse_speed   = self.get_parameter('reverse_speed').value
        self.reverse_dur     = self.get_parameter('reverse_duration_sec').value
        self.emerg_rotate    = self.get_parameter('emergency_allow_rotation').value
        self.emerg_max_ang   = self.get_parameter('emergency_max_angular').value

        # Distancia mínima al obstáculo más cercano en el cono frontal
        self.min_front      = float('inf')
        # trace(P_xy) del EKF: suma de varianzas de posición x e y
        self.trace_p_xy     = 0.0
        self._last_odom_t   = None
        self._last_teleop_t = None
        self._emergency_stop_active = False
        self.declare_parameter('teleop_timeout_sec', 0.5)
        self._teleop_timeout = self.get_parameter('teleop_timeout_sec').value

        # Timestamps para control del retroceso automático
        self._blocked_since   = None   # cuando empezó el emergency stop continuo
        self._reversing_since = None   # cuando empezó el retroceso activo

        # ── Subscriptions ────────────────────────────────────────────────────
        self.sub_scan_ = self.create_subscription(
            LaserScan, '/scan_stamped', self.scan_cb, qos_profile_sensor_data)
        self.sub_cmd_  = self.create_subscription(
            Twist, '/cmd_vel_in', self.cmd_cb, 10)
        self.sub_odom_ = self.create_subscription(
            Odometry, '/odom', self.odom_cb, 10)

        self.sub_teleop_ = self.create_subscription(
            Twist, '/cmd_vel_teleop', self._teleop_cb, 10)
        self.sub_emergency_ = self.create_subscription(
            Bool, '/emergency_stop', self._emergency_cb, 10)

        self.pub_cmd_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(0.05, self._emergency_loop)

        self.get_logger().info(
            f'obstacle_avoidance_node iniciado '
            f'(stop={self.stop_d} m, slow={self.slow_d} m, '
            f'cov_slow={self.cov_slow} m², cov_stop={self.cov_stop} m²)')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _teleop_cb(self, msg: Twist):
        self._last_teleop_t = _time.monotonic()
        if self._emergency_stop_active:
            self.pub_cmd_.publish(Twist())
            return
        self.pub_cmd_.publish(msg)

    def _emergency_cb(self, msg: Bool):
        self._emergency_stop_active = bool(msg.data)
        if self._emergency_stop_active:
            self.pub_cmd_.publish(Twist())

    def _emergency_loop(self):
        if self._emergency_stop_active:
            self.pub_cmd_.publish(Twist())

    def _teleop_active(self):
        if self._last_teleop_t is None:
            return False
        return (_time.monotonic() - self._last_teleop_t) < self._teleop_timeout

    def odom_cb(self, msg: Odometry):
        # Extrae la traza de la covarianza de posición xy del EKF
        self.trace_p_xy   = msg.pose.covariance[0] + msg.pose.covariance[7]
        self._last_odom_t = self.get_clock().now()

    def scan_cb(self, msg: LaserScan):
        # Calcula ángulo de cada rayo y filtra los del cono frontal ±front_angle_deg
        angles = np.arange(len(msg.ranges)) * msg.angle_increment + msg.angle_min
        ranges = np.array(msg.ranges)
        valid  = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
        front  = np.abs(angles) <= self.front_a
        mask   = valid & front
        # Si no hay rayos válidos en el cono, asume campo libre (inf)
        self.min_front = float(np.min(ranges[mask])) if mask.any() else float('inf')

    def cmd_cb(self, msg: Twist):
        if self._emergency_stop_active:
            self.pub_cmd_.publish(Twist())
            return
        if self._teleop_active():
            return

        out      = Twist()  # velocidad cero por defecto
        now_mono = _time.monotonic()

        # ── Prioridad 1: LiDAR — freno total + retroceso automático ──────────
        if self.min_front <= self.stop_d:
            if self._blocked_since is None:
                self._blocked_since = now_mono

            blocked_secs = now_mono - self._blocked_since

            if blocked_secs >= self.stuck_timeout:
                # Bloqueado demasiado tiempo → activar retroceso
                if self._reversing_since is None:
                    self._reversing_since = now_mono
                    self.get_logger().warn(
                        f'STUCK {blocked_secs:.1f}s → iniciando retroceso',
                        throttle_duration_sec=1.0)

                reverse_elapsed = now_mono - self._reversing_since
                if reverse_elapsed < self.reverse_dur:
                    out.linear.x  = -self.reverse_speed
                    out.angular.z = 0.20   # giro suave durante retroceso
                    self.pub_cmd_.publish(out)
                    self.get_logger().info(
                        f'REVERSING {reverse_elapsed:.1f}s/{self.reverse_dur}s',
                        throttle_duration_sec=0.5)
                    return
                else:
                    # Retroceso completado — resetear contadores
                    self.get_logger().info('Retroceso completado → reanudando navegación')
                    self._blocked_since   = None
                    self._reversing_since = None
                    self.pub_cmd_.publish(out)
                    return
            else:
                # Aún no cumple el timeout → frena el avance pero PERMITE girar
                # en su lugar para orientarse hacia un hueco libre (no congelar todo).
                self._reversing_since = None
                out.linear.x = 0.0
                if self.emerg_rotate:
                    out.angular.z = max(-self.emerg_max_ang,
                                        min(msg.angular.z, self.emerg_max_ang))
                self.pub_cmd_.publish(out)
                self.get_logger().warn(
                    f'EMERGENCY STOP — obstacle at {self.min_front:.2f} m '
                    f'→ giro={out.angular.z:.2f} rad/s '
                    f'(bloqueado {blocked_secs:.1f}s/{self.stuck_timeout}s)',
                    throttle_duration_sec=1.0)
                return
        else:
            # Frontal libre — resetear contadores de bloqueo
            self._blocked_since   = None
            self._reversing_since = None

        # ── Prioridad 2: timeout de localización (fuente /odom muerta) ────────
        odom_age = self._odom_age_sec()
        if odom_age is not None and odom_age > self.cov_timeout:
            self.pub_cmd_.publish(out)
            self.get_logger().warn(
                f'LOCALIZATION TIMEOUT — sin /odom por {odom_age:.1f}s → parado',
                throttle_duration_sec=2.0)
            return

        # ── Prioridad 3: localización perdida (P_xy demasiado grande) ─────────
        if self.trace_p_xy > self.cov_stop:
            self.pub_cmd_.publish(out)
            self.get_logger().warn(
                f'LOCALIZATION LOST — trace(P_xy)={self.trace_p_xy:.3f} m² '
                f'> {self.cov_stop} m² → parado',
                throttle_duration_sec=2.0)
            return

        # ── Factor de escala combinado (LiDAR + covarianza EKF) ───────────────
        # Toma el más restrictivo de los dos factores.

        # Fuente A: zona de frenado LiDAR
        if self.min_front <= self.slow_d and msg.linear.x > 0:
            lidar_scale = (self.min_front - self.stop_d) / (self.slow_d - self.stop_d)
            lidar_scale = max(0.0, min(lidar_scale, 1.0))
        else:
            lidar_scale = 1.0

        # Fuente B: covarianza del EKF — frenado gradual entre cov_slow y cov_stop
        if self.trace_p_xy > self.cov_slow:
            span      = max(self.cov_stop - self.cov_slow, 1e-6)
            cov_scale = 1.0 - (self.trace_p_xy - self.cov_slow) / span
            cov_scale = max(0.0, min(cov_scale, 1.0))
            self.get_logger().info(
                f'UNCERTAIN — trace(P_xy)={self.trace_p_xy:.3f} m² '
                f'→ vel_scale={cov_scale:.2f}',
                throttle_duration_sec=2.0)
        else:
            cov_scale = 1.0

        scale = min(lidar_scale, cov_scale)

        if scale < 1.0:
            out.linear.x  = msg.linear.x  * scale
            out.angular.z = msg.angular.z  * scale
        else:
            out = msg   # campo libre — pasa el comando sin modificar

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
