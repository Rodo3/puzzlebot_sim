"""
bug_navigation_node.py — Evasión reactiva de obstáculos dinámicos para Puzzlebot.

Arquitectura:
  1. Detecta obstáculo con LiDAR (N ciclos consecutivos para confirmar)
  2. Para el robot (bloquea /cmd_vel_steering → /cmd_vel_in)
  3. Inyecta blob en /map → A* replantea automáticamente (replan_on_new_map)
  4. Cuando llega una nueva ruta del A* → desbloquea el robot
  5. Si A* no responde en replan_timeout_sec → retrocede + reintenta

Sin rutas manuales: A* es el único que decide por dónde rodear el obstáculo.

Tópicos suscritos:
  /scan_stamped      (LaserScan)     — LiDAR
  /odom              (Odometry)      — pose EKF
  /goal_pose         (PoseStamped)   — waypoint actual (TRANSIENT_LOCAL)
  /map               (OccupancyGrid) — mapa estático base (TRANSIENT_LOCAL)
  /planned_path      (Path)          — rutas del A* (para detectar nueva ruta)
  /cmd_vel_steering  (Twist)         — pass-through al obstacle_avoidance

Tópicos publicados:
  /cmd_vel_in        (Twist)         — velocidad filtrada (stop o pass-through)
  /map               (OccupancyGrid) — mapa aumentado con obstáculos dinámicos
  /bug_nav/state     (String)        — estado: CLEAR | WAITING | RECOVERY
  /bug_nav/markers   (MarkerArray)   — visualización RViz
"""

import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,
                        qos_profile_sensor_data)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


# ── Estados FSM ───────────────────────────────────────────────────────────────
CLEAR     = 'CLEAR'      # sin obstáculo, A* controla libremente
WAITING   = 'WAITING'    # obstáculo inyectado, robot parado, esperando ruta de A*
FOLLOWING = 'FOLLOWING'  # siguiendo ruta bloqueada post-reroute (sin que A* la pise)
RECOVERY  = 'RECOVERY'   # A* tardó demasiado → retrocede para ganar espacio


class BugNavigationNode(Node):

    def __init__(self):
        super().__init__('bug_navigation_node')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('front_stop_distance',       0.50)
        self.declare_parameter('front_angle_deg',           30.0)
        self.declare_parameter('obstacle_confirm_cycles',   3)
        self.declare_parameter('obstacle_inject_radius_m',  0.45)
        self.declare_parameter('obstacle_inject_decay_sec', 12.0)
        self.declare_parameter('enable_rviz_markers',       True)
        # Tiempo máximo esperando que A* entregue una ruta nueva antes de recovery
        self.declare_parameter('replan_timeout_sec',        3.5)
        # Recovery: retroceso para ganar distancia del obstáculo
        self.declare_parameter('recovery_reverse_speed',    0.10)
        self.declare_parameter('recovery_reverse_sec',      1.5)

        self._front_stop  = self.get_parameter('front_stop_distance').value
        self._front_a     = math.radians(self.get_parameter('front_angle_deg').value)
        self._confirm_n   = self.get_parameter('obstacle_confirm_cycles').value
        self._inj_radius  = self.get_parameter('obstacle_inject_radius_m').value
        self._inj_decay   = self.get_parameter('obstacle_inject_decay_sec').value
        self._markers_en  = self.get_parameter('enable_rviz_markers').value
        self._replan_to   = self.get_parameter('replan_timeout_sec').value
        self._rev_speed   = self.get_parameter('recovery_reverse_speed').value
        self._rev_dur     = self.get_parameter('recovery_reverse_sec').value

        # ── Estado FSM ────────────────────────────────────────────────────────
        self._state          = CLEAR
        self._inject_t       = 0.0
        self._recovery_until = 0.0
        self._locked_path: Path = None   # ruta bloqueada durante FOLLOWING

        # ── Sensor / odometría ────────────────────────────────────────────────
        self._robot_x   = 0.0
        self._robot_y   = 0.0
        self._robot_th  = 0.0
        self._have_pose = False

        self._goal_x = None
        self._goal_y = None

        self._min_front  = float('inf')
        self._dist_left  = float('inf')
        self._dist_right = float('inf')
        self._scan_ok    = False

        # ── Mapa ──────────────────────────────────────────────────────────────
        self._base_map: OccupancyGrid = None
        # Lista de blobs activos: (x, y, timestamp, radius, decay_sec)
        self._injected_obs: list = []

        # ── Detección ─────────────────────────────────────────────────────────
        self._obs_count  = 0
        self._last_obs_x = None
        self._last_obs_y = None
        self._last_inj_t = 0.0

        # ── QoS ───────────────────────────────────────────────────────────────
        latched = QoSProfile(depth=1,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # ── Subscripciones ────────────────────────────────────────────────────
        self.create_subscription(Twist,         '/cmd_vel_steering', self._steering_cb, 10)
        self.create_subscription(LaserScan,     '/scan_stamped',     self._scan_cb,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry,      '/odom',             self._odom_cb,     10)
        self.create_subscription(PoseStamped,   '/goal_pose',        self._goal_cb,     latched)
        self.create_subscription(OccupancyGrid, '/map',              self._map_cb,      latched)
        self.create_subscription(Path,          '/planned_path',     self._path_cb,     10)

        # ── Publicadores ──────────────────────────────────────────────────────
        self._pub_cmd     = self.create_publisher(Twist,         '/cmd_vel_in',      10)
        self._pub_map     = self.create_publisher(OccupancyGrid, '/map',             latched)
        self._pub_path    = self.create_publisher(Path,          '/planned_path',    10)
        self._pub_state   = self.create_publisher(String,        '/bug_nav/state',   10)
        self._pub_markers = self.create_publisher(MarkerArray,   '/bug_nav/markers', 10)

        self.create_timer(0.10, self._loop)

        self.get_logger().info(
            f'bug_navigation_node OK | '
            f'front_stop={self._front_stop}m  inj_radius={self._inj_radius}m  '
            f'replan_timeout={self._replan_to}s  rev_dur={self._rev_dur}s')

    # ═══════════════════════════════════════════════════════════════════════════
    # Callbacks
    # ═══════════════════════════════════════════════════════════════════════════

    def _steering_cb(self, msg: Twist):
        now = time.monotonic()

        if self._state == RECOVERY:
            if now < self._recovery_until:
                out = Twist()
                out.linear.x = -self._rev_speed
                self._pub_cmd.publish(out)
            else:
                self.get_logger().info('[BugNav] Recovery completado — re-inyectando blob')
                self._last_obs_x = None
                self._last_obs_y = None
                self._obs_count  = 0
                self._transition(WAITING)
                self._inject_obstacle_from_scan()
                self._pub_cmd.publish(Twist())
            return

        if self._state == WAITING:
            self._pub_cmd.publish(Twist())
            return

        # CLEAR o FOLLOWING: pasar el comando (el steering sigue la ruta activa)
        self._pub_cmd.publish(msg)

    def _path_cb(self, msg: Path):
        """Cuando llega ruta del A* con waypoints y estamos esperando → bloquear y seguir."""
        if self._state == WAITING and len(msg.poses) > 0:
            self.get_logger().info(
                f'[BugNav] Nueva ruta del A* ({len(msg.poses)} wps) — '
                f'bloqueando ruta y siguiendo')
            self._locked_path = msg
            self._transition(FOLLOWING)

    def _odom_cb(self, msg: Odometry):
        self._robot_x  = msg.pose.pose.position.x
        self._robot_y  = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._robot_th = math.atan2(2*(q.w*q.z + q.x*q.y),
                                    1 - 2*(q.y*q.y + q.z*q.z))
        self._have_pose = True

    def _goal_cb(self, msg: PoseStamped):
        self._goal_x = msg.pose.position.x
        self._goal_y = msg.pose.position.y
        self.get_logger().info(f'[BugNav] Goal: ({self._goal_x:.2f}, {self._goal_y:.2f})')

    def _map_cb(self, msg: OccupancyGrid):
        if self._base_map is None:
            self._base_map = msg
            self.get_logger().info(
                f'[BugNav] Mapa base: {msg.info.width}x{msg.info.height} '
                f'@ {msg.info.resolution}m/px')
            self._pub_map.publish(msg)

    def _scan_cb(self, msg: LaserScan):
        self._scan_ok = True
        n = len(msg.ranges)
        if n == 0:
            return
        angles = np.arange(n) * msg.angle_increment + msg.angle_min
        r      = np.array(msg.ranges, dtype=float)
        valid  = np.isfinite(r) & (r > msg.range_min) & (r < msg.range_max)

        fm = valid & (np.abs(angles) <= self._front_a)
        self._min_front  = float(np.min(r[fm])) if fm.any() else float('inf')

        lm = valid & (angles >= math.radians(55)) & (angles <= math.radians(125))
        self._dist_left  = float(np.min(r[lm])) if lm.any() else float('inf')

        rm = valid & (angles >= math.radians(-125)) & (angles <= math.radians(-55))
        self._dist_right = float(np.min(r[rm])) if rm.any() else float('inf')

    # ═══════════════════════════════════════════════════════════════════════════
    # Loop principal
    # ═══════════════════════════════════════════════════════════════════════════

    def _loop(self):
        if not self._scan_ok or not self._have_pose:
            return

        now = time.monotonic()

        # 1. Expirar blobs viejos
        self._expire_obstacles(now)

        # 2. RECOVERY: esperar — _steering_cb maneja la velocidad
        if self._state == RECOVERY:
            self._publish_state()
            return

        # 3. WAITING: verificar timeout → recovery si A* no responde
        if self._state == WAITING:
            if (now - self._inject_t) > self._replan_to:
                self.get_logger().warn(
                    f'[BugNav] A* no respondió en {self._replan_to}s — recovery reverse')
                self._recovery_until = now + self._rev_dur
                self._transition(RECOVERY)
            self._publish_state()
            return

        # 4. FOLLOWING: republica la ruta bloqueada para que A* no la pise.
        #    Detecta nuevo obstáculo → vuelve a WAITING con nuevo blob.
        if self._state == FOLLOWING:
            # Verificar si llegamos al goal
            if self._goal_x is not None:
                dist_goal = math.hypot(self._goal_x - self._robot_x,
                                       self._goal_y - self._robot_y)
                if dist_goal < 0.15:
                    self.get_logger().info('[BugNav] Goal alcanzado — liberando ruta')
                    self._locked_path = None
                    self._transition(CLEAR)
                    self._publish_state()
                    return

            # Detectar nuevo obstáculo mientras seguimos la ruta bloqueada
            obstacle_now = (self._min_front < self._front_stop
                            and self._obstacle_toward_goal())
            if obstacle_now:
                self._obs_count += 1
            else:
                self._obs_count = 0

            if self._obs_count >= self._confirm_n:
                self.get_logger().warn(
                    '[BugNav] Nuevo obstáculo durante FOLLOWING — re-inyectando')
                self._obs_count   = 0
                self._locked_path = None
                self._last_obs_x  = None
                self._last_obs_y  = None
                self._transition(WAITING)
                self._inject_obstacle_from_scan()
                self._publish_state()
                return

            # Republica la ruta bloqueada para que el steering la siga
            if self._locked_path is not None:
                self._locked_path.header.stamp = self.get_clock().now().to_msg()
                self._pub_path.publish(self._locked_path)

            self._publish_state()
            if self._markers_en:
                self._publish_markers()
            return

        # 5. CLEAR: detectar obstáculo frontal
        obstacle_now = (self._min_front < self._front_stop
                        and self._obstacle_toward_goal())
        if obstacle_now:
            self._obs_count += 1
        else:
            self._obs_count = 0

        if self._obs_count >= self._confirm_n:
            self._obs_count = 0
            self._transition(WAITING)
            self._inject_obstacle_from_scan()

        self._publish_state()
        if self._markers_en:
            self._publish_markers()

    # ═══════════════════════════════════════════════════════════════════════════
    # Inyección de blobs
    # ═══════════════════════════════════════════════════════════════════════════

    def _inject_obstacle_from_scan(self):
        """Calcula posición del obstáculo desde LiDAR e inyecta blob en /map."""
        # Centroide de puntos frontales dentro de front_stop
        min_center = self._inj_radius + 0.20
        dist = max(self._min_front, min_center)
        obs_x = self._robot_x + dist * math.cos(self._robot_th)
        obs_y = self._robot_y + dist * math.sin(self._robot_th)

        now = time.monotonic()

        # No re-inyectar si ya hay un blob reciente muy cercano
        if (self._last_obs_x is not None
                and math.hypot(obs_x - self._last_obs_x,
                               obs_y - self._last_obs_y) < 0.60
                and (now - self._last_inj_t) < 15.0):
            self.get_logger().info('[BugNav] Blob reciente activo — no re-inyectando')
            self._inject_t = now  # resetear timeout de A*
            return

        self.get_logger().warn(
            f'[BugNav] Inyectando blob en ({obs_x:.2f},{obs_y:.2f}) '
            f'radio={self._inj_radius}m  '
            f'(obstáculo a {self._min_front:.2f}m)')

        self._injected_obs.append((obs_x, obs_y, now, self._inj_radius, self._inj_decay))
        self._last_obs_x = obs_x
        self._last_obs_y = obs_y
        self._last_inj_t = now
        self._inject_t   = now
        self._republish_map()

    def _expire_obstacles(self, now: float):
        before = len(self._injected_obs)
        self._injected_obs = [o for o in self._injected_obs
                              if (now - o[2]) < o[4]]
        if len(self._injected_obs) != before:
            self.get_logger().info(
                f'[BugNav] Blob(s) expirado(s) — quedan {len(self._injected_obs)}')
            self._republish_map()

    def _republish_map(self):
        if self._base_map is None:
            return
        info = self._base_map.info
        res  = info.resolution
        ox   = info.origin.position.x
        oy   = info.origin.position.y
        W, H = info.width, info.height

        data = list(self._base_map.data)

        for obs_x, obs_y, _, radius, _ in self._injected_obs:
            rc = max(1, int(math.ceil(radius / res)))
            cc_c = int((obs_x - ox) / res)
            rc_c = int((obs_y - oy) / res)
            for dr in range(-rc, rc + 1):
                for dc in range(-rc, rc + 1):
                    if math.hypot(dr, dc) > rc:
                        continue
                    rr = rc_c + dr
                    cc = cc_c + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        data[rr * W + cc] = 100

        new_map              = OccupancyGrid()
        new_map.header       = self._base_map.header
        new_map.header.stamp = self.get_clock().now().to_msg()
        new_map.info         = self._base_map.info
        new_map.data         = data
        self._pub_map.publish(new_map)

    # ═══════════════════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════════════════

    def _obstacle_toward_goal(self) -> bool:
        if self._goal_x is None or not self._have_pose:
            return True
        dist_goal = math.hypot(self._goal_x - self._robot_x,
                               self._goal_y - self._robot_y)
        if dist_goal < 0.20:
            return False
        angle_to_goal = math.atan2(self._goal_y - self._robot_y,
                                   self._goal_x - self._robot_x)
        angle_err = abs(_norm(angle_to_goal - self._robot_th))
        return angle_err < math.radians(90)

    def _transition(self, new_state: str):
        self.get_logger().warn(f'[BugNav] {self._state} → {new_state}')
        self._state = new_state

    def _publish_state(self):
        s = String()
        s.data = self._state
        self._pub_state.publish(s)

    def _publish_markers(self):
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()
        for i, (ox, oy, _, radius, _) in enumerate(self._injected_obs):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp    = now
            m.ns = 'bug_obstacles'
            m.id = i
            m.type   = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = ox
            m.pose.position.y = oy
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = radius * 2
            m.scale.z = 0.10
            m.color.r = 1.0
            m.color.g = 0.4
            m.color.a = 0.7
            arr.markers.append(m)
        self._pub_markers.publish(arr)


# ── Utilidades ────────────────────────────────────────────────────────────────

def _norm(a: float) -> float:
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


def main(args=None):
    rclpy.init(args=args)
    node = BugNavigationNode()
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
