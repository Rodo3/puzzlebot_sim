"""
bug_navigation_node.py — Bug2 Reactive Navigation para Puzzlebot.

Implementa el algoritmo Bug2 de la presentación de navegación reactiva:

  CLEAR:       Avanzar hacia el goal siguiendo la ruta del A*.
  WALL_FOLLOW: Al detectar obstáculo:
               1. Guardar "hit point" H y la línea start-goal (H → goal).
               2. Seguir la pared del obstáculo con control proporcional.
               3. Salir cuando el robot cruza la línea start-goal en un
                  punto más cercano al goal que H (leave point L).
  RECOVERY:    Si el frente sigue bloqueado al entrar en WALL_FOLLOW →
               retroceder brevemente para ganar espacio.

También inyecta blobs en /map para que A* aprenda la posición del
obstáculo y replantee la ruta futura.

Tópicos suscritos:
  /scan_stamped     (LaserScan)     — LiDAR
  /odom             (Odometry)      — pose actual
  /goal_pose        (PoseStamped)   — waypoint destino (TRANSIENT_LOCAL)
  /map              (OccupancyGrid) — mapa base (TRANSIENT_LOCAL)
  /cmd_vel_steering (Twist)         — velocidad del steering (pass-through en CLEAR)

Tópicos publicados:
  /cmd_vel_in       (Twist)         — velocidad filtrada / wall-follow
  /map              (OccupancyGrid) — mapa con blobs de obstáculos
  /bug_nav/state    (String)        — CLEAR | WALL_FOLLOW | RECOVERY
  /bug_nav/markers  (MarkerArray)   — visualización RViz
"""

import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,
                        qos_profile_sensor_data)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


# ── Estados FSM ───────────────────────────────────────────────────────────────
CLEAR       = 'CLEAR'        # avanzando hacia goal con A*
WALL_FOLLOW = 'WALL_FOLLOW'  # siguiendo pared del obstáculo (Bug2)
RECOVERY    = 'RECOVERY'     # retroceso de emergencia


class BugNavigationNode(Node):

    def __init__(self):
        super().__init__('bug_navigation_node')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('front_stop_distance',       0.45)
        self.declare_parameter('front_angle_deg',           30.0)
        self.declare_parameter('obstacle_confirm_cycles',   3)

        # Wall following
        self.declare_parameter('wall_follow_dist',          0.35)   # distancia objetivo a la pared [m]
        self.declare_parameter('wall_follow_speed',         0.10)   # velocidad lineal al seguir pared
        self.declare_parameter('wall_kp',                   1.20)   # ganancia proporcional del controlador
        self.declare_parameter('wall_angle_deg',            80.0)   # semióngulo del cono lateral para medir pared

        # Leave condition (Bug2): el robot debe estar sobre la línea start-goal
        # y más cerca del goal que el hit point
        self.declare_parameter('leave_line_tolerance',      0.20)   # distancia máxima a la línea [m]
        self.declare_parameter('leave_min_progress',        0.25)   # avance mínimo respecto al hit point [m]

        # Blob injection en /map (para que A* aprenda el obstáculo)
        self.declare_parameter('obstacle_inject_radius_m',  0.40)
        self.declare_parameter('obstacle_inject_decay_sec', 12.0)

        # Recovery
        self.declare_parameter('recovery_reverse_speed',    0.10)
        self.declare_parameter('recovery_reverse_sec',      1.5)

        self.declare_parameter('enable_rviz_markers',       True)

        # Leer parámetros
        self._front_stop   = self.get_parameter('front_stop_distance').value
        self._front_a      = math.radians(self.get_parameter('front_angle_deg').value)
        self._confirm_n    = self.get_parameter('obstacle_confirm_cycles').value

        self._wall_dist    = self.get_parameter('wall_follow_dist').value
        self._wall_speed   = self.get_parameter('wall_follow_speed').value
        self._wall_kp      = self.get_parameter('wall_kp').value
        self._wall_angle   = math.radians(self.get_parameter('wall_angle_deg').value)

        self._leave_tol    = self.get_parameter('leave_line_tolerance').value
        self._leave_prog   = self.get_parameter('leave_min_progress').value

        self._inj_radius   = self.get_parameter('obstacle_inject_radius_m').value
        self._inj_decay    = self.get_parameter('obstacle_inject_decay_sec').value

        self._rev_speed    = self.get_parameter('recovery_reverse_speed').value
        self._rev_dur      = self.get_parameter('recovery_reverse_sec').value
        self._markers_en   = self.get_parameter('enable_rviz_markers').value

        # ── Estado FSM ────────────────────────────────────────────────────────
        self._state            = CLEAR
        self._recovery_until   = 0.0
        self._wall_follow_start: float = 0.0   # timestamp de entrada a WALL_FOLLOW

        # Bug2: hit point y línea start-goal
        self._hit_x: float     = 0.0
        self._hit_y: float     = 0.0
        self._hit_dist_goal: float = float('inf')  # dist(hit, goal)
        self._follow_side: int = 1   # +1 = izquierda, -1 = derecha

        # Timeout máximo en WALL_FOLLOW (evita ciclos infinitos)
        self._wall_follow_timeout = 20.0   # segundos

        # ── Sensor / odometría ────────────────────────────────────────────────
        self._robot_x    = 0.0
        self._robot_y    = 0.0
        self._robot_th   = 0.0
        self._have_pose  = False

        self._goal_x: float = None
        self._goal_y: float = None

        self._min_front  = float('inf')
        self._dist_left  = float('inf')
        self._dist_right = float('inf')
        self._scan_ok    = False

        # ── Mapa ──────────────────────────────────────────────────────────────
        self._base_map: OccupancyGrid = None
        self._injected_obs: list = []   # (x, y, t, radius, decay)

        # ── Detección ─────────────────────────────────────────────────────────
        self._obs_count  = 0

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

        # ── Publicadores ──────────────────────────────────────────────────────
        self._pub_cmd     = self.create_publisher(Twist,         '/cmd_vel_in',      10)
        self._pub_map     = self.create_publisher(OccupancyGrid, '/map',             latched)
        self._pub_state   = self.create_publisher(String,        '/bug_nav/state',   10)
        self._pub_markers = self.create_publisher(MarkerArray,   '/bug_nav/markers', 10)

        self.create_timer(0.10, self._loop)

        self.get_logger().info(
            f'bug_navigation_node (Bug2) OK | '
            f'front_stop={self._front_stop}m  wall_dist={self._wall_dist}m  '
            f'leave_tol={self._leave_tol}m  leave_prog={self._leave_prog}m')

    # ═══════════════════════════════════════════════════════════════════════════
    # Callbacks
    # ═══════════════════════════════════════════════════════════════════════════

    def _steering_cb(self, msg: Twist):
        """En CLEAR pasa el comando del steering. En otros estados lo bloquea."""
        now = time.monotonic()

        if self._state == RECOVERY:
            if now < self._recovery_until:
                out = Twist()
                out.linear.x = -self._rev_speed
                self._pub_cmd.publish(out)
            else:
                self.get_logger().info('[Bug2] Recovery completado → WALL_FOLLOW')
                self._enter_wall_follow()
            return

        if self._state == WALL_FOLLOW:
            # El loop principal genera el cmd_vel de wall following
            # aquí no pasamos el steering del A*
            return

        # CLEAR: pasar el comando del steering controller (A* path following)
        self._pub_cmd.publish(msg)

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
        self.get_logger().info(f'[Bug2] Goal: ({self._goal_x:.2f}, {self._goal_y:.2f})')

    def _map_cb(self, msg: OccupancyGrid):
        if self._base_map is None:
            self._base_map = msg
            self.get_logger().info(
                f'[Bug2] Mapa base: {msg.info.width}x{msg.info.height}')
            self._pub_map.publish(msg)

    def _scan_cb(self, msg: LaserScan):
        self._scan_ok = True
        n = len(msg.ranges)
        if n == 0:
            return
        angles = np.arange(n) * msg.angle_increment + msg.angle_min
        r      = np.array(msg.ranges, dtype=float)
        valid  = np.isfinite(r) & (r > msg.range_min) & (r < msg.range_max)

        # Frente
        fm = valid & (np.abs(angles) <= self._front_a)
        self._min_front = float(np.min(r[fm])) if fm.any() else float('inf')

        # Lateral izquierda (para wall following)
        lm = valid & (angles >= (math.pi/2 - self._wall_angle/2)) \
                   & (angles <= (math.pi/2 + self._wall_angle/2))
        self._dist_left = float(np.min(r[lm])) if lm.any() else float('inf')

        # Lateral derecha
        rm = valid & (angles >= -(math.pi/2 + self._wall_angle/2)) \
                   & (angles <= -(math.pi/2 - self._wall_angle/2))
        self._dist_right = float(np.min(r[rm])) if rm.any() else float('inf')

    # ═══════════════════════════════════════════════════════════════════════════
    # Loop principal
    # ═══════════════════════════════════════════════════════════════════════════

    def _loop(self):
        if not self._scan_ok or not self._have_pose:
            return

        now = time.monotonic()

        # Expirar blobs
        self._expire_obstacles(now)

        # RECOVERY: gestionado en _steering_cb
        if self._state == RECOVERY:
            self._publish_state()
            return

        # ── CLEAR: detectar obstáculo hacia el goal ───────────────────────────
        if self._state == CLEAR:
            obstacle_ahead = (self._min_front < self._front_stop
                              and self._obstacle_toward_goal())
            if obstacle_ahead:
                self._obs_count += 1
            else:
                self._obs_count = 0

            if self._obs_count >= self._confirm_n:
                self._obs_count = 0
                self.get_logger().warn(
                    f'[Bug2] Obstáculo a {self._min_front:.2f}m → WALL_FOLLOW')
                # ¿Hay espacio para entrar directamente o necesitamos recovery?
                if self._min_front < 0.25:
                    self._recovery_until = now + self._rev_dur
                    self._transition(RECOVERY)
                else:
                    self._enter_wall_follow()

            self._publish_state()
            if self._markers_en:
                self._publish_markers()
            return

        # ── WALL_FOLLOW: Bug2 wall following ─────────────────────────────────
        if self._state == WALL_FOLLOW:
            # 0. Timeout de seguridad: si lleva demasiado tiempo siguiendo la pared
            #    sin encontrar el leave point → volver a CLEAR y dejar que A* decida
            if (now - self._wall_follow_start) > self._wall_follow_timeout:
                self.get_logger().warn(
                    f'[Bug2] Timeout wall following ({self._wall_follow_timeout}s) → CLEAR')
                self._transition(CLEAR)
                self._publish_state()
                return

            # 1. Verificar condición de salida Bug2:
            #    Robot en la línea start-goal Y más cerca del goal que el hit point
            if self._check_leave_condition():
                self.get_logger().info(
                    f'[Bug2] Leave point alcanzado — regresando a CLEAR')
                self._transition(CLEAR)
                self._publish_state()
                return

            # 2. Verificar goal alcanzado
            if self._goal_x is not None:
                dist_goal = math.hypot(self._goal_x - self._robot_x,
                                       self._goal_y - self._robot_y)
                if dist_goal < 0.15:
                    self.get_logger().info('[Bug2] Goal alcanzado → CLEAR')
                    self._transition(CLEAR)
                    self._publish_state()
                    return

            # 3. Generar comando de wall following
            cmd = self._wall_follow_cmd()
            self._pub_cmd.publish(cmd)

            self._publish_state()
            if self._markers_en:
                self._publish_markers()
            return

    # ═══════════════════════════════════════════════════════════════════════════
    # Bug2 — lógica de entrada y wall following
    # ═══════════════════════════════════════════════════════════════════════════

    def _enter_wall_follow(self):
        """Registra el hit point, elige lado y entra en WALL_FOLLOW."""
        self._wall_follow_start = time.monotonic()
        self._hit_x = self._robot_x
        self._hit_y = self._robot_y
        if self._goal_x is not None:
            self._hit_dist_goal = math.hypot(self._goal_x - self._hit_x,
                                             self._goal_y - self._hit_y)
        else:
            self._hit_dist_goal = float('inf')

        # Elegir lado: rodear por donde haya más espacio libre
        # +1 = pared a la DERECHA del robot (robot gira izq para rodearla)
        # -1 = pared a la IZQUIERDA del robot (robot gira der para rodearla)
        if self._dist_left >= self._dist_right:
            self._follow_side = -1   # obstáculo a la derecha → pared a la derecha
            side_name = 'DER'
        else:
            self._follow_side = 1    # obstáculo a la izquierda → pared a la izq
            side_name = 'IZQ'

        self.get_logger().warn(
            f'[Bug2] Hit point ({self._hit_x:.2f},{self._hit_y:.2f}) | '
            f'dist_goal={self._hit_dist_goal:.2f}m | lado={side_name}')

        # Inyectar blob para que A* aprenda el obstáculo
        self._inject_obstacle_blob()
        self._transition(WALL_FOLLOW)

    def _wall_follow_cmd(self) -> Twist:
        """
        Controlador proporcional de seguimiento de pared.

        Mantiene la distancia objetivo al obstáculo (wall_follow_dist) usando
        la lectura lateral del LiDAR. Si el frente está bloqueado, gira en
        dirección contraria al lado de la pared para encontrar espacio libre.
        """
        cmd = Twist()

        # Distancia al lado de la pared según el lado elegido
        side_dist = self._dist_right if self._follow_side == -1 else self._dist_left

        # Si el frente está obstruido → girar HACIA el espacio libre
        if self._min_front < self._front_stop:
            cmd.linear.x   = 0.0
            cmd.angular.z  = -self._follow_side * 0.6   # giro alejándose de la pared
            return cmd

        # Error respecto a la distancia objetivo
        error = side_dist - self._wall_dist
        # Si error > 0: demasiado lejos de la pared → girar HACIA la pared
        # Si error < 0: demasiado cerca → alejarse
        angular = self._follow_side * self._wall_kp * error

        cmd.linear.x  = self._wall_speed
        cmd.angular.z = float(np.clip(angular, -1.2, 1.2))
        return cmd

    def _check_leave_condition(self) -> bool:
        """
        Bug2 leave condition: el robot está sobre la línea start-goal
        (hit_point → goal) Y está más cerca del goal que el hit point.

        Distancia del robot a la línea start-goal < leave_line_tolerance
        Y dist(robot, goal) < dist(hit, goal) - leave_min_progress
        """
        if self._goal_x is None:
            return False

        # Vector de la línea: hit → goal
        lx = self._goal_x - self._hit_x
        ly = self._goal_y - self._hit_y
        line_len = math.hypot(lx, ly)
        if line_len < 0.1:
            return False

        # Proyección del robot sobre la línea
        dx = self._robot_x - self._hit_x
        dy = self._robot_y - self._hit_y
        t  = (dx * lx + dy * ly) / (line_len * line_len)

        # Solo considerar puntos adelante del hit point y antes del goal
        if t < 0.05:
            return False

        # Distancia perpendicular a la línea
        # Componente perpendicular: (dx,dy) - t*(lx,ly)
        perp_x = dx - t * lx
        perp_y = dy - t * ly
        dist_to_line = math.hypot(perp_x, perp_y)

        if dist_to_line > self._leave_tol:
            return False

        # ¿Está más cerca del goal que el hit point?
        dist_to_goal = math.hypot(self._goal_x - self._robot_x,
                                  self._goal_y - self._robot_y)

        return dist_to_goal < (self._hit_dist_goal - self._leave_prog)

    # ═══════════════════════════════════════════════════════════════════════════
    # Inyección de blobs en /map
    # ═══════════════════════════════════════════════════════════════════════════

    def _inject_obstacle_blob(self):
        """Inyecta un blob en /map en la posición del obstáculo detectado."""
        dist = max(self._min_front, self._inj_radius + 0.20)
        obs_x = self._robot_x + dist * math.cos(self._robot_th)
        obs_y = self._robot_y + dist * math.sin(self._robot_th)

        now = time.monotonic()
        self._injected_obs.append((obs_x, obs_y, now, self._inj_radius, self._inj_decay))
        self.get_logger().info(
            f'[Bug2] Blob inyectado en ({obs_x:.2f},{obs_y:.2f}) '
            f'radio={self._inj_radius}m')
        self._republish_map()

    def _expire_obstacles(self, now: float):
        before = len(self._injected_obs)
        self._injected_obs = [o for o in self._injected_obs if (now - o[2]) < o[4]]
        if len(self._injected_obs) != before:
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
        return abs(_norm(angle_to_goal - self._robot_th)) < math.radians(90)

    def _transition(self, new_state: str):
        self.get_logger().warn(f'[Bug2] {self._state} → {new_state}')
        self._state = new_state

    def _publish_state(self):
        s = String()
        s.data = self._state
        self._pub_state.publish(s)

    def _publish_markers(self):
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()

        # Blobs inyectados
        for i, (ox, oy, _, radius, _) in enumerate(self._injected_obs):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp    = now
            m.ns = 'bug2_blobs'
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = ox
            m.pose.position.y = oy
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = radius * 2
            m.scale.z = 0.10
            m.color.r = 1.0
            m.color.g = 0.4
            m.color.a = 0.6
            arr.markers.append(m)

        # Hit point (esfera roja)
        if self._state == WALL_FOLLOW:
            h = Marker()
            h.header.frame_id = 'map'
            h.header.stamp    = now
            h.ns = 'hit_point'
            h.id = 100
            h.type = Marker.SPHERE
            h.action = Marker.ADD
            h.pose.position.x = self._hit_x
            h.pose.position.y = self._hit_y
            h.pose.orientation.w = 1.0
            h.scale.x = h.scale.y = h.scale.z = 0.14
            h.color.r = 1.0
            h.color.a = 1.0
            arr.markers.append(h)

            # Línea start-goal
            if self._goal_x is not None:
                line = Marker()
                line.header.frame_id = 'map'
                line.header.stamp    = now
                line.ns = 'start_goal_line'
                line.id = 101
                line.type = Marker.LINE_STRIP
                line.action = Marker.ADD
                line.scale.x = 0.02
                line.color.r = 1.0
                line.color.g = 1.0
                line.color.a = 0.6
                from geometry_msgs.msg import Point as GPoint
                p1, p2 = GPoint(), GPoint()
                p1.x, p1.y = self._hit_x, self._hit_y
                p2.x, p2.y = self._goal_x, self._goal_y
                line.points = [p1, p2]
                arr.markers.append(line)

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
