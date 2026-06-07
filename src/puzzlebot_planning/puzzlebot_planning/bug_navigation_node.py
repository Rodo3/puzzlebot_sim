"""
bug_navigation_node.py — Evasión reactiva de obstáculos dinámicos para Puzzlebot.

Arquitectura simplificada:
  - NO intercepta /cmd_vel — el steering controller sigue manejando el movimiento
  - Detecta obstáculos con LiDAR
  - Inyecta el obstáculo en /map como celda ocupada
  - El path_planner_node replantea automáticamente al recibir el mapa nuevo
  - La nueva ruta evita el obstáculo por el lado con más espacio

Tópicos suscritos:
  /scan_stamped      (LaserScan)     — LiDAR
  /odom              (Odometry)      — pose EKF
  /goal_pose         (PoseStamped)   — waypoint actual (TRANSIENT_LOCAL)
  /map               (OccupancyGrid) — mapa estático base (TRANSIENT_LOCAL)
  /cmd_vel_steering  (Twist)         — pass-through al obstacle_avoidance

Tópicos publicados:
  /cmd_vel_in        (Twist)         — reenvío directo de cmd_vel_steering
  /map               (OccupancyGrid) — mapa aumentado con obstáculos dinámicos
  /bug_nav/state     (String)        — estado actual
  /bug_nav/markers   (MarkerArray)   — visualización RViz
"""

import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,
                        qos_profile_sensor_data)
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


class BugNavigationNode(Node):

    def __init__(self):
        super().__init__('bug_navigation_node')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('bug_algorithm',             'bug2')
        self.declare_parameter('front_stop_distance',       0.50)
        self.declare_parameter('front_angle_deg',           30.0)
        self.declare_parameter('obstacle_confirm_cycles',   3)
        self.declare_parameter('obstacle_inject_radius_m',  0.45)
        self.declare_parameter('obstacle_inject_decay_sec', 30.0)
        self.declare_parameter('blocked_inject_radius_m',   1.20)
        self.declare_parameter('blocked_inject_decay_sec',  60.0)
        self.declare_parameter('enable_rviz_markers',       True)
        self.declare_parameter('replan_stop_duration_sec',  2.5)
        # Pared virtual: N blobs adicionales perpendiculares al movimiento del robot.
        # wall_half_width_m: mitad del ancho de la pared (blobs a ±0, ±step, ±2*step)
        # wall_blob_radius_m: radio de cada blob de la pared (puede ser menor que el central)
        self.declare_parameter('wall_half_width_m',   0.75)   # cubre 1.5 m de ancho
        self.declare_parameter('wall_blob_step_m',    0.35)   # separación entre blobs
        self.declare_parameter('wall_blob_radius_m',  0.40)   # radio de cada blob lateral

        self._alg           = self.get_parameter('bug_algorithm').value
        self._front_stop    = self.get_parameter('front_stop_distance').value
        self._front_a       = math.radians(self.get_parameter('front_angle_deg').value)
        self._confirm_n     = self.get_parameter('obstacle_confirm_cycles').value
        self._inj_radius    = self.get_parameter('obstacle_inject_radius_m').value
        self._inj_decay     = self.get_parameter('obstacle_inject_decay_sec').value
        self._blk_radius    = self.get_parameter('blocked_inject_radius_m').value
        self._blk_decay     = self.get_parameter('blocked_inject_decay_sec').value
        self._markers_en    = self.get_parameter('enable_rviz_markers').value
        self._stop_dur      = self.get_parameter('replan_stop_duration_sec').value
        self._wall_hw       = self.get_parameter('wall_half_width_m').value
        self._wall_step     = self.get_parameter('wall_blob_step_m').value
        self._wall_r        = self.get_parameter('wall_blob_radius_m').value

        self._blocking_until: float = 0.0
        self._last_good_path_t: float = 0.0
        self._have_safe_path: bool = True
        self._map_injected_at: float = 0.0
        # Ruta de evasión activa. Mientras no sea None, se republica en cada ciclo
        # del loop para evitar que el path_planner la sobrescriba.
        self._evade_path: 'Path | None' = None
        # Distancia al primer waypoint de evasión para saber cuándo cancelarla.
        self._evade_wp1_x: float = 0.0
        self._evade_wp1_y: float = 0.0

        # ── Estado ────────────────────────────────────────────────────────────
        self._robot_x    = 0.0
        self._robot_y    = 0.0
        self._robot_th   = 0.0
        self._have_pose  = False

        self._goal_x     = None
        self._goal_y     = None

        self._min_front  = float('inf')
        self._dist_left  = float('inf')
        self._dist_right = float('inf')
        self._scan_ok    = False

        self._base_map   = None   # OccupancyGrid sin modificar — nunca se sobreescribe

        # Lista de obstáculos activos: (x, y, timestamp, radius, decay)
        self._injected_obs: list = []

        # Contador para confirmar obstáculo frontal sostenido
        self._obs_count  = 0

        # Última posición del obstáculo detectado (para no re-inyectar el mismo punto)
        self._last_obs_x = None
        self._last_obs_y = None
        self._last_inj_t = 0.0   # timestamp de la última inyección

        # ── QoS TRANSIENT_LOCAL — solo para /map (publicado latcheado por slam_node) ──
        map_qos = QoSProfile(depth=1,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # ── Subscripciones ────────────────────────────────────────────────────
        self.create_subscription(Twist,        '/cmd_vel_steering', self._steering_cb, 10)
        self.create_subscription(LaserScan,    '/scan_stamped',     self._scan_cb,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry,     '/odom',             self._odom_cb,     10)
        # /goal_pose viene de RViz (VOLATILE) — no usar TRANSIENT_LOCAL aquí
        self.create_subscription(PoseStamped,  '/goal_pose',        self._goal_cb,     10)
        self.create_subscription(OccupancyGrid, '/map',             self._map_cb,      map_qos)
        # Escuchar las rutas que llegan del A* para saber cuándo hay una ruta válida.
        # Nota: este nodo también publica en /planned_path (path vacío), pero la
        # suscripción solo actúa cuando llega una ruta CON waypoints (del path_planner).
        self.create_subscription(Path, '/planned_path', self._path_cb, 10)

        # ── Publicadores ──────────────────────────────────────────────────────
        self._pub_cmd     = self.create_publisher(Twist,        '/cmd_vel_in',    10)
        self._pub_map     = self.create_publisher(OccupancyGrid, '/map',          map_qos)
        self._pub_path    = self.create_publisher(Path,         '/planned_path',   10)
        self._pub_state   = self.create_publisher(String,       '/bug_nav/state',  10)
        self._pub_markers = self.create_publisher(MarkerArray,  '/bug_nav/markers', 10)

        # Timer principal: detecta obstáculos y gestiona expiración
        self.create_timer(0.10, self._loop)   # 10 Hz es suficiente

        self.get_logger().info(
            f'bug_navigation_node OK | alg={self._alg} '
            f'front_stop={self._front_stop}m inj_radius={self._inj_radius}m')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _path_cb(self, msg):
        # La ruta de evasión la publicamos nosotros mismos con _have_safe_path=True.
        # Las rutas del A* (que llegan después) se dejan pasar para retomar la
        # navegación normal cuando el robot ya rodeó el obstáculo.
        if len(msg.poses) > 0:
            self._last_good_path_t = time.monotonic()

    def _steering_cb(self, msg):
        now = time.monotonic()

        # Bloqueo activo: publicar cero y no pasar el comando del steering.
        # El bloqueo se activa en _inject_and_publish() y se libera en _path_cb()
        # cuando llega una ruta válida del A*. NO se renueva aquí — renovarlo aquí
        # causaría un deadlock si el obstáculo sigue presente mientras el robot
        # intenta seguir la ruta alternativa (que puede pasar cerca del obstáculo).
        if now < self._blocking_until:
            self._pub_cmd.publish(Twist())
            return

        # El obstacle_avoidance_node (downstream) es el último freno de emergencia
        # si el robot se acerca demasiado (< stop_distance = 0.20 m). No duplicar
        # esa lógica aquí para no bloquear la evasión lateral.
        self._pub_cmd.publish(msg)

    def _odom_cb(self, msg):
        self._robot_x  = msg.pose.pose.position.x
        self._robot_y  = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._robot_th = math.atan2(2*(q.w*q.z + q.x*q.y),
                                    1 - 2*(q.y*q.y + q.z*q.z))
        self._have_pose = True

    def _goal_cb(self, msg):
        self._goal_x = msg.pose.position.x
        self._goal_y = msg.pose.position.y
        self.get_logger().info(f'Goal recibido: ({self._goal_x:.2f}, {self._goal_y:.2f})')

    def _map_cb(self, msg):
        # Solo guardar el mapa base original — nunca sobreescribir con el aumentado
        if self._base_map is None:
            self._base_map = msg
            self.get_logger().info(
                f'Mapa base: {msg.info.width}x{msg.info.height} '
                f'@ {msg.info.resolution}m/px')
            # Publicar inmediatamente para que path_planner lo reciba
            self._pub_map.publish(msg)

    def _scan_cb(self, msg):
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

    # ── Loop principal ────────────────────────────────────────────────────────

    def _loop(self):
        if self._alg == 'none' or not self._scan_ok or not self._have_pose:
            return

        # 1. Expirar obstáculos viejos del mapa
        changed = self._expire_obstacles()

        # 2. Detectar obstáculo frontal que bloquea la dirección al goal
        obstacle_now = self._min_front < self._front_stop and self._obstacle_toward_goal()
        if obstacle_now:
            self._obs_count += 1
            # Activar bloqueo solo si aún no está activo (primera detección).
            # NO renovar en cada ciclo — eso causaría deadlock cuando el robot
            # intenta seguir la ruta de evasión con el obstáculo todavía al frente.
            # El bloqueo se libera en _path_cb() al recibir la ruta del A*.
            if self._blocking_until == 0.0:
                self._blocking_until = time.monotonic() + self._stop_dur
        else:
            self._obs_count = 0

        # 3. Si obstáculo confirmado Y no inyectamos uno muy recientemente
        if self._obs_count >= self._confirm_n:
            obs_x, obs_y = self._obstacle_world_pos()
            time_since_last = time.monotonic() - self._last_inj_t

            # No re-inyectar si ya hay uno cercano (< 0.70 m) reciente (< 20 s)
            already_covered = (
                self._last_obs_x is not None and
                math.hypot(obs_x - self._last_obs_x,
                           obs_y - self._last_obs_y) < 0.70 and
                time_since_last < 20.0
            )

            if not already_covered:
                dist_robot_to_obs = math.hypot(obs_x - self._robot_x,
                                               obs_y - self._robot_y)
                # Usar el radio configurado completo — el blob debe ser grande para
                # que el A* genere rutas con margen real. No recortar por distancia
                # al robot: el obstáculo se inyecta en la posición del obstáculo,
                # no en la del robot, por lo que el robot nunca queda dentro del blob
                # mientras esté a > inj_radius del obstáculo detectado.
                safe_radius = self._inj_radius

                self.get_logger().warn(
                    f'Obstáculo a {self._min_front:.2f}m → '
                    f'inyectando en ({obs_x:.2f},{obs_y:.2f}) '
                    f'radio={safe_radius:.2f}m (robot a {dist_robot_to_obs:.2f}m)')
                self._inject_and_publish(obs_x, obs_y, safe_radius, self._inj_decay)
                self._last_obs_x = obs_x
                self._last_obs_y = obs_y
                self._last_inj_t = time.monotonic()
                self._obs_count  = 0
                changed = True

        # 4. Mantener activa la ruta de evasión hasta que el robot alcance wp1.
        # El path_planner replantea cada ~1 s con replan_on_new_map — republicar
        # la ruta de evasión en cada ciclo evita que la sobrescriba.
        if self._evade_path is not None:
            dist_wp1 = math.hypot(self._evade_wp1_x - self._robot_x,
                                  self._evade_wp1_y - self._robot_y)
            if dist_wp1 < 0.30:
                # Robot llegó al primer waypoint lateral — dejar que el A* tome el control
                self.get_logger().info(
                    f'Evasión completada (wp1 alcanzado, dist={dist_wp1:.2f}m) '
                    f'— devolviendo control al A*')
                self._evade_path = None
            else:
                # Republicar la ruta de evasión para que el steering la siga
                self._evade_path.header.stamp = self.get_clock().now().to_msg()
                self._pub_path.publish(self._evade_path)

        # Estado para monitoreo
        blocked = self._min_front < self._front_stop
        evading = self._evade_path is not None
        state = 'EVADING' if evading else ('OBSTACLE_DETECTED' if blocked else 'CLEAR')
        s = String(); s.data = state
        self._pub_state.publish(s)

        if self._markers_en and self._last_obs_x is not None:
            self._publish_markers()

    # ── Inyección y gestión del mapa ──────────────────────────────────────────

    def _inject_and_publish(self, obs_x, obs_y, radius, decay):
        """Publica ruta de evasión lateral directa + inyecta blob en el mapa."""
        now_t = time.monotonic()
        self._blocking_until = now_t + self._stop_dur
        self._have_safe_path = False
        self._map_injected_at = now_t
        self.get_logger().warn('BLOCKING cmd_vel — generando ruta de evasión lateral')

        # Elegir lado de evasión: el lado con más espacio libre
        side = 1.0 if self._dist_left >= self._dist_right else -1.0
        side_name = 'IZQ' if side > 0 else 'DER'

        # Dirección perpendicular a la línea robot→goal
        if self._goal_x is not None:
            dir_to_goal = math.atan2(self._goal_y - self._robot_y,
                                     self._goal_x - self._robot_x)
        else:
            dir_to_goal = self._robot_th

        perp = dir_to_goal + math.pi / 2.0
        cos_d = math.cos(dir_to_goal)
        sin_d = math.sin(dir_to_goal)
        cos_p = math.cos(perp)
        sin_p = math.sin(perp)

        lat = self._wall_hw   # offset lateral (0.75 m)
        gx  = self._goal_x if self._goal_x is not None else self._robot_x
        gy  = self._goal_y if self._goal_y is not None else self._robot_y

        # Generar waypoints intermedios cada 0.20 m desde el robot hasta wp_lateral,
        # luego hasta wp_rebase, luego al goal. Así el steering nunca salta ninguno
        # porque cada punto siguiente está a > lookahead_distance del anterior.
        waypoints = []

        # Segmento 1: avanzar lateralmente desde el robot (N pasos de 0.20 m)
        step = 0.20
        n_lateral = max(2, int(math.ceil(lat / step)))
        for i in range(1, n_lateral + 1):
            t = i / n_lateral
            waypoints.append((
                self._robot_x + t * side * lat * cos_p,
                self._robot_y + t * side * lat * sin_p,
            ))

        # Segmento 2: avanzar en dirección al goal hasta rebasar el obstáculo
        # El punto de rebase está perpendicular al obs + adelante
        rebase_x = obs_x + side * lat * cos_p + 0.40 * cos_d
        rebase_y = obs_y + side * lat * sin_p + 0.40 * sin_d
        # Interpolar en 3 pasos desde el último lateral hasta el rebase
        lx, ly = waypoints[-1]
        for i in range(1, 4):
            t = i / 3.0
            waypoints.append((lx + t * (rebase_x - lx), ly + t * (rebase_y - ly)))

        # Punto final: goal original
        waypoints.append((gx, gy))

        # El primer waypoint lateral es el criterio de "evasión completada"
        evade_wp1_x, evade_wp1_y = waypoints[0]

        self.get_logger().warn(
            f'Evasión {side_name}: {len(waypoints)} wps | '
            f'lateral=({evade_wp1_x:.2f},{evade_wp1_y:.2f}) '
            f'rebase=({rebase_x:.2f},{rebase_y:.2f}) goal=({gx:.2f},{gy:.2f}) | '
            f'left={self._dist_left:.2f}m right={self._dist_right:.2f}m')

        evade_path = Path()
        evade_path.header.stamp    = self.get_clock().now().to_msg()
        evade_path.header.frame_id = 'map'
        for wx, wy in waypoints:
            ps = PoseStamped()
            ps.header = evade_path.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.orientation.w = 1.0
            evade_path.poses.append(ps)
        self._pub_path.publish(evade_path)
        self._evade_path  = evade_path
        self._evade_wp1_x = evade_wp1_x
        self._evade_wp1_y = evade_wp1_y
        self._have_safe_path = True
        self._blocking_until = now_t + self._stop_dur

        # Inyectar blob central en el mapa para evitar que el A* vuelva al mismo camino
        if self._base_map is not None:
            self._injected_obs.append((obs_x, obs_y, now_t, radius, decay))
            self._republish_map()

    def _expire_obstacles(self) -> bool:
        """Elimina obstáculos expirados. Retorna True si algo cambió."""
        now    = time.monotonic()
        before = len(self._injected_obs)
        self._injected_obs = [o for o in self._injected_obs if now - o[2] < o[4]]
        if len(self._injected_obs) != before:
            self.get_logger().info(
                f'Obstáculo expirado — quedan {len(self._injected_obs)}')
            self._republish_map()
            return True
        return False

    def _republish_map(self):
        """Construye el mapa con todos los obstáculos activos y lo publica."""
        if self._base_map is None:
            return

        info = self._base_map.info
        res  = info.resolution
        ox   = info.origin.position.x
        oy   = info.origin.position.y
        W    = info.width
        H    = info.height

        data = list(self._base_map.data)

        for obs_x, obs_y, _, radius, _ in self._injected_obs:
            rc = max(1, int(math.ceil(radius / res)))
            cc_center = int((obs_x - ox) / res)
            rc_center = int((obs_y - oy) / res)
            for dr in range(-rc, rc + 1):
                for dc in range(-rc, rc + 1):
                    if math.hypot(dr, dc) > rc:
                        continue
                    rr = rc_center + dr
                    cc = cc_center + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        data[rr * W + cc] = 100

        new_map              = OccupancyGrid()
        new_map.header       = self._base_map.header
        new_map.header.stamp = self.get_clock().now().to_msg()
        new_map.info         = self._base_map.info
        new_map.data         = data
        self._pub_map.publish(new_map)

        n = len(self._injected_obs)
        if n > 0:
            self.get_logger().info(f'Mapa publicado con {n} obstáculo(s) activo(s)')
        else:
            self.get_logger().info('Mapa limpio publicado (sin obstáculos dinámicos)')

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _obstacle_world_pos(self):
        # Empujar el centro del blob al menos (inj_radius + 0.25 m) por delante
        # del robot, para que el robot nunca quede dentro del blob aunque esté
        # muy cerca del obstáculo. Si el obstáculo medido está más lejos, usar
        # esa distancia real.
        min_center_dist = self._inj_radius + 0.25
        dist = max(self._min_front, min_center_dist)
        obs_x = self._robot_x + dist * math.cos(self._robot_th)
        obs_y = self._robot_y + dist * math.sin(self._robot_th)
        return obs_x, obs_y

    def _obstacle_toward_goal(self) -> bool:
        if self._goal_x is None or not self._have_pose:
            return True
        dist_goal = math.hypot(self._goal_x - self._robot_x,
                               self._goal_y - self._robot_y)
        if dist_goal < 0.20:
            return False
        angle_to_goal = math.atan2(self._goal_y - self._robot_y,
                                   self._goal_x - self._robot_x)
        angle_err = abs(self._norm(angle_to_goal - self._robot_th))
        return angle_err < math.radians(90)

    @staticmethod
    def _norm(a):
        while a >  math.pi: a -= 2 * math.pi
        while a < -math.pi: a += 2 * math.pi
        return a

    # ── Marcadores RViz ───────────────────────────────────────────────────────

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
            m.scale.z = 0.05
            m.color.r = 1.0
            m.color.g = 0.4
            m.color.a = 0.6
            arr.markers.append(m)

        # Posición del robot (referencia)
        rr = Marker()
        rr.header.frame_id = 'map'
        rr.header.stamp    = now
        rr.ns = 'bug_robot'
        rr.id = 100
        rr.type   = Marker.SPHERE
        rr.action = Marker.ADD
        rr.pose.position.x = self._robot_x
        rr.pose.position.y = self._robot_y
        rr.pose.orientation.w = 1.0
        rr.scale.x = rr.scale.y = rr.scale.z = 0.12
        rr.color.b = 1.0
        rr.color.a = 0.8
        arr.markers.append(rr)

        self._pub_markers.publish(arr)


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
