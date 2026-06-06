"""
dynamic_obstacle_manager.py — Gestión robusta de obstáculos dinámicos para Puzzlebot.

Arquitectura:
  - Mantiene current_global_goal persistente (nunca se pierde por un obstáculo)
  - FSM: NORMAL → BRAKE_FOR_REPLAN → REPLAN → FOLLOW_NEW_PATH →
          RECOVERY_REVERSE → RECOVERY_TURN → SAFE_STOP → NORMAL
  - Compuerta de velocidad directa: no depende de Path vacío para detener el robot
  - Capa dinámica temporal con clustering + TTL: no modifica el mapa base
  - Publica /augmented_map que el path_planner usa en lugar de /map
  - Verifica si la ruta actual intersecta obstáculos dinámicos → invalida y replantea
  - Recovery estructurado: retroceso → giro hacia espacio libre → replan

Tópicos suscritos:
  /scan_stamped       (LaserScan)      — LiDAR
  /odom               (Odometry)       — pose actual
  /goal_pose          (PoseStamped)    — goal persistente (TRANSIENT_LOCAL)
  /map                (OccupancyGrid)  — mapa base (TRANSIENT_LOCAL)
  /planned_path       (Path)           — ruta actual del A*
  /cmd_vel_steering   (Twist)          — velocidad del pure pursuit

Tópicos publicados:
  /cmd_vel_in         (Twist)          — velocidad filtrada (compuerta)
  /augmented_map      (OccupancyGrid)  — mapa base + obstáculos dinámicos (TRANSIENT_LOCAL)
  /replan_trigger     (PoseStamped)    — señal para que path_planner replantee ahora
  /dynamic_obstacles  (MarkerArray)    — marcadores RViz de obstáculos activos
  /dom/state          (String)         — estado FSM actual
  /dom/markers        (MarkerArray)    — marcadores completos (goal, estado, zonas)
"""

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped, Twist, Vector3
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,
                        qos_profile_sensor_data)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray


# ── Estados de la FSM ──────────────────────────────────────────────────────────
NORMAL          = 'NORMAL'
BRAKE_FOR_REPLAN = 'BRAKE_FOR_REPLAN'
REPLAN          = 'REPLAN'
FOLLOW_NEW_PATH = 'FOLLOW_NEW_PATH'
RECOVERY_REVERSE = 'RECOVERY_REVERSE'
RECOVERY_TURN   = 'RECOVERY_TURN'
SAFE_STOP       = 'SAFE_STOP'


@dataclass
class DynamicObstacle:
    x: float
    y: float
    radius: float
    inflation: float
    created_at: float
    last_seen: float
    ttl: float
    hit_count: int = 1
    active: bool = True

    def is_expired(self, now: float) -> bool:
        return (now - self.last_seen) > self.ttl

    def total_radius(self) -> float:
        return self.radius + self.inflation


class DynamicObstacleManager(Node):

    def __init__(self):
        super().__init__('dynamic_obstacle_manager')

        # ── Parámetros — detección ────────────────────────────────────────────
        self.declare_parameter('front_detection_angle_deg',  35.0)
        self.declare_parameter('front_stop_distance',         0.65)
        self.declare_parameter('front_slow_distance',         0.90)
        self.declare_parameter('emergency_stop_distance',     0.22)

        # ── Parámetros — obstáculos dinámicos ────────────────────────────────
        self.declare_parameter('dynamic_obstacle_radius_m',   0.35)
        self.declare_parameter('obstacle_inflation_radius_m', 0.70)
        self.declare_parameter('obstacle_cluster_radius_m',   0.45)
        self.declare_parameter('dynamic_obstacle_ttl_sec',    8.0)
        self.declare_parameter('obstacle_reinject_cooldown_sec', 5.0)
        self.declare_parameter('min_obstacle_points',         1)

        # ── Parámetros — replanning ───────────────────────────────────────────
        self.declare_parameter('replan_stop_hold_sec',        1.5)
        self.declare_parameter('max_replan_attempts',         3)
        self.declare_parameter('path_collision_check_enabled', True)
        self.declare_parameter('path_collision_sample_step_m', 0.05)
        self.declare_parameter('replan_timeout_sec',          4.0)

        # ── Parámetros — recovery ─────────────────────────────────────────────
        self.declare_parameter('recovery_enabled',            True)
        self.declare_parameter('reverse_speed',               0.07)
        self.declare_parameter('reverse_duration_sec',        1.5)
        self.declare_parameter('turn_speed',                  0.35)
        self.declare_parameter('turn_duration_sec',           1.2)
        self.declare_parameter('free_space_angle_deg',        70.0)

        # ── Parámetros — tópicos y frames ────────────────────────────────────
        self.declare_parameter('scan_topic',            '/scan_stamped')
        self.declare_parameter('odom_topic',            '/odom')
        self.declare_parameter('goal_topic',            '/goal_pose')
        self.declare_parameter('planned_path_topic',    '/planned_path')
        self.declare_parameter('base_map_topic',        '/map')
        self.declare_parameter('augmented_map_topic',   '/augmented_map')
        self.declare_parameter('cmd_vel_input_topic',   '/cmd_vel_steering')
        self.declare_parameter('cmd_vel_output_topic',  '/cmd_vel_in')
        self.declare_parameter('global_frame',          'map')

        self.declare_parameter('enable_rviz_markers',   True)
        self.declare_parameter('debug_logs',            True)

        # ── Discriminación estática vs dinámica ───────────────────────────────
        # ignore_points_matching_static_map: si True, los puntos del scan que
        # caen sobre celdas ya ocupadas en el mapa base se ignoran — son paredes
        # conocidas, no obstáculos dinámicos nuevos.
        # known_map_match_tolerance_m: margen de tolerancia en metros. Un punto
        # del scan se considera "ya en el mapa" si la celda correspondiente (±
        # este margen) está ocupada en el mapa base.
        self.declare_parameter('ignore_points_matching_static_map', True)
        self.declare_parameter('known_map_match_tolerance_m',       0.12)

        # ── LiDAR simulado — rango de detección ──────────────────────────────
        # En modo desktop (sin LiDAR real), los obstáculos inyectados via
        # /test_obstacle_pose se guardan como PENDIENTES y solo se activan
        # (inyectan en /augmented_map y disparan replanning) cuando el robot
        # se acerca a menos de simulated_lidar_range_m.
        # Simula el comportamiento real del LiDAR: el robot NO sabe que el
        # obstáculo existe hasta estar suficientemente cerca.
        # Típico LiDAR RPLidar A1: rango útil ~3-4 m en entornos cerrados.
        self.declare_parameter('simulated_lidar_range_m', 1.20)

        # Leer parámetros
        self._front_a     = math.radians(self.get_parameter('front_detection_angle_deg').value)
        self._front_stop  = self.get_parameter('front_stop_distance').value
        self._front_slow  = self.get_parameter('front_slow_distance').value
        self._emerg_stop  = self.get_parameter('emergency_stop_distance').value

        self._dyn_radius  = self.get_parameter('dynamic_obstacle_radius_m').value
        self._inflation   = self.get_parameter('obstacle_inflation_radius_m').value
        self._cluster_r   = self.get_parameter('obstacle_cluster_radius_m').value
        self._obs_ttl     = self.get_parameter('dynamic_obstacle_ttl_sec').value
        self._cooldown    = self.get_parameter('obstacle_reinject_cooldown_sec').value
        self._min_pts     = self.get_parameter('min_obstacle_points').value

        self._stop_hold   = self.get_parameter('replan_stop_hold_sec').value
        self._max_replan  = self.get_parameter('max_replan_attempts').value
        self._path_check  = self.get_parameter('path_collision_check_enabled').value
        self._path_step   = self.get_parameter('path_collision_sample_step_m').value
        self._replan_to   = self.get_parameter('replan_timeout_sec').value

        self._recovery_en  = self.get_parameter('recovery_enabled').value
        self._rev_speed    = self.get_parameter('reverse_speed').value
        self._rev_dur      = self.get_parameter('reverse_duration_sec').value
        self._turn_speed   = self.get_parameter('turn_speed').value
        self._turn_dur     = self.get_parameter('turn_duration_sec').value
        self._free_a       = math.radians(self.get_parameter('free_space_angle_deg').value)

        self._markers_en   = self.get_parameter('enable_rviz_markers').value
        self._debug        = self.get_parameter('debug_logs').value
        self._lidar_range  = self.get_parameter('simulated_lidar_range_m').value
        self._ignore_static = self.get_parameter('ignore_points_matching_static_map').value
        self._map_tol       = self.get_parameter('known_map_match_tolerance_m').value

        aug_topic  = self.get_parameter('augmented_map_topic').value
        scan_topic = self.get_parameter('scan_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        goal_topic = self.get_parameter('goal_topic').value
        path_topic = self.get_parameter('planned_path_topic').value
        map_topic  = self.get_parameter('base_map_topic').value
        cmd_in     = self.get_parameter('cmd_vel_input_topic').value
        cmd_out    = self.get_parameter('cmd_vel_output_topic').value

        # ── Estado del robot ──────────────────────────────────────────────────
        self._rx = 0.0
        self._ry = 0.0
        self._rth = 0.0
        self._have_pose = False

        # ── LiDAR ─────────────────────────────────────────────────────────────
        self._min_front   = float('inf')
        self._dist_left   = float('inf')
        self._dist_right  = float('inf')
        self._scan_ranges: Optional[np.ndarray] = None
        self._scan_angles: Optional[np.ndarray] = None
        self._scan_ok     = False

        # ── Goal persistente ──────────────────────────────────────────────────
        self._goal: Optional[PoseStamped] = None
        self._goal_x: Optional[float] = None
        self._goal_y: Optional[float] = None

        # ── Mapa ──────────────────────────────────────────────────────────────
        self._base_map: Optional[OccupancyGrid] = None
        self._dynamic_obstacles: List[DynamicObstacle] = []

        # ── Obstáculos pendientes (existen físicamente, robot aún no los "ve") ─
        # Se activan cuando el robot se acerca a < simulated_lidar_range_m.
        # Estructura: [(x, y, created_at)]
        self._pending_obstacles: list = []

        # ── Ruta actual ───────────────────────────────────────────────────────
        self._current_path: Optional[Path] = None
        self._path_valid  = True
        self._new_path_received = False
        self._path_received_t: float = 0.0

        # ── FSM ───────────────────────────────────────────────────────────────
        self._state       = NORMAL
        self._state_entry = time.monotonic()
        self._replan_attempts = 0
        self._blocking_until: float = 0.0

        # ── QoS ───────────────────────────────────────────────────────────────
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ── Subscripciones ────────────────────────────────────────────────────
        self.create_subscription(LaserScan,    scan_topic, self._scan_cb,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry,     odom_topic,  self._odom_cb,  10)
        self.create_subscription(PoseStamped,  goal_topic,  self._goal_cb,  latched)
        self.create_subscription(OccupancyGrid, map_topic,  self._map_cb,   latched)
        self.create_subscription(Path,         path_topic,  self._path_cb,  10)
        self.create_subscription(Twist,        cmd_in,      self._cmd_cb,   10)
        # /test_obstacle_pose: permite inyectar obstáculos simulados sin LiDAR real.
        # Publicado por dynamic_obstacle_spawner_node en modo desktop (sin Gazebo).
        # Formato: PoseStamped en frame 'map' con posición del obstáculo.
        self.create_subscription(PoseStamped, '/test_obstacle_pose',
                                 self._test_obstacle_cb, 10)

        # ── Publicadores ──────────────────────────────────────────────────────
        self._pub_cmd      = self.create_publisher(Twist,        cmd_out,         10)
        self._pub_aug_map  = self.create_publisher(OccupancyGrid, aug_topic,     latched)
        self._pub_replan   = self.create_publisher(PoseStamped,  '/replan_trigger', latched)
        self._pub_state    = self.create_publisher(String,       '/dom/state',     10)
        self._pub_markers  = self.create_publisher(MarkerArray,  '/dom/markers',   10)
        self._pub_dyn_obs  = self.create_publisher(MarkerArray,  '/dynamic_obstacles', 10)

        # Timer principal 10 Hz
        self.create_timer(0.10, self._loop)

        self.get_logger().info(
            f'dynamic_obstacle_manager OK | '
            f'front_stop={self._front_stop}m  inflation={self._inflation}m  '
            f'ttl={self._obs_ttl}s  cluster_r={self._cluster_r}m')

    # ═══════════════════════════════════════════════════════════════════════════
    # Callbacks
    # ═══════════════════════════════════════════════════════════════════════════

    def _odom_cb(self, msg: Odometry):
        self._rx  = msg.pose.pose.position.x
        self._ry  = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._rth = math.atan2(2 * (q.w * q.z + q.x * q.y),
                               1 - 2 * (q.y * q.y + q.z * q.z))
        self._have_pose = True

    def _goal_cb(self, msg: PoseStamped):
        self._goal   = msg
        self._goal_x = msg.pose.position.x
        self._goal_y = msg.pose.position.y
        self.get_logger().info(
            f'[DOM] Goal persistente guardado: ({self._goal_x:.2f}, {self._goal_y:.2f})')
        # Nuevo goal siempre fuerza replan y sale de SAFE_STOP
        if self._state == SAFE_STOP:
            self._transition(REPLAN)
        elif self._state in (NORMAL, FOLLOW_NEW_PATH):
            self._trigger_replan('nuevo goal recibido')

    def _map_cb(self, msg: OccupancyGrid):
        prev = self._base_map
        self._base_map = msg
        if prev is None or prev.info.width != msg.info.width or prev.info.height != msg.info.height:
            self.get_logger().info(
                f'[DOM] Mapa base actualizado: {msg.info.width}x{msg.info.height} '
                f'@ {msg.info.resolution:.3f} m/px')
        # Publicar siempre el mapa aumentado (con obstáculos activos)
        self._publish_augmented_map()

    def _path_cb(self, msg: Path):
        if len(msg.poses) > 0:
            self._current_path    = msg
            self._path_valid      = True
            self._new_path_received = True
            self._path_received_t = time.monotonic()
            if self._state in (REPLAN, BRAKE_FOR_REPLAN):
                self._transition(FOLLOW_NEW_PATH)
                self._log(f'Nueva ruta recibida ({len(msg.poses)} wps) — siguiendo')
        else:
            # Path vacío del A* significa que no hay ruta posible
            self._path_valid = False
            self._current_path = None

    def _test_obstacle_cb(self, msg: PoseStamped):
        """Recibe obstáculo simulado del spawner.
        Lo guarda como PENDIENTE — el robot no lo 've' hasta estar a < lidar_range_m.
        Esto simula el comportamiento real del LiDAR: detección por proximidad."""
        obs_x = msg.pose.position.x
        obs_y = msg.pose.position.y
        now   = time.monotonic()
        self._pending_obstacles.append((obs_x, obs_y, now))
        self.get_logger().info(
            f'[DOM] Obstáculo PENDIENTE en ({obs_x:.2f},{obs_y:.2f}) — '
            f'esperando que el robot se acerque a {self._lidar_range:.1f}m')

    def _scan_cb(self, msg: LaserScan):
        self._scan_ok = True
        n = len(msg.ranges)
        if n == 0:
            return
        angles = np.arange(n, dtype=float) * msg.angle_increment + msg.angle_min
        r      = np.array(msg.ranges, dtype=float)
        valid  = np.isfinite(r) & (r > msg.range_min) & (r < msg.range_max)

        self._scan_ranges = np.where(valid, r, np.inf)
        self._scan_angles = angles

        fm = valid & (np.abs(angles) <= self._front_a)
        self._min_front  = float(np.min(r[fm])) if fm.any() else float('inf')

        lm = valid & (angles >= math.radians(50)) & (angles <= math.radians(130))
        self._dist_left  = float(np.min(r[lm])) if lm.any() else float('inf')

        rm = valid & (angles >= math.radians(-130)) & (angles <= math.radians(-50))
        self._dist_right = float(np.min(r[rm])) if rm.any() else float('inf')

    def _cmd_cb(self, msg: Twist):
        """Compuerta de velocidad: filtra según estado FSM."""
        now = time.monotonic()

        # Emergencia absoluta — siempre pasa directo (obstacle_avoidance es el último freno)
        # Aquí solo aplicamos la lógica de la FSM
        if self._state in (BRAKE_FOR_REPLAN, REPLAN, SAFE_STOP):
            self._pub_cmd.publish(Twist())
            return

        if self._state == RECOVERY_REVERSE:
            out = Twist()
            out.linear.x  = -self._rev_speed
            out.angular.z = 0.0
            self._pub_cmd.publish(out)
            return

        if self._state == RECOVERY_TURN:
            out = Twist()
            out.angular.z = self._turn_dir * self._turn_speed
            self._pub_cmd.publish(out)
            return

        # NORMAL o FOLLOW_NEW_PATH — pasar el comando con posible frenado suave
        if self._min_front <= self._emerg_stop:
            self._pub_cmd.publish(Twist())
            return

        if self._min_front <= self._front_slow and msg.linear.x > 0:
            scale = (self._min_front - self._emerg_stop) / max(
                self._front_slow - self._emerg_stop, 1e-6)
            scale = max(0.0, min(1.0, scale))
            out = Twist()
            out.linear.x  = msg.linear.x  * scale
            out.angular.z = msg.angular.z * scale
            self._pub_cmd.publish(out)
            return

        self._pub_cmd.publish(msg)

    # ═══════════════════════════════════════════════════════════════════════════
    # Loop principal FSM
    # ═══════════════════════════════════════════════════════════════════════════

    def _loop(self):
        # _have_pose es el único requisito mínimo para que el loop corra.
        # _scan_ok NO es requisito aquí: en modo desktop no hay LiDAR real
        # y el chequeo de pendientes por proximidad debe funcionar igual.
        if not self._have_pose:
            return

        now = time.monotonic()

        # 0. Detectar obstáculos pendientes que el robot ya puede "ver" con LiDAR
        #    Funciona tanto con LiDAR real (_scan_ok=True) como en modo desktop.
        self._check_pending_lidar(now)

        # 1. Expirar obstáculos viejos
        expired = [o for o in self._dynamic_obstacles if o.is_expired(now)]
        if expired:
            self._dynamic_obstacles = [o for o in self._dynamic_obstacles
                                       if not o.is_expired(now)]
            self._log(f'{len(expired)} obstáculo(s) expirado(s) — '
                      f'quedan {len(self._dynamic_obstacles)}')
            self._publish_augmented_map()
            # Si había SAFE_STOP y se limpió el mapa, intentar replan
            if self._state == SAFE_STOP and self._goal is not None:
                self._transition(REPLAN)

        # 2. Verificar si ruta actual sigue válida
        if (self._state in (NORMAL, FOLLOW_NEW_PATH)
                and self._path_check
                and self._current_path is not None
                and self._dynamic_obstacles):
            if self._path_blocked_by_dynamic_obs():
                self._log('Ruta actual bloqueada por obstáculo dinámico — '
                          'triggering global replan to persistent goal')
                self._path_valid = False
                self._detect_and_add_obstacle()  # intenta refinar posición del obs
                self._transition(BRAKE_FOR_REPLAN)
                return

        # 3. Máquina de estados
        if self._state == NORMAL:
            self._state_normal(now)

        elif self._state == BRAKE_FOR_REPLAN:
            self._state_brake(now)

        elif self._state == REPLAN:
            self._state_replan(now)

        elif self._state == FOLLOW_NEW_PATH:
            self._state_follow(now)

        elif self._state == RECOVERY_REVERSE:
            self._state_recovery_reverse(now)

        elif self._state == RECOVERY_TURN:
            self._state_recovery_turn(now)

        elif self._state == SAFE_STOP:
            self._state_safe_stop(now)

        # 4. Forzar stop desde el loop (no solo desde _cmd_cb) cuando corresponde.
        # Esto garantiza que el robot pare aunque no llegue ningún cmd_vel_steering.
        if self._state in (BRAKE_FOR_REPLAN, REPLAN, SAFE_STOP):
            self._pub_cmd.publish(Twist())

        # 5. Publicar estado
        s = String()
        s.data = self._state
        self._pub_state.publish(s)

        # 5. Marcadores RViz
        if self._markers_en:
            self._publish_all_markers()

    # ═══════════════════════════════════════════════════════════════════════════
    # Estados FSM
    # ═══════════════════════════════════════════════════════════════════════════

    def _state_normal(self, now: float):
        if self._min_front < self._front_stop and self._obstacle_toward_goal():
            self._log(f'Obstáculo frontal a {self._min_front:.2f}m — verificando si es dinámico')
            if self._detect_and_add_obstacle():
                self._transition(BRAKE_FOR_REPLAN)

    def _state_brake(self, now: float):
        elapsed = now - self._state_entry
        if elapsed >= self._stop_hold:
            self._log('Stop hold completado → REPLAN')
            self._transition(REPLAN)
        # El _cmd_cb publica Twist() automáticamente en este estado

    def _state_replan(self, now: float):
        elapsed = now - self._state_entry
        # Publicar trigger de replan hacia el goal persistente
        if self._goal is not None:
            trigger = PoseStamped()
            trigger.header.stamp    = self.get_clock().now().to_msg()
            trigger.header.frame_id = 'map'
            trigger.pose = self._goal.pose
            self._pub_replan.publish(trigger)

        if self._new_path_received:
            self._new_path_received = False
            self._transition(FOLLOW_NEW_PATH)
            return

        if elapsed > self._replan_to:
            self._replan_attempts += 1
            self._log(f'Replan timeout — intento {self._replan_attempts}/{self._max_replan}')
            if self._replan_attempts >= self._max_replan:
                if self._recovery_en:
                    self._transition(RECOVERY_REVERSE)
                else:
                    self._transition(SAFE_STOP)
            else:
                # Re-entrar en REPLAN para otro intento
                self._state_entry = now

    def _state_follow(self, now: float):
        # Detectar nuevo obstáculo mientras seguimos ruta
        if self._min_front < self._front_stop and self._obstacle_toward_goal():
            self._log(f'Obstáculo a {self._min_front:.2f}m durante FOLLOW_NEW_PATH — verificando')
            if self._detect_and_add_obstacle():
                self._transition(BRAKE_FOR_REPLAN)
                return

        # Verificar que el goal no se ha alcanzado
        if self._goal_x is not None:
            dist_goal = math.hypot(self._goal_x - self._rx, self._goal_y - self._ry)
            if dist_goal < 0.15:
                self._log(f'Goal alcanzado (dist={dist_goal:.3f}m) → NORMAL')
                self._current_path = None
                self._transition(NORMAL)
                return

        # Si no hay ruta nueva pero llevamos tiempo sin recibirla → volver a REPLAN
        if (self._current_path is None
                and (now - self._path_received_t) > self._replan_to):
            self._transition(REPLAN)

    def _state_recovery_reverse(self, now: float):
        elapsed = now - self._state_entry
        if elapsed >= self._rev_dur:
            self._log('Retroceso completado → RECOVERY_TURN')
            self._choose_turn_direction()
            self._transition(RECOVERY_TURN)

    def _state_recovery_turn(self, now: float):
        elapsed = now - self._state_entry
        if elapsed >= self._turn_dur:
            self._log('Giro completado → REPLAN')
            self._replan_attempts = 0
            self._transition(REPLAN)

    def _state_safe_stop(self, now: float):
        # Vigilar si el obstáculo desaparece (expiró TTL) → intentar replan
        # Esta lógica está cubierta en el bloque de expiración del loop principal
        pass

    # ═══════════════════════════════════════════════════════════════════════════
    # Detección y clustering de obstáculos dinámicos
    # ═══════════════════════════════════════════════════════════════════════════

    def _check_pending_lidar(self, now: float):
        """
        Simula la detección LiDAR por proximidad.
        El robot no sabe que existe el obstáculo hasta estar a < simulated_lidar_range_m.
        Funciona en modo desktop (sin scan real) y con LiDAR real.
        """
        if not self._pending_obstacles:
            return

        still_pending = []
        for (obs_x, obs_y, created_at) in self._pending_obstacles:
            dist = math.hypot(obs_x - self._rx, obs_y - self._ry)
            # Log periódico para no saturar consola (cada ~2s a 10Hz)
            if int(now * 5) % 20 == 0:
                self.get_logger().info(
                    f'[DOM] pendiente ({obs_x:.2f},{obs_y:.2f}) '
                    f'dist={dist:.2f}m / {self._lidar_range:.1f}m')

            if dist <= self._lidar_range:
                self.get_logger().warn(
                    f'[DOM] *** LiDAR DETECTA obstáculo ({obs_x:.2f},{obs_y:.2f}) '
                    f'a {dist:.2f}m — ACTIVANDO en mapa ***')
                self._add_or_update_obstacle(obs_x, obs_y, now)
                # Siempre transicionar a BRAKE_FOR_REPLAN al detectar un obstáculo nuevo
                if self._state not in (BRAKE_FOR_REPLAN, REPLAN,
                                       RECOVERY_REVERSE, RECOVERY_TURN, SAFE_STOP):
                    self._transition(BRAKE_FOR_REPLAN)
            else:
                still_pending.append((obs_x, obs_y, created_at))

        self._pending_obstacles = still_pending

    def _is_static_map_point(self, wx: float, wy: float) -> bool:
        """Retorna True si el punto world (wx, wy) corresponde a una celda ocupada del mapa base.

        Usa una tolerancia de _map_tol metros para absorber pequeños errores de
        odometría y quantización del grid. Si el mapa base no está disponible,
        retorna False (conservador: no filtra nada).
        """
        if self._base_map is None or not self._ignore_static:
            return False
        info = self._base_map.info
        res  = info.resolution
        ox   = info.origin.position.x
        oy   = info.origin.position.y
        W    = info.width
        H    = info.height
        tol_cells = max(1, int(math.ceil(self._map_tol / res)))

        cc = int((wx - ox) / res)
        rc = int((wy - oy) / res)
        for dr in range(-tol_cells, tol_cells + 1):
            for dc in range(-tol_cells, tol_cells + 1):
                rr = rc + dr
                col = cc + dc
                if 0 <= rr < H and 0 <= col < W:
                    val = self._base_map.data[rr * W + col]
                    if val >= 50:
                        return True
        return False

    def _detect_and_add_obstacle(self) -> bool:
        """Detecta posición del obstáculo desde el LaserScan y lo agrega a la capa dinámica.

        Solo procesa puntos que NO están en el mapa base conocido (paredes estáticas).
        Retorna True si se creó o actualizó un obstáculo dinámico, False en caso contrario.
        """
        if not self._have_pose or self._scan_ranges is None:
            return False

        # Proyectar todos los puntos frontales del scan al frame world
        fm = np.abs(self._scan_angles) <= self._front_a
        front_indices = np.where(fm & np.isfinite(self._scan_ranges)
                                 & (self._scan_ranges < self._front_stop))[0]

        if len(front_indices) == 0:
            return False

        # Filtrar puntos que ya están en el mapa estático
        dynamic_pts = []
        for idx in front_indices:
            r     = float(self._scan_ranges[idx])
            angle = float(self._scan_angles[idx])
            wx    = self._rx + r * math.cos(self._rth + angle)
            wy    = self._ry + r * math.sin(self._rth + angle)
            if not self._is_static_map_point(wx, wy):
                dynamic_pts.append((wx, wy, r))

        if not dynamic_pts:
            self._log('Puntos frontales dentro de stop_distance son paredes estáticas — '
                      'ignorando, sin BRAKE_FOR_REPLAN')
            return False

        # Usar centroide de los puntos dinámicos detectados
        xs = [p[0] for p in dynamic_pts]
        ys = [p[1] for p in dynamic_pts]
        obs_x = float(np.mean(xs))
        obs_y = float(np.mean(ys))

        self.get_logger().warn(
            f'[DOM] LiDAR detecta obstáculo DINÁMICO en ({obs_x:.2f},{obs_y:.2f}) '
            f'— {len(dynamic_pts)} punto(s) no estáticos')

        now = time.monotonic()
        self._add_or_update_obstacle(obs_x, obs_y, now)
        return True

    def _add_or_update_obstacle(self, obs_x: float, obs_y: float, now: float):
        """Clustering: actualiza obstáculo existente o crea uno nuevo."""
        for obs in self._dynamic_obstacles:
            if not obs.active:
                continue
            if math.hypot(obs.x - obs_x, obs.y - obs_y) <= self._cluster_r:
                # Mismo cluster — actualizar posición promediada y timestamp
                alpha = 0.3
                obs.x         = (1 - alpha) * obs.x + alpha * obs_x
                obs.y         = (1 - alpha) * obs.y + alpha * obs_y
                obs.last_seen = now
                obs.hit_count += 1
                self._log(f'Obstáculo actualizado en ({obs.x:.2f},{obs.y:.2f}) '
                          f'hits={obs.hit_count}')
                self._publish_augmented_map()
                return

        # Nuevo obstáculo
        # Cooldown: no crear si hay uno reciente muy cercano (< cluster_r + 0.20)
        for obs in self._dynamic_obstacles:
            if math.hypot(obs.x - obs_x, obs.y - obs_y) <= (self._cluster_r + 0.20):
                if (now - obs.last_seen) < self._cooldown:
                    self._log(f'Cooldown activo — no re-inyectando obstáculo cercano')
                    return

        new_obs = DynamicObstacle(
            x=obs_x, y=obs_y,
            radius=self._dyn_radius,
            inflation=self._inflation,
            created_at=now,
            last_seen=now,
            ttl=self._obs_ttl,
        )
        self._dynamic_obstacles.append(new_obs)
        self.get_logger().warn(
            f'[DOM] Nuevo obstáculo dinámico en ({obs_x:.2f},{obs_y:.2f}) '
            f'radio={self._dyn_radius}m inflación={self._inflation}m')
        self._publish_augmented_map()

    # ═══════════════════════════════════════════════════════════════════════════
    # Verificación de ruta bloqueada
    # ═══════════════════════════════════════════════════════════════════════════

    def _path_blocked_by_dynamic_obs(self) -> bool:
        """Retorna True si algún segmento de la ruta actual cruza un obstáculo dinámico inflado."""
        if self._current_path is None or not self._dynamic_obstacles:
            return False

        poses = self._current_path.poses
        for i in range(len(poses)):
            wx = poses[i].pose.position.x
            wy = poses[i].pose.position.y
            # Samplear segmento hasta el siguiente waypoint
            if i + 1 < len(poses):
                nx = poses[i + 1].pose.position.x
                ny = poses[i + 1].pose.position.y
                seg_len = math.hypot(nx - wx, ny - wy)
                steps = max(1, int(seg_len / self._path_step))
                for s in range(steps + 1):
                    t  = s / max(steps, 1)
                    px = wx + t * (nx - wx)
                    py = wy + t * (ny - wy)
                    for obs in self._dynamic_obstacles:
                        if math.hypot(px - obs.x, py - obs.y) < obs.total_radius():
                            return True
            else:
                for obs in self._dynamic_obstacles:
                    if math.hypot(wx - obs.x, wy - obs.y) < obs.total_radius():
                        return True
        return False

    # ═══════════════════════════════════════════════════════════════════════════
    # Mapa aumentado
    # ═══════════════════════════════════════════════════════════════════════════

    def _publish_augmented_map(self):
        """Publica /augmented_map = mapa base + obstáculos dinámicos inflados."""
        if self._base_map is None:
            return

        info = self._base_map.info
        res  = info.resolution
        ox   = info.origin.position.x
        oy   = info.origin.position.y
        W    = info.width
        H    = info.height

        data = list(self._base_map.data)

        for obs in self._dynamic_obstacles:
            total_r  = obs.total_radius()
            rc_total = max(1, int(math.ceil(total_r / res)))
            cc_c     = int((obs.x - ox) / res)
            rc_c     = int((obs.y - oy) / res)

            for dr in range(-rc_total, rc_total + 1):
                for dc in range(-rc_total, rc_total + 1):
                    dist_m = math.hypot(dr, dc) * res
                    if dist_m > total_r:
                        continue
                    rr = rc_c + dr
                    cc = cc_c + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        # Ocupancia gradual: centro=100, borde inflado=70
                        if dist_m <= obs.radius:
                            occ = 100
                        else:
                            frac = (dist_m - obs.radius) / max(obs.inflation, 1e-6)
                            occ  = int(100 - frac * 30)
                        if data[rr * W + cc] < occ:
                            data[rr * W + cc] = occ

        aug              = OccupancyGrid()
        aug.header       = self._base_map.header
        aug.header.stamp = self.get_clock().now().to_msg()
        aug.info         = self._base_map.info
        aug.data         = data
        self._pub_aug_map.publish(aug)

        n = len(self._dynamic_obstacles)
        if n > 0:
            self._log(f'Mapa aumentado publicado con {n} obstáculo(s) dinámico(s)')

    # ═══════════════════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════════════════

    def _trigger_replan(self, reason: str):
        """Envía trigger de replan inmediato sin cambiar estado FSM."""
        if self._goal is None:
            return
        trigger = PoseStamped()
        trigger.header.stamp    = self.get_clock().now().to_msg()
        trigger.header.frame_id = 'map'
        trigger.pose            = self._goal.pose
        self._pub_replan.publish(trigger)
        self._log(f'Replan trigger enviado: {reason}')

    def _obstacle_toward_goal(self) -> bool:
        if self._goal_x is None or not self._have_pose:
            return True
        dist_goal = math.hypot(self._goal_x - self._rx, self._goal_y - self._ry)
        if dist_goal < 0.20:
            return False
        angle_to_goal = math.atan2(self._goal_y - self._ry, self._goal_x - self._rx)
        angle_err = abs(_norm_angle(angle_to_goal - self._rth))
        return angle_err < math.radians(100)

    def _choose_turn_direction(self):
        """Elige sentido de giro hacia el lado con más espacio libre."""
        self._turn_dir = 1.0 if self._dist_left >= self._dist_right else -1.0
        side = 'IZQ' if self._turn_dir > 0 else 'DER'
        self._log(f'Giro de recovery hacia {side} '
                  f'(left={self._dist_left:.2f}m right={self._dist_right:.2f}m)')

    def _transition(self, new_state: str):
        old = self._state
        self._state       = new_state
        self._state_entry = time.monotonic()
        self.get_logger().warn(f'[DOM] FSM: {old} → {new_state}')
        if new_state == REPLAN:
            self._new_path_received = False
            self._publish_augmented_map()
            if self._goal is not None:
                self._trigger_replan(f'transición a {new_state}')
        if new_state == SAFE_STOP:
            self.get_logger().warn(
                '[DOM] SAFE_STOP — robot detenido. Goal persistente mantenido. '
                'Esperando que el obstáculo desaparezca o nuevo goal desde RViz.')

    def _log(self, msg: str):
        if self._debug:
            self.get_logger().info(f'[DOM] {msg}')

    # ═══════════════════════════════════════════════════════════════════════════
    # Marcadores RViz
    # ═══════════════════════════════════════════════════════════════════════════

    def _publish_all_markers(self):
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()

        # Obstáculos dinámicos: radio físico (rojo) + radio inflado (naranja)
        for i, obs in enumerate(self._dynamic_obstacles):
            # Radio físico
            m = _make_cylinder_marker(
                i * 2, 'dyn_obs_phys', obs.x, obs.y, obs.radius * 2, now,
                ColorRGBA(r=1.0, g=0.2, b=0.2, a=0.7))
            arr.markers.append(m)
            # Radio inflado
            m2 = _make_cylinder_marker(
                i * 2 + 1, 'dyn_obs_infl', obs.x, obs.y, obs.total_radius() * 2, now,
                ColorRGBA(r=1.0, g=0.6, b=0.0, a=0.35))
            arr.markers.append(m2)

        # Obstáculos PENDIENTES — el robot aún no los "ve" (gris punteado)
        for i, (px, py, _) in enumerate(self._pending_obstacles):
            # Círculo gris semitransparente = "obstáculo físico, fuera del rango LiDAR"
            pending_m = _make_cylinder_marker(
                500 + i, 'pending_obs', px, py, self._lidar_range * 2, now,
                ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.15))
            arr.markers.append(pending_m)
            # Cubo gris = posición del obstáculo (visible para el operador, no para el robot)
            cube = Marker()
            cube.header.frame_id = 'map'
            cube.header.stamp    = now
            cube.ns   = 'pending_cube'
            cube.id   = 500 + i
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.pose.position.x    = px
            cube.pose.position.y    = py
            cube.pose.position.z    = 0.25
            cube.pose.orientation.w = 1.0
            cube.scale.x = cube.scale.y = 0.20
            cube.scale.z = 0.50
            cube.color = ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.5)
            arr.markers.append(cube)
            # Texto "NO VISTO" sobre el obstáculo
            label = Marker()
            label.header.frame_id = 'map'
            label.header.stamp    = now
            label.ns   = 'pending_label'
            label.id   = 500 + i
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x    = px
            label.pose.position.y    = py
            label.pose.position.z    = 0.65
            label.pose.orientation.w = 1.0
            label.scale.z = 0.10
            label.text    = f'⚠ pendiente\n{math.hypot(px-self._rx, py-self._ry):.1f}m'
            label.color   = ColorRGBA(r=0.8, g=0.8, b=0.2, a=0.9)
            arr.markers.append(label)

        # Goal persistente (estrella verde)
        if self._goal_x is not None:
            g = Marker()
            g.header.frame_id = 'map'
            g.header.stamp    = now
            g.ns   = 'persistent_goal'
            g.id   = 200
            g.type = Marker.SPHERE
            g.action = Marker.ADD
            g.pose.position.x = self._goal_x
            g.pose.position.y = self._goal_y
            g.pose.orientation.w = 1.0
            g.scale.x = g.scale.y = g.scale.z = 0.18
            g.color = ColorRGBA(r=0.0, g=1.0, b=0.2, a=1.0)
            arr.markers.append(g)

        # Estado FSM como texto
        txt = Marker()
        txt.header.frame_id = 'map'
        txt.header.stamp    = now
        txt.ns   = 'fsm_state'
        txt.id   = 300
        txt.type = Marker.TEXT_VIEW_FACING
        txt.action = Marker.ADD
        txt.pose.position.x = self._rx
        txt.pose.position.y = self._ry + 0.30
        txt.pose.orientation.w = 1.0
        txt.scale.z = 0.15
        txt.text    = self._state
        txt.color   = _state_color(self._state)
        arr.markers.append(txt)

        # Robot position
        rr = _make_sphere_marker(400, 'robot_pos', self._rx, self._ry, 0.10, now,
                                 ColorRGBA(r=0.2, g=0.4, b=1.0, a=0.8))
        arr.markers.append(rr)

        self._pub_markers.publish(arr)

        # Tópico separado /dynamic_obstacles solo con los obstáculos
        dyn_arr = MarkerArray()
        for i, obs in enumerate(self._dynamic_obstacles):
            m = _make_cylinder_marker(
                i, 'dynamic_obstacles', obs.x, obs.y, obs.total_radius() * 2, now,
                ColorRGBA(r=1.0, g=0.4, b=0.0, a=0.5))
            dyn_arr.markers.append(m)
        self._pub_dyn_obs.publish(dyn_arr)


# ── Utilidades de marcadores ───────────────────────────────────────────────────

def _make_cylinder_marker(mid, ns, x, y, diameter, stamp, color):
    m = Marker()
    m.header.frame_id = 'map'
    m.header.stamp    = stamp
    m.ns     = ns
    m.id     = mid
    m.type   = Marker.CYLINDER
    m.action = Marker.ADD
    m.pose.position.x = x
    m.pose.position.y = y
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = diameter
    m.scale.z = 0.05
    m.color   = color
    return m


def _make_sphere_marker(mid, ns, x, y, size, stamp, color):
    m = Marker()
    m.header.frame_id = 'map'
    m.header.stamp    = stamp
    m.ns     = ns
    m.id     = mid
    m.type   = Marker.SPHERE
    m.action = Marker.ADD
    m.pose.position.x = x
    m.pose.position.y = y
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = m.scale.z = size
    m.color   = color
    return m


def _state_color(state: str) -> ColorRGBA:
    colors = {
        NORMAL:           ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
        BRAKE_FOR_REPLAN: ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0),
        REPLAN:           ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),
        FOLLOW_NEW_PATH:  ColorRGBA(r=0.0, g=0.8, b=1.0, a=1.0),
        RECOVERY_REVERSE: ColorRGBA(r=1.0, g=0.0, b=0.5, a=1.0),
        RECOVERY_TURN:    ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0),
        SAFE_STOP:        ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
    }
    return colors.get(state, ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0))


def _norm_angle(a: float) -> float:
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


def main(args=None):
    rclpy.init(args=args)
    node = DynamicObstacleManager()
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
