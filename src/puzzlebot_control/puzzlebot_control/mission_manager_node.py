"""
mission_manager_node.py — FSM logística del Puzzlebot.

FUNCIÓN:
  Coordina la misión completa de robot logístico:
  localización → búsqueda de pallet (QR) → pick → navegación a expedición
  → búsqueda de tráiler (YOLO logo) → dock → drop → salida.

ESTRATEGIA DE BÚSQUEDA (barrido en serpentina):
  Las zonas de pickup (conveyors/racks) y docks no tienen posiciones exactas
  conocidas de antemano. En vez de ir a puntos fijos, el robot genera una
  trayectoria de barrido en serpentina dentro del bounding box de cada zona
  definido en el YAML. El barrido se interrumpe en cuanto se detecta el QR
  o el logo. Esto hace la búsqueda robusta ante posiciones desconocidas.

  Bounding boxes definidos en YAML (zona_conveyor_*, zona_rack_*, zona_dock_*):
    zona_conveyor_x_min/max, zona_conveyor_y_min/max  — zona roja
    zona_rack_x_min/max,     zona_rack_y_min/max      — zona verde
    zona_dock_x_min/max,     zona_dock_y_min/max      — zona morada
    sweep_step_m   [0.50]  — separación entre pasadas del barrido
    sweep_approach_yaw     — orientación del robot durante el barrido en cada zona

ESTADOS PRINCIPALES:
  INIT_SYSTEM            — espera que subsistemas estén listos
  LOCALIZATION_CHECK     — verifica que la localización sea fiable
  SELECT_MISSION         — elige Misión 1 (conveyor) o Misión 2 (rack)
  NAVIGATE_TO_ZONE       — navega al borde de entrada de la zona de barrido
  EXPLORE_PICKUP_AREA    — barrido en serpentina buscando QR
  APPROACH_PALLET        — se acerca al pallet detectado
  ALIGN_TO_PALLET        — alineación fina con el pallet
  PRE_PICK_VALIDATION    — valida distancia y ángulo antes de tomar
  PICK_MANEUVER          — sube horquillas (maniobra hardcodeada)
  VERIFY_PICK            — confirma que el pick fue exitoso
  NAVIGATE_TO_EXPEDITION — navega a la zona de expedición
  SEARCH_TRAILER_LOGO    — barrido en serpentina buscando logo con YOLO
  MATCH_TRAILER_CLIENT   — confirma que el logo coincide con el cliente del QR
  ALIGN_TO_DOCK          — alineación fina con la entrada del tráiler
  ENTER_TRAILER          — avanza dentro del tráiler
  DROP_PALLET            — baja horquillas
  EXIT_TRAILER           — retrocede y sale
  MISSION_COMPLETE       — misión terminada

ESTADOS DE RECUPERACIÓN:
  LOCALIZATION_RECOVERY  — intenta relocalizarse (gira buscando ArUco)
  RECOVER_QR_VIEW        — gira en sitio para recuperar vista del QR
  RECOVER_LOGO_VIEW      — gira en sitio para recuperar vista del logo
  FAIL_SAFE_STOP         — parada de emergencia, espera RESET
  FAIL_LOCALIZATION      — fallo irrecuperable de localización
  FAIL_QR_NOT_FOUND      — agotó el barrido completo sin encontrar QR
  FAIL_TRAILER_NOT_FOUND — agotó el barrido de docks sin match
  FAIL_PICK              — pick fallido (validación negativa)

TÓPICOS SUSCRITOS:
  /localization/status    (std_msgs/String)              — "OK" | "LOST" | "INITIALIZING"
  /qr/detected            (std_msgs/Bool)                — QR visible en este momento
  /qr/client              (std_msgs/String)              — nombre del cliente leído del QR
  /qr/pose                (geometry_msgs/PoseStamped)    — pose del QR en frame base_footprint
  /detections             (vision_msgs/Detection2DArray) — detecciones YOLO (logos)
  /odom                   (nav_msgs/Odometry)            — pose actual del robot
  /fork/status            (std_msgs/Bool)                — True=horquillas arriba, False=abajo
  /mission_state_in       (std_msgs/String)              — RESET | START_M1 | START_M2 | ABORT

TÓPICOS PUBLICADOS:
  /mission_state          (std_msgs/String)              — estado FSM actual a 2 Hz
  /mission_client         (std_msgs/String)              — cliente activo (del QR)
  /goal_pose              (geometry_msgs/PoseStamped)    — meta de navegación para A* (QoS TRANSIENT_LOCAL)
  /fork/command           (std_msgs/Bool)                — True=subir, False=bajar horquillas
  /cmd_vel_in             (geometry_msgs/Twist)          — maniobras directas (barrido, dock, exit, align);
                                                           pasa por obstacle_avoidance_node → /cmd_vel

PARÁMETROS GENERALES:
  mission_number          [1]      1=conveyor, 2=rack
  localization_timeout    [5.0]    segundos esperando loc OK antes de recovery
  NOTA: el umbral de covarianza que decide OK/LOST vive en kalman_filter_node
        (parámetro loc_lost_thresh). La FSM solo lee el string /localization/status.
  approach_distance       [0.40]   distancia objetivo al acercarse al pallet (m)
  align_angle_thresh      [0.08]   error angular máximo para considerar alineado (rad)
  align_dist_thresh       [0.05]   error de distancia máximo para pick (m)
  dock_center_thresh      [0.05]   tolerancia de centrado del logo en imagen
                                   (|bbox_cx-0.5|) para ALIGN_TO_DOCK por visión
  logo_confidence_thresh  [0.65]   confianza mínima de YOLO para confirmar logo
  logo_confirm_count      [3]      detecciones consecutivas requeridas para match
  dock_forward_speed      [0.10]   velocidad de entrada al tráiler (m/s)
  dock_forward_time       [3.0]    tiempo de avance dentro del tráiler (s)
  exit_backward_speed     [0.10]   velocidad de retroceso al salir (m/s)
  exit_backward_time      [3.0]    tiempo de retroceso al salir (s)
  pick_lift_time          [2.0]    tiempo maniobra de subida (s)
  waypoints_file          ['']     ruta al YAML de waypoints (para expedition_*)
  qr_search_timeout       [60.0]   segundos máximos en EXPLORE_PICKUP_AREA
  logo_search_timeout     [60.0]   segundos máximos en SEARCH_TRAILER_LOGO
  recovery_rotate_speed   [0.3]    velocidad angular en recovery (rad/s)
  recovery_rotate_time    [8.0]    duración de la rotación de recovery (s)

PARÁMETROS DE BARRIDO (zona conveyor — zona roja):
  conveyor_x_min  [0.60]   límite X mínimo del bounding box
  conveyor_x_max  [3.20]   límite X máximo
  conveyor_y_min  [3.80]   límite Y mínimo (el robot barre desde aquí)
  conveyor_y_max  [4.40]   límite Y máximo (hasta la pared)
  conveyor_sweep_yaw [-1.57]  orientación del robot durante el barrido

PARÁMETROS DE BARRIDO (zona rack — zona verde):
  rack_x_min      [1.10]
  rack_x_max      [2.30]
  rack_y_min      [2.10]
  rack_y_max      [3.60]
  rack_sweep_yaw  [0.0]    mira hacia la derecha (+X) al barrer

PARÁMETROS DE BARRIDO (zona dock — zona morada):
  dock_x_min      [1.00]
  dock_x_max      [2.80]
  dock_y_min      [0.80]
  dock_y_max      [1.30]
  dock_sweep_yaw  [1.57]   mira hacia arriba (+Y) hacia los tráilers

PARÁMETRO DE BARRIDO COMÚN:
  sweep_step_m    [0.50]   separación entre pasadas paralelas del barrido
  sweep_wp_tol    [0.20]   tolerancia de llegada a cada waypoint del barrido (m)

PUNTO DE EXPEDICIÓN:
  expedition_x    [1.88]   X del punto intermedio antes de los docks
  expedition_y    [1.80]   Y del punto intermedio
  expedition_yaw  [1.57]
"""

import enum
import math
import os
import time

import rclpy
import yaml
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from vision_msgs.msg import Detection2DArray


# ─── FSM states ──────────────────────────────────────────────────────────────

class State(enum.Enum):
    INIT_SYSTEM            = 'INIT_SYSTEM'
    LOCALIZATION_CHECK     = 'LOCALIZATION_CHECK'
    SELECT_MISSION         = 'SELECT_MISSION'
    NAVIGATE_TO_ZONE       = 'NAVIGATE_TO_ZONE'
    EXPLORE_PICKUP_AREA    = 'EXPLORE_PICKUP_AREA'
    APPROACH_PALLET        = 'APPROACH_PALLET'
    ALIGN_TO_PALLET        = 'ALIGN_TO_PALLET'
    PRE_PICK_VALIDATION    = 'PRE_PICK_VALIDATION'
    PICK_MANEUVER          = 'PICK_MANEUVER'
    VERIFY_PICK            = 'VERIFY_PICK'
    NAVIGATE_TO_EXPEDITION = 'NAVIGATE_TO_EXPEDITION'
    SEARCH_TRAILER_LOGO    = 'SEARCH_TRAILER_LOGO'
    MATCH_TRAILER_CLIENT   = 'MATCH_TRAILER_CLIENT'
    ALIGN_TO_DOCK          = 'ALIGN_TO_DOCK'
    ENTER_TRAILER          = 'ENTER_TRAILER'
    DROP_PALLET            = 'DROP_PALLET'
    EXIT_TRAILER           = 'EXIT_TRAILER'
    MISSION_COMPLETE       = 'MISSION_COMPLETE'
    # Recovery
    LOCALIZATION_RECOVERY  = 'LOCALIZATION_RECOVERY'
    RECOVER_QR_VIEW        = 'RECOVER_QR_VIEW'
    RECOVER_LOGO_VIEW      = 'RECOVER_LOGO_VIEW'
    FAIL_SAFE_STOP         = 'FAIL_SAFE_STOP'
    FAIL_LOCALIZATION      = 'FAIL_LOCALIZATION'
    FAIL_QR_NOT_FOUND      = 'FAIL_QR_NOT_FOUND'
    FAIL_TRAILER_NOT_FOUND = 'FAIL_TRAILER_NOT_FOUND'
    FAIL_PICK              = 'FAIL_PICK'


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _pose_stamped(x: float, y: float, yaw: float, frame: str = 'map') -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = frame
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.orientation.z = math.sin(yaw / 2.0)
    msg.pose.orientation.w = math.cos(yaw / 2.0)
    return msg


def _yaw_from_odom(odom: Odometry) -> float:
    q = odom.pose.pose.orientation
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def _dist(odom: Odometry, x: float, y: float) -> float:
    dx = odom.pose.pose.position.x - x
    dy = odom.pose.pose.position.y - y
    return math.hypot(dx, dy)


# ─── Main node ────────────────────────────────────────────────────────────────

class MissionManagerNode(Node):

    def __init__(self):
        super().__init__('mission_manager_node')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('mission_number',         1)
        self.declare_parameter('localization_timeout',   5.0)
        self.declare_parameter('approach_distance',      0.40)
        self.declare_parameter('align_angle_thresh',     0.08)
        self.declare_parameter('align_dist_thresh',      0.05)
        # Tolerancia de centrado del logo en la imagen para ALIGN_TO_DOCK:
        # |bbox_cx - 0.5| por debajo de esto se considera centrado (fracción del ancho)
        self.declare_parameter('dock_center_thresh',     0.05)
        self.declare_parameter('logo_confidence_thresh', 0.65)
        self.declare_parameter('logo_confirm_count',     3)
        self.declare_parameter('dock_forward_speed',     0.10)
        self.declare_parameter('dock_forward_time',      3.0)
        self.declare_parameter('exit_backward_speed',    0.10)
        self.declare_parameter('exit_backward_time',     3.0)
        self.declare_parameter('pick_lift_time',         2.0)
        self.declare_parameter('waypoints_file',         '')
        self.declare_parameter('qr_search_timeout',      60.0)
        self.declare_parameter('logo_search_timeout',    60.0)
        self.declare_parameter('recovery_rotate_speed',  0.3)
        self.declare_parameter('recovery_rotate_time',   8.0)

        # Bounding boxes de barrido
        self.declare_parameter('conveyor_x_min',    0.60)
        self.declare_parameter('conveyor_x_max',    3.20)
        self.declare_parameter('conveyor_y_min',    3.80)
        self.declare_parameter('conveyor_y_max',    4.40)
        self.declare_parameter('conveyor_sweep_yaw', -1.57)
        self.declare_parameter('rack_x_min',        1.10)
        self.declare_parameter('rack_x_max',        2.30)
        self.declare_parameter('rack_y_min',        2.10)
        self.declare_parameter('rack_y_max',        3.60)
        self.declare_parameter('rack_sweep_yaw',    0.0)
        self.declare_parameter('dock_x_min',        1.00)
        self.declare_parameter('dock_x_max',        2.80)
        self.declare_parameter('dock_y_min',        0.80)
        self.declare_parameter('dock_y_max',        1.30)
        self.declare_parameter('dock_sweep_yaw',    1.57)
        self.declare_parameter('sweep_step_m',      0.50)
        self.declare_parameter('sweep_wp_tol',      0.20)
        self.declare_parameter('sweep_speed',       0.10)   # m/s durante el barrido

        # Punto de expedición
        self.declare_parameter('expedition_x',   1.88)
        self.declare_parameter('expedition_y',   1.80)
        self.declare_parameter('expedition_yaw', 1.57)

        p = self.get_parameter
        self._mission_number        = p('mission_number').value
        self._loc_timeout           = p('localization_timeout').value
        self._approach_dist         = p('approach_distance').value
        self._align_angle_thresh    = p('align_angle_thresh').value
        self._align_dist_thresh     = p('align_dist_thresh').value
        self._dock_center_thresh    = p('dock_center_thresh').value
        self._logo_conf_thresh      = p('logo_confidence_thresh').value
        self._logo_confirm_count    = p('logo_confirm_count').value
        self._dock_fwd_speed        = p('dock_forward_speed').value
        self._dock_fwd_time         = p('dock_forward_time').value
        self._exit_bwd_speed        = p('exit_backward_speed').value
        self._exit_bwd_time         = p('exit_backward_time').value
        self._pick_lift_time        = p('pick_lift_time').value
        self._qr_search_timeout     = p('qr_search_timeout').value
        self._logo_search_timeout   = p('logo_search_timeout').value
        self._recovery_rot_speed    = p('recovery_rotate_speed').value
        self._recovery_rot_time     = p('recovery_rotate_time').value
        self._sweep_step            = p('sweep_step_m').value
        self._sweep_wp_tol          = p('sweep_wp_tol').value
        self._sweep_speed           = p('sweep_speed').value

        # Bounding boxes
        self._bbox = {
            'conveyor': {
                'x_min': p('conveyor_x_min').value, 'x_max': p('conveyor_x_max').value,
                'y_min': p('conveyor_y_min').value, 'y_max': p('conveyor_y_max').value,
                'yaw':   p('conveyor_sweep_yaw').value,
            },
            'rack': {
                'x_min': p('rack_x_min').value, 'x_max': p('rack_x_max').value,
                'y_min': p('rack_y_min').value, 'y_max': p('rack_y_max').value,
                'yaw':   p('rack_sweep_yaw').value,
            },
            'dock': {
                'x_min': p('dock_x_min').value, 'x_max': p('dock_x_max').value,
                'y_min': p('dock_y_min').value, 'y_max': p('dock_y_max').value,
                'yaw':   p('dock_sweep_yaw').value,
            },
        }

        # Punto de expedición (desde parámetros, no desde YAML)
        self._expedition_wp: dict = {
            'x': p('expedition_x').value,
            'y': p('expedition_y').value,
            'yaw': p('expedition_yaw').value,
            'name': 'expedition_center',
        }

        # ── Waypoints ─────────────────────────────────────────────────────────
        self._wps: dict = self._load_waypoints(p('waypoints_file').value)

        # Trayectorias de barrido generadas en SELECT_MISSION
        self._sweep_wps:     list = []   # lista de (x, y, yaw) para el barrido actual
        self._sweep_wp_idx:  int  = 0
        self._dock_sweep_wps:    list = []
        self._dock_sweep_idx:    int  = 0

        # Dock donde se encontró el logo (para ALIGN_TO_DOCK)
        self._matched_dock_pose: dict | None = None
        # Centro horizontal normalizado [0,1] del logo confirmado en la imagen
        # (0.5 = centrado). ALIGN_TO_DOCK lo usa para centrarse sobre el tráiler.
        self._matched_logo_cx: float = 0.5

        # ── Estado FSM ────────────────────────────────────────────────────────
        self._state: State            = State.INIT_SYSTEM
        self._prev_state: State | None = None
        self._state_entry_time: float = time.monotonic()

        # ── Contexto de misión ────────────────────────────────────────────────
        self._loc_status:        str   = 'INITIALIZING'
        self._odom:              Odometry | None = None
        self._qr_detected:       bool  = False
        self._qr_client:         str   = ''
        self._qr_pose:           PoseStamped | None = None
        self._fork_up:           bool  = False
        self._active_client:     str   = ''
        self._logo_detections:   list  = []
        self._logo_confirm_buf:  int   = 0
        self._last_det_time:     float = 0.0

        # Timestamps para timeouts dentro de estados
        self._search_start:   float = 0.0
        self._maneuver_start: float = 0.0

        # ── Suscripciones ─────────────────────────────────────────────────────
        self.create_subscription(String,          '/localization/status', self._loc_cb,     10)
        self.create_subscription(Bool,            '/qr/detected',         self._qr_det_cb,  10)
        self.create_subscription(String,          '/qr/client',           self._qr_cli_cb,  10)
        self.create_subscription(PoseStamped,     '/qr/pose',             self._qr_pose_cb, 10)
        self.create_subscription(Detection2DArray,'/detections',          self._det_cb,     10)
        self.create_subscription(Odometry,        '/odom',                self._odom_cb,    10)
        self.create_subscription(Bool,            '/fork/status',         self._fork_cb,    10)
        self.create_subscription(String,          '/mission_state_in',    self._override_cb,10)

        # ── Publicadores ──────────────────────────────────────────────────────
        # /goal_pose con QoS TRANSIENT_LOCAL + RELIABLE: bug_navigation_node lo
        # suscribe "latched", y un publisher VOLATILE es INCOMPATIBLE con un
        # subscriber TRANSIENT_LOCAL (no se establece la conexión). path_planner
        # (VOLATILE depth 10) sigue recibiendo sin problema porque TRANSIENT_LOCAL
        # es compatible hacia atrás con suscriptores VOLATILE.
        goal_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._pub_state  = self.create_publisher(String,      '/mission_state',   10)
        self._pub_client = self.create_publisher(String,      '/mission_client',  10)
        self._pub_goal   = self.create_publisher(PoseStamped, '/goal_pose',       goal_qos)
        self._pub_fork   = self.create_publisher(Bool,        '/fork/command',    10)
        # Maniobras directas (barrido, alineación, dock, exit) salen por /cmd_vel_in,
        # NO por /cmd_vel: así pasan por obstacle_avoidance_node (dueño de /cmd_vel)
        # en vez de competir con él como segundo publicador del mismo tópico. Esto
        # además da protección de obstáculos (LiDAR + covarianza) durante el barrido.
        self._pub_vel    = self.create_publisher(Twist,       '/cmd_vel_in',      10)

        # Timer principal FSM (10 Hz) y publicación de estado (2 Hz)
        self.create_timer(0.10, self._fsm_step)
        self.create_timer(0.50, self._publish_state)

        self.get_logger().info(
            f'mission_manager_node iniciado — misión {self._mission_number}')

    # ── Carga de waypoints ────────────────────────────────────────────────────

    def _load_waypoints(self, path: str) -> dict:
        if not path or not os.path.isfile(path):
            self.get_logger().warn(
                f'waypoints_file no encontrado: {path!r} — usando waypoints vacíos')
            return {}
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        wps = data.get('waypoints', data)
        self.get_logger().info(f'Waypoints cargados: {list(wps.keys())}')
        return wps

    def _get_waypoints_by_prefix(self, prefix: str) -> list:
        result = [
            {'name': k, **v}
            for k, v in self._wps.items()
            if k.startswith(prefix)
        ]
        result.sort(key=lambda w: w['name'])
        return result

    def _generate_sweep(self, zone: str) -> list:
        """
        Genera la secuencia de puntos para el barrido lineal de la zona.

        conveyor / dock  → barrido en X a Y constante
          El robot llega a la Y de la zona y se mueve solo en X de un extremo
          al otro. La cámara apunta hacia la zona durante todo el recorrido.

        rack             → barrido en Y a X constante
          El robot llega a la X del rack y se mueve solo en Y de extremo a extremo.

        Cada punto de la lista es el goal que el robot debe alcanzar antes de
        continuar al siguiente. La detección del QR/logo interrumpe el barrido
        en cualquier momento entre puntos.

        Devuelve lista de dicts {'x', 'y', 'yaw', 'name'}.
        """
        bb   = self._bbox[zone]
        step = self._sweep_step
        yaw  = bb['yaw']
        wps  = []

        if zone in ('conveyor', 'dock'):
            # Y fija en el borde interior de la zona (robot mira hacia la pared/tráiler)
            y_fixed = bb['y_min'] + 0.15
            x = bb['x_min']
            i = 0
            while x <= bb['x_max'] + 1e-3:
                wps.append({'name': f'{zone}_{i}',
                            'x': round(x, 3), 'y': round(y_fixed, 3), 'yaw': yaw})
                x += step
                i += 1

        else:  # rack — X fija, barrido en Y
            x_fixed = bb['x_min'] + 0.15
            y = bb['y_min']
            i = 0
            while y <= bb['y_max'] + 1e-3:
                wps.append({'name': f'{zone}_{i}',
                            'x': round(x_fixed, 3), 'y': round(y, 3), 'yaw': yaw})
                y += step
                i += 1

        self.get_logger().info(
            f'Barrido "{zone}": {len(wps)} puntos  '
            f'bbox=[{bb["x_min"]:.2f}–{bb["x_max"]:.2f}] × '
            f'[{bb["y_min"]:.2f}–{bb["y_max"]:.2f}]  '
            f'step={step} m')
        return wps

    # ── Callbacks de tópicos ──────────────────────────────────────────────────

    def _loc_cb(self, msg: String):
        self._loc_status = msg.data.strip().upper()

    def _qr_det_cb(self, msg: Bool):
        self._qr_detected = msg.data

    def _qr_cli_cb(self, msg: String):
        if msg.data:
            self._qr_client = msg.data.strip()

    def _qr_pose_cb(self, msg: PoseStamped):
        self._qr_pose = msg

    def _det_cb(self, msg: Detection2DArray):
        self._logo_detections = list(msg.detections)
        self._last_det_time   = time.monotonic()

    def _odom_cb(self, msg: Odometry):
        self._odom = msg

    def _fork_cb(self, msg: Bool):
        self._fork_up = msg.data

    def _override_cb(self, msg: String):
        cmd = msg.data.strip().upper()
        if cmd == 'RESET':
            self._transition(State.INIT_SYSTEM)
        elif cmd == 'START_M1':
            self._mission_number = 1
            self._transition(State.SELECT_MISSION)
        elif cmd == 'START_M2':
            self._mission_number = 2
            self._transition(State.SELECT_MISSION)
        elif cmd == 'ABORT':
            self._transition(State.FAIL_SAFE_STOP)
        else:
            self.get_logger().warn(f'Comando desconocido: {msg.data!r}')

    # ── Publicadores activos ──────────────────────────────────────────────────

    def _publish_state(self):
        msg = String()
        msg.data = self._state.value
        self._pub_state.publish(msg)

        if self._active_client:
            cmsg = String()
            cmsg.data = self._active_client
            self._pub_client.publish(cmsg)

    def _send_goal(self, wp: dict):
        msg = _pose_stamped(wp['x'], wp['y'], wp.get('yaw', 0.0))
        msg.header.stamp = self.get_clock().now().to_msg()
        self._pub_goal.publish(msg)
        self.get_logger().info(
            f'Goal → {wp.get("name","?")}  x={wp["x"]:.2f} y={wp["y"]:.2f}')

    def _send_fork(self, up: bool):
        msg = Bool()
        msg.data = up
        self._pub_fork.publish(msg)

    def _send_vel(self, linear: float, angular: float):
        msg = Twist()
        msg.linear.x  = linear
        msg.angular.z = angular
        self._pub_vel.publish(msg)

    def _stop(self):
        self._send_vel(0.0, 0.0)

    # ── Transición de estado ──────────────────────────────────────────────────

    def _transition(self, new_state: State):
        if new_state == self._state:
            return
        self.get_logger().info(
            f'FSM: {self._state.value} → {new_state.value}')
        self._prev_state = self._state
        self._state = new_state
        self._state_entry_time = time.monotonic()

    def _time_in_state(self) -> float:
        return time.monotonic() - self._state_entry_time

    # ── Paso principal de la FSM ──────────────────────────────────────────────

    def _fsm_step(self):
        s = self._state

        if s == State.INIT_SYSTEM:
            self._step_init_system()
        elif s == State.LOCALIZATION_CHECK:
            self._step_localization_check()
        elif s == State.SELECT_MISSION:
            self._step_select_mission()
        elif s == State.NAVIGATE_TO_ZONE:
            self._step_navigate_to_zone()
        elif s == State.EXPLORE_PICKUP_AREA:
            self._step_explore_pickup_area()
        elif s == State.APPROACH_PALLET:
            self._step_approach_pallet()
        elif s == State.ALIGN_TO_PALLET:
            self._step_align_to_pallet()
        elif s == State.PRE_PICK_VALIDATION:
            self._step_pre_pick_validation()
        elif s == State.PICK_MANEUVER:
            self._step_pick_maneuver()
        elif s == State.VERIFY_PICK:
            self._step_verify_pick()
        elif s == State.NAVIGATE_TO_EXPEDITION:
            self._step_navigate_to_expedition()
        elif s == State.SEARCH_TRAILER_LOGO:
            self._step_search_trailer_logo()
        elif s == State.MATCH_TRAILER_CLIENT:
            self._step_match_trailer_client()
        elif s == State.ALIGN_TO_DOCK:
            self._step_align_to_dock()
        elif s == State.ENTER_TRAILER:
            self._step_enter_trailer()
        elif s == State.DROP_PALLET:
            self._step_drop_pallet()
        elif s == State.EXIT_TRAILER:
            self._step_exit_trailer()
        elif s == State.MISSION_COMPLETE:
            pass  # estado terminal, solo publica estado
        elif s == State.LOCALIZATION_RECOVERY:
            self._step_localization_recovery()
        elif s == State.RECOVER_QR_VIEW:
            self._step_recover_qr_view()
        elif s == State.RECOVER_LOGO_VIEW:
            self._step_recover_logo_view()
        elif s in (State.FAIL_SAFE_STOP, State.FAIL_LOCALIZATION,
                   State.FAIL_QR_NOT_FOUND, State.FAIL_TRAILER_NOT_FOUND,
                   State.FAIL_PICK):
            self._stop()

    # ── Implementaciones de estados ───────────────────────────────────────────

    def _step_init_system(self):
        # Espera mínimo 2 s para que los nodos de localización arranquen,
        # luego pasa a verificar localización.
        if self._time_in_state() > 2.0:
            self._transition(State.LOCALIZATION_CHECK)

    def _step_localization_check(self):
        if self._loc_status == 'OK':
            self._transition(State.SELECT_MISSION)
            return
        if self._loc_status == 'LOST' and self._time_in_state() > self._loc_timeout:
            self._transition(State.LOCALIZATION_RECOVERY)
            return
        # Si sigue INITIALIZING esperamos sin timeout (el ArUco llegará)

    def _step_localization_recovery(self):
        # Gira lentamente para que la cámara barra el entorno buscando ArUco
        self._send_vel(0.0, self._recovery_rot_speed)
        if self._loc_status == 'OK':
            self._stop()
            self._transition(State.LOCALIZATION_CHECK)
            return
        if self._time_in_state() > self._recovery_rot_time:
            self._stop()
            self.get_logger().error('Recovery de localización agotado.')
            self._transition(State.FAIL_LOCALIZATION)

    def _step_select_mission(self):
        zone = 'conveyor' if self._mission_number == 1 else 'rack'

        # Genera la serpentina de pickup y la de docks
        self._sweep_wps      = self._generate_sweep(zone)
        self._sweep_wp_idx   = 0
        self._dock_sweep_wps = self._generate_sweep('dock')
        self._dock_sweep_idx = 0

        self._active_client    = ''
        self._qr_client        = ''
        self._logo_confirm_buf = 0
        self._matched_dock_pose = None

        self.get_logger().info(
            f'Misión {self._mission_number} ({zone}) — '
            f'{len(self._sweep_wps)} wp pickup, '
            f'{len(self._dock_sweep_wps)} wp dock')

        # Navega al primer waypoint del barrido
        self._send_goal(self._sweep_wps[0])
        self._transition(State.NAVIGATE_TO_ZONE)

    def _step_navigate_to_zone(self):
        # Usa A* para llegar al punto de entrada de la zona de barrido.
        # Una vez ahí, el robot se alinea con el yaw correcto y pasa al barrido.
        if self._odom is None:
            return
        entry = self._sweep_wps[0]
        if _dist(self._odom, entry['x'], entry['y']) < self._sweep_wp_tol:
            # Alinea el yaw antes de iniciar el barrido lineal
            target_yaw  = entry['yaw']
            current_yaw = _yaw_from_odom(self._odom)
            err = math.atan2(math.sin(target_yaw - current_yaw),
                             math.cos(target_yaw - current_yaw))
            if abs(err) > self._align_angle_thresh:
                self._send_vel(0.0, max(-0.4, min(0.4, err * 1.5)))
                return
            self._stop()
            self._sweep_wp_idx = 0
            self._search_start = time.monotonic()
            self.get_logger().info(
                f'Zona alcanzada — iniciando barrido lineal '
                f'({len(self._sweep_wps)} puntos)')
            self._transition(State.EXPLORE_PICKUP_AREA)

    def _step_explore_pickup_area(self):
        # QR detectado → para inmediatamente y pasa a acercarse
        if self._qr_detected and self._qr_pose is not None:
            self._stop()
            self._transition(State.APPROACH_PALLET)
            return

        elapsed = time.monotonic() - self._search_start
        if elapsed > self._qr_search_timeout:
            self._stop()
            self.get_logger().warn(
                f'QR no encontrado tras {elapsed:.0f} s de barrido → FAIL')
            self._transition(State.FAIL_QR_NOT_FOUND)
            return

        if self._odom is None:
            return

        wp = self._sweep_wps[self._sweep_wp_idx]
        rx = self._odom.pose.pose.position.x
        ry = self._odom.pose.pose.position.y
        dist_to_wp = math.hypot(wp['x'] - rx, wp['y'] - ry)

        if dist_to_wp < self._sweep_wp_tol:
            next_idx = self._sweep_wp_idx + 1
            if next_idx >= len(self._sweep_wps):
                self._stop()
                self.get_logger().warn('Barrido completo sin QR — RECOVER_QR_VIEW')
                self._transition(State.RECOVER_QR_VIEW)
                return
            self._sweep_wp_idx = next_idx
            self.get_logger().info(
                f'Barrido [{self._sweep_wp_idx}/{len(self._sweep_wps)-1}] '
                f'→ ({self._sweep_wps[self._sweep_wp_idx]["x"]:.2f}, '
                f'{self._sweep_wps[self._sweep_wp_idx]["y"]:.2f})')
            return  # siguiente tick calcula velocidad con wp ya actualizado

        # ── Barrido lineal directo ────────────────────────────────────────────
        # El robot diferencial siempre avanza en su propio eje X (forward).
        # Para movernos en el mapa usamos pure pursuit simple: calculamos el
        # ángulo hacia el waypoint en frame mapa, lo comparamos con el yaw
        # actual, y combinamos avance + corrección angular.
        #
        # Esto funciona para ambos casos (barrido en X y en Y del mapa) sin
        # necesidad de distinguir ejes: el robot siempre mira hacia el siguiente
        # punto y avanza recto.

        desired_yaw = math.atan2(wp['y'] - ry, wp['x'] - rx)
        current_yaw = _yaw_from_odom(self._odom)
        angle_err   = math.atan2(
            math.sin(desired_yaw - current_yaw),
            math.cos(desired_yaw - current_yaw))

        # Velocidad lineal: se frena si el ángulo es grande (gira antes de avanzar)
        angle_factor = max(0.0, 1.0 - abs(angle_err) / (math.pi / 4))
        v   = self._sweep_speed * angle_factor
        ang = max(-0.5, min(0.5, angle_err * 2.0))

        self._send_vel(v, ang)

    def _step_approach_pallet(self):
        # /qr/pose está en frame base_footprint (relativo al robot).
        # La distancia al QR es directamente la norma del vector de posición.
        if self._qr_pose is None:
            self._transition(State.RECOVER_QR_VIEW)
            return

        # Distancia directa desde base_footprint (frame del qr_pose)
        qx_rel = self._qr_pose.pose.position.x   # adelante del robot
        qy_rel = self._qr_pose.pose.position.y   # lateral
        dist   = math.hypot(qx_rel, qy_rel)

        if dist <= self._approach_dist:
            self._stop()
            self._transition(State.ALIGN_TO_PALLET)
            return

        # Avance directo hacia el QR usando control proporcional en base_footprint:
        # linear.x = avanzar, angular.z = corregir ángulo lateral
        angle_to_qr = math.atan2(qy_rel, qx_rel)
        angle_factor = max(0.0, 1.0 - abs(angle_to_qr) / (math.pi / 3))
        v   = min(self._sweep_speed * 1.5, dist * 0.5) * angle_factor
        ang = max(-0.5, min(0.5, angle_to_qr * 2.0))
        self._send_vel(v, ang)

    def _step_align_to_pallet(self):
        # /qr/pose en base_footprint: el ángulo al QR es atan2(y_rel, x_rel).
        # Giramos hasta que el QR quede casi recto al frente (ángulo ≈ 0).
        if self._qr_pose is None:
            self._transition(State.RECOVER_QR_VIEW)
            return

        qx_rel = self._qr_pose.pose.position.x
        qy_rel = self._qr_pose.pose.position.y
        error_angle = math.atan2(qy_rel, qx_rel)

        if abs(error_angle) < self._align_angle_thresh:
            self._stop()
            self._transition(State.PRE_PICK_VALIDATION)
            return

        ang_vel = max(-0.3, min(0.3, error_angle * 1.5))
        self._send_vel(0.0, ang_vel)

    def _step_pre_pick_validation(self):
        # Valida usando /qr/pose en base_footprint (coordenadas relativas al robot)
        if self._qr_pose is None:
            self._transition(State.RECOVER_QR_VIEW)
            return

        qx_rel  = self._qr_pose.pose.position.x
        qy_rel  = self._qr_pose.pose.position.y
        dist    = math.hypot(qx_rel, qy_rel)
        angle_err = abs(math.atan2(qy_rel, qx_rel))

        dist_ok  = dist < (self._approach_dist + self._align_dist_thresh)
        angle_ok = angle_err < self._align_angle_thresh

        if dist_ok and angle_ok:
            self.get_logger().info(
                f'PRE_PICK OK — dist={dist:.3f} m  angle_err={math.degrees(angle_err):.1f}°')
            self._transition(State.PICK_MANEUVER)
        else:
            self.get_logger().warn(
                f'PRE_PICK fallo — dist={dist:.3f} m  angle={math.degrees(angle_err):.1f}° → realinear')
            self._transition(State.ALIGN_TO_PALLET)

    def _step_pick_maneuver(self):
        # Primer tick: registrar tiempo de inicio y mandar comando de subida
        if self._time_in_state() < 0.05:
            self._maneuver_start = time.monotonic()
            self._send_fork(True)   # sube horquillas
            self.get_logger().info('PICK: subiendo horquillas…')
            return

        if time.monotonic() - self._maneuver_start >= self._pick_lift_time:
            self._transition(State.VERIFY_PICK)

    def _step_verify_pick(self):
        # La maniobra de subida ya se completó (venimos de PICK_MANEUVER).
        # Esperamos confirmación de /fork/status == True.
        # Si la Jetson no publica /fork/status, _fork_up arranca False y se
        # esperaría el timeout — por eso el timeout es generoso (pick_lift_time + 2 s).
        # Cuando exista limit switch físico, este estado funciona sin cambios.
        if self._fork_up:
            self.get_logger().info('VERIFY_PICK: horquillas arriba ✓')
            self._active_client = self._qr_client or 'UNKNOWN'
            self.get_logger().info(f'Cliente del pallet: "{self._active_client}"')
            self._send_goal(self._expedition_wp)
            self._transition(State.NAVIGATE_TO_EXPEDITION)
        elif self._time_in_state() > self._pick_lift_time + 2.0:
            self.get_logger().error('VERIFY_PICK: timeout — horquillas no confirmaron')
            self._send_fork(False)
            self._transition(State.FAIL_PICK)

    def _step_navigate_to_expedition(self):
        # Usa A* para llegar al punto intermedio de expedición.
        # Una vez ahí, el barrido de docks es lineal (sin A*).
        if self._odom is None:
            return
        wp = self._expedition_wp
        if _dist(self._odom, wp['x'], wp['y']) < 0.30:
            self._search_start     = time.monotonic()
            self._dock_sweep_idx   = 0
            self._logo_confirm_buf = 0
            self._transition(State.SEARCH_TRAILER_LOGO)

    def _step_search_trailer_logo(self):
        # La detección tiene MÁXIMA prioridad — se evalúa antes que cualquier
        # lógica de movimiento para no perder un logo en el tick de llegada a wp.
        best_score, _, _ = self._best_logo_detection()
        if best_score >= self._logo_conf_thresh:
            self._stop()
            self._transition(State.MATCH_TRAILER_CLIENT)
            return

        elapsed = time.monotonic() - self._search_start
        if elapsed > self._logo_search_timeout:
            self._stop()
            self.get_logger().warn(
                f'Logo no encontrado tras {elapsed:.0f} s → FAIL')
            self._transition(State.FAIL_TRAILER_NOT_FOUND)
            return

        if self._odom is None:
            return

        wp = self._dock_sweep_wps[self._dock_sweep_idx]
        rx = self._odom.pose.pose.position.x
        ry = self._odom.pose.pose.position.y
        dist_to_wp = math.hypot(wp['x'] - rx, wp['y'] - ry)

        if dist_to_wp < self._sweep_wp_tol:
            next_idx = self._dock_sweep_idx + 1
            if next_idx >= len(self._dock_sweep_wps):
                self._stop()
                self.get_logger().warn('Barrido dock completo sin logo — RECOVER_LOGO_VIEW')
                self._transition(State.RECOVER_LOGO_VIEW)
                return
            self._dock_sweep_idx = next_idx
            self.get_logger().info(
                f'Barrido dock [{self._dock_sweep_idx}/{len(self._dock_sweep_wps)-1}] '
                f'→ ({self._dock_sweep_wps[self._dock_sweep_idx]["x"]:.2f}, '
                f'{self._dock_sweep_wps[self._dock_sweep_idx]["y"]:.2f})')
            return  # siguiente tick calcula velocidad con wp actualizado

        # Pure-pursuit hacia el waypoint actual
        desired_yaw = math.atan2(wp['y'] - ry, wp['x'] - rx)
        current_yaw = _yaw_from_odom(self._odom)
        angle_err   = math.atan2(
            math.sin(desired_yaw - current_yaw),
            math.cos(desired_yaw - current_yaw))
        angle_factor = max(0.0, 1.0 - abs(angle_err) / (math.pi / 4))
        v   = self._sweep_speed * angle_factor
        ang = max(-0.5, min(0.5, angle_err * 2.0))
        self._send_vel(v, ang)

    def _step_match_trailer_client(self):
        # Acumula logo_confirm_count detecciones DISTINTAS (no el mismo frame repetido).
        # Usamos el timestamp del último mensaje de detecciones para evitar contar
        # el mismo frame múltiples veces entre ticks del FSM.
        if not hasattr(self, '_last_det_stamp_confirmed'):
            self._last_det_stamp_confirmed = -1.0

        best_score, best_class, best_cx = self._best_logo_detection()

        client_match = (
            not self._active_client
            or best_class.lower() == self._active_client.lower()
        )

        if best_score >= self._logo_conf_thresh and client_match:
            # Solo cuenta si es un frame nuevo (timestamp distinto al último confirmado)
            det_stamp = getattr(self, '_last_det_time', 0.0)
            if det_stamp != self._last_det_stamp_confirmed:
                self._last_det_stamp_confirmed = det_stamp
                self._logo_confirm_buf += 1
                self.get_logger().info(
                    f'Logo match {self._logo_confirm_buf}/{self._logo_confirm_count} '
                    f'— "{best_class}" conf={best_score:.2f}')

            if self._logo_confirm_buf >= self._logo_confirm_count:
                self._logo_confirm_buf = 0
                wp = self._dock_sweep_wps[self._dock_sweep_idx]
                self._matched_dock_pose = wp
                # Guarda el centro horizontal del logo en la imagen para que
                # ALIGN_TO_DOCK centre el robot lateralmente sobre el tráiler
                # (no basta el yaw fijo del bbox: el dock puede estar desplazado).
                self._matched_logo_cx = best_cx
                self.get_logger().info(
                    f'Tráiler confirmado: "{best_class}" en '
                    f'({wp["x"]:.2f}, {wp["y"]:.2f})  bbox_cx={best_cx:.2f}')
                self._transition(State.ALIGN_TO_DOCK)
        else:
            self._logo_confirm_buf = 0
            self._last_det_stamp_confirmed = -1.0
            self._transition(State.SEARCH_TRAILER_LOGO)

    def _best_logo_detection(self) -> tuple:
        """
        Devuelve (mejor_score, clase, bbox_cx) de las detecciones YOLO actuales.

        bbox_cx es el centro horizontal normalizado [0,1] del bbox del mejor
        logo (0.5 = centrado en la imagen). yolo_node publica
        bbox.center.position.x ya normalizado por el ancho de la imagen. Se usa
        en ALIGN_TO_DOCK para centrar el robot lateralmente sobre el tráiler.
        """
        best_score = 0.0
        best_class = ''
        best_cx    = 0.5
        for det in self._logo_detections:
            for hyp in det.results:
                if hyp.hypothesis.score > best_score:
                    best_score = hyp.hypothesis.score
                    best_class = hyp.hypothesis.class_id
                    best_cx    = det.bbox.center.position.x
        return best_score, best_class, best_cx

    def _step_align_to_dock(self):
        # Alineación con el tráiler en dos prioridades:
        #
        #   1. CENTRADO POR VISIÓN (preferente): si el logo sigue visible y la
        #      detección es reciente, giramos hasta que el bbox del logo quede
        #      centrado en la imagen (bbox_cx ≈ 0.5). Esto apunta el robot al
        #      tráiler REAL aunque esté lateralmente desplazado respecto al punto
        #      del barrido donde se confirmó. Es más preciso que el yaw fijo.
        #
        #   2. RESPALDO POR YAW: si el logo ya no es visible (p.ej. el robot se
        #      acercó tanto que el logo salió del cuadro), caemos al yaw del dock
        #      del bbox como antes.
        if self._matched_dock_pose is None or self._odom is None:
            return

        # ¿Hay una detección de logo fresca para centrar por visión?
        logo_fresh = (time.monotonic() - self._last_det_time) < 0.5
        best_score, _, best_cx = self._best_logo_detection()

        if logo_fresh and best_score >= self._logo_conf_thresh:
            # cx en [0,1]; error > 0 → logo a la derecha → girar a la derecha (ang<0)
            cx_err = best_cx - 0.5
            if abs(cx_err) < self._dock_center_thresh:
                self._stop()
                self._maneuver_start = time.monotonic()
                self._transition(State.ENTER_TRAILER)
                return
            ang_vel = max(-0.3, min(0.3, -cx_err * 1.5))
            self._send_vel(0.0, ang_vel)
            return

        # ── Respaldo: alinear al yaw del dock ────────────────────────────────
        desired_yaw = self._matched_dock_pose.get('yaw',
                          self._bbox['dock']['yaw'])
        current_yaw = _yaw_from_odom(self._odom)
        error_angle = math.atan2(
            math.sin(desired_yaw - current_yaw),
            math.cos(desired_yaw - current_yaw))

        if abs(error_angle) < self._align_angle_thresh:
            self._stop()
            self._maneuver_start = time.monotonic()
            self._transition(State.ENTER_TRAILER)
            return

        ang_vel = max(-0.3, min(0.3, error_angle * 1.5))
        self._send_vel(0.0, ang_vel)

    def _step_enter_trailer(self):
        if self._time_in_state() < self._dock_fwd_time:
            self._send_vel(self._dock_fwd_speed, 0.0)
        else:
            self._stop()
            self._transition(State.DROP_PALLET)

    def _step_drop_pallet(self):
        if self._time_in_state() < 0.05:
            self._send_fork(False)
            self.get_logger().info('DROP: bajando horquillas…')
            return
        # Espera confirmación (fork_up==False) o timeout antes de salir
        # Nota: no transiciona en el primer tick (< 0.05 s) para dar tiempo
        # a que llegue el primer mensaje de /fork/status tras el comando.
        fork_confirmed = not self._fork_up
        timed_out      = self._time_in_state() > self._pick_lift_time + 2.0
        if fork_confirmed or timed_out:
            if timed_out and not fork_confirmed:
                self.get_logger().warn('DROP: timeout — saliendo sin confirmación')
            self._transition(State.EXIT_TRAILER)

    def _step_exit_trailer(self):
        if self._time_in_state() < self._exit_bwd_time:
            self._send_vel(-self._exit_bwd_speed, 0.0)
        else:
            self._stop()
            self._transition(State.MISSION_COMPLETE)
            self.get_logger().info('¡Misión completada!')

    # ── Recovery states ───────────────────────────────────────────────────────

    def _step_recover_qr_view(self):
        # Gira en sitio buscando el QR. Si lo encuentra → APPROACH_PALLET.
        # Si agota el tiempo → reinicia el barrido lineal desde el inicio.
        self._send_vel(0.0, self._recovery_rot_speed * 0.8)
        if self._qr_detected and self._qr_pose is not None:
            self._stop()
            self._transition(State.APPROACH_PALLET)
            return
        if self._time_in_state() > self._recovery_rot_time:
            self._stop()
            self._sweep_wp_idx = 0
            self._search_start = time.monotonic()
            # El barrido es lineal; no mandamos goal al A*.
            # EXPLORE_PICKUP_AREA calculará velocidad directa hacia sweep_wps[0].
            self._transition(State.EXPLORE_PICKUP_AREA)

    def _step_recover_logo_view(self):
        # Gira en sitio buscando el logo. Si lo detecta → SEARCH_TRAILER_LOGO
        # para que confirme. Si agota el tiempo → reinicia el barrido de docks.
        self._send_vel(0.0, self._recovery_rot_speed * 0.8)
        best_score, _, _ = self._best_logo_detection()
        if best_score >= self._logo_conf_thresh:
            self._stop()
            self._transition(State.MATCH_TRAILER_CLIENT)
            return
        if self._time_in_state() > self._recovery_rot_time:
            self._stop()
            self._dock_sweep_idx = 0
            self._search_start   = time.monotonic()
            self._transition(State.SEARCH_TRAILER_LOGO)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = MissionManagerNode()
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
