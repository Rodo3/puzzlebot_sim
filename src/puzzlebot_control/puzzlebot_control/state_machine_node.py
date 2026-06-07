"""High-level mission state machine for the Puzzlebot warehouse delivery.

Coordina la misión logística completa:
    escanear QR → recoger pallet (montacargas) → navegar a docks →
    identificar tráiler por logo → depositar pallet.

Estados
-------
IDLE                 Esperando comando de inicio de misión.
WAITING_FOR_GOAL     (Solo Misión 2) Espera un /goal_pose (click en mapa) como
                     punto de inicio.
GOING_TO_START       (Solo Misión 2) Navega al punto elegido por el usuario.
SCANNING_QR          Patrulla patrol_waypoints buscando el QR del cliente.
FORKLIFT_UP          Sube el montacargas para recoger el pallet (stub).
NAVIGATING_TO_DOCKS  Navega al waypoint dock_scan (único hardcodeado).
SCANNING_LOGOS       Desde dock_scan, compara /logo_detection/result con el
                     target leído del QR.
FORKLIFT_DOWN        Baja el montacargas para depositar el pallet (stub).
DONE                 Misión completada — regresa a IDLE.
ERROR                Fallo irrecuperable — publica velocidad cero y espera reset.
MAPPING              (Conservado del esqueleto anterior) Exploración SLAM.

Transiciones
------------
  /mission_start "1"  → SCANNING_QR
  /mission_start "2"  → WAITING_FOR_GOAL → (/goal_pose) → GOING_TO_START →
                        (llegada por /odom) → SCANNING_QR
  SCANNING_QR     ─(QR estable)→ FORKLIFT_UP ─(timeout)→ NAVIGATING_TO_DOCKS
  NAVIGATING_TO_DOCKS ─(llegada)→ SCANNING_LOGOS ─(match logo)→ FORKLIFT_DOWN
  FORKLIFT_DOWN   ─(timeout)→ DONE → IDLE
  /mission_start "stop" (cualquier estado) → IDLE

SEGURIDAD: el nodo NO planea ni evade — delega la navegación publicando nombres
de waypoint en /navigate_to_waypoint (waypoint_navigator → planner). La
detección de llegada se hace comparando /odom con las coordenadas del
waypoint cargadas desde waypoints.yaml. NUNCA publica /initialpose.
"""

import enum
import json
import math
import os
import time

import rclpy
import yaml
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


class State(enum.Enum):
    IDLE                = 'IDLE'
    WAITING_FOR_GOAL    = 'WAITING_FOR_GOAL'
    GOING_TO_START      = 'GOING_TO_START'
    SCANNING_QR         = 'SCANNING_QR'
    FORKLIFT_UP         = 'FORKLIFT_UP'
    NAVIGATING_TO_DOCKS = 'NAVIGATING_TO_DOCKS'
    SCANNING_LOGOS      = 'SCANNING_LOGOS'
    FORKLIFT_DOWN       = 'FORKLIFT_DOWN'
    DONE                = 'DONE'
    ERROR               = 'ERROR'
    MAPPING             = 'MAPPING'


# Colores RGB [0,1] de los markers de RViz (coinciden con el dashboard).
_QR_MARKER_COLOR = (0.13, 0.83, 0.93)        # cyan
_LOGO_MARKER_COLORS = {
    'Walmart': (0.29, 0.84, 0.50),           # verde
    'Pepsi':   (0.97, 0.44, 0.44),           # rojo
    'Amazon':  (0.98, 0.70, 0.13),           # ámbar
}

# Defaults usados si mission_config.yaml no los define.
_DEFAULTS = {
    'qr_logo_map':               {'wolmar': 'Walmart', 'popsi': 'Pepsi', 'emezon': 'Amazon'},
    'dock_waypoint':             'dock_scan',
    'patrol_waypoints':          ['estacion_a', 'estacion_b', 'estacion_c', 'estacion_d', 'estacion_e'],
    'scan_timeout_sec':          8.0,
    'logo_confidence_threshold': 0.70,
    'goal_reached_tolerance':    0.20,
    'qr_stable_frames':          3,
    'logo_stable_frames':        5,
    'forklift_timeout_sec':      3.0,
    'logo_scan_timeout_sec':     30.0,
}


class StateMachineNode(Node):
    def __init__(self):
        super().__init__('state_machine_node')

        # ── Parámetros ─────────────────────────────────────────────────────
        self.declare_parameter('mission_config_file', '')
        self.declare_parameter('waypoints_file', '')
        # Tópico de velocidad para la parada de seguridad (ERROR / stop).
        # DEBE coincidir con el tópico final del robot:
        #   Robot real → /cmd_vel   |   Gazebo → /model/puzzlebot/cmd_vel
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('done_dwell_sec', 3.0)         # tiempo en DONE antes de IDLE
        # Markers de detección para RViz.
        self.declare_parameter('marker_frame', 'odom')        # frame de los markers (RViz los lleva a 'map' por TF)
        self.declare_parameter('marker_forward_offset', 0.30) # m frente al robot donde se planta el marker

        cfg_file       = self.get_parameter('mission_config_file').get_parameter_value().string_value
        wp_param       = self.get_parameter('waypoints_file').get_parameter_value().string_value
        cmd_vel_topic  = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        self._done_dwell  = float(self.get_parameter('done_dwell_sec').value)
        self._marker_frame  = self.get_parameter('marker_frame').get_parameter_value().string_value
        self._marker_offset = float(self.get_parameter('marker_forward_offset').value)

        # ── Cargar mission_config.yaml ─────────────────────────────────────
        self._cfg = dict(_DEFAULTS)
        self._load_mission_config(cfg_file)

        self._qr_logo_map = {k.lower(): v for k, v in self._cfg['qr_logo_map'].items()}
        self._dock_wp     = self._cfg['dock_waypoint']
        self._patrol      = list(self._cfg['patrol_waypoints'])
        self._scan_dwell  = float(self._cfg['scan_timeout_sec'])
        self._logo_thresh = float(self._cfg['logo_confidence_threshold'])
        self._goal_tol    = float(self._cfg['goal_reached_tolerance'])
        self._qr_need     = int(self._cfg['qr_stable_frames'])
        self._logo_need   = int(self._cfg['logo_stable_frames'])
        self._fork_to     = float(self._cfg['forklift_timeout_sec'])
        self._logo_to     = float(self._cfg['logo_scan_timeout_sec'])

        # ── Cargar waypoints.yaml (coordenadas para detección de llegada) ──
        wp_file = wp_param or str(self._cfg.get('waypoints_file', '') or '')
        self._waypoints = {}
        self._load_waypoints(wp_file)

        # ── Estado interno ─────────────────────────────────────────────────
        self._state          = State.IDLE
        self._mission        = None        # '1' | '2' | None
        self._target_logo    = None        # 'Walmart' | 'Pepsi' | 'Amazon'
        self._current_pose   = None        # {'x', 'y', 'theta'} desde /odom
        self._nav_target     = None        # {'x', 'y'} destino de navegación actual
        self._state_enter    = time.monotonic()

        # Markers de RViz (persistentes; se republican completos en cada alta)
        self._markers        = []
        self._marker_id      = 0

        # Patrulla
        self._patrol_index   = 0
        self._patrol_phase   = 'traveling'  # 'traveling' | 'dwelling'
        self._dwell_start    = 0.0

        # Estabilidad de detecciones
        self._qr_candidate   = None
        self._qr_count       = 0
        self._logo_count     = 0

        # ── Suscripciones ──────────────────────────────────────────────────
        self.create_subscription(String,      '/mission_start',          self._mission_cb, 10)
        self.create_subscription(String,      '/qr/detections',          self._qr_cb,      10)
        self.create_subscription(String,      '/logo_detection/result',  self._logo_cb,    10)
        self.create_subscription(Odometry,    '/odom',                   self._odom_cb,    10)
        self.create_subscription(PoseStamped, '/goal_pose',              self._goal_cb,    10)

        # ── Publicadores ───────────────────────────────────────────────────
        self._pub_state = self.create_publisher(String, '/mission_state',         10)
        self._pub_navwp = self.create_publisher(String, '/navigate_to_waypoint',  10)
        self._pub_fork  = self.create_publisher(String, '/forklift/command',      10)
        self._pub_estop = self.create_publisher(Twist,  cmd_vel_topic,            10)
        self._pub_markers = self.create_publisher(MarkerArray, '/mission/markers', 10)

        # ── Timers ─────────────────────────────────────────────────────────
        self.create_timer(0.2, self._tick)            # FSM @ 5 Hz
        self.create_timer(0.5, self._publish_state)   # heartbeat de estado @ 2 Hz

        self.get_logger().info(
            f'state_machine_node listo — estado: IDLE | '
            f'patrol={self._patrol} | dock={self._dock_wp} | '
            f'waypoints cargados: {len(self._waypoints)}')

    # ====================================================================== #
    #  Carga de configuración
    # ====================================================================== #

    def _load_mission_config(self, path: str) -> None:
        if not path or not os.path.isfile(path):
            self.get_logger().warn(
                f'mission_config_file no encontrado ("{path}") — usando defaults.')
            return
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f) or {}
            for key in _DEFAULTS:
                if key in data and data[key] is not None:
                    self._cfg[key] = data[key]
            if 'waypoints_file' in data:
                self._cfg['waypoints_file'] = data['waypoints_file']
            self.get_logger().info(f'mission_config cargado desde {path}')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Error leyendo mission_config: {exc} — usando defaults.')

    def _load_waypoints(self, path: str) -> None:
        if not path or not os.path.isfile(path):
            self.get_logger().warn(
                f'waypoints_file no encontrado ("{path}"). '
                'La detección de llegada no funcionará hasta proveerlo.')
            return
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f) or {}
            for name, cfg in (data.get('waypoints', {}) or {}).items():
                try:
                    self._waypoints[name.lower()] = {
                        'x': float(cfg['x']),
                        'y': float(cfg['y']),
                    }
                except (KeyError, TypeError):
                    self.get_logger().warn(f'Waypoint "{name}" mal formateado, omitido.')
            self.get_logger().info(
                f'Waypoints cargados desde {path}: {list(self._waypoints.keys())}')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Error leyendo waypoints: {exc}')

    # ====================================================================== #
    #  Callbacks de entrada
    # ====================================================================== #

    def _mission_cb(self, msg: String) -> None:
        cmd = msg.data.strip().lower()
        if cmd == 'stop':
            self.get_logger().info('Comando: DETENER misión')
            self._abort_to_idle()
            return
        if cmd == '1':
            if self._state != State.IDLE:
                self.get_logger().warn('Misión ya en curso — ignorando "1". Envía "stop" primero.')
                return
            self._mission = '1'
            self.get_logger().info('Misión 1 iniciada')
            self._clear_markers()
            self._transition(State.SCANNING_QR)
        elif cmd == '2':
            if self._state != State.IDLE:
                self.get_logger().warn('Misión ya en curso — ignorando "2". Envía "stop" primero.')
                return
            self._mission = '2'
            self.get_logger().info('Misión 2 iniciada — esperando click en el mapa')
            self._clear_markers()
            self._transition(State.WAITING_FOR_GOAL)
        else:
            self.get_logger().warn(f'Comando de misión desconocido: {msg.data!r}')

    def _goal_cb(self, msg: PoseStamped) -> None:
        # Solo se captura como punto de inicio durante la Misión 2.
        if self._state != State.WAITING_FOR_GOAL:
            return
        self._nav_target = {
            'x': float(msg.pose.position.x),
            'y': float(msg.pose.position.y),
        }
        self.get_logger().info(
            f'Punto de inicio recibido (click): '
            f'({self._nav_target["x"]:.2f}, {self._nav_target["y"]:.2f}) — navegando')
        self._transition(State.GOING_TO_START)

    def _odom_cb(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._current_pose = {
            'x': float(msg.pose.pose.position.x),
            'y': float(msg.pose.pose.position.y),
            'theta': math.atan2(siny, cosy),
        }

    def _qr_cb(self, msg: String) -> None:
        if self._state != State.SCANNING_QR:
            return
        qr = self._first_known_qr(msg.data)
        if qr is None:
            return
        if qr == self._qr_candidate:
            self._qr_count += 1
        else:
            self._qr_candidate = qr
            self._qr_count = 1
        if self._qr_count >= self._qr_need:
            self._target_logo = self._qr_logo_map[qr]
            self.get_logger().info(
                f'QR "{qr}" confirmado ({self._qr_count} frames) → '
                f'target logo: {self._target_logo}')
            self._add_marker(f'QR: {qr}', _QR_MARKER_COLOR)
            self._transition(State.FORKLIFT_UP)

    def _logo_cb(self, msg: String) -> None:
        if self._state != State.SCANNING_LOGOS or self._target_logo is None:
            return
        if self._logo_present(msg.data, self._target_logo):
            self._logo_count += 1
        else:
            self._logo_count = 0
        if self._logo_count >= self._logo_need:
            self.get_logger().info(
                f'Logo "{self._target_logo}" confirmado ({self._logo_count} frames) → depositar')
            color = _LOGO_MARKER_COLORS.get(self._target_logo, (0.66, 0.55, 0.98))
            self._add_marker(self._target_logo, color)
            self._transition(State.FORKLIFT_DOWN)

    # ── Parsing helpers ───────────────────────────────────────────────────

    def _first_known_qr(self, raw: str):
        try:
            dets = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(dets, list):
            return None
        for det in dets:
            data = str(det.get('data', '')).strip().lower()
            if data in self._qr_logo_map:
                return data
        return None

    def _logo_present(self, raw: str, target: str) -> bool:
        try:
            dets = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return False
        if not isinstance(dets, list):
            return False
        for det in dets:
            if (str(det.get('class_name', '')) == target
                    and float(det.get('confidence', 0.0)) >= self._logo_thresh):
                return True
        return False

    # ====================================================================== #
    #  FSM tick — lógica dependiente de tiempo / llegada
    # ====================================================================== #

    def _tick(self) -> None:
        st = self._state
        now = time.monotonic()

        if st == State.GOING_TO_START:
            if self._arrived():
                self.get_logger().info('Llegó al punto de inicio — comenzando búsqueda de QR')
                self._transition(State.SCANNING_QR)

        elif st == State.SCANNING_QR:
            self._tick_scanning_qr(now)

        elif st == State.FORKLIFT_UP:
            if now - self._state_enter >= self._fork_to:
                self.get_logger().info('Montacargas arriba — navegando a docks')
                self._transition(State.NAVIGATING_TO_DOCKS)

        elif st == State.NAVIGATING_TO_DOCKS:
            if self._arrived():
                self.get_logger().info('Llegó a dock_scan — escaneando logos')
                self._transition(State.SCANNING_LOGOS)

        elif st == State.SCANNING_LOGOS:
            if now - self._state_enter >= self._logo_to:
                self.get_logger().error(
                    f'Timeout buscando logo "{self._target_logo}" — ERROR')
                self._transition(State.ERROR)

        elif st == State.FORKLIFT_DOWN:
            if now - self._state_enter >= self._fork_to:
                self.get_logger().info('Montacargas abajo — misión completada')
                self._transition(State.DONE)

        elif st == State.DONE:
            if now - self._state_enter >= self._done_dwell:
                self._transition(State.IDLE)

    def _tick_scanning_qr(self, now: float) -> None:
        if not self._patrol:
            return  # sin waypoints de patrulla, solo se queda escaneando en sitio
        if self._patrol_phase == 'traveling':
            if self._arrived():
                self._patrol_phase = 'dwelling'
                self._dwell_start = now
                self.get_logger().info(
                    f'En "{self._patrol[self._patrol_index]}" — escaneando QR '
                    f'({self._scan_dwell:.0f}s)')
        elif self._patrol_phase == 'dwelling':
            if now - self._dwell_start >= self._scan_dwell:
                self._patrol_index = (self._patrol_index + 1) % len(self._patrol)
                if self._patrol_index == 0:
                    self.get_logger().info('Patrulla completa sin QR — reiniciando ciclo')
                self._start_patrol_leg()

    def _start_patrol_leg(self) -> None:
        name = self._patrol[self._patrol_index]
        self._patrol_phase = 'traveling'
        self._navigate_to_waypoint(name)

    # ====================================================================== #
    #  Navegación / llegada
    # ====================================================================== #

    def _navigate_to_waypoint(self, name: str) -> None:
        self._pub_navwp.publish(String(data=name))
        wp = self._waypoints.get(name.lower())
        if wp is not None:
            self._nav_target = {'x': wp['x'], 'y': wp['y']}
        else:
            self._nav_target = None
            self.get_logger().warn(
                f'Waypoint "{name}" sin coordenadas — no se podrá detectar llegada.')
        self.get_logger().info(f'navigate_to_waypoint → {name}')

    def _arrived(self) -> bool:
        if self._nav_target is None or self._current_pose is None:
            return False
        dx = self._current_pose['x'] - self._nav_target['x']
        dy = self._current_pose['y'] - self._nav_target['y']
        return math.hypot(dx, dy) <= self._goal_tol

    # ====================================================================== #
    #  Transiciones
    # ====================================================================== #

    def _transition(self, new_state: State) -> None:
        if new_state == self._state:
            return
        self.get_logger().info(f'Estado: {self._state.value} → {new_state.value}')
        self._state = new_state
        self._state_enter = time.monotonic()
        self._on_enter(new_state)
        self._publish_state()

    def _on_enter(self, st: State) -> None:
        if st == State.SCANNING_QR:
            self._qr_candidate = None
            self._qr_count = 0
            self._patrol_index = 0
            self._patrol_phase = 'traveling'
            if self._patrol:
                self._start_patrol_leg()

        elif st == State.FORKLIFT_UP:
            self._pub_fork.publish(String(data='up'))
            self.get_logger().info('forklift/command → up (stub)')

        elif st == State.NAVIGATING_TO_DOCKS:
            self._navigate_to_waypoint(self._dock_wp)

        elif st == State.SCANNING_LOGOS:
            self._logo_count = 0

        elif st == State.FORKLIFT_DOWN:
            self._pub_fork.publish(String(data='down'))
            self.get_logger().info('forklift/command → down (stub)')

        elif st == State.DONE:
            self.get_logger().info(
                f'✓ Misión {self._mission} completada (target: {self._target_logo})')

        elif st == State.ERROR:
            self._pub_navwp.publish(String(data='stop'))   # cancela navegación en curso
            self._pub_estop.publish(Twist())               # parada de seguridad (cmd_vel_topic)
            self.get_logger().error('Estado ERROR — navegación cancelada y velocidad cero.')

        elif st == State.IDLE:
            self._reset_mission_state()

    def _abort_to_idle(self) -> None:
        # Cancela navegación en curso y detiene el robot.
        self._pub_navwp.publish(String(data='stop'))
        self._pub_estop.publish(Twist())
        self._transition(State.IDLE)

    def _reset_mission_state(self) -> None:
        self._mission       = None
        self._target_logo   = None
        self._nav_target    = None
        self._patrol_index  = 0
        self._patrol_phase  = 'traveling'
        self._qr_candidate  = None
        self._qr_count      = 0
        self._logo_count    = 0

    # ====================================================================== #
    #  Markers de RViz (un puntito de color + etiqueta por detección)
    # ====================================================================== #

    def _add_marker(self, text: str, color) -> None:
        """Planta un marker (esfera + etiqueta) frente al robot, en su pose
        actual. Se republica el set completo para que RViz al reconectar lo
        vea. Si no hay pose aún, no hace nada."""
        if self._current_pose is None:
            self.get_logger().warn(f'Sin pose para marcar "{text}" en el mapa.')
            return

        th = self._current_pose.get('theta', 0.0)
        x = self._current_pose['x'] + self._marker_offset * math.cos(th)
        y = self._current_pose['y'] + self._marker_offset * math.sin(th)
        r, g, b = color
        stamp = self.get_clock().now().to_msg()

        sphere = Marker()
        sphere.header.frame_id = self._marker_frame
        sphere.header.stamp = stamp
        sphere.ns = 'mission_detections'
        sphere.id = self._marker_id
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = x
        sphere.pose.position.y = y
        sphere.pose.position.z = 0.12
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.12
        sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = r, g, b, 1.0

        label = Marker()
        label.header.frame_id = self._marker_frame
        label.header.stamp = stamp
        label.ns = 'mission_detections'
        label.id = self._marker_id + 1
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = x
        label.pose.position.y = y
        label.pose.position.z = 0.28
        label.pose.orientation.w = 1.0
        label.scale.z = 0.14
        label.color.r, label.color.g, label.color.b, label.color.a = r, g, b, 1.0
        label.text = text

        self._markers.extend([sphere, label])
        self._marker_id += 2
        self._pub_markers.publish(MarkerArray(markers=self._markers))
        self.get_logger().info(f'Marker "{text}" en ({x:.2f}, {y:.2f}) [{self._marker_frame}]')

    def _clear_markers(self) -> None:
        clear = Marker()
        clear.action = Marker.DELETEALL
        clear.ns = 'mission_detections'
        self._pub_markers.publish(MarkerArray(markers=[clear]))
        self._markers = []
        self._marker_id = 0

    # ====================================================================== #
    #  Publicación de estado
    # ====================================================================== #

    def _publish_state(self) -> None:
        self._pub_state.publish(String(data=self._state.value))


def main(args=None):
    rclpy.init(args=args)
    node = StateMachineNode()
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
