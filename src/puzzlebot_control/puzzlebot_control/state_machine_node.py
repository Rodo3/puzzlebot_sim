"""
state_machine_node.py — Máquina de estados de misión del Puzzlebot.

FUNCIÓN:
  Coordina a alto nivel en qué modo está operando el robot. No controla
  directamente los motores; administra transiciones y reenvía goals.

ESTADOS:
  IDLE       — robot detenido, esperando una misión
  MAPPING    — ejecutando SLAM exploratorio
  NAVIGATING — siguiendo una ruta planificada hacia un goal
  DOCKING    — acercándose a un objetivo detectado (ArUco / YOLO)
  DONE       — misión completada
  ERROR      — falla irrecuperable; publica velocidad cero y espera reset

TRANSICIONES AUTOMÁTICAS:
  /goal_pose recibido      → IDLE/DONE → NAVIGATING
  Detección con conf. alta → NAVIGATING → DOCKING
  /mission_state_in = 'RESET' → cualquier estado → IDLE

TOPICS SUSCRITOS:
  /goal_pose         (geometry_msgs/PoseStamped)   — meta de navegación
  /planned_path      (nav_msgs/Path)               — ruta calculada
  /detections        (vision_msgs/Detection2DArray) — objetos detectados por YOLO/ArUco
  /mission_state_in  (std_msgs/String)             — override externo ('RESET','MAP','DONE','ERROR')

TOPICS PUBLICADOS:
  /mission_state     (std_msgs/String)             — estado actual a 1 Hz
  /goal_pose_out     (geometry_msgs/PoseStamped)   — reenvío del goal al planner
  /cmd_vel           (geometry_msgs/Twist)         — solo en ERROR (velocidad cero)

PARÁMETROS:
  docking_confidence_thresh [0.60] — umbral de confianza para activar DOCKING
  goal_reached_tolerance    [0.10 m]
"""

import enum

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray


class State(enum.Enum):
    IDLE       = 'IDLE'
    MAPPING    = 'MAPPING'
    NAVIGATING = 'NAVIGATING'
    DOCKING    = 'DOCKING'
    DONE       = 'DONE'
    ERROR      = 'ERROR'


class StateMachineNode(Node):
    def __init__(self):
        super().__init__('state_machine_node')
        self.declare_parameter('docking_confidence_thresh', 0.60)
        self.declare_parameter('goal_reached_tolerance',    0.10)  # metres

        self.dock_conf_thresh  = self.get_parameter('docking_confidence_thresh').value
        self.goal_tol          = self.get_parameter('goal_reached_tolerance').value

        self._state            = State.IDLE
        self._current_path     = None
        self._path_index       = 0

        # Subscriptions
        self.create_subscription(PoseStamped,       '/goal_pose',       self._goal_cb,       10)
        self.create_subscription(Path,              '/planned_path',    self._path_cb,       10)
        self.create_subscription(Detection2DArray,  '/detections',      self._detections_cb, 10)
        self.create_subscription(String,            '/mission_state_in', self._override_cb,  10)

        # Publishers
        self.pub_state  = self.create_publisher(String, '/mission_state', 10)
        self.pub_goal   = self.create_publisher(PoseStamped, '/goal_pose_out', 10)
        self.pub_estop  = self.create_publisher(Twist, '/cmd_vel', 10)  # emergency stop

        # Status timer — publishes current state at 1 Hz
        self.create_timer(1.0, self._publish_state)

        self.get_logger().info('state_machine_node started — state: IDLE')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _goal_cb(self, msg: PoseStamped):
        if self._state in (State.IDLE, State.DONE):
            self._transition(State.NAVIGATING)
            # Forward goal to path planner
            self.pub_goal.publish(msg)

    def _path_cb(self, msg: Path):
        if self._state == State.NAVIGATING:
            self._current_path = msg
            self._path_index   = 0
            self.get_logger().info(f'Path received: {len(msg.poses)} waypoints')

    def _detections_cb(self, msg: Detection2DArray):
        if self._state != State.NAVIGATING:
            return
        for det in msg.detections:
            for hyp in det.results:
                if hyp.hypothesis.score >= self.dock_conf_thresh:
                    self.get_logger().info(
                        f'Target detected (class={hyp.hypothesis.class_id}, '
                        f'conf={hyp.hypothesis.score:.2f}) — switching to DOCKING')
                    self._transition(State.DOCKING)
                    return

    def _override_cb(self, msg: String):
        cmd = msg.data.strip().upper()
        if cmd == 'RESET':
            self._transition(State.IDLE)
        elif cmd == 'MAP':
            self._transition(State.MAPPING)
        elif cmd == 'DONE':
            self._transition(State.DONE)
        elif cmd == 'ERROR':
            self._transition(State.ERROR)
        else:
            self.get_logger().warn(f'Unknown mission command: {msg.data!r}')

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _transition(self, new_state: State):
        if new_state == self._state:
            return
        self.get_logger().info(f'State: {self._state.value} → {new_state.value}')
        self._state = new_state

        if new_state == State.ERROR:
            self._publish_estop()

    def _publish_estop(self):
        self.pub_estop.publish(Twist())

    def _publish_state(self):
        msg = String()
        msg.data = self._state.value
        self.pub_state.publish(msg)


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
