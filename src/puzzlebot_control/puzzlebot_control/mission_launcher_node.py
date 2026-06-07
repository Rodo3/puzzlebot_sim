"""
mission_launcher_node — recibe /mission_start y lanza ros2 launch como subproceso.

Misión 1: lanza inmediatamente mission1.launch.py
Misión 2: publica WAITING_FOR_GOAL, espera click en el mapa (/goal_pose),
          luego lanza mission2.launch.py start_x:=X start_y:=Y
"""

import os
import signal
import subprocess

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String


_LATCHED = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
)


class MissionLauncherNode(Node):

    def __init__(self):
        super().__init__('mission_launcher_node')

        self._state            = 'IDLE'
        self._process          = None
        self._waiting_for_goal = False
        self._auto_idle_timer  = None

        self._pub_state = self.create_publisher(String, '/mission_state', 10)

        # Suscribe a /goal_pose con TRANSIENT_LOCAL para capturar el click del usuario.
        # El mensaje cacheado al arrancar llega con _waiting_for_goal=False → ignorado.
        self._sub_goal = self.create_subscription(
            PoseStamped, '/goal_pose', self._goal_cb, _LATCHED)
        self._sub_start = self.create_subscription(
            String, '/mission_start', self._start_cb, 10)

        self.create_timer(1.0, self._check_process)

        self._publish_state('IDLE')
        self.get_logger().info('mission_launcher_node listo')

    # ── callbacks ──────────────────────────────────────────────────────────────

    def _start_cb(self, msg: String):
        cmd = msg.data.strip()
        if cmd == 'stop':
            self._stop()
        elif cmd == '1':
            self._launch_mission1()
        elif cmd == '2':
            self._enter_waiting_for_goal()

    def _goal_cb(self, msg: PoseStamped):
        if not self._waiting_for_goal:
            return
        self._waiting_for_goal = False
        x = round(msg.pose.position.x, 3)
        y = round(msg.pose.position.y, 3)
        self.get_logger().info(f'Waypoint recibido: ({x}, {y}) — lanzando Misión 2')
        self._launch_mission2(x, y)

    # ── lógica de misiones ─────────────────────────────────────────────────────

    def _enter_waiting_for_goal(self):
        self._kill_process()
        self._waiting_for_goal = True
        self._publish_state('WAITING_FOR_GOAL')
        self.get_logger().info('Misión 2: haz click en el mapa para elegir el punto de inicio')

    def _launch_mission1(self):
        self._kill_process()
        self._waiting_for_goal = False
        self.get_logger().info('Lanzando Misión 1…')
        self._process = subprocess.Popen(
            ['ros2', 'launch', 'puzzlebot_control', 'mission1.launch.py'],
            start_new_session=True,
        )
        self._publish_state('RUNNING_M1')

    def _launch_mission2(self, x: float, y: float):
        self._kill_process()
        self.get_logger().info(f'Lanzando Misión 2 (inicio: {x}, {y})…')
        self._process = subprocess.Popen(
            [
                'ros2', 'launch', 'puzzlebot_control', 'mission2.launch.py',
                f'start_x:={x}', f'start_y:={y}',
            ],
            start_new_session=True,
        )
        self._publish_state('RUNNING_M2')

    def _stop(self):
        self._kill_process()
        self._waiting_for_goal = False
        self._publish_state('IDLE')
        self.get_logger().info('Misión detenida')

    # ── utilidades ─────────────────────────────────────────────────────────────

    def _check_process(self):
        if self._process is not None and self._process.poll() is not None:
            self.get_logger().info('Proceso de misión finalizado')
            self._process = None
            self._publish_state('DONE')
            # Auto-reset a IDLE después de 3 s para poder iniciar otra misión
            self._auto_idle_timer = self.create_timer(3.0, self._auto_idle)

    def _auto_idle(self):
        if self._auto_idle_timer is not None:
            self._auto_idle_timer.cancel()
            self._auto_idle_timer = None
        if self._state == 'DONE':
            self._publish_state('IDLE')
            self.get_logger().info('Listo para nueva misión')

    def _kill_process(self):
        if self._process is None:
            return
        if self._process.poll() is None:
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                self._process.wait(timeout=4)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
        self._process = None

    def _publish_state(self, state: str):
        self._state = state
        msg = String()
        msg.data = state
        self._pub_state.publish(msg)

    def destroy_node(self):
        self._kill_process()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MissionLauncherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
