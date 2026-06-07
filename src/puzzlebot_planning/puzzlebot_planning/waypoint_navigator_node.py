"""
waypoint_navigator_node.py — Navegador a waypoints nombrados.

Carga un archivo YAML con waypoints (nombre → x, y, yaw) y publica
geometry_msgs/PoseStamped en /goal_pose cuando recibe el nombre del
waypoint por topic o por servicio.

Topics suscritos:
  /navigate_to_waypoint  (std_msgs/String)  — nombre del waypoint destino

Topics publicados:
  /goal_pose             (geometry_msgs/PoseStamped) — goal para path_planner
  /waypoint_status       (std_msgs/String)           — estado actual

Parámetros:
  waypoints_file  — ruta absoluta al YAML de waypoints
  frame_id        — frame del mapa (default: 'map')
"""

import math
import os

import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Bool, String


class WaypointNavigatorNode(Node):

    def __init__(self):
        super().__init__('waypoint_navigator_node')

        self.declare_parameter('waypoints_file', '')
        self.declare_parameter('frame_id', 'map')

        waypoints_file = self.get_parameter('waypoints_file').value
        self.frame_id  = self.get_parameter('frame_id').value

        self._waypoints: dict = {}
        self._current_goal: str = ''

        self._load_waypoints(waypoints_file)

        # QoS latched para /goal_pose (path_planner usa TRANSIENT_LOCAL)
        goal_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._goal_pub   = self.create_publisher(PoseStamped, '/goal_pose', goal_qos)
        self._status_pub = self.create_publisher(String, '/waypoint_status', 10)

        self._cmd_sub = self.create_subscription(
            String, '/navigate_to_waypoint', self._on_command, 10)
        self._cancel_sub = self.create_subscription(
            Bool, '/navigation/cancel', self._on_cancel, 10)
        self._goal_reached_sub = self.create_subscription(
            Bool, '/navigation/goal_reached', self._on_goal_reached, 10)

        self.get_logger().info(
            f'WaypointNavigator listo — {len(self._waypoints)} waypoints cargados: '
            f'{list(self._waypoints.keys())}')

        self._print_waypoints()

    # ──────────────────────────────────────────────────────────────────────────

    def _load_waypoints(self, path: str) -> None:
        if not path or not os.path.isfile(path):
            self.get_logger().error(
                f'waypoints_file no encontrado: "{path}". '
                'Pasa el parámetro waypoints_file con la ruta correcta.')
            return

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        wps = data.get('waypoints', {})
        for name, cfg in wps.items():
            try:
                self._waypoints[name] = {
                    'x':   float(cfg['x']),
                    'y':   float(cfg['y']),
                    'yaw': float(cfg.get('yaw', 0.0)),
                    'description': cfg.get('description', ''),
                }
            except (KeyError, TypeError) as e:
                self.get_logger().warn(f'Waypoint "{name}" mal formateado: {e}')

        self.get_logger().info(f'Waypoints cargados desde {path}')

    def _print_waypoints(self) -> None:
        self.get_logger().info('══════════════════════════════════════════')
        self.get_logger().info(' WAYPOINTS DISPONIBLES:')
        for name, wp in self._waypoints.items():
            self.get_logger().info(
                f'  {name:20s}  x={wp["x"]:.2f}  y={wp["y"]:.2f}'
                f'  yaw={wp["yaw"]:.2f}  — {wp["description"]}')
        self.get_logger().info('══════════════════════════════════════════')
        self.get_logger().info('Publica el nombre en /navigate_to_waypoint para navegar')

    def _on_command(self, msg: String) -> None:
        name = msg.data.strip().lower()

        # Comando especial: listar waypoints disponibles
        if name in ('list', 'lista', '?'):
            self._publish_status('WAYPOINTS: ' + ', '.join(self._waypoints.keys()))
            return

        # Comando especial: cancelar navegación actual
        if name in ('stop', 'cancel', 'parar'):
            self._current_goal = ''
            self._publish_status('CANCELADO')
            self.get_logger().info('Navegación cancelada')
            return

        if name not in self._waypoints:
            self.get_logger().warn(
                f'Waypoint desconocido: "{name}". '
                f'Disponibles: {list(self._waypoints.keys())}')
            self._publish_status(f'ERROR: waypoint "{name}" no existe')
            return

        self._current_goal = name
        self._send_goal(name)

    def _on_cancel(self, msg: Bool) -> None:
        if not msg.data:
            return
        self._current_goal = ''
        self._publish_status('CANCELADO')
        self.get_logger().info('Navegación cancelada')

    def _on_goal_reached(self, msg: Bool) -> None:
        if not msg.data or not self._current_goal:
            return
        reached = self._current_goal
        self._current_goal = ''
        self._publish_status(f'COMPLETADO: {reached}')
        self.get_logger().info(f'Waypoint completado: {reached}')

    def _send_goal(self, name: str) -> None:
        wp = self._waypoints[name]

        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = wp['x']
        msg.pose.position.y = wp['y']
        msg.pose.position.z = 0.0

        # Quaternion desde yaw
        half = wp['yaw'] / 2.0
        msg.pose.orientation.z = math.sin(half)
        msg.pose.orientation.w = math.cos(half)

        self._goal_pub.publish(msg)
        status = f'NAVEGANDO → {name} ({wp["description"]}) x={wp["x"]:.2f} y={wp["y"]:.2f}'
        self._publish_status(status)
        self.get_logger().info(status)

    def _publish_status(self, text: str) -> None:
        msg = String()
        msg.data = text
        self._status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointNavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
