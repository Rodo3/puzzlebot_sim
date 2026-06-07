"""
mission_stub_runner — publica una secuencia de goal_pose para simular una misión.

Parámetros:
  waypoints_x   (double_array)  lista de coordenadas X de los waypoints
  waypoints_y   (double_array)  lista de coordenadas Y de los waypoints
  start_x       (double)        si != nan, se antepone como primer waypoint
  start_y       (double)        ídem para Y
  step_timeout  (double)        segundos entre waypoints (default: 20.0)
"""

import math

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


class MissionStubRunner(Node):

    def __init__(self):
        super().__init__('mission_stub_runner')

        self.declare_parameter('waypoints_x',  [3.5, 0.5])
        self.declare_parameter('waypoints_y',  [2.0, 0.5])
        self.declare_parameter('start_x',      math.nan)
        self.declare_parameter('start_y',      math.nan)
        self.declare_parameter('step_timeout', 20.0)

        xs = list(self.get_parameter('waypoints_x').get_parameter_value().double_array_value)
        ys = list(self.get_parameter('waypoints_y').get_parameter_value().double_array_value)
        sx = self.get_parameter('start_x').get_parameter_value().double_value
        sy = self.get_parameter('start_y').get_parameter_value().double_value

        # Si se proveyó un punto de inicio, anteponerlo a los waypoints
        if not (math.isnan(sx) or math.isnan(sy)):
            xs.insert(0, sx)
            ys.insert(0, sy)

        self._waypoints    = list(zip(xs, ys))
        self._step_timeout = self.get_parameter('step_timeout').get_parameter_value().double_value
        self._idx          = 0
        self._timer        = None

        self._pub_goal  = self.create_publisher(PoseStamped, '/goal_pose', _LATCHED)
        self._pub_state = self.create_publisher(String, '/mission_state', 10)

        self.get_logger().info(
            f'mission_stub_runner: {len(self._waypoints)} waypoints, '
            f'timeout={self._step_timeout}s'
        )

        # Publicar primer waypoint tras 1 s (da tiempo al nav stack a estar listo)
        self._timer = self.create_timer(1.0, self._send_next)

    def _send_next(self):
        # Cancelar el timer único de esta llamada
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

        if self._idx >= len(self._waypoints):
            self.get_logger().info('Todos los waypoints completados — misión terminada')
            msg = String()
            msg.data = 'DONE'
            self._pub_state.publish(msg)
            raise SystemExit

        x, y = self._waypoints[self._idx]
        self._idx += 1

        goal = PoseStamped()
        goal.header.stamp    = self.get_clock().now().to_msg()
        goal.header.frame_id = 'map'
        goal.pose.position.x = float(x)
        goal.pose.position.y = float(y)
        goal.pose.orientation.w = 1.0

        self._pub_goal.publish(goal)
        self.get_logger().info(
            f'[stub] Waypoint {self._idx}/{len(self._waypoints)}: ({x:.2f}, {y:.2f})'
        )

        msg = String()
        msg.data = f'NAVIGATING_TO_WP_{self._idx}'
        self._pub_state.publish(msg)

        # Programar siguiente waypoint después de step_timeout
        self._timer = self.create_timer(self._step_timeout, self._send_next)


def main(args=None):
    rclpy.init(args=args)
    node = MissionStubRunner()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
