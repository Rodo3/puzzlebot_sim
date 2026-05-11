"""
Keyboard teleoperation node for the Puzzlebot.

Controls
--------
  W / S   : increase / decrease linear velocity
  A / D   : turn left / right (angular velocity)
  Space   : full stop
  Q / ESC : quit

The node publishes geometry_msgs/Twist on /cmd_vel at ~20 Hz.
All motion stops automatically if the key is released (hold-to-move behaviour).
"""

import sys
import tty
import termios
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

BANNER = """
┌─────────────────────────────────┐
│   Puzzlebot Keyboard Teleop     │
│  W/S : forward / backward       │
│  A/D : turn left / right        │
│  SPC : stop                     │
│  Q   : quit                     │
└─────────────────────────────────┘
"""

KEY_MAP = {
    'w': ( 1,  0),
    's': (-1,  0),
    'a': ( 0,  1),
    'd': ( 0, -1),
    ' ': ( 0,  0),  # stop
}


def get_key(fd: int) -> str:
    """Read a single character from the terminal (raw mode, non-blocking 0.1 s)."""
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        # Read up to 3 bytes to handle arrow-key escape sequences without blocking
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            ch2 = sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            return ch + ch2 + ch3
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class TeleopNode(Node):
    def __init__(self):
        super().__init__('teleop_node')

        self.declare_parameter('linear_speed',  0.20)   # m/s
        self.declare_parameter('angular_speed', 0.80)   # rad/s
        self.declare_parameter('cmd_topic',     '/cmd_vel')

        self._lin_max  = self.get_parameter('linear_speed').value
        self._ang_max  = self.get_parameter('angular_speed').value
        topic          = self.get_parameter('cmd_topic').value

        self._pub = self.create_publisher(Twist, topic, 10)
        self._timer = self.create_timer(0.05, self._publish)   # 20 Hz

        self._lin = 0.0
        self._ang = 0.0
        self._running = True

        self._kbd_thread = threading.Thread(target=self._read_keys, daemon=True)
        self._kbd_thread.start()

        print(BANNER)
        self.get_logger().info(
            f"teleop ready  topic={topic}  v_max={self._lin_max} m/s  "
            f"w_max={self._ang_max} rad/s"
        )

    # ── keyboard loop (runs in background thread) ─────────────────────
    def _read_keys(self):
        fd = sys.stdin.fileno()
        while self._running and rclpy.ok():
            try:
                key = get_key(fd)
            except Exception:
                break

            key_lower = key.lower()

            if key_lower in ('q', '\x1b'):   # Q or ESC
                self.get_logger().info('Quit requested — stopping robot.')
                self._lin = 0.0
                self._ang = 0.0
                self._running = False
                rclpy.shutdown()
                break

            if key_lower in KEY_MAP:
                lin_dir, ang_dir = KEY_MAP[key_lower]
                self._lin = lin_dir * self._lin_max
                self._ang = ang_dir * self._ang_max
                self._print_status()
            else:
                # Unknown key → stop (safe default)
                self._lin = 0.0
                self._ang = 0.0

    # ── publish loop (ROS timer) ──────────────────────────────────────
    def _publish(self):
        msg = Twist()
        msg.linear.x  = self._lin
        msg.angular.z = self._ang
        self._pub.publish(msg)

    def _print_status(self):
        print(f'\r  v={self._lin:+.2f} m/s   w={self._ang:+.2f} rad/s    ', end='', flush=True)

    def destroy_node(self):
        self._running = False
        # Publish a final zero-velocity before shutting down
        stop = Twist()
        self._pub.publish(stop)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
