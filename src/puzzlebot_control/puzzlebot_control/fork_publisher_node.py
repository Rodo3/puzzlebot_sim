"""
fork_publisher_node.py — Control manual de horquillas desde el PC operador.

FUNCIÓN:
  Publica /fork/command (std_msgs/Bool) con un valor booleano:
    True  → subir horquillas (PICK)
    False → bajar horquillas (DROP / neutro)

  La Jetson tiene un nodo receptor que actúa sobre el actuador real.
  Este nodo NO controla el actuador directamente; solo publica el comando.

USO INTERACTIVO (teclado):
  ros2 run puzzlebot_control fork_publisher_node

  Teclas:
    u / U → subir horquillas  (publica True)
    d / D → bajar horquillas  (publica False)
    q / Q → salir

USO CON ARGUMENTO (one-shot para scripting o launch):
  ros2 run puzzlebot_control fork_publisher_node --ros-args -p one_shot:=true -p one_shot_value:=true

PARÁMETROS:
  one_shot        [false]  Si true, publica una vez y termina
  one_shot_value  [false]  Valor a publicar en modo one_shot
  publish_rate    [10.0]   Hz de publicación continua del estado actual
  topic           [/fork/command]  Tópico de salida

TÓPICOS PUBLICADOS:
  /fork/command   (std_msgs/Bool)  — True=arriba, False=abajo
"""

import sys
import termios
import threading
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


class ForkPublisherNode(Node):

    def __init__(self):
        super().__init__('fork_publisher_node')

        self.declare_parameter('one_shot',       False)
        self.declare_parameter('one_shot_value', False)
        self.declare_parameter('publish_rate',   10.0)
        self.declare_parameter('topic',          '/fork/command')

        self._one_shot       = self.get_parameter('one_shot').value
        self._one_shot_value = self.get_parameter('one_shot_value').value
        topic                = self.get_parameter('topic').value
        rate                 = self.get_parameter('publish_rate').value

        self._fork_up = False
        self._pub = self.create_publisher(Bool, topic, 10)

        if self._one_shot:
            self._publish(self._one_shot_value)
            self.get_logger().info(
                f'one_shot: publicado {"True (arriba)" if self._one_shot_value else "False (abajo)"} en {topic}')
            raise SystemExit(0)

        self.create_timer(1.0 / rate, self._timer_cb)
        self.get_logger().info(
            f'fork_publisher_node iniciado en {topic} a {rate} Hz\n'
            '  u/U → subir  |  d/D → bajar  |  q/Q → salir')

    def _publish(self, value: bool):
        msg = Bool()
        msg.data = value
        self._pub.publish(msg)

    def _timer_cb(self):
        self._publish(self._fork_up)

    def set_fork(self, up: bool):
        self._fork_up = up
        label = 'ARRIBA ↑' if up else 'ABAJO  ↓'
        self.get_logger().info(f'Horquillas → {label}')
        self._publish(up)


def _get_key(fd, old_settings):
    """Lee un carácter del terminal sin eco."""
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def main(args=None):
    rclpy.init(args=args)
    node = ForkPublisherNode()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    print('\n─── Fork Publisher ───────────────────────────')
    print('  u → subir horquillas    (publica True)')
    print('  d → bajar horquillas    (publica False)')
    print('  q → salir')
    print('──────────────────────────────────────────────\n')

    stop_event = threading.Event()

    def keyboard_thread():
        # Hilo separado para leer teclado sin bloquear rclpy.spin()
        while not stop_event.is_set():
            try:
                key = _get_key(fd, old_settings)
            except Exception:
                break
            if key in ('u', 'U'):
                node.set_fork(True)
            elif key in ('d', 'D'):
                node.set_fork(False)
            elif key in ('q', 'Q', '\x03'):
                stop_event.set()
                break

    t = threading.Thread(target=keyboard_thread, daemon=True)
    t.start()

    try:
        while rclpy.ok() and not stop_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.05)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
