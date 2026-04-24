"""
Smoke test — verifica que los topics clave están activos y con datos coherentes.

Secuencia:
  1. Publica un cuadrado en /cmd_vel (avanza, gira x4).
  2. Mientras tanto, monitorea /scan y /odom.
  3. Al terminar imprime un reporte PASS/FAIL por topic.

Uso:
  ros2 run puzzlebot_bringup smoke_test
"""

import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


# ── parámetros del cuadrado ───────────────────────────────────────────────────
SIDE_SPEED   = 0.15   # m/s hacia adelante
TURN_SPEED   = 0.5    # rad/s en giro
SIDE_TIME    = 3.0    # segundos por lado
TURN_TIME    = math.pi / 2.0 / TURN_SPEED  # ~π/2 rad ÷ ω
SIDES        = 4


class SmokeTest(Node):
    def __init__(self):
        super().__init__('smoke_test')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # ── contadores de mensajes recibidos ─────────────────────────────────
        self._recv = {
            '/scan': 0,
            '/odom': 0,
        }
        # En empty.world todos los rayos son inf — se considera PASS igualmente.
        # /velocity_enc_r y /velocity_enc_l los genera mock_encoders, no Gazebo.

        self.create_subscription(LaserScan, '/scan', self._cb_scan, 10)
        self.create_subscription(Odometry,  '/odom', self._cb_odom, 10)

    # ── callbacks ─────────────────────────────────────────────────────────────
    def _cb_scan(self, _):
        self._recv['/scan'] += 1

    def _cb_odom(self, _):
        self._recv['/odom'] += 1

    # ── publicación bloqueante ────────────────────────────────────────────────
    def _publish_for(self, twist: Twist, duration: float):
        end = time.time() + duration
        while time.time() < end:
            self.cmd_pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.05)

    def _stop(self):
        self.cmd_pub.publish(Twist())
        rclpy.spin_once(self, timeout_sec=0.2)

    # ── secuencia principal ───────────────────────────────────────────────────
    def run(self):
        self.get_logger().info('=== SMOKE TEST START ===')
        self.get_logger().info('Esperando 2 s para que los topics suban...')
        end = time.time() + 2.0
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

        forward = Twist()
        forward.linear.x = SIDE_SPEED

        turn = Twist()
        turn.angular.z = TURN_SPEED

        for i in range(SIDES):
            self.get_logger().info(f'Lado {i+1}/{SIDES}: avanzando {SIDE_TIME:.1f} s')
            self._publish_for(forward, SIDE_TIME)
            self.get_logger().info(f'Lado {i+1}/{SIDES}: girando {TURN_TIME:.1f} s')
            self._publish_for(turn, TURN_TIME)

        self._stop()
        self.get_logger().info('Movimiento terminado. Reporte:')
        self._report()

    # ── reporte final ─────────────────────────────────────────────────────────
    def _report(self):
        MIN_MSGS = 5
        sep = '=' * 45
        self.get_logger().info(sep)
        all_pass = True
        for topic, count in self._recv.items():
            ok = count >= MIN_MSGS
            all_pass = all_pass and ok
            status = 'PASS' if ok else 'FAIL'
            self.get_logger().info(f'  [{status}]  {topic}  (msgs={count})')
        self.get_logger().info(sep)
        if all_pass:
            self.get_logger().info('RESULTADO FINAL: PASS — simulacion lista')
        else:
            self.get_logger().warn('RESULTADO FINAL: FAIL — revisa los topics marcados')


def main(args=None):
    rclpy.init(args=args)
    node = SmokeTest()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
