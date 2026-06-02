"""
obstacle_avoidance_node.py — Capa de seguridad reactiva ante obstáculos.

POSICIÓN EN EL PIPELINE:
  steering_controller_node  →  /cmd_vel_in  →  [ESTE NODO]  →  /cmd_vel  →  robot/Gazebo

FUNCIÓN:
  Intercepta el comando de velocidad del controlador antes de enviarlo al robot.
  Lee el LiDAR (/scan) y decide si el comando es seguro:
    - Obstáculo < stop_distance  → publica velocidad cero (freno total)
    - Obstáculo < slow_distance  → escala la velocidad linealmente (frenado gradual)
    - Sin obstáculo cercano      → pasa el comando sin modificar

TOPICS SUSCRITOS:
  /cmd_vel_in  (geometry_msgs/Twist)  — velocidad calculada por el controlador
  /scan        (sensor_msgs/LaserScan) — datos del LiDAR

TOPICS PUBLICADOS:
  /cmd_vel     (geometry_msgs/Twist)  — velocidad final hacia el robot

PARÁMETROS:
  stop_distance   [0.30 m] — distancia mínima; por debajo → freno total
  slow_distance   [0.60 m] — zona de frenado gradual
  front_angle_deg [30.0°]  — semi-ángulo del cono frontal que se monitorea
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class ObstacleAvoidanceNode(Node):

    def __init__(self):
        super().__init__('obstacle_avoidance_node')
        self.declare_parameter('stop_distance',   0.30)   # metros — freno total
        self.declare_parameter('slow_distance',   0.60)   # metros — inicio frenado gradual
        self.declare_parameter('front_angle_deg', 30.0)   # semi-ángulo del cono frontal

        self.stop_d  = self.get_parameter('stop_distance').value
        self.slow_d  = self.get_parameter('slow_distance').value
        self.front_a = math.radians(self.get_parameter('front_angle_deg').value)

        # Distancia mínima al obstáculo más cercano en el cono frontal.
        # Inicia en inf para no frenar antes de recibir el primer scan.
        self.min_front = float('inf')

        self.sub_scan_ = self.create_subscription(
            LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.sub_cmd_  = self.create_subscription(
            Twist, '/cmd_vel_in', self.cmd_cb, 10)
        self.pub_cmd_  = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info(
            f'obstacle_avoidance_node iniciado '
            f'(stop={self.stop_d} m, slow={self.slow_d} m, cono=±{math.degrees(self.front_a):.0f}°)')

    def scan_cb(self, msg: LaserScan):
        # Calcula el ángulo de cada rayo y filtra los que están en el cono frontal.
        # Se descartan rangos infinitos/NaN y los fuera del rango válido del sensor.
        angles = np.arange(len(msg.ranges)) * msg.angle_increment + msg.angle_min
        ranges = np.array(msg.ranges)
        valid  = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
        front  = np.abs(angles) <= self.front_a
        mask   = valid & front
        # Si no hay rayos válidos en el cono frontal, asume campo libre (inf).
        self.min_front = float(np.min(ranges[mask])) if mask.any() else float('inf')

    def cmd_cb(self, msg: Twist):
        out = Twist()  # velocidad cero por defecto

        if self.min_front <= self.stop_d:
            # Obstáculo demasiado cerca — freno total, ignora el comando del controlador.
            self.pub_cmd_.publish(out)
            return

        if self.min_front <= self.slow_d and msg.linear.x > 0:
            # Zona de frenado gradual: escala la velocidad lineal de forma proporcional.
            # A stop_d → scale=0 (parado), a slow_d → scale=1 (velocidad completa).
            scale = (self.min_front - self.stop_d) / (self.slow_d - self.stop_d)
            out.linear.x  = msg.linear.x * scale
            out.angular.z = msg.angular.z  # giro sin escalar para mantener la trayectoria
        else:
            out = msg  # campo libre — pasa el comando sin modificar

        self.pub_cmd_.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidanceNode()
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
