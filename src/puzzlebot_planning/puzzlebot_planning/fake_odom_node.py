"""
fake_odom_node.py — Odometría simulada para pruebas sin Gazebo ni encoders.

Integra /cmd_vel y publica /odom + TF odom→base_footprint.
Permite probar el stack A* + DOM + steering con teleop o goals de RViz
sin necesidad de robot físico ni Gazebo.

Pose inicial configurable via parámetros o /initialpose de RViz.
"""

import math
import time

import rclpy
from geometry_msgs.msg import (PoseWithCovarianceStamped, Quaternion,
                                TransformStamped, Twist)
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


def _yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw / 2.0)
    q.z = math.sin(yaw / 2.0)
    return q


class FakeOdomNode(Node):

    def __init__(self):
        super().__init__('fake_odom_node')

        self.declare_parameter('initial_x',    0.30)
        self.declare_parameter('initial_y',    0.30)
        self.declare_parameter('initial_yaw',  1.5708)
        self.declare_parameter('odom_frame',   'odom')
        self.declare_parameter('base_frame',   'base_footprint')
        self.declare_parameter('publish_tf',   True)

        self._x   = self.get_parameter('initial_x').value
        self._y   = self.get_parameter('initial_y').value
        self._th  = self.get_parameter('initial_yaw').value
        self._odom_frame = self.get_parameter('odom_frame').value
        self._base_frame = self.get_parameter('base_frame').value
        self._pub_tf     = self.get_parameter('publish_tf').value

        self._vx  = 0.0
        self._wz  = 0.0
        self._last_t = time.monotonic()

        self._tf_br = TransformBroadcaster(self)
        self._pub   = self.create_publisher(Odometry, '/odom', 10)

        self.create_subscription(Twist, '/cmd_vel', self._cmd_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose',
                                 self._init_pose_cb, 10)

        # 50 Hz update
        self.create_timer(0.02, self._update)

        self.get_logger().info(
            f'fake_odom_node OK — pose inicial ({self._x:.2f},{self._y:.2f},{self._th:.2f})')

    def _cmd_cb(self, msg: Twist):
        self._vx = msg.linear.x
        self._wz = msg.angular.z

    def _init_pose_cb(self, msg: PoseWithCovarianceStamped):
        self._x  = msg.pose.pose.position.x
        self._y  = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._th = math.atan2(2 * (q.w * q.z + q.x * q.y),
                              1 - 2 * (q.y * q.y + q.z * q.z))
        self.get_logger().info(
            f'Pose inicial recibida: ({self._x:.2f},{self._y:.2f},{math.degrees(self._th):.1f}°)')

    def _update(self):
        now_t  = time.monotonic()
        dt     = now_t - self._last_t
        self._last_t = now_t

        # Integrar cinemática diferencial
        self._th += self._wz * dt
        self._x  += self._vx * math.cos(self._th) * dt
        self._y  += self._vx * math.sin(self._th) * dt

        now_msg = self.get_clock().now().to_msg()
        q = _yaw_to_quat(self._th)

        # Publicar /odom
        odom = Odometry()
        odom.header.stamp    = now_msg
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id  = self._base_frame
        odom.pose.pose.position.x  = self._x
        odom.pose.pose.position.y  = self._y
        odom.pose.pose.orientation = q
        odom.twist.twist.linear.x  = self._vx
        odom.twist.twist.angular.z = self._wz
        self._pub.publish(odom)

        # Publicar TF odom → base_footprint
        if self._pub_tf:
            tf = TransformStamped()
            tf.header.stamp    = now_msg
            tf.header.frame_id = self._odom_frame
            tf.child_frame_id  = self._base_frame
            tf.transform.translation.x = self._x
            tf.transform.translation.y = self._y
            tf.transform.rotation      = q
            self._tf_br.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = FakeOdomNode()
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
