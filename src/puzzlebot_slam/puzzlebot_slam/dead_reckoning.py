"""
Dead-reckoning odometry node for a differential-drive robot.

Sim mode  — input_source='joint_states' (default):
  Subscribes /joint_states (sensor_msgs/JointState) remapped in the launch file
  to /world/<world>/model/puzzlebot/joint_state from ros_gz_bridge.

Real robot — input_source='encoders':
  Subscribes /velocity_enc_r and /velocity_enc_l (std_msgs/Float32, rad/s).

Outputs:
  /odom  (nav_msgs/Odometry)
  TF     odom → base_footprint

Parameters:
  wheel_radius     float  0.05  [m]
  wheel_separation float  0.19  [m]  centre-to-centre
  odom_frame       str    'odom'
  base_frame       str    'base_footprint'
  input_source     str    'joint_states' | 'encoders'
"""

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
import tf2_ros


def _euler_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


class DeadReckoning(Node):
    def __init__(self):
        super().__init__('dead_reckoning')

        self.declare_parameter('wheel_radius',     0.05)
        self.declare_parameter('wheel_separation', 0.19)
        self.declare_parameter('odom_frame',       'odom')
        self.declare_parameter('base_frame',       'base_footprint')
        self.declare_parameter('input_source',     'joint_states')

        self._r          = self.get_parameter('wheel_radius').value
        self._l          = self.get_parameter('wheel_separation').value
        self._odom_frame = self.get_parameter('odom_frame').value
        self._base_frame = self.get_parameter('base_frame').value
        self._source     = self.get_parameter('input_source').value

        self._x   = 0.0
        self._y   = 0.0
        self._yaw = 0.0
        self._last_stamp = None

        self._wl = 0.0
        self._wr = 0.0

        self._odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        if self._source == 'joint_states':
            self.create_subscription(JointState, '/joint_states', self._joint_states_cb, 10)
            self.get_logger().info('dead_reckoning: input=joint_states (simulation)')
        else:
            self.create_subscription(Float32, '/velocity_enc_r', self._enc_r_cb, 10)
            self.create_subscription(Float32, '/velocity_enc_l', self._enc_l_cb, 10)
            self.create_timer(0.05, self._encoder_timer_cb)
            self.get_logger().info('dead_reckoning: input=encoders (real robot)')

    # ── Simulation path ──────────────────────────────────────────────────

    def _joint_states_cb(self, msg: JointState):
        stamp = msg.header.stamp
        if self._last_stamp is None:
            self._last_stamp = stamp
            return
        dt = (stamp.sec - self._last_stamp.sec) + \
             (stamp.nanosec - self._last_stamp.nanosec) * 1e-9
        self._last_stamp = stamp
        if dt <= 0.0 or dt > 0.5:
            return

        wl = wr = None
        for i, name in enumerate(msg.name):
            if name == 'wheel_l_joint' and i < len(msg.velocity):
                wl = msg.velocity[i]
            elif name == 'wheel_r_joint' and i < len(msg.velocity):
                wr = msg.velocity[i]
        if wl is None or wr is None:
            return
        self._integrate(wl, wr, dt, stamp)

    # ── Real-robot path ──────────────────────────────────────────────────

    def _enc_r_cb(self, msg: Float32):
        self._wr = msg.data

    def _enc_l_cb(self, msg: Float32):
        self._wl = msg.data

    def _encoder_timer_cb(self):
        now = self.get_clock().now().to_msg()
        if self._last_stamp is None:
            self._last_stamp = now
            return
        dt = (now.sec - self._last_stamp.sec) + \
             (now.nanosec - self._last_stamp.nanosec) * 1e-9
        self._last_stamp = now
        if dt <= 0.0 or dt > 0.5:
            return
        self._integrate(self._wl, self._wr, dt, now)

    # ── Kinematics ──────────────────────────────────────────────────────

    def _integrate(self, wl: float, wr: float, dt: float, stamp):
        v = self._r * (wr + wl) / 2.0
        w = self._r * (wr - wl) / self._l
        self._x   += v * math.cos(self._yaw) * dt
        self._y   += v * math.sin(self._yaw) * dt
        self._yaw += w * dt
        self._yaw  = math.atan2(math.sin(self._yaw), math.cos(self._yaw))
        self._publish(v, w, stamp)

    def _publish(self, v: float, w: float, stamp):
        q = _euler_to_quaternion(self._yaw)

        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id  = self._base_frame
        odom.pose.pose.position.x  = self._x
        odom.pose.pose.position.y  = self._y
        odom.pose.pose.orientation = q
        odom.twist.twist.linear.x  = v
        odom.twist.twist.angular.z = w
        self._odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp            = stamp
        tf.header.frame_id         = self._odom_frame
        tf.child_frame_id          = self._base_frame
        tf.transform.translation.x = self._x
        tf.transform.translation.y = self._y
        tf.transform.rotation      = q
        self._tf_broadcaster.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = DeadReckoning()
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
