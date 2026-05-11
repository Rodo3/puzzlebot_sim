"""
Ground-truth odometry publisher for Gazebo Fortress simulation.

Subscribes to /world/<world>/dynamic_pose/info (bridged from ignition.msgs.Pose_V)
and publishes the puzzlebot's true world-frame pose as /odom (nav_msgs/Odometry)
plus the TF odom -> base_footprint.

Purpose: provides slip-free, drift-free pose so the SLAM mapping node can build
a coherent map without wheel-encoder integration errors. Use this only in
simulation. On the real robot use dead_reckoning.py instead.
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, TransformStamped, Quaternion
from nav_msgs.msg import Odometry
import tf2_ros


class GroundTruthOdom(Node):
    def __init__(self):
        super().__init__('ground_truth_odom')

        self.declare_parameter('model_name',  'puzzlebot')
        self.declare_parameter('odom_frame',  'odom')
        self.declare_parameter('base_frame',  'base_footprint')
        self.declare_parameter('pose_topic',  '/world/maze/dynamic_pose/info')

        self._model_name = self.get_parameter('model_name').value
        self._odom_frame = self.get_parameter('odom_frame').value
        self._base_frame = self.get_parameter('base_frame').value
        pose_topic       = self.get_parameter('pose_topic').value

        self._odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self._prev_x = None
        self._prev_y = None
        self._prev_yaw = None
        self._prev_stamp = None

        self.create_subscription(PoseArray, pose_topic, self._pose_cb, 10)
        self.get_logger().info(
            f'ground_truth_odom listening on {pose_topic} '
            f'for model "{self._model_name}"')

    def _pose_cb(self, msg: PoseArray):
        # PoseArray bridged from ignition.msgs.Pose_V loses model names.  The
        # dynamic_pose topic only contains moving entities, so the spawned robot
        # is expected to be the first pose in this simulation.
        if not msg.poses:
            return
        pose = msg.poses[0]

        x = pose.position.x
        y = pose.position.y
        q = pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9

        v = 0.0
        w = 0.0
        if self._prev_stamp is not None:
            dt = t - self._prev_stamp
            if 0.0 < dt < 0.5:
                dx = x - self._prev_x
                dy = y - self._prev_y
                # Linear velocity in robot frame (forward only)
                v = math.hypot(dx, dy) / dt
                dyaw = math.atan2(math.sin(yaw - self._prev_yaw),
                                  math.cos(yaw - self._prev_yaw))
                w = dyaw / dt

        self._prev_x, self._prev_y, self._prev_yaw, self._prev_stamp = x, y, yaw, t

        # Publish Odometry
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id  = self._base_frame
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation = q
        odom.twist.twist.linear.x  = v
        odom.twist.twist.angular.z = w
        self._odom_pub.publish(odom)

        # Publish TF odom -> base_footprint
        tf = TransformStamped()
        tf.header.stamp    = stamp
        tf.header.frame_id = self._odom_frame
        tf.child_frame_id  = self._base_frame
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.rotation = q
        self._tf_broadcaster.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthOdom()
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
