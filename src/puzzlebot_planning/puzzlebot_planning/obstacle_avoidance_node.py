import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class ObstacleAvoidanceNode(Node):
    """
    Reactive obstacle avoidance layer.
    Subscribes to /cmd_vel_in (from the steering controller) and /scan.
    If an obstacle is detected within stop_distance, outputs zero velocity.
    Otherwise passes /cmd_vel_in through to /cmd_vel unchanged.
    """

    def __init__(self):
        super().__init__('obstacle_avoidance_node')
        self.declare_parameter('stop_distance',  0.30)   # metres — halt below this
        self.declare_parameter('slow_distance',  0.60)   # metres — scale speed below this
        self.declare_parameter('front_angle_deg', 30.0)  # half-cone in front of robot

        self.stop_d  = self.get_parameter('stop_distance').value
        self.slow_d  = self.get_parameter('slow_distance').value
        self.front_a = math.radians(self.get_parameter('front_angle_deg').value)

        self.min_front = float('inf')

        self.sub_scan_ = self.create_subscription(LaserScan, '/scan',
                          self.scan_cb, 10)
        self.sub_cmd_  = self.create_subscription(Twist, '/cmd_vel_in',
                          self.cmd_cb, 10)
        self.pub_cmd_  = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info(
            f'obstacle_avoidance_node started '
            f'(stop={self.stop_d} m, slow={self.slow_d} m)')

    def scan_cb(self, msg: LaserScan):
        import math
        import numpy as np
        angles = np.arange(len(msg.ranges)) * msg.angle_increment + msg.angle_min
        ranges = np.array(msg.ranges)
        valid  = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
        front  = np.abs(angles) <= self.front_a
        mask   = valid & front
        self.min_front = float(np.min(ranges[mask])) if mask.any() else float('inf')

    def cmd_cb(self, msg: Twist):
        out = Twist()
        if self.min_front <= self.stop_d:
            # Full stop — publish zero velocity
            self.pub_cmd_.publish(out)
            return
        if self.min_front <= self.slow_d and msg.linear.x > 0:
            # Scale linear speed proportionally
            scale = (self.min_front - self.stop_d) / (self.slow_d - self.stop_d)
            out.linear.x  = msg.linear.x  * scale
            out.angular.z = msg.angular.z
        else:
            out = msg
        self.pub_cmd_.publish(out)


import math  # noqa: E402 — needed by scan_cb at runtime


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
