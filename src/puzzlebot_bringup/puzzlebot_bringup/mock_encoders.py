import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


class MockEncoders(Node):
    """Publishes fake encoder angular velocities [rad/s] to simulate motion."""

    def __init__(self):
        super().__init__('mock_encoders')
        self.declare_parameter('linear_vel',  0.2)   # m/s
        self.declare_parameter('angular_vel', 0.0)   # rad/s
        self.declare_parameter('wheel_radius', 0.05)
        self.declare_parameter('wheelbase',    0.18)

        v = self.get_parameter('linear_vel').value
        w = self.get_parameter('angular_vel').value
        r = self.get_parameter('wheel_radius').value
        l = self.get_parameter('wheelbase').value

        vR = (v + w * l / 2.0) / r
        vL = (v - w * l / 2.0) / r

        self.pub_r = self.create_publisher(Float32, '/velocity_enc_r', 10)
        self.pub_l = self.create_publisher(Float32, '/velocity_enc_l', 10)
        self.create_timer(0.05, lambda: [
            self.pub_r.publish(Float32(data=vR)),
            self.pub_l.publish(Float32(data=vL)),
        ])
        self.get_logger().info(f'mock_encoders: vR={vR:.3f} rad/s, vL={vL:.3f} rad/s')


def main(args=None):
    rclpy.init(args=args)
    node = MockEncoders()
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
