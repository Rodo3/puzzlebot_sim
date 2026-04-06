
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
import transforms3d
import numpy as np

class PuzzlebotPublisher(Node):

    def __init__(self):
        super().__init__('frame_publisher')

        self.radius = 0.5         # radio de la trayectoria circular [m]
        self.omega = 0.3          # velocidad angular del robot [rad/s] (negativo = CW, rueda izq mas rapida)
        self.wheel_radius = 0.05  # radio de la rueda [m]
        self.wheel_sep = 0.19     # separacion entre ruedas (2 * 0.095) [m]


        self.define_TF()

        self.static_br = StaticTransformBroadcaster(self)
        self.tf_br = TransformBroadcaster(self)

        #Publish static map->odom once
        self.static_br.sendTransform(self.map_to_odom_tf)

        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)

        self.start_time = self.get_clock().now().nanoseconds / 1e9

        #Create a Timer
        timer_period = 0.005  # seconds
        self.timer = self.create_timer(timer_period, self.timer_cb)


    #Timer Callback
    def timer_cb(self):

        time = self.get_clock().now().nanoseconds / 1e9 - self.start_time

        #Dynamic TF: odom -> base_footprint (trayectoria circular)
        self.odom_to_base_footprint_tf.header.stamp = self.get_clock().now().to_msg()
        self.odom_to_base_footprint_tf.transform.translation.x = self.radius * np.cos(self.omega * time)
        self.odom_to_base_footprint_tf.transform.translation.y = self.radius * np.sin(self.omega * time)
        self.odom_to_base_footprint_tf.transform.translation.z = 0.0
        yaw = self.omega * time + np.pi / 2  
        q = transforms3d.euler.euler2quat(0, 0, yaw)  # euler2quat(roll, pitch, yaw) -> [w,x,y,z]
        self.odom_to_base_footprint_tf.transform.rotation.x = q[1]
        self.odom_to_base_footprint_tf.transform.rotation.y = q[2]
        self.odom_to_base_footprint_tf.transform.rotation.z = q[3]
        self.odom_to_base_footprint_tf.transform.rotation.w = q[0]

        self.tf_br.sendTransform(self.odom_to_base_footprint_tf)


        # Para ver la rueda izquierda mas rapida cambia self.omega a negativo
        v_r = (self.radius + self.wheel_sep / 2.0) * self.omega
        v_l = (self.radius - self.wheel_sep / 2.0) * self.omega
        omega_r = v_r / self.wheel_radius
        omega_l = v_l / self.wheel_radius

        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = ['wheel_r_joint', 'wheel_l_joint']
        joint_state.position = [omega_r * time, omega_l * time]
        joint_state.velocity = [omega_r, omega_l]

        self.joint_pub.publish(joint_state)


    def define_TF(self):

        self.map_to_odom_tf = TransformStamped()
        self.map_to_odom_tf.header.stamp = self.get_clock().now().to_msg()
        self.map_to_odom_tf.header.frame_id = 'map'
        self.map_to_odom_tf.child_frame_id = 'odom'
        self.map_to_odom_tf.transform.translation.x = 1.0
        self.map_to_odom_tf.transform.translation.y = 0.5
        self.map_to_odom_tf.transform.translation.z = 0.0
        q = transforms3d.euler.euler2quat(0, 0, 0)
        self.map_to_odom_tf.transform.rotation.x = q[1]
        self.map_to_odom_tf.transform.rotation.y = q[2]
        self.map_to_odom_tf.transform.rotation.z = q[3]
        self.map_to_odom_tf.transform.rotation.w = q[0]

        #Dynamic TF: odom -> base_footprint 
        self.odom_to_base_footprint_tf = TransformStamped()
        self.odom_to_base_footprint_tf.header.frame_id = 'odom'
        self.odom_to_base_footprint_tf.child_frame_id = 'base_footprint'


def main(args=None):
    rclpy.init(args=args)

    node = PuzzlebotPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():  # Ensure shutdown is only called once
            rclpy.shutdown()
        node.destroy_node()


if __name__ == '__main__':
    main()