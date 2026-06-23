import math
import time as _time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

class VelocityMuxNode(Node):
    def __init__(self):
        super().__init__('velocity_mux_node')

        self.last_teleop_command = Twist()
        self.last_steering_command = Twist()
        self.TELEOP_MOVING = None
        self.STEERING_MOVING = None 

        self.sub_cmd_ = self.create_subscription(Twist,'/cmd_vel_teleop',self.cmd_teleop_cb,10)
        self.sub_path_ = self.create_subscription(Twist, '/cmd_vel_in',self.cmd_steering_cb,10)

        self.pub_cmd = self.create_publisher(Twist,'/model/puzzlebot/cmd_vel',10)
        self.timer = self.create_timer(0.05,self.mux_logic)

        self.get_logger().info(
            f'Velocity mux initiated'
        )

    def cmd_teleop_cb(self,msg: Twist):
        self.last_teleop_command = msg

    def cmd_steering_cb(self,msg:Twist):
        self.last_steering_command = msg

    def mux_logic(self):
        
        self.get_logger().info(
            f'Mux logic entered'
        )
        self.STEERING_MOVING = (self.last_steering_command.linear.x != 0 or self.last_steering_command.angular.z != 0)
        self.TELEOP_MOVING = (self.last_teleop_command.linear.x != 0 or self.last_teleop_command.angular.z != 0 )

        if (self.STEERING_MOVING):
            self.pub_cmd.publish((self.last_steering_command))
            self.get_logger().info(f'Steering Commanding')

        elif (self.TELEOP_MOVING):
            self.pub_cmd.publish((self.last_teleop_command))
            self.get_logger().info(f'Teleop Commanding')
        else:
            self.pub_cmd.publish((Twist()))
            self.get_logger().info(f'None of them')

def main(args=None):
    rclpy.init(args=args)
    node = VelocityMuxNode()
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


    