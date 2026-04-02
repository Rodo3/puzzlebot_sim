import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

from launch.actions import  EmitEvent, LogInfo, RegisterEventHandler
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.events import Shutdown
from launch.substitutions import EnvironmentVariable, LocalSubstitution


def generate_launch_description():

    pkg_share = get_package_share_directory('puzzlebot_sim')
    urdf_file = os.path.join(pkg_share, 'urdf', 'puzzlebot.urdf')
    rviz_file = os.path.join(pkg_share, 'rviz', 'puzzlebot_rviz.rviz')

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}]
    )

    joint_state_publisher_node = Node(
        name='joint_state_publisher',
        package='puzzlebot_sim',
        executable='joint_state_publisher'
    )

    rviz_node = Node(
        name='rviz',
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_file]
    )

    l_d = LaunchDescription([
        robot_state_publisher_node,
        joint_state_publisher_node,
        rviz_node
    ])

    return l_d