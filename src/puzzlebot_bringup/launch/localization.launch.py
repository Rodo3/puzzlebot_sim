import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('puzzlebot_bringup'), 'config')

    odometry_node = Node(
        package='puzzlebot_localization',
        executable='odometry_node',
        name='odometry_node',
        parameters=[
            os.path.join(cfg, 'robot_params.yaml'),
            {'use_sim_time': True},
        ],
        output='screen',
    )

    kalman_filter_node = Node(
        package='puzzlebot_localization',
        executable='kalman_filter_node',
        name='kalman_filter_node',
        parameters=[
            os.path.join(cfg, 'robot_params.yaml'),
            os.path.join(cfg, 'kalman_params.yaml'),
            {'use_sim_time': True},
        ],
        output='screen',
    )

    return LaunchDescription([
        odometry_node,
        kalman_filter_node,
    ])
