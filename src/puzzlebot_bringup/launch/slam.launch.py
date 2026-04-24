import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    cfg      = os.path.join(get_package_share_directory('puzzlebot_bringup'), 'config')
    bringup  = get_package_share_directory('puzzlebot_bringup')

    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup, 'launch', 'localization.launch.py')
        )
    )

    slam_node = Node(
        package='puzzlebot_slam',
        executable='slam_node',
        name='slam_node',
        parameters=[os.path.join(cfg, 'slam_params.yaml')],
        output='screen',
    )

    return LaunchDescription([
        localization,
        slam_node,
    ])
