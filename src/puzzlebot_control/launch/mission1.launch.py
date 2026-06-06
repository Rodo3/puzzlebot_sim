"""
Misión 1 (stub) — cruza el mapa de esquina a esquina:
  WP1: (3.5, 2.0)  →  WP2: (0.5, 0.5)
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='puzzlebot_control',
            executable='mission_stub_runner',
            name='mission1_runner',
            output='screen',
            parameters=[{
                'waypoints_x':  [3.5, 0.5],
                'waypoints_y':  [2.0, 0.5],
                'step_timeout': 20.0,
            }],
        ),
    ])
