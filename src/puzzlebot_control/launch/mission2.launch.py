"""
Misión 2 (stub) — va al punto elegido por el usuario y luego cruza al otro extremo.
Args:
  start_x  (float)  coordenada X del punto de inicio elegido en el mapa
  start_y  (float)  coordenada Y del punto de inicio elegido en el mapa
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('start_x', default_value='1.0',
                              description='X del punto de inicio (click en mapa)'),
        DeclareLaunchArgument('start_y', default_value='1.0',
                              description='Y del punto de inicio (click en mapa)'),

        Node(
            package='puzzlebot_control',
            executable='mission_stub_runner',
            name='mission2_runner',
            output='screen',
            parameters=[{
                'start_x':      LaunchConfiguration('start_x'),
                'start_y':      LaunchConfiguration('start_y'),
                'waypoints_x':  [3.5, 0.5],
                'waypoints_y':  [2.0, 0.5],
                'step_timeout': 20.0,
            }],
        ),
    ])
