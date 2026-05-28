"""
mock_test.launch.py — Lanza todos los nodos necesarios para probar el dashboard
sin el robot físico.

Nodos lanzados:
  1. mock_robot      — publica /odom, /scan, /cmd_vel, /cmd_vel_in, /map
  2. mock_camera     — publica /camera/image/compressed
  3. bridge_node     — WebSocket bridge + POST /audio para voz

Uso desde la raíz del workspace:
  colcon build --base-paths src mock --packages-select puzzlebot_mock puzzlebot_web_bridge
  source install/setup.bash
  ros2 launch puzzlebot_mock mock_test.launch.py

Con modelos de voz:
  ros2 launch puzzlebot_mock mock_test.launch.py \\
    artifact_dir:=src/puzzlebot_voice_commands/artifacts_final

Sin modelos (solo datos del robot):
  ros2 launch puzzlebot_mock mock_test.launch.py artifact_dir:=''

Desde otra computadora, abrir el dashboard en:
  http://<IP_DE_ESTA_MAQUINA>:5173
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    arg_artifact_dir = DeclareLaunchArgument(
        'artifact_dir',
        default_value='src/puzzlebot_voice_commands/artifacts_final',
        description='Ruta a los modelos de voz. Dejar vacío para deshabilitar inferencia.',
    )
    arg_ws_port = DeclareLaunchArgument(
        'websocket_port',
        default_value='8000',
        description='Puerto del WebSocket bridge.',
    )

    artifact_dir = LaunchConfiguration('artifact_dir')
    ws_port      = LaunchConfiguration('websocket_port')

    mock_robot = Node(
        package='puzzlebot_mock',
        executable='mock_robot',
        name='mock_robot_publisher',
        output='screen',
    )

    mock_camera = Node(
        package='puzzlebot_mock',
        executable='mock_camera',
        name='mock_camera_publisher',
        output='screen',
    )

    bridge = Node(
        package='puzzlebot_web_bridge',
        executable='bridge_node',
        name='puzzlebot_web_bridge',
        output='screen',
        parameters=[{
            'artifact_dir':   artifact_dir,
            'websocket_port': ws_port,
        }],
    )

    return LaunchDescription([
        arg_artifact_dir,
        arg_ws_port,
        mock_robot,
        mock_camera,
        bridge,
    ])
