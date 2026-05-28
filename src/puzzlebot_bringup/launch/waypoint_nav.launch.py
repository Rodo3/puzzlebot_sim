"""
waypoint_nav.launch.py — Navegación autónoma a waypoints nombrados.

Lanza el stack completo de navegación A* + el nodo de waypoints.
El robot puede ser dirigido a cualquier punto del YAML con un simple
comando de texto.

════════════════════════════════════════════════════════════════
 CÓMO USAR (flujo completo)
════════════════════════════════════════════════════════════════

  PASO 1 — Lanza Gazebo + SLAM + waypoints en una terminal:
  ─────────────────────────────────────────────────────────────
    ros2 launch puzzlebot_bringup gz_sim.launch.py \\
      world:=real_arena mode:=mapping odom_source:=ground_truth \\
      navigation:=true rviz:=true

  PASO 2 — En otra terminal, lanza este launch:
  ─────────────────────────────────────────────────────────────
    ros2 launch puzzlebot_bringup waypoint_nav.launch.py

  PASO 3 — Construye el mapa con teleop:
  ─────────────────────────────────────────────────────────────
    ros2 run teleop_twist_keyboard teleop_twist_keyboard \\
      --ros-args --remap cmd_vel:=/model/puzzlebot/cmd_vel

  PASO 4 — Navega a un waypoint (en otra terminal):
  ─────────────────────────────────────────────────────────────
    # Ver waypoints disponibles:
    ros2 topic pub --once /navigate_to_waypoint std_msgs/String "data: 'list'"

    # Navegar al centro:
    ros2 topic pub --once /navigate_to_waypoint std_msgs/String "data: 'centro'"

    # Navegar a estación C:
    ros2 topic pub --once /navigate_to_waypoint std_msgs/String "data: 'estacion_c'"

    # Detener navegación:
    ros2 topic pub --once /navigate_to_waypoint std_msgs/String "data: 'stop'"

  PASO 5 — Monitorea el estado:
  ─────────────────────────────────────────────────────────────
    ros2 topic echo /waypoint_status

════════════════════════════════════════════════════════════════
 ARGUMENTOS
════════════════════════════════════════════════════════════════

  use_sim_time   [true]    Reloj de simulación
  cmd_vel_topic  [/model/puzzlebot/cmd_vel]  Topic final de velocidad
  waypoints_file [waypoints.yaml]  Archivo de waypoints a cargar
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    bringup_pkg     = get_package_share_directory('puzzlebot_bringup')
    nav_launch_file = os.path.join(bringup_pkg, 'launch', 'navigation.launch.py')

    arg_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Usar reloj de simulación')

    arg_cmd_vel = DeclareLaunchArgument(
        'cmd_vel_topic', default_value='/model/puzzlebot/cmd_vel',
        description='Topic final de velocidad. Gazebo: /model/puzzlebot/cmd_vel | Real: /cmd_vel')

    arg_waypoints = DeclareLaunchArgument(
        'waypoints_file',
        default_value=os.path.join(bringup_pkg, 'config', 'waypoints.yaml'),
        description='Ruta al archivo YAML de waypoints')

    use_sim_time   = LaunchConfiguration('use_sim_time')
    cmd_vel_topic  = LaunchConfiguration('cmd_vel_topic')
    waypoints_file = LaunchConfiguration('waypoints_file')

    # ── Stack de navegación A* (path_planner + steering + avoidance) ──────────
    navigation_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav_launch_file),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'cmd_vel_topic': cmd_vel_topic,
        }.items(),
    )

    # ── Waypoint Navigator ────────────────────────────────────────────────────
    # Suscribe: /navigate_to_waypoint (String)
    # Publica:  /goal_pose (PoseStamped) → path_planner_node
    #           /waypoint_status (String)
    waypoint_navigator = Node(
        package='puzzlebot_planning',
        executable='waypoint_navigator_node',
        name='waypoint_navigator_node',
        parameters=[{
            'use_sim_time':   use_sim_time,
            'waypoints_file': waypoints_file,
            'frame_id':       'map',
        }],
        output='screen',
    )

    return LaunchDescription([
        arg_sim_time,
        arg_cmd_vel,
        arg_waypoints,
        navigation_stack,
        waypoint_navigator,
    ])
