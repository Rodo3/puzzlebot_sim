"""
Autonomous navigation launch — Puzzlebot (Gazebo Fortress / robot real).

Conecta el stack completo de navegación A* + control + obstacle avoidance:

  /map  ──────────────────────→ path_planner_node (A*)
  /goal_pose ─────────────────→ path_planner_node
  /odom ──────────────────────→ path_planner_node (TF lookup)
                                │
                                ▼ /planned_path
                        steering_controller_node (pure pursuit)
  /odom ──────────────────────→ steering_controller_node
                                │
                                ▼ /cmd_vel_in
                        obstacle_avoidance_node
  /scan ──────────────────────→ obstacle_avoidance_node
                                │
                                ▼ /cmd_vel
                        (Gazebo DiffDrive plugin / MCU motor driver)

════════════════════════════════════════════════════════════════
 ARGUMENTOS
════════════════════════════════════════════════════════════════

  use_sim_time   [true]   Usar reloj de simulación
  use_pd         [false]  Usar pd_controller en lugar de steering_controller
                          (útil para navegación punto a punto sin path)

════════════════════════════════════════════════════════════════
 CÓMO USAR
════════════════════════════════════════════════════════════════

  # Con Gazebo (desde otra terminal ya corriendo gz_sim.launch.py):
  ros2 launch puzzlebot_bringup navigation.launch.py use_sim_time:=true

  # Luego en RViz:
  # 1. Usa "2D Nav Goal" (tecla G) para enviar /goal_pose
  # 2. Observa /planned_path en RViz (tipo Path, frame=map, color=verde)

  # Robot real:
  ros2 launch puzzlebot_bringup navigation.launch.py use_sim_time:=false

════════════════════════════════════════════════════════════════
 SAFETY PARAMETERS (configurar en controller_params.yaml)
════════════════════════════════════════════════════════════════

  obstacle_avoidance_node:
    stop_distance:   0.30 m  — detiene el robot completamente
    slow_distance:   0.60 m  — escala velocidad linealmente
    front_angle_deg: 30.0°   — semióngulo del cono de detección frontal

  path_planner_node:
    inflation_radius: 0.15 m — inflación de obstáculos antes de planear

  steering_controller_node:
    goal_tolerance:   0.10 m — radio de aceptación del goal
    max_linear_vel:   0.30 m/s
    max_angular_vel:  1.50 rad/s
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup_pkg = get_package_share_directory('puzzlebot_bringup')

    controller_cfg = os.path.join(bringup_pkg, 'config', 'controller_params.yaml')

    arg_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use /clock topic from Gazebo')
    arg_use_pd = DeclareLaunchArgument(
        'use_pd', default_value='false',
        description='Use pd_controller_node instead of steering_controller_node')
    # En Gazebo el DiffDrive escucha /model/puzzlebot/cmd_vel, no /cmd_vel.
    # En robot real, el bridge de micro-ROS escucha /cmd_vel directamente.
    arg_cmd_vel_topic = DeclareLaunchArgument(
        'cmd_vel_topic', default_value='/model/puzzlebot/cmd_vel',
        description='Topic final de velocidad. Gazebo: /model/puzzlebot/cmd_vel  |  Real: /cmd_vel')

    use_sim_time   = LaunchConfiguration('use_sim_time')
    use_pd         = LaunchConfiguration('use_pd')
    cmd_vel_topic  = LaunchConfiguration('cmd_vel_topic')

    # ── A* Path Planner ───────────────────────────────────────────────────
    # Suscribe: /map, /goal_pose, /initialpose (fallback pose)
    # Publica:  /planned_path (nav_msgs/Path)
    # TF:       lee map→base_footprint para pose actual del robot
    path_planner = Node(
        package='puzzlebot_planning',
        executable='path_planner_node',
        name='path_planner_node',
        parameters=[{
            'use_sim_time':       use_sim_time,
            'inflation_radius':   0.15,      # metros — radio inflación obstáculos
            'occupied_threshold': 50,        # celdas con valor > 50 se tratan como ocupadas
            'map_frame':          'map',
            'base_frame':         'base_footprint',
            'use_tf_pose':        True,      # pose desde TF map→base_footprint
            'replan_on_new_map':  False,     # False: replana solo cuando llega un goal nuevo
        }],
        output='screen',
    )

    # ── Steering Controller (pure pursuit) — DEFAULT ──────────────────────
    # Suscribe: /odom, /planned_path
    # Publica:  /cmd_vel_in
    steering_controller = Node(
        package='puzzlebot_controller',
        executable='steering_controller_node',
        name='steering_controller_node',
        parameters=[controller_cfg, {'use_sim_time': use_sim_time}],
        output='screen',
        condition=UnlessCondition(use_pd),
    )

    # ── PD Controller — ALTERNATIVA para navegación punto a punto ─────────
    # Suscribe: /odom, /goal_pose (directo, no usa /planned_path)
    # Publica:  /cmd_vel_in
    pd_controller = Node(
        package='puzzlebot_controller',
        executable='pd_controller_node',
        name='pd_controller_node',
        parameters=[controller_cfg, {'use_sim_time': use_sim_time}],
        output='screen',
        condition=IfCondition(use_pd),
    )

    # ── Obstacle Avoidance (safety layer entre controller y cmd_vel) ───────
    # Suscribe: /scan, /cmd_vel_in
    # Publica:  /cmd_vel
    # Lógica:   si obstáculo < stop_distance → velocidad cero
    #           si obstáculo < slow_distance → escala velocidad
    obstacle_avoidance = Node(
        package='puzzlebot_planning',
        executable='obstacle_avoidance_node',
        name='obstacle_avoidance_node',
        parameters=[controller_cfg, {'use_sim_time': use_sim_time}],
        # Remapea /cmd_vel → topic real del robot (Gazebo o hardware)
        remappings=[('/cmd_vel', cmd_vel_topic)],
        output='screen',
    )

    return LaunchDescription([
        arg_sim_time,
        arg_use_pd,
        arg_cmd_vel_topic,
        path_planner,
        steering_controller,
        pd_controller,
        obstacle_avoidance,
    ])
