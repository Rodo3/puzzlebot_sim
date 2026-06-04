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
from launch.substitutions import LaunchConfiguration, PythonExpression
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
    arg_scan_topic = DeclareLaunchArgument(
        'scan_topic', default_value='/scan_stamped',
        description='Topic del LiDAR. Gazebo: /scan  |  Real: /scan_stamped')
    # obstacle_manager: qué nodo gestiona los obstáculos dinámicos.
    #   dynamic  → dynamic_obstacle_manager (FSM + goal persistente + capa dinámica) [RECOMENDADO]
    #   legacy   → bug_navigation_node (modo viejo, inyección directa en /map)
    #   none     → sin gestor de obstáculos dinámicos (solo obstacle_avoidance de emergencia)
    arg_obstacle_manager = DeclareLaunchArgument(
        'obstacle_manager', default_value='dynamic',
        description='Gestor de obstáculos: dynamic | legacy | none')

    use_sim_time      = LaunchConfiguration('use_sim_time')
    use_pd            = LaunchConfiguration('use_pd')
    cmd_vel_topic     = LaunchConfiguration('cmd_vel_topic')
    scan_topic        = LaunchConfiguration('scan_topic')
    obstacle_manager  = LaunchConfiguration('obstacle_manager')

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
            'map_frame':          'map',
            'base_frame':         'base_footprint',
            'use_tf_pose':        True,
            'replan_on_new_map':  True,

            # ── Obstacle inflation ──────────────────────────────────────────
            # Cells within this radius of a wall are BLOCKED — the A* never
            # routes through them.  Keep this small enough so narrow corridors
            # between internal structures remain open for navigation.
            # 0.15 m = just enough to keep the robot body off the wall.
            # The distance costmap (below) handles keeping the path centred.
            'inflation_radius':    0.15,

            # Cells with occupancy > this threshold are treated as obstacles.
            'occupied_threshold':  50,

            # Unknown cells (-1 in the map) treated as occupied.
            'unknown_as_occupied': True,

            # Morphological closing disabled — walls stay as mapped so all
            # corridors between internal structures remain open.
            'wall_closing_radius': 0,

            # ── Distance costmap ────────────────────────────────────────────
            # Adds a per-cell penalty inversely proportional to distance to
            # the nearest wall.  A* routes through corridor centres instead
            # of grazing walls.
            #
            # cost_weight: multiplier on the per-cell cost added to each A*
            #   step.  Higher → stronger wall avoidance.
            #   0.0  → classic A* (no wall avoidance, shortest path only)
            #   1.0  → light avoidance
            #   2.0  → moderate (recommended default)
            #   4.0+ → strong avoidance (paths through corridor centres)
            #
            # max_obstacle_cost: cap on the per-cell cost value.
            #   Higher → sharper gradient near walls.
            'use_cost_map':       True,
            'cost_weight':        2.0,
            'max_obstacle_cost':  5.0,

            # ── Goal safety ─────────────────────────────────────────────────
            # Goals closer than min_goal_obstacle_distance to a wall are
            # moved to the nearest safe free cell automatically.
            # Set allow_goal_reprojection: false to reject instead of move.
            'min_goal_obstacle_distance': 0.15,
            'allow_goal_reprojection':    True,
            'goal_reprojection_radius':   0.50,

            # ── Path safety warning ─────────────────────────────────────────
            # Log a warning if any waypoint ends up closer than this to a wall.
            'min_path_obstacle_distance': 0.15,
        }],
        output='screen',
    )

    # ── Steering Controller (pure pursuit) — DEFAULT ──────────────────────
    # Suscribe: /odom, /planned_path
    # Publica:  /cmd_vel_steering  (antes /cmd_vel_in — ahora va al bug_navigation_node)
    steering_controller = Node(
        package='puzzlebot_controller',
        executable='steering_controller_node',
        name='steering_controller_node',
        parameters=[controller_cfg, {'use_sim_time': use_sim_time}],
        remappings=[('/cmd_vel_in', '/cmd_vel_steering')],
        output='screen',
        condition=UnlessCondition(use_pd),
    )

    # ── PD Controller — ALTERNATIVA para navegación punto a punto ─────────
    # Suscribe: /odom, /goal_pose (directo, no usa /planned_path)
    # Publica:  /cmd_vel_steering
    pd_controller = Node(
        package='puzzlebot_controller',
        executable='pd_controller_node',
        name='pd_controller_node',
        parameters=[controller_cfg, {'use_sim_time': use_sim_time}],
        remappings=[('/cmd_vel_in', '/cmd_vel_steering')],
        output='screen',
        condition=IfCondition(use_pd),
    )

    # ── Dynamic Obstacle Manager [RECOMENDADO] ────────────────────────────────
    # FSM: NORMAL→BRAKE_FOR_REPLAN→REPLAN→FOLLOW_NEW_PATH→RECOVERY→SAFE_STOP
    # Publica /augmented_map para que A* evite obstáculos dinámicos.
    # Goal persistente: nunca cancela el objetivo de RViz.
    # Activo cuando obstacle_manager:=dynamic
    dynamic_obstacle_manager = Node(
        package='puzzlebot_planning',
        executable='dynamic_obstacle_manager',
        name='dynamic_obstacle_manager',
        parameters=[controller_cfg, {'use_sim_time': use_sim_time}],
        remappings=[('/scan_stamped', scan_topic)],
        output='screen',
        condition=IfCondition(
            PythonExpression(["'", obstacle_manager, "' == 'dynamic'"])
        ),
    )

    # ── Bug Navigation (modo legacy) ──────────────────────────────────────────
    # Modo anterior: inyección directa en /map + ruta de evasión manual.
    # Activo cuando obstacle_manager:=legacy
    # Mantener para comparación y rollback.
    bug_navigation = Node(
        package='puzzlebot_planning',
        executable='bug_navigation_node',
        name='bug_navigation_node',
        parameters=[controller_cfg, {'use_sim_time': use_sim_time}],
        remappings=[('/scan_stamped', scan_topic)],
        output='screen',
        condition=IfCondition(
            PythonExpression(["'", obstacle_manager, "' == 'legacy'"])
        ),
    )

    # ── Obstacle Avoidance (safety layer final entre bug_nav y cmd_vel) ───
    # Suscribe: /scan_stamped, /cmd_vel_in, /odom
    # Publica:  /cmd_vel
    # Lógica:   parada de emergencia si LiDAR < stop_distance (último filtro)
    obstacle_avoidance = Node(
        package='puzzlebot_planning',
        executable='obstacle_avoidance_node',
        name='obstacle_avoidance_node',
        parameters=[controller_cfg, {'use_sim_time': use_sim_time}],
        remappings=[
            ('/cmd_vel', cmd_vel_topic),
            ('/scan',    scan_topic),
        ],
        output='screen',
    )

    return LaunchDescription([
        arg_sim_time,
        arg_use_pd,
        arg_cmd_vel_topic,
        arg_scan_topic,
        arg_obstacle_manager,
        path_planner,
        steering_controller,
        pd_controller,
        dynamic_obstacle_manager,
        bug_navigation,
        obstacle_avoidance,
    ])
