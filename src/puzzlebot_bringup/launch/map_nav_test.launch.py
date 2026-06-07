"""
map_nav_test.launch.py — Prueba de navegación con mapa PNG. Sin Gazebo.

Carga el mapa SLAM guardado, publica /map, lanza A* + DOM + steering.
El robot se mueve con teleop o con goals de RViz (2D Nav Goal).
Los obstáculos dinámicos se inyectan automáticamente cada N segundos.

════════════════════════════════════════════════════════════════
 USO RÁPIDO
════════════════════════════════════════════════════════════════

  source ~/puzzlebot_sim/install/setup.bash

  # Lanzar:
  ros2 launch puzzlebot_bringup map_nav_test.launch.py

  # Teleop (terminal separada):
  ros2 run teleop_twist_keyboard teleop_twist_keyboard \\
    --ros-args --remap cmd_vel:=/cmd_vel

  # En RViz:
  #   1. "2D Pose Estimate" (P) → clic donde está el robot en el mapa
  #   2. "2D Nav Goal" (G) → clic en el destino
  #   3. Observar ruta verde + evasión automática de obstáculos

════════════════════════════════════════════════════════════════
 ARGUMENTOS
════════════════════════════════════════════════════════════════

  map_file            Ruta al PNG del mapa
  map_resolution      [0.05]   m/px
  map_origin_x        [-0.25]  X esquina SW en world frame
  map_origin_y        [-0.25]  Y esquina SW en world frame
  initial_x           [0.30]   Pose inicial robot X
  initial_y           [0.30]   Pose inicial robot Y
  initial_yaw         [1.5708] Pose inicial robot yaw (π/2 = norte)
  dynamic_obstacles   [true]   Activar spawner de obstáculos
  spawn_interval_sec  [45.0]   Segundos entre spawns
  spawn_mode          [on_path] on_path | near_path | random_free | fixed_sequence
  rviz                [true]

════════════════════════════════════════════════════════════════
 INYECTAR OBSTÁCULO MANUALMENTE
════════════════════════════════════════════════════════════════

  ros2 topic pub --once /test_obstacle_pose geometry_msgs/msg/PoseStamped \\
    '{header: {frame_id: map}, pose: {position: {x: 1.5, y: 2.0}, orientation: {w: 1.0}}}'
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup_pkg = get_package_share_directory('puzzlebot_bringup')
    desc_pkg    = get_package_share_directory('puzzlebot_description')

    rviz_cfg = os.path.join(desc_pkg, 'rviz', 'nav_test_rviz.rviz')

    import glob as _glob
    ws_root = os.path.abspath(os.path.join(bringup_pkg, '..', '..', '..', '..'))
    _maps = sorted(_glob.glob(os.path.join(ws_root, 'slam_map_*.png')))
    default_map = _maps[-1] if _maps else ''

    # ── Argumentos ─────────────────────────────────────────────────────────────
    arg_map_file = DeclareLaunchArgument(
        'map_file', default_value=default_map,
        description='Ruta absoluta al PNG del mapa SLAM')
    arg_map_res = DeclareLaunchArgument(
        'map_resolution', default_value='0.05',
        description='Metros por pixel del mapa')
    arg_origin_x = DeclareLaunchArgument(
        'map_origin_x', default_value='-0.25',
        description='X de la esquina SW del mapa [m]')
    arg_origin_y = DeclareLaunchArgument(
        'map_origin_y', default_value='-0.25',
        description='Y de la esquina SW del mapa [m]')
    arg_init_x = DeclareLaunchArgument(
        'initial_x', default_value='0.30',
        description='Pose inicial X del robot [m]')
    arg_init_y = DeclareLaunchArgument(
        'initial_y', default_value='0.30',
        description='Pose inicial Y del robot [m]')
    arg_init_yaw = DeclareLaunchArgument(
        'initial_yaw', default_value='1.5708',
        description='Pose inicial yaw del robot [rad]')
    arg_dyn_obs = DeclareLaunchArgument(
        'dynamic_obstacles', default_value='true',
        description='Activar spawner de obstáculos dinámicos')
    arg_spawn_interval = DeclareLaunchArgument(
        'spawn_interval_sec', default_value='45.0',
        description='Segundos entre spawns de obstáculos')
    arg_spawn_mode = DeclareLaunchArgument(
        'spawn_mode', default_value='on_path',
        description='on_path | near_path | random_free | fixed_sequence')
    arg_rviz = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Lanzar RViz')

    # ── Variables ──────────────────────────────────────────────────────────────
    map_file       = LaunchConfiguration('map_file')
    map_res        = LaunchConfiguration('map_resolution')
    origin_x       = LaunchConfiguration('map_origin_x')
    origin_y       = LaunchConfiguration('map_origin_y')
    init_x         = LaunchConfiguration('initial_x')
    init_y         = LaunchConfiguration('initial_y')
    init_yaw       = LaunchConfiguration('initial_yaw')
    dyn_obs_en     = LaunchConfiguration('dynamic_obstacles')
    spawn_interval = LaunchConfiguration('spawn_interval_sec')
    spawn_mode     = LaunchConfiguration('spawn_mode')
    rviz_en        = LaunchConfiguration('rviz')

    # ── 1. Map Server — carga el PNG como /map ─────────────────────────────────
    map_server = Node(
        package='puzzlebot_slam',
        executable='map_server_node',
        name='map_server_node',
        parameters=[{
            'use_sim_time':    False,
            'map_path':        map_file,
            'map_resolution':  map_res,
            'map_origin_x':    origin_x,
            'map_origin_y':    origin_y,
            'map_frame':       'map',
            'occupied_thresh': 0.65,   # pixel > 166 → libre (blanco)
            'free_thresh':     0.196,  # pixel < 50  → ocupado (negro)
        }],
        output='screen',
    )

    # ── 2. TF estático map → odom ──────────────────────────────────────────────
    # En pruebas de escritorio el origen del mundo = origen de odom.
    # El fake_odom_node publica odom→base_footprint dinámicamente.
    map_odom_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_tf',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen',
    )

    # ── 3. Odometría simulada ─────────────────────────────────────────────────
    # Integra /cmd_vel → publica /odom + TF odom→base_footprint.
    # /initialpose de RViz ("2D Pose Estimate") resetea la pose del robot.
    fake_odom = Node(
        package='puzzlebot_planning',
        executable='fake_odom_node',
        name='fake_odom_node',
        parameters=[{
            'use_sim_time': False,
            'initial_x':    init_x,
            'initial_y':    init_y,
            'initial_yaw':  init_yaw,
            'odom_frame':   'odom',
            'base_frame':   'base_footprint',
            'publish_tf':   True,
        }],
        output='screen',
    )

    # ── 4. Robot State Publisher ───────────────────────────────────────────────
    urdf_file = os.path.join(desc_pkg, 'urdf', 'puzzlebot.urdf')
    try:
        with open(urdf_file, 'r') as f:
            robot_description = f.read()
    except FileNotFoundError:
        robot_description = '<robot name="puzzlebot"><link name="base_footprint"/></robot>'

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': False}],
        output='screen',
    )

    # ── 5. A* Path Planner ────────────────────────────────────────────────────
    # inflation_radius=0.10m: mapa 86×108px a 0.05m/px — corredores estrechos.
    # unknown_as_occupied=false: el mapa guardado tiene zonas grises fuera de la
    # pista que no son obstáculos reales; tratarlas como libres.
    path_planner = Node(
        package='puzzlebot_planning',
        executable='path_planner_node',
        name='path_planner_node',
        parameters=[{
            'use_sim_time':               False,
            'map_frame':                  'map',
            'base_frame':                 'base_footprint',
            'use_tf_pose':                True,
            'replan_on_new_map':          True,
            'inflation_radius':           0.10,
            'occupied_threshold':         50,
            'unknown_as_occupied':        False,
            'wall_closing_radius':        1,
            'use_cost_map':               True,
            'cost_weight':                2.0,
            'max_obstacle_cost':          5.0,
            'min_goal_obstacle_distance': 0.10,
            'allow_goal_reprojection':    True,
            'goal_reprojection_radius':   0.40,
            'min_path_obstacle_distance': 0.08,
        }],
        output='screen',
    )

    # ── 6. Steering Controller ────────────────────────────────────────────────
    steering_controller = Node(
        package='puzzlebot_controller',
        executable='steering_controller_node',
        name='steering_controller_node',
        parameters=[{
            'use_sim_time':       False,
            'lookahead_distance': 0.30,
            'max_linear_vel':     0.14,
            'max_angular_vel':    0.80,
            'goal_tolerance':     0.12,
            'control_frequency':  20.0,
        }],
        remappings=[('/cmd_vel_in', '/cmd_vel_steering')],
        output='screen',
    )

    # ── 7. Dynamic Obstacle Manager ───────────────────────────────────────────
    # Inflación reducida a 0.35m para no cubrir la mitad del mapa pequeño.
    # TTL corto: el obstáculo de prueba desaparece en 60s para ver recuperación.
    dynamic_obstacle_manager = Node(
        package='puzzlebot_planning',
        executable='dynamic_obstacle_manager',
        name='dynamic_obstacle_manager',
        parameters=[{
            'use_sim_time':               False,
            'front_detection_angle_deg':  35.0,
            'front_stop_distance':        0.65,
            'front_slow_distance':        0.90,
            'emergency_stop_distance':    0.22,
            'dynamic_obstacle_radius_m':  0.20,   # radio físico pequeño
            'obstacle_inflation_radius_m': 0.35,  # inflación razonable para mapa 4×5m
            'obstacle_cluster_radius_m':  0.40,
            'dynamic_obstacle_ttl_sec':   60.0,   # desaparece en 60s
            'obstacle_reinject_cooldown_sec': 5.0,
            'min_obstacle_points':        1,
            'replan_stop_hold_sec':       1.5,
            'max_replan_attempts':        3,
            'replan_timeout_sec':         5.0,
            'path_collision_check_enabled': True,
            'path_collision_sample_step_m': 0.05,
            'recovery_enabled':           True,
            'reverse_speed':              0.07,
            'reverse_duration_sec':       1.5,
            'turn_speed':                 0.35,
            'turn_duration_sec':          1.2,
            'free_space_angle_deg':       70.0,
            'scan_topic':                 '/scan_stamped',
            'odom_topic':                 '/odom',
            'goal_topic':                 '/goal_pose',
            'planned_path_topic':         '/planned_path',
            'base_map_topic':             '/map',
            'augmented_map_topic':        '/augmented_map',
            'cmd_vel_input_topic':        '/cmd_vel_steering',
            'cmd_vel_output_topic':       '/cmd_vel_in',
            'global_frame':               'map',
            'enable_rviz_markers':        True,
            'debug_logs':                 True,
            # LiDAR simulado: el robot "detecta" el obstáculo cuando está
            # a menos de esta distancia. Simula el rango real del RPLidar A1.
            # Reducir para que el robot reaccione más tarde (más dramático).
            # Aumentar para que reaccione antes (más margen de frenado).
            'simulated_lidar_range_m':    1.20,
        }],
        output='screen',
    )

    # ── 8. Obstacle Avoidance — pass-through en modo desktop ─────────────────
    # Sin LiDAR real: stop_distance muy bajo para no bloquear la navegación.
    # Solo actúa de relay /cmd_vel_in → /cmd_vel.
    obstacle_avoidance = Node(
        package='puzzlebot_planning',
        executable='obstacle_avoidance_node',
        name='obstacle_avoidance_node',
        parameters=[{
            'use_sim_time':       False,
            'stop_distance':      0.05,    # casi desactivado — sin LiDAR
            'slow_distance':      0.08,
            'front_angle_deg':    30.0,
            'cov_slow_threshold': 9999.0,  # desactivar freno por covarianza
            'cov_stop_threshold': 9999.0,
            'cov_timeout_sec':    9999.0,
            'stuck_timeout_sec':  9999.0,
        }],
        remappings=[
            ('/scan', '/scan_stamped'),
            ('/cmd_vel', '/cmd_vel'),
        ],
        output='screen',
    )

    # ── 9. Dynamic Obstacle Spawner ───────────────────────────────────────────
    # desktop_mode=true: publica en /test_obstacle_pose en vez de Gazebo.
    # El DOM recibe la posición y la inyecta en /augmented_map.
    # El spawner espera 10s antes de arrancar para que el robot tenga pose válida.
    dynamic_obstacle_spawner = Node(
        package='puzzlebot_planning',
        executable='dynamic_obstacle_spawner_node',
        name='dynamic_obstacle_spawner_node',
        parameters=[{
            'use_sim_time':             False,
            'enabled':                  dyn_obs_en,
            'spawn_interval_sec':       spawn_interval,
            'obstacle_ttl_sec':         60.0,
            'max_active_obstacles':     2,
            'spawn_mode':               spawn_mode,
            'obstacle_shape':           'box',
            'obstacle_length_m':        0.25,
            'obstacle_width_m':         0.25,
            'obstacle_height_m':        0.50,
            'min_distance_from_robot_m': 0.70,
            'min_distance_from_goal_m':  0.40,
            'min_distance_from_walls_m': 0.20,
            'on_path_forward_offset_m':  0.80,
            'use_current_planned_path':  True,
            'planned_path_topic':        '/planned_path',
            'map_topic':                 '/map',
            'goal_topic':                '/goal_pose',
            'world_name':               'real_arena',
            'global_frame':             'map',
            'robot_frame':              'base_footprint',
            'enable_rviz_markers':       True,
            'auto_remove':               True,
            'desktop_mode':              True,   # sin Gazebo → /test_obstacle_pose
            # Posiciones fijas para fixed_sequence
            'fixed_obstacles.0.x':  1.50,
            'fixed_obstacles.0.y':  2.00,
            'fixed_obstacles.1.x':  2.50,
            'fixed_obstacles.1.y':  3.00,
        }],
        output='screen',
        condition=IfCondition(dyn_obs_en),
    )

    # ── 10. RViz ──────────────────────────────────────────────────────────────
    rviz_node = TimerAction(
        period=2.0,
        actions=[Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_cfg],
            parameters=[{'use_sim_time': False}],
            output='screen',
        )],
        condition=IfCondition(rviz_en),
    )

    return LaunchDescription([
        # Argumentos
        arg_map_file, arg_map_res, arg_origin_x, arg_origin_y,
        arg_init_x, arg_init_y, arg_init_yaw,
        arg_dyn_obs, arg_spawn_interval, arg_spawn_mode, arg_rviz,

        # Infraestructura (arrancan inmediatamente)
        rsp,
        map_odom_tf,
        map_server,
        fake_odom,

        # Stack de navegación (con pequeño delay para que /map llegue primero)
        TimerAction(period=1.5, actions=[path_planner]),
        TimerAction(period=1.5, actions=[steering_controller]),
        TimerAction(period=1.5, actions=[dynamic_obstacle_manager]),
        TimerAction(period=1.5, actions=[obstacle_avoidance]),

        # Spawner arranca después de 10s para que el robot tenga pose válida
        TimerAction(period=10.0, actions=[dynamic_obstacle_spawner]),

        # RViz
        rviz_node,
    ])
