"""
dynamic_obstacle_test.launch.py — Prueba de evasión de obstáculos dinámicos.

Arquitectura de percepción (sin trampa):
─────────────────────────────────────────
  Gazebo spawner
    → crea obstáculo físico (SDF con visual + collision)
    → NO informa al planner
    → NO modifica mapas

  LiDAR simulado (Gazebo)
    → detecta el obstáculo cuando entra en rango/FOV real
    → publica /scan

  dynamic_obstacle_manager
    → lee SOLO LaserScan + TF
    → filtra puntos que ya estaban en el mapa base (paredes estáticas)
    → clusteriza puntos NUEVOS → obstáculo dinámico temporal
    → publica /augmented_map

  path_planner_node
    → usa /augmented_map para replantear al mismo goal

Mapa base fijo:
  Se carga slam_map_20260529_235356.png via MCL (publica /mcl/map → /map)
  y via slam_node en localization_only (publica TF map→odom via scan matching).
  El mapa NO se modifica. Los obstáculos dinámicos van en /augmented_map.

════════════════════════════════════════════════════════════════
 ARGUMENTOS
════════════════════════════════════════════════════════════════

  use_sim_time          [true]        Reloj de simulación
  world                 [real_arena]  Mundo Gazebo
  map_file              [<ruta>]      PNG del mapa base para MCL y planner
  localization          [mcl]         mcl | slam_loconly | ground_truth
  navigation            [true]        Lanza A* + steering + DOM + avoidance
  dynamic_obstacles     [true]        Lanza spawner de obstáculos dinámicos
  spawn_interval_sec    [60.0]        Intervalo entre spawns [s]
  spawn_mode            [on_path]     on_path | near_path | random_free | fixed_sequence
  spawn_shape           [box]         box | cylinder
  obstacle_ttl_sec      [90.0]        TTL del obstáculo [s]
  max_active_obstacles  [3]           Máximo simultáneo
  rviz                  [true]        Lanzar RViz
  gui                   [true]        GUI de Gazebo

════════════════════════════════════════════════════════════════
 USO
════════════════════════════════════════════════════════════════

  # Build primero:
  cd ~/puzzlebot_sim && colcon build --symlink-install
  source install/setup.bash

  # Prueba con MCL + mapa fijo + obstáculos cada 60s (recomendado):
  ros2 launch puzzlebot_bringup dynamic_obstacle_test.launch.py

  # Prueba rápida cada 20s, spawn sobre la ruta:
  ros2 launch puzzlebot_bringup dynamic_obstacle_test.launch.py \\
    spawn_interval_sec:=20.0 spawn_mode:=on_path

  # Localización por ground truth (más simple, para debug inicial):
  ros2 launch puzzlebot_bringup dynamic_obstacle_test.launch.py \\
    localization:=ground_truth

════════════════════════════════════════════════════════════════
 FLUJO DE PRUEBA
════════════════════════════════════════════════════════════════

  1. Gazebo levanta real_arena + robot en esquina SW
  2. MCL carga el mapa PNG — partículas visibles en RViz
  3. En RViz: "2D Nav Goal" (G) → clic en el mapa
  4. A* genera ruta verde
  5. Robot empieza a moverse
  6. Pasado spawn_interval_sec, el spawner crea un objeto físico en Gazebo
     (el planner NO sabe que existe todavía)
  7. Cuando el robot se acerca, el LiDAR real lo detecta
     → solo puntos que NO estaban en el mapa base se agregan a /augmented_map
     → DOM frena → replantea al mismo goal → robot rodea
  8. Robot llega al goal original

════════════════════════════════════════════════════════════════
 MONITOREO
════════════════════════════════════════════════════════════════

  ros2 topic echo /dom/state
  ros2 topic hz /augmented_map
  ros2 topic echo /planned_path --field header
  ros2 topic hz /cmd_vel

  # Confirmar que el planner NO usa ground truth:
  ros2 node info /path_planner_node
  ros2 node info /dynamic_obstacle_manager
  # dynamic_obstacle_manager debe suscribirse a /scan, NO a topics del spawner.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                             IncludeLaunchDescription, SetEnvironmentVariable,
                             TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    bringup_pkg = get_package_share_directory('puzzlebot_bringup')
    desc_pkg    = get_package_share_directory('puzzlebot_description')
    ros_gz_sim  = get_package_share_directory('ros_gz_sim')

    ctrl_cfg    = os.path.join(bringup_pkg, 'config', 'controller_params.yaml')
    slam_cfg    = os.path.join(bringup_pkg, 'config', 'slam_params.yaml')
    rviz_file   = os.path.join(desc_pkg, 'rviz', 'dynamic_obs_rviz.rviz')
    sdf_file    = os.path.join(desc_pkg, 'sdf',  'puzzlebot_gz.sdf')
    urdf_file   = os.path.join(desc_pkg, 'urdf', 'puzzlebot_gz.urdf')
    world_arena = os.path.join(desc_pkg, 'worlds', 'real_arena.sdf')
    world_flat  = os.path.join(desc_pkg, 'worlds', 'flat_plane.sdf')
    world_maze  = os.path.join(desc_pkg, 'worlds', 'maze.sdf')

    import glob as _glob
    ws_root = os.path.abspath(os.path.join(bringup_pkg, '..', '..', '..', '..'))
    _maps = sorted(_glob.glob(os.path.join(ws_root, 'slam_map_*.png')))
    default_map = _maps[-1] if _maps else ''

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # ── IGN_GAZEBO_RESOURCE_PATH ──────────────────────────────────────────────
    desc_parent  = os.path.dirname(desc_pkg)
    existing_ign = os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')
    ign_path     = (desc_parent + ':' + existing_ign) if existing_ign else desc_parent

    set_resource_path = SetEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', ign_path)

    # ── Argumentos ─────────────────────────────────────────────────────────────
    arg_sim_time = DeclareLaunchArgument('use_sim_time', default_value='true')
    arg_world    = DeclareLaunchArgument('world', default_value='real_arena',
                                         description='flat_plane | maze | real_arena')
    arg_map      = DeclareLaunchArgument('map_file', default_value=default_map,
                                         description='Ruta al PNG del mapa base')
    arg_loc      = DeclareLaunchArgument(
        'localization', default_value='mcl',
        description='mcl | slam_loconly | ground_truth')
    arg_nav      = DeclareLaunchArgument('navigation', default_value='true')
    arg_dyn      = DeclareLaunchArgument('dynamic_obstacles', default_value='true')
    arg_interval = DeclareLaunchArgument('spawn_interval_sec', default_value='60.0')
    arg_mode     = DeclareLaunchArgument('spawn_mode', default_value='on_path',
                                          description='on_path|near_path|random_free|fixed_sequence')
    arg_shape    = DeclareLaunchArgument('spawn_shape', default_value='box')
    arg_ttl      = DeclareLaunchArgument('obstacle_ttl_sec', default_value='90.0')
    arg_maxobs   = DeclareLaunchArgument('max_active_obstacles', default_value='3')
    arg_rviz     = DeclareLaunchArgument('rviz', default_value='true')
    arg_gui      = DeclareLaunchArgument('gui', default_value='true')

    world      = LaunchConfiguration('world')
    map_file   = LaunchConfiguration('map_file')
    loc_mode   = LaunchConfiguration('localization')
    nav_en     = LaunchConfiguration('navigation')
    dyn_en     = LaunchConfiguration('dynamic_obstacles')
    interval   = LaunchConfiguration('spawn_interval_sec')
    spawn_mode = LaunchConfiguration('spawn_mode')
    spawn_shape = LaunchConfiguration('spawn_shape')
    obs_ttl    = LaunchConfiguration('obstacle_ttl_sec')
    max_obs    = LaunchConfiguration('max_active_obstacles')
    rviz_en    = LaunchConfiguration('rviz')
    gui_en     = LaunchConfiguration('gui')

    # ── Gazebo Fortress ───────────────────────────────────────────────────────
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': PythonExpression([
                "'-r ' + ('" + world_maze + "' if '",
                world, "' == 'maze' else ('" + world_flat + "' if '",
                world, "' == 'flat_plane' else '" + world_arena + "'))"
            ]),
            'gz_version': '6',
        }.items(),
    )

    # ── robot_state_publisher ─────────────────────────────────────────────────
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
        output='screen',
    )

    # ── ROS ↔ Gazebo bridge (real_arena) ─────────────────────────────────────
    # /scan es el tópico estándar del LiDAR en Gazebo Fortress.
    # El DOM suscribe /scan directamente (configurado en controller_params.yaml).
    bridge_arena = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_bridge',
        arguments=[
            '/model/puzzlebot/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist',
            '/model/puzzlebot/odometry@nav_msgs/msg/Odometry@ignition.msgs.Odometry',
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            '/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan',
            '/world/real_arena/model/puzzlebot/joint_state'
            '@sensor_msgs/msg/JointState[ignition.msgs.Model',
            '/world/real_arena/dynamic_pose/info'
            '@geometry_msgs/msg/PoseArray[ignition.msgs.Pose_V',
        ],
        condition=IfCondition(PythonExpression(["'", world, "' == 'real_arena'"])),
        output='screen',
    )

    bridge_flat = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_bridge',
        arguments=[
            '/model/puzzlebot/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist',
            '/model/puzzlebot/odometry@nav_msgs/msg/Odometry@ignition.msgs.Odometry',
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            '/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan',
            '/world/flat_plane/model/puzzlebot/joint_state'
            '@sensor_msgs/msg/JointState[ignition.msgs.Model',
            '/world/flat_plane/dynamic_pose/info'
            '@geometry_msgs/msg/PoseArray[ignition.msgs.Pose_V',
        ],
        condition=IfCondition(PythonExpression(["'", world, "' == 'flat_plane'"])),
        output='screen',
    )

    bridge_maze = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_bridge',
        arguments=[
            '/model/puzzlebot/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist',
            '/model/puzzlebot/odometry@nav_msgs/msg/Odometry@ignition.msgs.Odometry',
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            '/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan',
            '/world/maze/model/puzzlebot/joint_state'
            '@sensor_msgs/msg/JointState[ignition.msgs.Model',
        ],
        condition=IfCondition(PythonExpression(["'", world, "' == 'maze'"])),
        output='screen',
    )

    # ── Joint relay → /joint_states ───────────────────────────────────────────
    joint_relay_arena = Node(
        package='ros_gz_bridge', executable='parameter_bridge', name='joint_relay',
        arguments=['/world/real_arena/model/puzzlebot/joint_state'
                   '@sensor_msgs/msg/JointState[ignition.msgs.Model'],
        remappings=[('/world/real_arena/model/puzzlebot/joint_state', '/joint_states')],
        condition=IfCondition(PythonExpression(["'", world, "' == 'real_arena'"])),
        output='screen',
    )

    joint_relay_flat = Node(
        package='ros_gz_bridge', executable='parameter_bridge', name='joint_relay',
        arguments=['/world/flat_plane/model/puzzlebot/joint_state'
                   '@sensor_msgs/msg/JointState[ignition.msgs.Model'],
        remappings=[('/world/flat_plane/model/puzzlebot/joint_state', '/joint_states')],
        condition=IfCondition(PythonExpression(["'", world, "' == 'flat_plane'"])),
        output='screen',
    )

    joint_relay_maze = Node(
        package='ros_gz_bridge', executable='parameter_bridge', name='joint_relay',
        arguments=['/world/maze/model/puzzlebot/joint_state'
                   '@sensor_msgs/msg/JointState[ignition.msgs.Model'],
        remappings=[('/world/maze/model/puzzlebot/joint_state', '/joint_states')],
        condition=IfCondition(PythonExpression(["'", world, "' == 'maze'"])),
        output='screen',
    )

    # ── TF estáticos ─────────────────────────────────────────────────────────
    lidar_tf = Node(
        package='tf2_ros', executable='static_transform_publisher', name='lidar_frame_fix',
        arguments=['0', '0', '0', '0', '0', '0', 'lidar_link', 'puzzlebot/base_footprint/lidar'],
        output='screen',
    )
    camera_tf = Node(
        package='tf2_ros', executable='static_transform_publisher', name='camera_frame_fix',
        arguments=['0', '0', '0', '0', '0', '0', 'camera_link', 'puzzlebot/base_footprint/camera'],
        output='screen',
    )

    # ── Spawn del robot ───────────────────────────────────────────────────────
    spawn_arena = TimerAction(period=5.0, actions=[ExecuteProcess(
        cmd=['ign', 'service', '-s', '/world/real_arena/create',
             '--reqtype', 'ignition.msgs.EntityFactory',
             '--reptype', 'ignition.msgs.Boolean', '--timeout', '5000',
             '--req',
             f'sdf_filename: "{sdf_file}", name: "puzzlebot", '
             f'pose: {{position: {{x: 0.30, y: 0.30, z: 0.05}}, '
             f'orientation: {{z: 0.7071068, w: 0.7071068}}}}'],
        additional_env={'IGN_GAZEBO_RESOURCE_PATH': ign_path},
        output='screen',
    )], condition=IfCondition(PythonExpression(["'", world, "' == 'real_arena'"])))

    spawn_flat = TimerAction(period=5.0, actions=[ExecuteProcess(
        cmd=['ign', 'service', '-s', '/world/flat_plane/create',
             '--reqtype', 'ignition.msgs.EntityFactory',
             '--reptype', 'ignition.msgs.Boolean', '--timeout', '5000',
             '--req', f'sdf_filename: "{sdf_file}", name: "puzzlebot", '
                      f'pose: {{position: {{z: 0.05}}}}'],
        additional_env={'IGN_GAZEBO_RESOURCE_PATH': ign_path},
        output='screen',
    )], condition=IfCondition(PythonExpression(["'", world, "' == 'flat_plane'"])))

    spawn_maze = TimerAction(period=5.0, actions=[ExecuteProcess(
        cmd=['ign', 'service', '-s', '/world/maze/create',
             '--reqtype', 'ignition.msgs.EntityFactory',
             '--reptype', 'ignition.msgs.Boolean', '--timeout', '5000',
             '--req', f'sdf_filename: "{sdf_file}", name: "puzzlebot", '
                      f'pose: {{position: {{z: 0.05}}}}'],
        additional_env={'IGN_GAZEBO_RESOURCE_PATH': ign_path},
        output='screen',
    )], condition=IfCondition(PythonExpression(["'", world, "' == 'maze'"])))

    # ── Odometría de ruedas → /odom ───────────────────────────────────────────
    # Publica TF odom→base_footprint. map→odom lo publica MCL o slam_loconly.
    wheel_odom = Node(
        package='puzzlebot_localization',
        executable='odometry_node',
        name='odometry_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'wheel_radius': 0.05,
            'wheel_separation': 0.19,
            'odom_topic': '/odom',
            'odom_frame': 'odom',
            'base_frame': 'base_footprint',
            'input_source': 'joint_states',
            'publish_tf': True,
        }],
    )

    # ── Ground truth odom (solo para localization:=ground_truth) ─────────────
    ground_truth_odom = Node(
        package='puzzlebot_localization',
        executable='ground_truth_odom',
        name='ground_truth_odom',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'model_name': 'puzzlebot',
            'odom_frame': 'odom',
            'base_frame': 'base_footprint',
            'pose_topic': '/world/real_arena/dynamic_pose/info',
        }],
        condition=IfCondition(
            PythonExpression(["'", loc_mode, "' == 'ground_truth'"])
        ),
    )

    # ── TF map→odom estático (solo ground_truth) ──────────────────────────────
    map_odom_static = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_odom_tf',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        condition=IfCondition(
            PythonExpression(["'", loc_mode, "' == 'ground_truth'"])
        ),
        output='screen',
    )

    # ── MCL — Localización con partículas sobre mapa fijo ────────────────────
    # Publica: /mcl/particles (PoseArray), /mcl/map (OccupancyGrid), TF map→odom
    # El planner usa /map. /mcl/map se remapea a /map aquí para que ambos coincidan.
    # MCL NO modifica el mapa: solo localiza el robot sobre él.
    mcl_node = Node(
        package='puzzlebot_slam',
        executable='mcl',
        name='mcl',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'map_path':       map_file,
            'map_resolution':  0.05,
            'map_origin_x':   -0.25,
            'map_origin_y':   -0.25,
            'num_particles':   500,
            'top_k':           150,
            'noise_xy':        0.05,
            'noise_theta':     0.05,
            'score_rays':      36,
            'map_frame':      'map',
            'odom_frame':     'odom',
        }],
        # /mcl/map → /map  para que path_planner y DOM reciban el mapa base fijo
        remappings=[('/mcl/map', '/map')],
        condition=IfCondition(PythonExpression(["'", loc_mode, "' == 'mcl'"])),
    )

    # ── slam_node en modo localization-only ───────────────────────────────────
    # Alternativa a MCL: carga el mapa PNG, hace scan matching para localizar,
    # NO modifica el mapa. Publica TF map→odom directamente.
    slam_loconly = Node(
        package='puzzlebot_slam',
        executable='slam_node',
        name='slam_node',
        output='screen',
        parameters=[slam_cfg, {
            'use_sim_time':          True,
            'localization_only':     True,
            'localization_map_path': map_file,
            'publish_map_odom_tf':   True,
        }],
        condition=IfCondition(PythonExpression(["'", loc_mode, "' == 'slam_loconly'"])),
    )

    # ── Navegación A* + steering + obstacle_avoidance ─────────────────────────
    nav_launch_file = os.path.join(bringup_pkg, 'launch', 'navigation.launch.py')
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav_launch_file),
        launch_arguments={
            'use_sim_time':     'true',
            'cmd_vel_topic':    '/model/puzzlebot/cmd_vel',
            'obstacle_manager': 'dynamic',
        }.items(),
        condition=IfCondition(nav_en),
    )

    # ── Dynamic Obstacle Manager ──────────────────────────────────────────────
    # scan_topic=/scan: en Gazebo el bridge publica en /scan directamente.
    # ignore_points_matching_static_map=true: filtra paredes del mapa base.
    # Esto garantiza que SOLO obstáculos nuevos (no en el mapa) activan el replan.
    dynamic_obstacle_manager = Node(
        package='puzzlebot_planning',
        executable='dynamic_obstacle_manager',
        name='dynamic_obstacle_manager',
        parameters=[ctrl_cfg, {
            'use_sim_time':                      True,
            'scan_topic':                        '/scan',
            'ignore_points_matching_static_map': True,
            'known_map_match_tolerance_m':       0.12,
        }],
        output='screen',
        condition=IfCondition(nav_en),
    )

    # ── Dynamic Obstacle Spawner ──────────────────────────────────────────────
    # desktop_mode=False: crea modelo físico en Gazebo via ign service.
    # NO publica la posición al planner ni al DOM.
    # El DOM solo sabrá del obstáculo cuando el LiDAR lo detecte en /scan.
    dynamic_obstacle_spawner = Node(
        package='puzzlebot_planning',
        executable='dynamic_obstacle_spawner_node',
        name='dynamic_obstacle_spawner_node',
        parameters=[ctrl_cfg, {
            'use_sim_time':         True,
            'desktop_mode':         False,
            'spawn_interval_sec':   interval,
            'spawn_mode':           spawn_mode,
            'obstacle_shape':       spawn_shape,
            'obstacle_ttl_sec':     obs_ttl,
            'max_active_obstacles': max_obs,
            'world_name':           world,
            'spawn_delay_after_path_sec': 5.0,
        }],
        output='screen',
        condition=IfCondition(dyn_en),
    )

    # ── RViz ─────────────────────────────────────────────────────────────────
    # Usa dynamic_obs_rviz.rviz — muestra /map, /augmented_map, /scan,
    # /planned_path, /mcl/particles, /dom/markers, /dynamic_obstacles
    rviz_node = TimerAction(
        period=15.0,
        actions=[Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_file],
            parameters=[{'use_sim_time': True}],
            output='screen',
        )],
        condition=IfCondition(rviz_en),
    )

    return LaunchDescription([
        set_resource_path,
        # Argumentos
        arg_sim_time, arg_world, arg_map, arg_loc, arg_nav, arg_dyn,
        arg_interval, arg_mode, arg_shape, arg_ttl, arg_maxobs, arg_rviz, arg_gui,
        # Gazebo
        gz_sim,
        rsp,
        # Bridges
        bridge_arena, bridge_flat, bridge_maze,
        # Joint relays
        joint_relay_arena, joint_relay_flat, joint_relay_maze,
        # TF estáticos
        lidar_tf, camera_tf,
        # Spawn del robot
        spawn_arena, spawn_flat, spawn_maze,
        # Odometría
        TimerAction(period=6.0, actions=[wheel_odom]),
        # Localización (elige según arg)
        TimerAction(period=7.0, actions=[ground_truth_odom, map_odom_static]),
        TimerAction(period=7.0, actions=[mcl_node]),
        TimerAction(period=7.0, actions=[slam_loconly]),
        # Navegación (+8s para que el mapa MCL esté disponible)
        TimerAction(period=10.0, actions=[navigation]),
        # DOM (+1s sobre nav para que /map ya esté latched)
        TimerAction(period=11.0, actions=[dynamic_obstacle_manager]),
        # Spawner (arranca cuando todo esté funcionando)
        TimerAction(period=15.0, actions=[dynamic_obstacle_spawner]),
        # RViz
        rviz_node,
    ])
