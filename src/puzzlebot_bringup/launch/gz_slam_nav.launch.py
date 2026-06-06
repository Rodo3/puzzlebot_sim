"""
gz_slam_nav.launch.py — Gazebo Fortress + SLAM mapping + Navegación A* + Obstáculos dinámicos

El SLAM arranca con el mapa PNG ya conocido (paredes de la pista) y sigue mapeando
en tiempo real mientras el robot navega. El LiDAR detecta obstáculos físicos en Gazebo
y los integra al mapa vivo → A* replantea automáticamente al verlos en /augmented_map.

Flujo de uso:
─────────────

  # Navegación con mapa preexistente + mapeo continuo + obstáculos dinámicos
  ros2 launch puzzlebot_bringup gz_slam_nav.launch.py

  En otra terminal (teleop para explorar si quieres):
    ros2 run teleop_twist_keyboard teleop_twist_keyboard \
         --ros-args -r /cmd_vel:=/model/puzzlebot/cmd_vel

  En RViz:
    - Presiona G → "2D Nav Goal" → click en el destino
    - La ruta verde aparece inmediatamente (mapa PNG ya cargado)
    - El spawner coloca obstáculos físicos en la ruta
    - El LiDAR del robot los detecta → SLAM actualiza /map → A* replantea → robot rodea

  ARGUMENTOS
  ──────────
  world             [real_arena]   World de Gazebo: flat_plane | maze | real_arena
  initial_map       [<ruta PNG>]   PNG de mapa previo para inicializar SLAM
  navigation        [true]         Lanza A* + steering + DOM
  dynamic_obstacles [true]         Lanza dynamic_obstacle_spawner
  obstacle_manager  [dynamic]      dynamic | legacy | none
  spawn_mode        [on_path]      on_path | near_path | random_free | fixed_sequence
  use_sim_time      [true]         Reloj de simulación
  rviz              [true]         Lanzar RViz
  web_bridge        [true]         Lanza bridge WebSocket para el dashboard
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    desc_pkg    = get_package_share_directory('puzzlebot_description')
    bringup_pkg = get_package_share_directory('puzzlebot_bringup')
    ros_gz_sim  = get_package_share_directory('ros_gz_sim')

    sdf_file    = os.path.join(desc_pkg, 'sdf',    'puzzlebot_gz.sdf')
    urdf_file   = os.path.join(desc_pkg, 'urdf',   'puzzlebot_gz.urdf')
    world_arena = os.path.join(desc_pkg, 'worlds', 'real_arena.sdf')
    world_flat  = os.path.join(desc_pkg, 'worlds', 'flat_plane.sdf')
    world_maze  = os.path.join(desc_pkg, 'worlds', 'maze.sdf')
    rviz_file   = os.path.join(desc_pkg, 'rviz',   'mapping_rviz.rviz')
    slam_cfg    = os.path.join(bringup_pkg, 'config', 'slam_params.yaml')
    ctrl_cfg    = os.path.join(bringup_pkg, 'config', 'controller_params.yaml')
    kalman_cfg  = os.path.join(bringup_pkg, 'config', 'kalman_params.yaml')
    aruco_map_yaml = os.path.join(bringup_pkg, 'config', 'aruco_map.yaml')

    # Workspace root: src/puzzlebot_bringup/share/puzzlebot_bringup → ../../../../
    ws_root = os.path.abspath(os.path.join(bringup_pkg, '..', '..', '..', '..'))

    # Mapa PNG previo — busca el más reciente en el workspace root, o vacío si no hay.
    import glob as _glob
    _maps = sorted(_glob.glob(os.path.join(ws_root, 'slam_map_*.png')))
    default_map_png = _maps[-1] if _maps else ''

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # ── IGN_GAZEBO_RESOURCE_PATH ──────────────────────────────────────────────
    desc_parent = os.path.dirname(desc_pkg)
    existing_ign = os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')
    ign_path = (desc_parent + ':' + existing_ign) if existing_ign else desc_parent

    set_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=ign_path,
    )

    # ── Argumentos ────────────────────────────────────────────────────────────
    arg_world = DeclareLaunchArgument(
        'world', default_value='real_arena',
        description='World: flat_plane | maze | real_arena')
    arg_initial_map = DeclareLaunchArgument(
        'initial_map', default_value=default_map_png,
        description='PNG del mapa previo para inicializar SLAM. Vacío = empezar desde cero.')
    arg_nav = DeclareLaunchArgument(
        'navigation', default_value='false',
        description='Lanzar A* + steering + DOM')
    arg_dyn = DeclareLaunchArgument(
        'dynamic_obstacles', default_value='false',
        description='Lanzar dynamic_obstacle_spawner')
    arg_obs_mgr = DeclareLaunchArgument(
        'obstacle_manager', default_value='dynamic',
        description='dynamic | legacy | none')
    arg_spawn_mode = DeclareLaunchArgument(
        'spawn_mode', default_value='on_path',
        description='on_path | near_path | random_free | fixed_sequence')
    arg_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true')
    arg_rviz = DeclareLaunchArgument(
        'rviz', default_value='true')
    arg_web_bridge = DeclareLaunchArgument(
        'web_bridge', default_value='true',
        description='Lanza puzzlebot_web_bridge para el dashboard')
    arg_artifact_dir = DeclareLaunchArgument(
        'artifact_dir', default_value='',
        description='Ruta a artifacts_final/ con modelos de voz. Vacío = auto-detectar o deshabilitar.')

    world       = LaunchConfiguration('world')
    initial_map = LaunchConfiguration('initial_map')
    nav_en      = LaunchConfiguration('navigation')
    dyn_en      = LaunchConfiguration('dynamic_obstacles')
    obs_mgr     = LaunchConfiguration('obstacle_manager')
    spawn_mode  = LaunchConfiguration('spawn_mode')
    sim_time    = LaunchConfiguration('use_sim_time')
    rviz_en     = LaunchConfiguration('rviz')
    web_bridge_en = LaunchConfiguration('web_bridge')
    artifact_dir = LaunchConfiguration('artifact_dir')

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

    # ── ROS ↔ Gazebo bridge (real_arena) ────────────────────────────────────
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_bridge',
        arguments=[
            '/model/puzzlebot/cmd_vel'
            '@geometry_msgs/msg/Twist@ignition.msgs.Twist',
            '/model/puzzlebot/odometry'
            '@nav_msgs/msg/Odometry@ignition.msgs.Odometry',
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            '/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan',
            '/world/real_arena/model/puzzlebot/joint_state'
            '@sensor_msgs/msg/JointState[ignition.msgs.Model',
            '/world/real_arena/dynamic_pose/info'
            '@geometry_msgs/msg/PoseArray[ignition.msgs.Pose_V',
            '/camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
        ],
        parameters=[{
            'qos_overrides./model/puzzlebot.subscriber.reliability': 'reliable',
        }],
        condition=IfCondition(
            PythonExpression(["'", world, "' == 'real_arena'"])
        ),
        output='screen',
    )

    bridge_flat = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_bridge',
        arguments=[
            '/model/puzzlebot/cmd_vel'
            '@geometry_msgs/msg/Twist@ignition.msgs.Twist',
            '/model/puzzlebot/odometry'
            '@nav_msgs/msg/Odometry@ignition.msgs.Odometry',
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            '/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan',
            '/world/flat_plane/model/puzzlebot/joint_state'
            '@sensor_msgs/msg/JointState[ignition.msgs.Model',
            '/world/flat_plane/dynamic_pose/info'
            '@geometry_msgs/msg/PoseArray[ignition.msgs.Pose_V',
        ],
        condition=IfCondition(
            PythonExpression(["'", world, "' == 'flat_plane'"])
        ),
        output='screen',
    )

    bridge_maze = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_bridge',
        arguments=[
            '/model/puzzlebot/cmd_vel'
            '@geometry_msgs/msg/Twist@ignition.msgs.Twist',
            '/model/puzzlebot/odometry'
            '@nav_msgs/msg/Odometry@ignition.msgs.Odometry',
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            '/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan',
            '/world/maze/model/puzzlebot/joint_state'
            '@sensor_msgs/msg/JointState[ignition.msgs.Model',
            '/world/maze/dynamic_pose/info'
            '@geometry_msgs/msg/PoseArray[ignition.msgs.Pose_V',
        ],
        condition=IfCondition(
            PythonExpression(["'", world, "' == 'maze'"])
        ),
        output='screen',
    )

    # ── Joint relay → /joint_states ───────────────────────────────────────────
    joint_relay_arena = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='joint_relay',
        arguments=[
            '/world/real_arena/model/puzzlebot/joint_state'
            '@sensor_msgs/msg/JointState[ignition.msgs.Model',
        ],
        remappings=[('/world/real_arena/model/puzzlebot/joint_state', '/joint_states')],
        condition=IfCondition(PythonExpression(["'", world, "' == 'real_arena'"])),
        output='screen',
    )

    joint_relay_flat = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='joint_relay',
        arguments=[
            '/world/flat_plane/model/puzzlebot/joint_state'
            '@sensor_msgs/msg/JointState[ignition.msgs.Model',
        ],
        remappings=[('/world/flat_plane/model/puzzlebot/joint_state', '/joint_states')],
        condition=IfCondition(PythonExpression(["'", world, "' == 'flat_plane'"])),
        output='screen',
    )

    joint_relay_maze = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='joint_relay',
        arguments=[
            '/world/maze/model/puzzlebot/joint_state'
            '@sensor_msgs/msg/JointState[ignition.msgs.Model',
        ],
        remappings=[('/world/maze/model/puzzlebot/joint_state', '/joint_states')],
        condition=IfCondition(PythonExpression(["'", world, "' == 'maze'"])),
        output='screen',
    )

    # ── TF estáticos LiDAR y cámara ─────────────────────────────────────────
    lidar_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='lidar_frame_fix',
        arguments=['0', '0', '0', '0', '0', '0',
                   'lidar_link', 'puzzlebot/base_footprint/lidar'],
        output='screen',
    )

    camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_frame_fix',
        arguments=['0', '0', '0', '0', '0', '0',
                   'camera_link', 'puzzlebot/base_footprint/camera'],
        output='screen',
    )

    # ── Spawn del robot (5 s de retraso para que Gazebo arranque) ────────────
    spawn_arena = TimerAction(
        period=5.0,
        actions=[ExecuteProcess(
            cmd=[
                'ign', 'service',
                '-s', '/world/real_arena/create',
                '--reqtype', 'ignition.msgs.EntityFactory',
                '--reptype', 'ignition.msgs.Boolean',
                '--timeout', '5000',
                '--req',
                f'sdf_filename: "{sdf_file}", name: "puzzlebot", '
                f'pose: {{position: {{x: 0.30, y: 0.30, z: 0.05}}, '
                f'orientation: {{z: 0.7071068, w: 0.7071068}}}}',
            ],
            additional_env={'IGN_GAZEBO_RESOURCE_PATH': ign_path},
            output='screen',
        )],
        condition=IfCondition(PythonExpression(["'", world, "' == 'real_arena'"])),
    )

    spawn_flat = TimerAction(
        period=5.0,
        actions=[ExecuteProcess(
            cmd=[
                'ign', 'service',
                '-s', '/world/flat_plane/create',
                '--reqtype', 'ignition.msgs.EntityFactory',
                '--reptype', 'ignition.msgs.Boolean',
                '--timeout', '5000',
                '--req',
                f'sdf_filename: "{sdf_file}", name: "puzzlebot", '
                f'pose: {{position: {{z: 0.05}}}}',
            ],
            additional_env={'IGN_GAZEBO_RESOURCE_PATH': ign_path},
            output='screen',
        )],
        condition=IfCondition(PythonExpression(["'", world, "' == 'flat_plane'"])),
    )

    spawn_maze = TimerAction(
        period=5.0,
        actions=[ExecuteProcess(
            cmd=[
                'ign', 'service',
                '-s', '/world/maze/create',
                '--reqtype', 'ignition.msgs.EntityFactory',
                '--reptype', 'ignition.msgs.Boolean',
                '--timeout', '5000',
                '--req',
                f'sdf_filename: "{sdf_file}", name: "puzzlebot", '
                f'pose: {{position: {{z: 0.05}}}}',
            ],
            additional_env={'IGN_GAZEBO_RESOURCE_PATH': ign_path},
            output='screen',
        )],
        condition=IfCondition(PythonExpression(["'", world, "' == 'maze'"])),
    )

    # ── Odometría de encoders → /odom_raw (predicción EKF, sin TF) ──────────
    # publish_tf: False porque el TF odom→base_footprint lo publica kalman_filter_node
    # después de fusionar con ArUco. El slam_node lee ese TF corregido.
    wheel_odom_raw = TimerAction(
        period=6.0,
        actions=[Node(
            package='puzzlebot_localization',
            executable='odometry_node',
            name='odometry_node',
            output='screen',
            parameters=[{
                'use_sim_time':    True,
                'wheel_radius':    0.05,
                'wheel_separation': 0.19,
                'odom_topic':      '/odom_raw',
                'odom_frame':      'odom',
                'base_frame':      'base_footprint',
                'input_source':    'joint_states',
                'publish_tf':      False,
            }],
        )],
    )

    # ── ArUco Oracle — Ground truth de Gazebo → /aruco/pose sintético ────────
    # Lee la pose real del robot de Gazebo y simula detecciones ArUco con ruido.
    # Requiere que el bridge ya publique /world/real_arena/dynamic_pose/info.
    aruco_oracle = TimerAction(
        period=6.0,
        actions=[Node(
            package='puzzlebot_localization',
            executable='aruco_oracle',
            name='aruco_oracle',
            output='screen',
            parameters=[{
                'use_sim_time':       True,
                'pose_topic':         '/world/real_arena/dynamic_pose/info',
                'aruco_map_file':     aruco_map_yaml,
                'max_detection_dist': 2.0,
                'sigma_lateral':      0.015,
                'sigma_depth_base':   0.020,
                'sigma_yaw':          0.015,
                'publish_rate_hz':    10.0,
                'detection_prob':     1.0,
            }],
        )],
    )

    # ── Kalman EKF — fusiona /odom_raw + /aruco/pose → /odom + TF ───────────
    # init_from_aruco: false → arranca inmediatamente con la pose del spawn (0.30, 0.30, π/2).
    # Así el SLAM tiene TF válido desde el primer scan en lugar de esperar la primera
    # detección ArUco, que puede tardar si el robot no está frente a un marcador.
    kalman = TimerAction(
        period=6.0,
        actions=[Node(
            package='puzzlebot_localization',
            executable='kalman_filter_node',
            name='kalman_filter_node',
            output='screen',
            parameters=[kalman_cfg, {
                'use_sim_time':   True,
                'init_from_aruco': False,
                'initial_x':      0.30,
                'initial_y':      0.30,
                'initial_theta':  1.5708,
            }],
        )],
    )

    # ── SLAM mapping con mapa inicial ─────────────────────────────────────────
    # Arranca 8 s después para que kalman haya publicado al menos un TF
    # odom→base_footprint antes de que llegue el primer scan.
    # scan_match_updates_map_odom: True → el matcher corrige drift acumulado.
    slam_node = TimerAction(
        period=8.0,
        actions=[Node(
            package='puzzlebot_slam',
            executable='slam_node',
            name='slam_node',
            parameters=[slam_cfg, {
                'use_sim_time':               True,
                'localization_map_path':      initial_map,
                'localization_only':          False,
                'scan_match_updates_map_odom': True,
            }],
            output='screen',
        )],
    )

    # ── Navegación A* + steering + obstacle_avoidance + DOM ─────────────────
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_pkg, 'launch', 'navigation.launch.py')
        ),
        launch_arguments={
            'use_sim_time':     'true',
            'cmd_vel_topic':    '/model/puzzlebot/cmd_vel',
            'obstacle_manager': obs_mgr,
            'scan_topic':       '/scan',
        }.items(),
        condition=IfCondition(nav_en),
    )

    # ── Dynamic obstacle spawner (Gazebo mode, no desktop) ───────────────────
    spawner = Node(
        package='puzzlebot_planning',
        executable='dynamic_obstacle_spawner_node',
        name='dynamic_obstacle_spawner_node',
        parameters=[ctrl_cfg, {
            'use_sim_time':   True,
            'desktop_mode':   False,
            'spawn_mode':     spawn_mode,
            'spawn_delay_after_path_sec': 5.0,
            'world_name':     world,
        }],
        output='screen',
        condition=IfCondition(dyn_en),
    )

    web_bridge = Node(
        package='puzzlebot_web_bridge',
        executable='bridge_node',
        name='puzzlebot_web_bridge',
        output='screen',
        parameters=[{
            'use_sim_time':       True,
            'cmd_vel_out_topic':  '/model/puzzlebot/cmd_vel',
            'artifact_dir':       artifact_dir,
        }],
        condition=IfCondition(web_bridge_en),
    )

    # ── RViz (con retraso para que el mapa empiece a aparecer) ───────────────
    rviz = TimerAction(
        period=10.0,
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
        arg_world, arg_initial_map, arg_nav, arg_dyn, arg_obs_mgr, arg_spawn_mode, arg_sim_time, arg_rviz,
        arg_web_bridge, arg_artifact_dir,
        # Gazebo
        gz_sim,
        rsp,
        # Bridges
        bridge,
        bridge_flat,
        bridge_maze,
        # Joint relays
        joint_relay_arena,
        joint_relay_flat,
        joint_relay_maze,
        # TF estáticos
        lidar_tf,
        camera_tf,
        # Spawn del robot
        spawn_arena,
        spawn_flat,
        spawn_maze,
        # Odometría de encoders (predicción EKF)
        wheel_odom_raw,
        # ArUco oracle (ground truth → /aruco/pose sintético)
        aruco_oracle,
        # Kalman EKF (fusiona encoders + ArUco → /odom + TF)
        kalman,
        # SLAM (arranca 8s después, cuando kalman ya tiene TF publicado)
        slam_node,
        # Navegación (condicional)
        navigation,
        # Spawner de obstáculos (condicional)
        spawner,
        # Web dashboard bridge (condicional)
        web_bridge,
        # RViz
        rviz,
    ])
