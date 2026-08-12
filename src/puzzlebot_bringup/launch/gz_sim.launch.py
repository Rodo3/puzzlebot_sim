"""
Gazebo Fortress (ignition-gazebo 6) simulation launch.

Stack: ros-humble-ros-gz (Fortress bridge) — the official ROS 2 Humble pairing.
Binary: ign gazebo (gz_version=6).  Never mix with gz sim / Harmonic binaries.
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
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    desc_pkg    = get_package_share_directory('puzzlebot_description')
    slam_pkg    = get_package_share_directory('puzzlebot_slam')
    bringup_pkg = get_package_share_directory('puzzlebot_bringup')
    ros_gz_sim  = get_package_share_directory('ros_gz_sim')

    urdf_file    = os.path.join(desc_pkg, 'urdf', 'puzzlebot_gz.urdf')
    sdf_file     = os.path.join(desc_pkg, 'sdf',    'puzzlebot_gz.sdf')
    world_flat   = os.path.join(desc_pkg, 'worlds', 'flat_plane.sdf')
    world_maze   = os.path.join(desc_pkg, 'worlds', 'maze.sdf')
    world_arena  = os.path.join(desc_pkg, 'worlds', 'real_arena.sdf')
    world_almacen = os.path.join(desc_pkg, 'worlds', 'almacen_racks.sdf')
    rviz_flat    = os.path.join(desc_pkg, 'rviz',   'puzzlebot_rviz.rviz')
    rviz_maze    = os.path.join(desc_pkg, 'rviz',   'mcl_rviz.rviz')
    rviz_mapping = os.path.join(desc_pkg, 'rviz',   'mapping_rviz.rviz')
    map_file      = os.path.join(slam_pkg,    'puzzlebot_slam', 'maze_map.png')
    slam_cfg      = os.path.join(bringup_pkg, 'config', 'slam_params.yaml')
    kalman_cfg    = os.path.join(bringup_pkg, 'config', 'kalman_params.yaml')
    aruco_map_yaml = os.path.join(bringup_pkg, 'config', 'aruco_map.yaml')
    camera_calib_yaml = os.path.join(bringup_pkg, 'config', 'camera_calibration.yaml')
    camera_extr_yaml  = os.path.join(bringup_pkg, 'config', 'camera_extrinsics.yaml')

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    desc_share_parent = os.path.dirname(desc_pkg)
    existing = os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')
    ign_resource_path = (desc_share_parent + ':' + existing) if existing else desc_share_parent

    set_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=ign_resource_path,
    )

    arg_world = DeclareLaunchArgument('world', default_value='real_arena',
                                      description="'flat_plane', 'maze', 'real_arena' (pista física 3.76×4.86 m), o 'almacen' (almacén con racks y ArUcos)")
    arg_gui   = DeclareLaunchArgument('gui',   default_value='true')
    arg_slam  = DeclareLaunchArgument('slam',  default_value='true')
    arg_rviz  = DeclareLaunchArgument('rviz',  default_value='true')
    arg_mode  = DeclareLaunchArgument('mode',  default_value='mapping',
                                      description="'mapping' or 'mcl'")
    
    arg_odom_source = DeclareLaunchArgument(
        'odom_source',
        default_value='dead_reckoning',
        description="'ground_truth' or 'dead_reckoning' for mode:=mapping",
    )

    # ── Argumentos de odometría Kalman (solo world:=real_arena) ──────────
    # kalman:=true  → odometry_node publica /odom_raw + kalman_filter_node
    #                 produce /odom y TF odom→base_footprint
    #                 (Estrategia A: predicción pura sin ArUco)
    # kalman:=false → odometry_node publica /odom directamente (default)
    arg_kalman = DeclareLaunchArgument(
        'kalman',
        default_value='true',
        description='[real_arena] Usa kalman_filter_node entre odom_raw y odom',
    )

    # aruco_oracle:=true → aruco_oracle lee ground truth de Gazebo y publica
    #                       /aruco/pose sintético → kalman lo fusiona (Estrategia B,
    #                       debug del EKF sin depender de detección real)
    # aruco_oracle:=false (default) → aruco_node hace detección real por cámara
    #                       + OpenCV, con su incertidumbre real. Comportamiento
    #                       por defecto: lo más fiel posible al robot físico.
    # Requiere kalman:=true para tener efecto útil.
    arg_aruco_oracle = DeclareLaunchArgument(
        'aruco_oracle',
        default_value='false',
        description='Publica /aruco/pose sintético desde ground truth (Gazebo) en vez de '
                    'usar detección real por cámara. true = debug/ground-truth, '
                    'false = visión real (default).',
    )

    # navigation:=true → lanza navigation.launch.py (A* + steering_controller + obstacle_avoidance)
    # Requiere que SLAM haya construido /map (mode:=mapping o que /map ya esté disponible).
    # Usa 2D Nav Goal en RViz para enviar /goal_pose al planner.
    arg_navigation = DeclareLaunchArgument(
        'navigation',
        default_value='true',
        description='Lanza navegación autónoma A* + steering_controller + obstacle_avoidance. '
                    'Envía /goal_pose desde RViz (tecla G → 2D Nav Goal).',
    )

    # arg_web_bridge = DeclareLaunchArgument(
    #     'web_bridge',
    #     default_value='false',
    #     description='Lanza puzzlebot_web_bridge (WebSocket dashboard). '
    #                 'Deshabilitar con web_bridge:=false si no se usa el dashboard.',
    # )

    world_name    = LaunchConfiguration('world')
    slam_en       = LaunchConfiguration('slam')
    rviz_en       = LaunchConfiguration('rviz')
    mode          = LaunchConfiguration('mode')
    odom_source   = LaunchConfiguration('odom_source')
    kalman_en     = LaunchConfiguration('kalman')
    oracle_en     = LaunchConfiguration('aruco_oracle')
    nav_en        = LaunchConfiguration('navigation')
    web_bridge_en = LaunchConfiguration('web_bridge')

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': PythonExpression([
                "'-r ' + ('" + world_maze + "' if '",
                world_name,
                "' == 'maze' else ('" + world_arena + "' if '",
                world_name,
                "' == 'real_arena' else ('" + world_almacen + "' if '",
                world_name,
                "' == 'almacen' else '" + world_flat + "')))"
            ]),
            'gz_version': '6',
        }.items(),
    )

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
        output='screen',
    )

    # ── 3. ros_gz_bridge ─────────────────────────────────────────────────
    # Camera topics added: /camera/image_raw and /camera/camera_info
    # Fortress syntax: '[' means Gazebo→ROS only (subscribe from Gazebo)

    # ── real_arena bridge ────────────────────────────────────────────────
    # Igual que bridge_maze pero con world name 'real_arena'.
    # Activo solo cuando world:=real_arena.
    bridge_arena = Node(
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
            PythonExpression(["'", world_name, "' == 'real_arena'"])
        ),
        output='screen',
    )

    joint_relay_arena = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='joint_relay',
        arguments=[
            '/world/real_arena/model/puzzlebot/joint_state'
            '@sensor_msgs/msg/JointState[ignition.msgs.Model',
        ],
        remappings=[
            ('/world/real_arena/model/puzzlebot/joint_state', '/joint_states'),
        ],
        condition=IfCondition(
            PythonExpression(["'", world_name, "' == 'real_arena'"])
        ),
        output='screen',
    )

    # ── almacén bridge ───────────────────────────────────────────────────
    # Igual que bridge_arena pero con world name 'almacen_racks'.
    # Activo solo cuando world:=almacen.
    bridge_almacen = Node(
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
            '/world/almacen_racks/model/puzzlebot/joint_state'
            '@sensor_msgs/msg/JointState[ignition.msgs.Model',
            '/world/almacen_racks/dynamic_pose/info'
            '@geometry_msgs/msg/PoseArray[ignition.msgs.Pose_V',
            '/camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
        ],
        parameters=[{
            'qos_overrides./model/puzzlebot.subscriber.reliability': 'reliable',
        }],
        condition=IfCondition(
            PythonExpression(["'", world_name, "' == 'almacen'"])
        ),
        output='screen',
    )

    joint_relay_almacen = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='joint_relay',
        arguments=[
            '/world/almacen_racks/model/puzzlebot/joint_state'
            '@sensor_msgs/msg/JointState[ignition.msgs.Model',
        ],
        remappings=[
            ('/world/almacen_racks/model/puzzlebot/joint_state', '/joint_states'),
        ],
        condition=IfCondition(
            PythonExpression(["'", world_name, "' == 'almacen'"])
        ),
        output='screen',
    )

    # Robot spawn: esquina SW de la pista, 30 cm del borde, mirando al norte (yaw=π/2).
    # Coincide con la pose inicial configurada en kalman_params.yaml.
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
            additional_env={'IGN_GAZEBO_RESOURCE_PATH': ign_resource_path},
            output='screen',
        )],
        condition=IfCondition(
            PythonExpression(["'", world_name, "' == 'real_arena'"])
        ),
    )

    # Robot spawn: almacén, frente al marker ID 21 (pared sur, x=0.79, y=0.035),
    # a 70 cm del marker, mirando al sur (yaw=-π/2) para que la cámara lo vea
    # de inmediato al arrancar — cumple el criterio de spawn en lugar conocido.
    spawn_almacen = TimerAction(
        period=5.0,
        actions=[ExecuteProcess(
            cmd=[
                'ign', 'service',
                '-s', '/world/almacen_racks/create',
                '--reqtype', 'ignition.msgs.EntityFactory',
                '--reptype', 'ignition.msgs.Boolean',
                '--timeout', '5000',
                '--req',
                f'sdf_filename: "{sdf_file}", name: "puzzlebot", '
                f'pose: {{position: {{x: 0.79, y: 0.70, z: 0.10}}, '
                f'orientation: {{z: -0.7071068, w: 0.7071068}}}}',
            ],
            additional_env={'IGN_GAZEBO_RESOURCE_PATH': ign_resource_path},
            output='screen',
        )],
        condition=IfCondition(
            PythonExpression(["'", world_name, "' == 'almacen'"])
        ),
    )

    # ── Odometría de ruedas — variante A: publicación directa a /odom ───────
    # Activa cuando kalman:=false (comportamiento estándar dead_reckoning).
    # publish_tf:true → dueño de odom→base_footprint.
    # wheel_separation: 0.19 m coincide con el SDF del robot en Gazebo.
    wheel_odom_arena_direct = Node(
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
        remappings=[('/joint_states', '/world/real_arena/model/puzzlebot/joint_state')],
        condition=IfCondition(PythonExpression([
            "'", world_name, "' == 'real_arena' and '", slam_en, "' == 'true' and ",
            "('", mode, "' != 'mapping' or '", odom_source, "' == 'dead_reckoning') and ",
            "'", kalman_en, "' == 'false'",
        ])),
    )

    # ── Odometría de ruedas — variante B: alimenta /odom_raw al Kalman ──────
    # Activa cuando kalman:=true.
    # publish_tf:false → el kalman_filter_node es dueño de odom→base_footprint.
    # Para simular el error de wheel_separation del robot real, cambia 0.19→0.172.
    wheel_odom_arena_raw = Node(
        package='puzzlebot_localization',
        executable='odometry_node',
        name='odometry_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'wheel_radius': 0.05,
            'wheel_separation': 0.172,   # ← cambiar a 0.172 para simular error real
            'odom_topic': '/odom_raw',
            'odom_frame': 'odom',
            'base_frame': 'base_footprint',
            'input_source': 'joint_states',
            'publish_tf': False,
        }],
        remappings=[('/joint_states', '/world/real_arena/model/puzzlebot/joint_state')],
        condition=IfCondition(PythonExpression([
            "'", world_name, "' == 'real_arena' and '", slam_en, "' == 'true' and ",
            "('", mode, "' != 'mapping' or '", odom_source, "' == 'dead_reckoning') and ",
            "'", kalman_en, "' == 'true'",
        ])),
    )

    # ── Ground truth odom para real_arena (referencia perfecta) ─────────────
    # Activo solo con odom_source:=ground_truth (sin Kalman, sin dead_reckoning).
    ground_truth_arena = Node(
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
        condition=IfCondition(PythonExpression([
            "'", world_name, "' == 'real_arena' and '", slam_en, "' == 'true' and ",
            "'", mode, "' == 'mapping' and '", odom_source, "' == 'ground_truth'",
        ])),
    )

    # ── Odometría de ruedas — almacén, variante A: publicación directa a /odom ──
    wheel_odom_almacen_direct = Node(
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
        remappings=[('/joint_states', '/world/almacen_racks/model/puzzlebot/joint_state')],
        condition=IfCondition(PythonExpression([
            "'", world_name, "' == 'almacen' and '", slam_en, "' == 'true' and ",
            "('", mode, "' != 'mapping' or '", odom_source, "' == 'dead_reckoning') and ",
            "'", kalman_en, "' == 'false'",
        ])),
    )

    # ── Odometría de ruedas — almacén, variante B: alimenta /odom_raw al Kalman ──
    wheel_odom_almacen_raw = Node(
        package='puzzlebot_localization',
        executable='odometry_node',
        name='odometry_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'wheel_radius': 0.05,
            'wheel_separation': 0.172,
            'odom_topic': '/odom_raw',
            'odom_frame': 'odom',
            'base_frame': 'base_footprint',
            'input_source': 'joint_states',
            'publish_tf': False,
        }],
        remappings=[('/joint_states', '/world/almacen_racks/model/puzzlebot/joint_state')],
        condition=IfCondition(PythonExpression([
            "'", world_name, "' == 'almacen' and '", slam_en, "' == 'true' and ",
            "('", mode, "' != 'mapping' or '", odom_source, "' == 'dead_reckoning') and ",
            "'", kalman_en, "' == 'true'",
        ])),
    )

    # ── Ground truth odom para almacén (referencia perfecta) ────────────────
    ground_truth_almacen = Node(
        package='puzzlebot_localization',
        executable='ground_truth_odom',
        name='ground_truth_odom',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'model_name': 'puzzlebot',
            'odom_frame': 'odom',
            'base_frame': 'base_footprint',
            'pose_topic': '/world/almacen_racks/dynamic_pose/info',
        }],
        condition=IfCondition(PythonExpression([
            "'", world_name, "' == 'almacen' and '", slam_en, "' == 'true' and ",
            "'", mode, "' == 'mapping' and '", odom_source, "' == 'ground_truth'",
        ])),
    )

    # ── Kalman filter — Estrategia A y B ─────────────────────────────────────
    # Activo cuando world:=real_arena AND kalman:=true.
    #
    # Estrategia A (aruco_oracle:=false):
    #   init_from_aruco=false → arranca desde (0.30, 0.30, π/2) sin esperar ArUco.
    #   Solo predice con ruedas. Debe comportarse igual que dead_reckoning_direct
    #   si el EKF es correcto — sirve para validar la implementación C++.
    #
    # Estrategia B (aruco_oracle:=true):
    #   init_from_aruco=true → espera el primer /aruco/pose del oracle para
    #   inicializar el estado. Luego funde predicción de ruedas + corrección oracle.
    #
    # Pose inicial: coincide con spawn_arena (0.30, 0.30, π/2 = mirando norte).
    _kalman_init_from_aruco = ParameterValue(
        PythonExpression(["'", oracle_en, "' == 'true'"]),
        value_type=bool,
    )
    kalman_arena = Node(
        package='puzzlebot_localization',
        executable='kalman_filter_node',
        name='kalman_filter_node',
        output='screen',
        parameters=[kalman_cfg, {
            'use_sim_time': True,
            'initial_x':     0.30,
            'initial_y':     0.30,
            'initial_theta': 1.5708,   # π/2 — mirando norte, igual que spawn
            'init_from_aruco': _kalman_init_from_aruco,
        }],
        condition=IfCondition(PythonExpression([
            "'", world_name, "' == 'real_arena' and '", kalman_en, "' == 'true' and ",
            "('", mode, "' != 'mapping' or '", odom_source, "' == 'dead_reckoning')",
        ])),
    )

    # ── ArUco Oracle — Estrategia B ───────────────────────────────────────────
    # Activo cuando world:=real_arena AND kalman:=true AND aruco_oracle:=true.
    # Lee pose real del robot desde Gazebo y publica /aruco/pose sintético con
    # ruido gaussiano configurable — simula lo que vería aruco_node en el físico.
    #
    # Parámetros de ruido ajustables para simular distintas calidades de cámara:
    #   sigma_lateral=0.015 m  (ruido en eje perpendicular al marcador)
    #   sigma_depth_base=0.020 m (ruido en eje de profundidad, crece con oblicuidad)
    #   sigma_yaw=0.015 rad
    #   detection_prob=1.0 (bajar a 0.7-0.8 para simular pérdidas de detección)
    aruco_oracle_arena = Node(
        package='puzzlebot_localization',
        executable='aruco_oracle',
        name='aruco_oracle',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'pose_topic':          '/world/real_arena/dynamic_pose/info',
            'aruco_map_file':      aruco_map_yaml,
            'max_detection_dist':  3.0,   # ampliado: centro arena >1.9 m de todos los markers
            'max_incidence_deg':   75.0,
            'sigma_lateral':       0.015,
            'sigma_depth_base':    0.020,
            'sigma_yaw':           0.015,
            'publish_rate_hz':     10.0,
            'detection_prob':      1.0,
            # Hint de spawn — coincide con spawn_arena (0.30, 0.30).
            'robot_spawn_x':       0.30,
            'robot_spawn_y':       0.30,
        }],
        condition=IfCondition(PythonExpression([
            "'", world_name, "' == 'real_arena' and '",
            kalman_en, "' == 'true' and '", oracle_en, "' == 'true'",
        ])),
    )

    # ── Kalman filter — almacén ──────────────────────────────────────────────
    # Activo cuando world:=almacen AND kalman:=true.
    # Pose inicial: coincide con spawn_almacen (0.79, 0.70, -π/2 = mirando sur,
    # de frente al marker ID 21).
    #
    # init_from_aruco: true siempre que kalman:=true, sin importar aruco_oracle.
    # aruco_node (visión real por cámara) corre incondicionalmente y publica
    # /aruco/pose — es la fuente por defecto. aruco_oracle:=true la reemplaza
    # por ground-truth sintético (debug del EKF sin depender de detección real).
    # En ambos casos el EKF debe esperar la primera detección, no arrancar en
    # una pose fija — así se comporta igual que el robot real.
    _kalman_init_from_aruco_almacen = True
    kalman_almacen = Node(
        package='puzzlebot_localization',
        executable='kalman_filter_node',
        name='kalman_filter_node',
        output='screen',
        parameters=[kalman_cfg, {
            'use_sim_time': True,
            'initial_x':     0.79,
            'initial_y':     0.70,
            'initial_theta': -1.5708,   # -π/2 — mirando sur, igual que spawn_almacen
            'init_from_aruco': _kalman_init_from_aruco_almacen,
        }],
        condition=IfCondition(PythonExpression([
            "'", world_name, "' == 'almacen' and '", kalman_en, "' == 'true' and ",
            "('", mode, "' != 'mapping' or '", odom_source, "' == 'dead_reckoning')",
        ])),
    )

    # ── ArUco Oracle — almacén ───────────────────────────────────────────────
    # Activo cuando world:=almacen AND kalman:=true AND aruco_oracle:=true.
    aruco_oracle_almacen = Node(
        package='puzzlebot_localization',
        executable='aruco_oracle',
        name='aruco_oracle',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'pose_topic':          '/world/almacen_racks/dynamic_pose/info',
            'aruco_map_file':      aruco_map_yaml,
            'max_detection_dist':  3.0,
            'max_incidence_deg':   75.0,
            'sigma_lateral':       0.015,
            'sigma_depth_base':    0.020,
            'sigma_yaw':           0.015,
            'publish_rate_hz':     10.0,
            'detection_prob':      1.0,
            # Hint de spawn — coincide con spawn_almacen (0.79, 0.70).
            # El robot no es poses[0] en el almacén; el oracle lo identifica
            # por cercanía a este punto en el primer frame.
            'robot_spawn_x':       0.79,
            'robot_spawn_y':       0.70,
        }],
        condition=IfCondition(PythonExpression([
            "'", world_name, "' == 'almacen' and '",
            kalman_en, "' == 'true' and '", oracle_en, "' == 'true'",
        ])),
    )

    lidar_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='lidar_frame_fix',
        arguments=['0', '0', '0', '0', '0', '0',
                   'lidar_link', 'puzzlebot/base_footprint/lidar'],
        output='screen',
    )

    # ── Static TF: camera_link ───────────────────────────────────────────
    # Fortress scopes the camera frame as 'puzzlebot/base_footprint/camera'
    # internally. Publish a zero-offset alias so aruco_node can find it.
    camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_frame_fix',
        arguments=['0', '0', '0', '0', '0', '0',
                   'camera_link', 'puzzlebot/base_footprint/camera'],
        output='screen',
    )

    slam_mapping = Node(
        package='puzzlebot_slam',
        executable='slam_node',
        name='slam_node',
        output='screen',
        parameters=[slam_cfg, {'use_sim_time': True}],
        condition=IfCondition(PythonExpression([
            "'", slam_en, "' == 'true' and '", mode, "' == 'mapping'"
        ])),
    )

    mcl = Node(
        package='puzzlebot_slam',
        executable='mcl',
        name='mcl',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'map_path':       map_file,
            'map_resolution': 0.05,
            'map_origin_x':  -5.54,
            'map_origin_y':  -8.10,
            'num_particles':  500,
            'top_k':          150,
            'noise_xy':       0.05,
            'noise_theta':    0.05,
            'score_rays':     36,
        }],
        condition=IfCondition(PythonExpression([
            "'", world_name, "' == 'maze' and '",
            slam_en, "' == 'true' and '", mode, "' == 'mcl'"
        ])),
    )

    # ── Perception nodes ─────────────────────────────────────────────────
    # Visión real por cámara. Publica /aruco/pose — mismo topic que
    # aruco_oracle_almacen. NO deben correr juntos en almacén (doble
    # publisher → EKF recibe mezcla caótica de detección real + sintética):
    # activo salvo cuando world:=almacen Y el oracle sintético está encendido.
    aruco_node = Node(
        package='puzzlebot_perception',
        executable='aruco_node',
        name='aruco_node',
        parameters=[{
            'use_sim_time':   True,
            'marker_map_file': aruco_map_yaml,
            'marker_length':   0.09,
            # Default del nodo es '/camera/image/compressed' (robot físico).
            # En Gazebo el bridge publica '/camera/image_raw' (sensor_msgs/Image
            # sin comprimir) — sin este override aruco_node nunca recibe imagen.
            'image_topic':      '/camera/image_raw',
            'camera_info_file': camera_calib_yaml,
            'extrinsics_file':  camera_extr_yaml,
            # Las texturas en puzzlebot_description/textures/arucos/ son del
            # diccionario ARUCO_ORIGINAL, no DICT_4X4_50 (default del nodo).
            # Con el diccionario equivocado el detector encuentra el contorno
            # del marker pero falla al decodificar los bits → 0 detecciones.
            'dictionary':       'DICT_ARUCO_ORIGINAL',
        }],
        output='screen',
        condition=IfCondition(PythonExpression([
            "not ('", world_name, "' == 'almacen' and '", oracle_en, "' == 'true')"
        ])),
    )

    # Fallback genérico — solo para worlds sin variante propia (flat_plane, maze).
    # real_arena y almacen ya tienen su kalman_arena/kalman_almacen dedicado;
    # este NO debe correr en paralelo con esos o se duplica la TF odom→base_footprint.
    kalman_node = Node(
        package='puzzlebot_localization',
        executable='kalman_filter_node',
        name='kalman_filter_node',
        parameters=[{'use_sim_time': True}],
        output='screen',
        condition=IfCondition(PythonExpression([
            "'", world_name, "' != 'real_arena' and '", world_name, "' != 'almacen'"
        ])),
    )

    rviz_mapping_node = TimerAction(
        period=15.0,
        actions=[Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_mapping],
            parameters=[{'use_sim_time': True}],
            output='screen',
        )],
        condition=IfCondition(PythonExpression([
            "'", rviz_en, "' == 'true' and '", mode, "' == 'mapping'"
        ])),
    )

    # ── Navegación autónoma A* (opcional, navigation:=true) ──────────────
    # Incluye: path_planner_node + steering_controller + obstacle_avoidance
    # Conecta: /map → A* → /planned_path → steering → /cmd_vel_in → avoidance → /cmd_vel
    nav_launch_file = os.path.join(bringup_pkg, 'launch', 'navigation.launch.py')
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav_launch_file),
        launch_arguments={
            'use_sim_time':   'true',
            # 'cmd_vel_topic':  '/model/puzzlebot/cmd_vel',  # DiffDrive de Fortress
            'remapping': 'true',
        }.items(),
        condition=IfCondition(nav_en),
    )

    # ── scan_restamper (requerido para navigation en Gazebo) ─────────────
    # Los nodos de navegación (bug_navigation, obstacle_avoidance) suscriben /scan_stamped.
    # En Gazebo el scan llega en /scan directamente. Este nodo reempaqueta el timestamp
    # y cambia el frame_id a lidar_link para que la cadena TF sea válida.
    scan_restamper = Node(
        package='puzzlebot_localization',
        executable='scan_restamper',
        name='scan_restamper',
        output='screen',
        parameters=[{
            'use_sim_time':    True,
            'input_topic':     '/scan',
            'target_frame':    'lidar_link',
            'invert_angles':   False,
            'angle_offset_rad': 0.0,
        }],
        condition=IfCondition(nav_en),
    )

    # ── Web dashboard bridge (opcional, web_bridge:=true) ────────────────
    # Expone ws://0.0.0.0:8000/ws para el dashboard React.
    # cmd_vel_out_topic apunta al tópico del DiffDrive de Gazebo para que el teleop funcione.
    # web_bridge = Node(
    #     package='puzzlebot_web_bridge',
    #     executable='bridge_node',
    #     name='puzzlebot_web_bridge',
    #     output='screen',
    #     parameters=[{
    #         'use_sim_time':       True,
    #         'cmd_vel_out_topic':  '/model/puzzlebot/cmd_vel',
    #         'artifact_dir':       '',
    #     }],
    #     condition=IfCondition(web_bridge_en),
    # )

    return LaunchDescription([
        set_resource_path,
        # Argumentos
        arg_world, arg_gui, arg_slam, arg_rviz, arg_mode, arg_odom_source,
        arg_kalman, arg_aruco_oracle, arg_navigation,
        gz_sim,
        rsp,
        # Bridges (uno activo según world)
        bridge_arena,
        bridge_almacen,
        # Joint relays (uno activo según world)
        joint_relay_arena,
        joint_relay_almacen,
        # TF estáticos (siempre activos)
        lidar_tf,
        camera_tf,
        # Spawns del robot (uno activo según world)
        spawn_arena,
        spawn_almacen,
        # Odometría de ruedas flat/maze (sin Kalman)
        # Odometría real_arena — variante directa (kalman:=false)
        wheel_odom_arena_direct,
        # Odometría real_arena — variante raw para Kalman (kalman:=true)
        wheel_odom_arena_raw,
        # Ground truth odom (uno activo según world+mode+odom_source)
        ground_truth_arena,
        # Kalman filter (real_arena + kalman:=true) — Estrategia A y B
        kalman_arena,
        # ArUco Oracle (real_arena + kalman:=true + aruco_oracle:=true) — Estrategia B
        aruco_oracle_arena,
        # Odometría de ruedas — almacén (misma lógica que real_arena)
        wheel_odom_almacen_direct,
        wheel_odom_almacen_raw,
        ground_truth_almacen,
        kalman_almacen,
        aruco_oracle_almacen,
        # SLAM / localización
        slam_mapping,
        mcl,
        # Percepción sim (skeleton — sin activar por defecto)
        aruco_node,
        kalman_node,
        # RViz
        rviz_mapping_node,
        # Navegación autónoma A* (navigation:=true)
        navigation,
        # scan_restamper: adapta /scan → /scan_stamped para nodos de navegación en Gazebo
        scan_restamper,
        # Web dashboard bridge (web_bridge:=true)
        # web_bridge,
    ])