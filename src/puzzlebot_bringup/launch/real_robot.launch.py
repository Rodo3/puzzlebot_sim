"""
Real-robot launch — Puzzlebot differential drive (Jetson Orin).

Corre completamente en el PC del operador. La Jetson solo publica sensores;
todo el cómputo (odometría, SLAM, percepción, control) ocurre en el PC.

════════════════════════════════════════════════════════════════
 ARGUMENTOS
════════════════════════════════════════════════════════════════

  slam        [true]   slam_node: construye /map en tiempo real
  mcl         [false]  MCL: localización con mapa PNG guardado
                       (usa slam:=false mcl:=true en sesión 2)
  kalman      [false]  Kalman EKF entre encoders y ArUco
                         false → odometry_node publica /odom directamente
                         true  → odometry_node publica /odom_raw sin TF;
                                 kalman_filter_node produce /odom + TF odom→base_footprint
                       Estrategia A: kalman:=true aruco:=false (solo predicción de ruedas)
                       Estrategia B: kalman:=true aruco:=true  (fusión encoders + ArUco)
                       NOTA: cuando kalman:=true, aruco_map_odom se desactiva
                             automáticamente (el Kalman ya incorpora la corrección ArUco).
  avoidance   [false]  obstacle_avoidance_node
  aruco       [true]   aruco_node (detección visual de marcadores)
                         Con kalman:=false → también activa aruco_map_odom (map→odom TF)
                         Con kalman:=true  → /aruco/pose va directo al Kalman
  viewer      [false]  image_viewer_node con corrección de distorsión
  rviz        [true]   RViz2
  lidar_topic [/Lidar] /Lidar (micro-ROS) o /scan (sllidar directo)
  invert_lidar [false] Invierte izquierda/derecha si el LiDAR está espejeado
  lidar_yaw_offset [3.14159265359] Rota el scan; π corrige frente/atrás invertido

════════════════════════════════════════════════════════════════
 COMBINACIONES TÍPICAS
════════════════════════════════════════════════════════════════

  # Sesión 1 — Mapeo clásico con corrección ArUco via aruco_map_odom:
  ros2 launch puzzlebot_bringup real_robot.launch.py slam:=true aruco:=true

  # Sesión 1 — Mapeo con Kalman EKF + ArUco (Estrategia B):
  ros2 launch puzzlebot_bringup real_robot.launch.py slam:=true kalman:=true aruco:=true

  # Sesión 2 — Localización MCL:
  ros2 launch puzzlebot_bringup real_robot.launch.py slam:=false mcl:=true aruco:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    bringup_pkg = get_package_share_directory('puzzlebot_bringup')
    desc_pkg    = get_package_share_directory('puzzlebot_description')

    # ── Rutas de configuración ────────────────────────────────────────────
    controller_cfg  = os.path.join(bringup_pkg, 'config', 'controller_params.yaml')
    robot_cfg       = os.path.join(bringup_pkg, 'config', 'robot_params.yaml')
    slam_cfg        = os.path.join(bringup_pkg, 'config', 'slam_params.yaml')
    mcl_cfg         = os.path.join(bringup_pkg, 'config', 'mcl_params.yaml')
    kalman_cfg      = os.path.join(bringup_pkg, 'config', 'kalman_params.yaml')
    calib_yaml      = os.path.join(bringup_pkg, 'config', 'camera_calibration.yaml')
    extrinsics_yaml = os.path.join(bringup_pkg, 'config', 'camera_extrinsics.yaml')
    aruco_map_yaml  = os.path.join(bringup_pkg, 'config', 'aruco_map.yaml')
    urdf_file       = os.path.join(desc_pkg,    'urdf',   'puzzlebot_gz.urdf')
    rviz_cfg        = os.path.join(desc_pkg,    'rviz',   'puzzlebot_rviz.rviz')

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # ── Argumentos del launch ─────────────────────────────────────────────
    arg_slam        = DeclareLaunchArgument('slam',        default_value='true',
                          description='Enable slam_node (mapeo). Usa slam:=false con mcl:=true.')
    arg_mcl         = DeclareLaunchArgument('mcl',         default_value='false',
                          description='Enable MCL (localización con mapa PNG). Usa slam:=false mcl:=true.')
    arg_kalman      = DeclareLaunchArgument('kalman',      default_value='false',
                          description='Kalman EKF: true → odom_raw→kalman→odom+TF. '
                                       'Estrategia A: kalman:=true aruco:=false. '
                                       'Estrategia B: kalman:=true aruco:=true')
    arg_avoidance   = DeclareLaunchArgument('avoidance',   default_value='false',
                          description='Enable obstacle_avoidance_node')
    arg_aruco       = DeclareLaunchArgument('aruco',       default_value='true',
                          description='Enable aruco_node. Con kalman:=false activa también aruco_map_odom.')
    arg_viewer      = DeclareLaunchArgument('viewer',      default_value='false',
                          description='Enable image_viewer_node con corrección de distorsión')
    arg_rviz        = DeclareLaunchArgument('rviz',        default_value='true',
                          description='Open RViz2')
    arg_lidar_topic = DeclareLaunchArgument('lidar_topic', default_value='/Lidar',
                          description='Tópico LiDAR: /Lidar (micro-ROS) o /scan (sllidar directo)')
    arg_invert_lidar = DeclareLaunchArgument('invert_lidar', default_value='false',
                          description='Invert LaserScan angles when left/right are mirrored')
    arg_lidar_yaw_offset = DeclareLaunchArgument('lidar_yaw_offset', default_value='3.14159265359',
                          description='LaserScan angular offset in radians; pi flips front/back')
    arg_navigation = DeclareLaunchArgument('navigation', default_value='false',
                          description='Navegación autónoma A* + steering_controller + obstacle_avoidance. '
                                      'Requiere /map disponible. Enviar /goal_pose por RViz (G → 2D Nav Goal).')

    slam_en      = LaunchConfiguration('slam')
    mcl_en       = LaunchConfiguration('mcl')
    kalman_en    = LaunchConfiguration('kalman')
    avoidance_en = LaunchConfiguration('avoidance')
    nav_en       = LaunchConfiguration('navigation')
    aruco_en     = LaunchConfiguration('aruco')
    viewer_en    = LaunchConfiguration('viewer')
    rviz_en      = LaunchConfiguration('rviz')
    lidar_topic  = LaunchConfiguration('lidar_topic')
    invert_lidar = LaunchConfiguration('invert_lidar')
    lidar_yaw_offset = LaunchConfiguration('lidar_yaw_offset')

    # slam_node publica map→odom cuando NO hay nadie más dueño de ese TF.
    # Con kalman:=true, el Kalman absorbe la corrección ArUco internamente
    # en el TF odom→base_footprint; el slam_node aún puede publicar map→odom
    # pero no necesita scan matching (solo acumula el mapa).
    slam_publishes_map_odom = ParameterValue(
        PythonExpression([
            "'", aruco_en, "' == 'false' and '", mcl_en, "' == 'false' and '",
            kalman_en, "' == 'false'"
        ]),
        value_type=bool,
    )

    # ── 1. Robot State Publisher ──────────────────────────────────────────
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': False,
        }],
        output='screen',
    )

    # ── 2. TF estáticos ───────────────────────────────────────────────────
    # camera_optical_tf: camera_link (ROS: x frente, y izquierda, z arriba)
    # → camera_optical_frame (OpenCV: x derecha, y abajo, z frente).
    # aruco_node usa esta misma conversión internamente para solvePnP.
    camera_optical_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_optical_tf',
        arguments=['0.0', '0.0', '0.0',
                   '-1.57079632679', '0.0', '-1.57079632679',
                   'camera_link', 'camera_optical_frame'],
        output='screen',
    )

    # laser_tf: lidar_link → laser
    #   El driver sllidar_ros2 publica scans con frame_id='laser' por default.
    #   Este TF conecta ese frame con lidar_link (que viene del URDF).
    #   scan_restamper reescribe el frame_id a 'lidar_link', pero el TF tree
    #   debe contener 'laser' de todas formas para que tf2 no reporte errores.
    #   Si el LiDAR tiene un offset rotacional (montado girado), ajustar yaw aquí.
    laser_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='laser_tf',
        arguments=['0.0', '0.0', '0.0',
                   '0.0', '0.0', '0.0',
                   'lidar_link', 'laser'],
        output='screen',
    )

    # ── 3a. Odometría de ruedas — modo directo (kalman:=false) ───────────────
    # Dueña del frame odom→base_footprint. Publica /odom con TF.
    # Comportamiento clásico: las correcciones globales viven en map→odom
    # (via aruco_map_odom o slam_node).
    odometry_direct = Node(
        package='puzzlebot_localization',
        executable='odometry_node',
        name='odometry_node',
        output='screen',
        parameters=[robot_cfg, {
            'use_sim_time': False,
            'input_source': 'encoders',
            'odom_topic':   '/odom',
            'publish_tf':   True,
        }],
        remappings=[
            ('velocity_enc_r', '/VelocityEncR'),
            ('velocity_enc_l', '/VelocityEncL'),
        ],
        condition=IfCondition(PythonExpression(["'", kalman_en, "' == 'false'"])),
    )

    # ── 3b. Odometría de ruedas — modo raw para Kalman (kalman:=true) ────────
    # Publica /odom_raw SIN TF. El kalman_filter_node es el dueño de
    # odom→base_footprint y fusiona /odom_raw + /aruco/pose.
    odometry_raw = Node(
        package='puzzlebot_localization',
        executable='odometry_node',
        name='odometry_node',
        output='screen',
        parameters=[robot_cfg, {
            'use_sim_time': False,
            'input_source': 'encoders',
            'odom_topic':   '/odom_raw',
            'publish_tf':   False,
        }],
        remappings=[
            ('velocity_enc_r', '/VelocityEncR'),
            ('velocity_enc_l', '/VelocityEncL'),
        ],
        condition=IfCondition(PythonExpression(["'", kalman_en, "' == 'true'"])),
    )

    # ── 3c. Kalman filter EKF (kalman:=true) ──────────────────────────────────
    # Fusiona /odom_raw (predicción) + /aruco/pose (corrección, si aruco:=true).
    # Publica /odom + TF odom→base_footprint.
    # Con init_from_aruco:true espera el primer ArUco para inicializar el estado.
    # Con aruco:=false solo predice (Estrategia A — debug de encoders).
    kalman = Node(
        package='puzzlebot_localization',
        executable='kalman_filter_node',
        name='kalman_filter_node',
        output='screen',
        parameters=[kalman_cfg, {'use_sim_time': False}],
        condition=IfCondition(PythonExpression(["'", kalman_en, "' == 'true'"])),
    )

    # ── 4. ArUco map→odom correction ─────────────────────────────────────
    # Convierte /aruco/pose (pose absoluta en map) + /odom (ruedas) en una
    # corrección global map→odom via TF. Solo activo cuando:
    #   • aruco:=true  — hay detecciones ArUco disponibles
    #   • mcl:=false   — MCL no está compitiendo por el TF map→odom
    #   • kalman:=false — el Kalman incorpora la corrección ArUco internamente
    #                     en odom→base_footprint; aruco_map_odom sería redundante
    #                     y causaría doble corrección.
    aruco_map_odom = Node(
        package='puzzlebot_localization',
        executable='aruco_map_odom',
        name='aruco_map_odom',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'odom_topic': '/odom',
            'aruco_pose_topic': '/aruco/pose',
            'map_to_odom_topic': '/map_to_odom',
            'correction_alpha': 0.35,
            'map_min_x': 0.0,
            'map_max_x': 3.76,
            'map_min_y': 0.0,
            'map_max_y': 4.86,
            'map_bounds_margin': 0.25,
        }],
        condition=IfCondition(PythonExpression([
            "'", aruco_en, "' == 'true' and '",
            mcl_en, "' == 'false' and '",
            kalman_en, "' == 'false'",
        ])),
    )

    # ── 5. ArUco pose estimation ──────────────────────────────────────────
    aruco = Node(
        package='puzzlebot_perception',
        executable='aruco_node',
        name='aruco_node',
        output='screen',
        parameters=[{
            'use_sim_time':      False,
            'image_topic':       '/camera/image/compressed',
            'camera_info_file':  calib_yaml,
            'extrinsics_file':   extrinsics_yaml,
            'marker_map_file':   aruco_map_yaml,
            'marker_length':     0.10,
            'max_detection_distance': 1.8,
            'max_incidence_angle_deg': 65.0,
            'map_min_x': 0.0,
            'map_max_x': 3.76,
            'map_min_y': 0.0,
            'map_max_y': 4.86,
            'map_bounds_margin': 0.25,
        }],
        condition=IfCondition(aruco_en),
    )

    # ── 6. Camera viewer ─────────────────────────────────────────────────
    viewer = Node(
        package='puzzlebot_perception',
        executable='image_viewer_node',
        name='image_viewer_node',
        output='screen',
        parameters=[{
            'use_sim_time':  False,
            'topic':         '/camera/image/compressed',
            'rectify':       True,
            'calib_yaml':    calib_yaml,
            'window_title':  'Puzzlebot — Camara Rectificada',
        }],
        additional_env={'QT_QPA_PLATFORM': 'xcb'},
        condition=IfCondition(viewer_en),
    )

    # ── 7. Scan restamper ─────────────────────────────────────────────────
    scan_restamper = Node(
        package='puzzlebot_localization',
        executable='scan_restamper',
        name='scan_restamper',
        output='screen',
        parameters=[{
            'input_topic':  lidar_topic,
            'target_frame': 'lidar_link',
            'invert_angles': invert_lidar,
            'angle_offset_rad': lidar_yaw_offset,
        }],
    )

    # ── 8a. SLAM mapping (sesión 1) ───────────────────────────────────────
    # scan_matching_enabled: false en slam_params.yaml para mapeo limpio.
    # El nodo construye /map usando /odom continuo y la corrección externa
    # /map_to_odom cuando ArUco está activo. No publica map→odom en real_robot
    # cuando aruco_map_odom o MCL son los dueños de esa transformada.
    # scan_match_updates_map_odom: false cuando ArUco o MCL son dueños de map→odom.
    # Evita que el scan matcher sobreescriba la corrección absoluta de ArUco.
    # scan_match_updates_map_odom: el slam_node solo sobreescribe map→odom
    # con su scan matcher cuando nadie más es dueño de ese TF.
    # Con aruco_map_odom, MCL o Kalman activos, el scan matcher solo
    # construye el mapa pero no publica el TF (evita sobrescribir correcciones).
    slam_match_updates_odom = ParameterValue(
        PythonExpression([
            "'", aruco_en, "' == 'false' and '",
            mcl_en, "' == 'false' and '",
            kalman_en, "' == 'false'"
        ]),
        value_type=bool,
    )

    slam = Node(
        package='puzzlebot_slam',
        executable='slam_node',
        name='slam_node',
        output='screen',
        parameters=[slam_cfg, {
            'use_sim_time': False,
            'publish_map_odom_tf': slam_publishes_map_odom,
            'scan_match_updates_map_odom': slam_match_updates_odom,
        }],
        remappings=[('/scan', '/scan_stamped')],
        condition=IfCondition(slam_en),
    )

    # ── 8b. MCL localización (sesión 2) ──────────────────────────────────
    # Requiere mapa PNG guardado con map_saver + convertido.
    # Publica map→odom TF a partir de partículas contra el mapa guardado.
    # IMPORTANTE: correr con slam:=false mcl:=true para que no haya
    # conflicto de dos nodos publicando el TF map→odom.
    mcl = Node(
        package='puzzlebot_slam',
        executable='mcl',
        name='mcl',
        output='screen',
        parameters=[mcl_cfg, {'use_sim_time': False}],
        remappings=[('/scan', '/scan_stamped'),   # usa el scan re-sellado
                    ('/odom', '/odom')],           # odom continuo de ruedas
        condition=IfCondition(mcl_en),
    )

    # ── 9. Obstacle Avoidance ─────────────────────────────────────────────
    obstacle_avoidance = Node(
        package='puzzlebot_planning',
        executable='obstacle_avoidance_node',
        name='obstacle_avoidance_node',
        output='screen',
        parameters=[controller_cfg, {'use_sim_time': False}],
        remappings=[('/scan', '/scan_stamped')],
        condition=IfCondition(avoidance_en),
    )

    # ── 10. RViz ──────────────────────────────────────────────────────────
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_cfg],
        parameters=[{'use_sim_time': False}],
        remappings=[('/scan', '/scan_stamped')],
        condition=IfCondition(rviz_en),
        output='screen',
    )

    # ── Navegación autónoma (navigation:=true) ────────────────────────────
    # Incluye path_planner_node (A*) + steering_controller + obstacle_avoidance.
    # Si navigation:=true y avoidance:=true simultáneamente, se lanzan dos
    # instancias de obstacle_avoidance. Usar uno u otro, no ambos.
    nav_launch_file = os.path.join(bringup_pkg, 'launch', 'navigation.launch.py')
    navigation_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav_launch_file),
        launch_arguments={
            'use_sim_time':  'false',
            'cmd_vel_topic': '/cmd_vel',  # micro-ROS bridge escucha /cmd_vel
        }.items(),
        condition=IfCondition(nav_en),
    )

    return LaunchDescription([
        # Argumentos
        arg_slam,
        arg_mcl,
        arg_kalman,
        arg_avoidance,
        arg_navigation,
        arg_aruco,
        arg_viewer,
        arg_rviz,
        arg_lidar_topic,
        arg_invert_lidar,
        arg_lidar_yaw_offset,
        # Infraestructura base (siempre activos)
        rsp,
        camera_optical_tf,
        laser_tf,
        scan_restamper,
        # Odometría — modo directo (kalman:=false) o raw+EKF (kalman:=true)
        odometry_direct,
        odometry_raw,
        kalman,
        # Corrección map→odom via ArUco (solo kalman:=false)
        aruco_map_odom,
        # Percepción (condicional)
        aruco,
        viewer,
        # Navegación / SLAM
        slam,
        mcl,
        obstacle_avoidance,
        # Navegación autónoma A* completa (navigation:=true)
        navigation_stack,
        # Visualización
        rviz,
    ])
