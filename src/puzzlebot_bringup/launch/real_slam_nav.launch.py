"""
real_slam_nav.launch.py — Robot real: SLAM con mapa conocido + Navegación A* + Bug2

Análogo en hardware real de `gz_slam_nav.launch.py` (Gazebo). Aplica EL MISMO
algoritmo que usa la simulación, pero sobre el Puzzlebot físico:

  • Odometría de encoders  → /odom_raw (predicción EKF, sin TF)
  • Kalman EKF             → fusiona /odom_raw + /aruco/pose (ArUco real) → /odom + TF odom→base_footprint
  • SLAM con mapa conocido → arranca con el PNG previo y sigue mapeando en vivo;
                             el LiDAR integra obstáculos físicos al /map → A* replantea
  • Navegación A*          → path_planner_node (replanea sobre /map vivo)
  • Bug2                   → bug_navigation_node (wall following reactivo, obstacle_manager:=legacy)
  • obstacle_avoidance     → capa final de parada de emergencia → /cmd_vel

Diferencias con la simulación
─────────────────────────────
  • aruco_oracle (ground truth de Gazebo) → reemplazado por aruco_node (cámara real)
  • bridge / spawn / dynamic_obstacle_spawner de Gazebo → no aplican
  • use_sim_time = False en todos los nodos
  • LiDAR real vía scan_restamper (/scan o /Lidar → /scan_stamped)

Ownership de TF (idéntico a la simulación)
──────────────────────────────────────────
  odom→base_footprint : kalman_filter_node (fusiona encoders + ArUco)
  map→odom            : slam_node (scan matcher, scan_match_updates_map_odom=True)
  → NO se usa aruco_map_odom: el ArUco entra al EKF, igual que en gz_slam_nav.

Flujo de uso
────────────

  # En la Jetson (publica sensores): micro-ROS / sllidar / cámara
  # En el PC del operador:
  ros2 launch puzzlebot_bringup real_slam_nav.launch.py lidar_topic:=/scan

  En RViz:
    - G → "2D Nav Goal" → click en el destino
    - La ruta verde aparece sobre el mapa conocido
    - El LiDAR detecta obstáculos físicos → /map se actualiza → A* replantea
    - Bug2 rodea reactivamente si un obstáculo bloquea la ruta

ARGUMENTOS
──────────
  initial_map      [slam_map_20260529_235356.png] PNG del mapa precargado para inicializar SLAM
  navigation       [true]      Lanza A* + steering + bug2 + obstacle_avoidance
  obstacle_manager [legacy]    legacy (Bug2) | dynamic | none
  aruco            [true]      aruco_node (detección visual real → corrige el EKF)
  rviz             [true]      Lanzar RViz2
  lidar_topic      [/scan]     /scan (sllidar directo) o /Lidar (micro-ROS)
  invert_lidar     [false]     Invierte izquierda/derecha si el LiDAR está espejeado
  lidar_yaw_offset [3.14159265359] Rota el scan; pi corrige frente/atrás invertido
"""
import os
import glob

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    bringup_pkg = get_package_share_directory('puzzlebot_bringup')
    desc_pkg    = get_package_share_directory('puzzlebot_description')

    # ── Rutas de configuración ────────────────────────────────────────────
    robot_cfg       = os.path.join(bringup_pkg, 'config', 'robot_params.yaml')
    slam_cfg        = os.path.join(bringup_pkg, 'config', 'slam_params.yaml')
    kalman_cfg      = os.path.join(bringup_pkg, 'config', 'kalman_params.yaml')
    calib_yaml      = os.path.join(bringup_pkg, 'config', 'camera_calibration.yaml')
    extrinsics_yaml = os.path.join(bringup_pkg, 'config', 'camera_extrinsics.yaml')
    aruco_map_yaml  = os.path.join(bringup_pkg, 'config', 'aruco_map.yaml')
    urdf_file       = os.path.join(desc_pkg, 'urdf', 'puzzlebot_gz.urdf')
    rviz_file       = os.path.join(desc_pkg, 'rviz', 'mcl_rviz.rviz')

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # Workspace root: src/puzzlebot_bringup/share/puzzlebot_bringup → ../../../../
    ws_root = os.path.abspath(os.path.join(bringup_pkg, '..', '..', '..', '..'))
    # Mapa precargado fijo: slam_map_20260529_235356.png (el del arena conocido).
    # Si no existe, cae al más reciente; si tampoco hay, mapea desde cero.
    preloaded_map = os.path.join(ws_root, 'slam_map_20260529_235356.png')
    if os.path.exists(preloaded_map):
        default_map_png = preloaded_map
    else:
        _maps = sorted(glob.glob(os.path.join(ws_root, 'slam_map_*.png')))
        default_map_png = _maps[-1] if _maps else ''

    # ── Argumentos ────────────────────────────────────────────────────────
    arg_initial_map = DeclareLaunchArgument(
        'initial_map', default_value=default_map_png,
        description='PNG del mapa previo para inicializar SLAM. Vacío = mapear desde cero.')
    arg_nav = DeclareLaunchArgument(
        'navigation', default_value='true',
        description='Lanzar A* + steering + bug2 + obstacle_avoidance')
    arg_obs_mgr = DeclareLaunchArgument(
        'obstacle_manager', default_value='legacy',
        description='legacy (Bug2) | dynamic | none')
    arg_aruco = DeclareLaunchArgument(
        'aruco', default_value='true',
        description='aruco_node: detección visual real que corrige el EKF')
    arg_rviz = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Lanzar RViz2')
    arg_lidar_topic = DeclareLaunchArgument(
        'lidar_topic', default_value='/scan',
        description='Tópico LiDAR: /scan (sllidar directo) o /Lidar (micro-ROS)')
    arg_invert_lidar = DeclareLaunchArgument(
        'invert_lidar', default_value='false',
        description='Invierte ángulos del LaserScan si izquierda/derecha están espejeados')
    arg_lidar_yaw_offset = DeclareLaunchArgument(
        'lidar_yaw_offset', default_value='3.14159265359',
        description='Offset angular del LaserScan en radianes; pi invierte frente/atrás')

    initial_map      = LaunchConfiguration('initial_map')
    nav_en           = LaunchConfiguration('navigation')
    obs_mgr          = LaunchConfiguration('obstacle_manager')
    aruco_en         = LaunchConfiguration('aruco')
    rviz_en          = LaunchConfiguration('rviz')
    lidar_topic      = LaunchConfiguration('lidar_topic')
    invert_lidar     = LaunchConfiguration('invert_lidar')
    lidar_yaw_offset = LaunchConfiguration('lidar_yaw_offset')

    # ── 1. Robot State Publisher ──────────────────────────────────────────
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': False}],
        output='screen',
    )

    # ── 2. TF estáticos ───────────────────────────────────────────────────
    # camera_link → camera_optical_frame (convención OpenCV para solvePnP).
    camera_optical_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_optical_tf',
        arguments=['0.0', '0.0', '0.0',
                   '-1.57079632679', '0.0', '-1.57079632679',
                   'camera_link', 'camera_optical_frame'],
        output='screen',
    )

    # lidar_link → laser (frame_id por default del driver sllidar_ros2).
    laser_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='laser_tf',
        arguments=['0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
                   'lidar_link', 'laser'],
        output='screen',
    )

    # ── 3. Scan restamper — LiDAR real → /scan_stamped (frame lidar_link) ──
    scan_restamper = Node(
        package='puzzlebot_localization',
        executable='scan_restamper',
        name='scan_restamper',
        output='screen',
        parameters=[{
            'input_topic':      lidar_topic,
            'target_frame':     'lidar_link',
            'invert_angles':    invert_lidar,
            'angle_offset_rad': lidar_yaw_offset,
        }],
    )

    # ── 4. Odometría de encoders → /odom_raw (predicción EKF, sin TF) ─────
    # publish_tf=False: el kalman_filter_node es el dueño de odom→base_footprint.
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
    )

    # ── 5. ArUco real → /aruco/pose (corrección del EKF) ──────────────────
    # Reemplaza al aruco_oracle de la simulación (que leía ground truth de Gazebo).
    aruco = Node(
        package='puzzlebot_perception',
        executable='aruco_node',
        name='aruco_node',
        output='screen',
        parameters=[{
            'use_sim_time':            False,
            'image_topic':             '/camera/image/compressed',
            'camera_info_file':        calib_yaml,
            'extrinsics_file':         extrinsics_yaml,
            'marker_map_file':         aruco_map_yaml,
            'marker_length':           0.10,
            'max_detection_distance':  2.5,
            'max_incidence_angle_deg': 75.0,
            'max_processing_hz':       8.0,
            'max_position_jump':       0.25,
            'map_min_x': 0.0, 'map_max_x': 3.76,
            'map_min_y': 0.0, 'map_max_y': 4.86,
            'map_bounds_margin': 0.25,
        }],
        condition=IfCondition(aruco_en),
    )

    # ── 6. Kalman EKF — /odom_raw + /aruco/pose → /odom + TF odom→base ─────
    # Mismo rol que en gz_slam_nav: dueño de odom→base_footprint. El ArUco entra
    # al EKF (NO se usa aruco_map_odom). init_from_aruco/initial_* vienen del YAML.
    kalman = Node(
        package='puzzlebot_localization',
        executable='kalman_filter_node',
        name='kalman_filter_node',
        output='screen',
        parameters=[kalman_cfg, {'use_sim_time': False}],
    )

    # ── 7. SLAM con mapa conocido + mapeo en vivo ─────────────────────────
    # MAPPING+INIT: carga el PNG previo y sigue integrando scans → /map vivo.
    #
    # IMPORTANTE (difiere de la simulación): con el EKF activo, el scan matcher
    # alimenta al Kalman vía /scan_match/pose (corrige odom→base_footprint). Por
    # eso scan_match_updates_map_odom=False: si fuera True, el MISMO dato del scan
    # matcher se aplicaría también a map→odom → doble corrección → pose "rara"/saltona
    # peleando contra ArUco. El EKF (encoders+ArUco+scan_match) es la única fuente
    # global; slam_node publica map→odom estático (sin sobreescribirlo con el matcher).
    slam_node = Node(
        package='puzzlebot_slam',
        executable='slam_node',
        name='slam_node',
        output='screen',
        parameters=[slam_cfg, {
            'use_sim_time':                False,
            'localization_map_path':       initial_map,
            'localization_only':           False,
            'publish_map_odom_tf':         True,
            'scan_match_updates_map_odom': False,
        }],
        remappings=[('/scan', '/scan_stamped')],
    )

    # ── 8. Navegación A* + steering + Bug2 + obstacle_avoidance ───────────
    # cmd_vel_topic=/cmd_vel: el bridge micro-ROS de la Jetson escucha /cmd_vel.
    # scan_topic=/scan_stamped: el scan re-sellado del LiDAR real.
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_pkg, 'launch', 'navigation.launch.py')
        ),
        launch_arguments={
            'use_sim_time':     'false',
            'cmd_vel_topic':    '/cmd_vel',
            'scan_topic':       '/scan_stamped',
            'obstacle_manager': obs_mgr,
        }.items(),
        condition=IfCondition(nav_en),
    )

    # ── 9. RViz (Fixed Frame: map, herramienta 2D Nav Goal) ───────────────
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_file],
        parameters=[{'use_sim_time': False}],
        remappings=[('/scan', '/scan_stamped')],
        condition=IfCondition(rviz_en),
        output='screen',
    )

    return LaunchDescription([
        # Argumentos
        arg_initial_map, arg_nav, arg_obs_mgr, arg_aruco, arg_rviz,
        arg_lidar_topic, arg_invert_lidar, arg_lidar_yaw_offset,
        # Infraestructura base
        rsp,
        camera_optical_tf,
        laser_tf,
        scan_restamper,
        # Localización: encoders → EKF (+ ArUco real)
        odometry_raw,
        aruco,
        kalman,
        # SLAM con mapa conocido + mapeo en vivo
        slam_node,
        # Navegación A* + Bug2
        navigation,
        # Visualización
        rviz,
    ])
