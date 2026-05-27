"""
Real-robot launch — Puzzlebot differential drive (Jetson Orin).

Corre completamente en el PC del operador. La Jetson solo publica sensores;
todo el cómputo (odometría, SLAM, percepción, control) ocurre en el PC.

════════════════════════════════════════════════════════════════
 PASO 1 — EN LA JETSON (via SSH, una terminal por servicio)
════════════════════════════════════════════════════════════════

  # Mismo ROS_DOMAIN_ID en Jetson y PC (añadir al ~/.bashrc de ambas):
  export ROS_DOMAIN_ID=42

── Terminal 1: micro-ROS agent ───────────────────────────────
  ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0 -b 921600

── Terminal 2: LiDAR (sllidar directo, sin micro-ROS) ────────
  cd ~/sllidar_ros2-main && source install/setup.bash
  ros2 launch sllidar_ros2 sllidar_a1_launch.py frame_id:=lidar_link

── Terminal 3: Cámara ────────────────────────────────────────
  cd ~/ros2_ws && source install/setup.bash
  ros2 run <camera_package> <camera_node>

════════════════════════════════════════════════════════════════
 PASO 2 — EN EL PC
════════════════════════════════════════════════════════════════

  cd ~/Documents/puzzlebot_sim && source install/setup.bash

  ── SESIÓN 1: Mapeo (sin scan_matching) ─────────────────────
  ros2 launch puzzlebot_bringup real_robot.launch.py \\
    avoidance:=false viewer:=false lidar_topic:=/scan

  Cuando el mapa esté bien (RViz), guárdalo desde otra terminal:
    ros2 run nav2_map_server map_saver_cli -f ~/puzzlebot_map
  Convierte a PNG para MCL:
    python3 -c "
    from PIL import Image
    Image.open('/home/jesus/puzzlebot_map.pgm').save('/home/jesus/puzzlebot_map.png')
    print('Mapa guardado')"
  Copia map_origin_x/y y map_resolution del .yaml al mcl_params.yaml.

  ── SESIÓN 2: Localización con MCL ──────────────────────────
  ros2 launch puzzlebot_bringup real_robot.launch.py \\
    slam:=false mcl:=true avoidance:=false viewer:=false lidar_topic:=/scan

════════════════════════════════════════════════════════════════
 ARGUMENTOS
════════════════════════════════════════════════════════════════

  slam        [true]   slam_node: construye /map en tiempo real
  mcl         [false]  MCL: localización con mapa PNG guardado
                       (usa slam:=false mcl:=true en sesión 2)
  avoidance   [false]  obstacle_avoidance_node
  aruco       [true]   aruco_node + aruco_map_odom (publica map→odom)
  viewer      [false]  image_viewer_node con corrección de distorsión
  rviz        [true]   RViz2
  lidar_topic [/Lidar] /Lidar (micro-ROS) o /scan (sllidar directo)
  invert_lidar [false] Invierte izquierda/derecha si el LiDAR está espejeado
  lidar_yaw_offset [3.14159265359] Rota el scan; π corrige frente/atrás invertido
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
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
    arg_avoidance   = DeclareLaunchArgument('avoidance',   default_value='false',
                          description='Enable obstacle_avoidance_node')
    arg_aruco       = DeclareLaunchArgument('aruco',       default_value='true',
                          description='Enable aruco_node + aruco_map_odom')
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

    slam_en      = LaunchConfiguration('slam')
    mcl_en       = LaunchConfiguration('mcl')
    avoidance_en = LaunchConfiguration('avoidance')
    aruco_en     = LaunchConfiguration('aruco')
    viewer_en    = LaunchConfiguration('viewer')
    rviz_en      = LaunchConfiguration('rviz')
    lidar_topic  = LaunchConfiguration('lidar_topic')
    invert_lidar = LaunchConfiguration('invert_lidar')
    lidar_yaw_offset = LaunchConfiguration('lidar_yaw_offset')
    slam_publishes_map_odom = ParameterValue(
        PythonExpression([
            "'", aruco_en, "' == 'false' and '", mcl_en, "' == 'false'"
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

    # ── 3. Odometría de ruedas (C++) ──────────────────────────────────────
    # Dueña del frame continuo odom → base_footprint. No debe recibir
    # correcciones globales; esas viven en map → odom.
    odometry = Node(
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
    )

    # ── 4. ArUco map→odom correction ─────────────────────────────────────
    # Convierte /aruco/pose (pose absoluta en map) + /odom (ruedas) en una
    # corrección global map → odom. Se desactiva automáticamente en modo MCL,
    # porque MCL también publica map → odom.
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
            "'", aruco_en, "' == 'true' and '", mcl_en, "' == 'false'"
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
    slam = Node(
        package='puzzlebot_slam',
        executable='slam_node',
        name='slam_node',
        output='screen',
        parameters=[slam_cfg, {
            'use_sim_time': False,
            'publish_map_odom_tf': slam_publishes_map_odom,
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

    return LaunchDescription([
        # Argumentos
        arg_slam,
        arg_mcl,
        arg_avoidance,
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
        odometry,
        aruco_map_odom,
        # Percepción (condicional)
        aruco,
        viewer,
        # Navegación / SLAM
        slam,
        mcl,
        obstacle_avoidance,
        # Visualización
        rviz,
    ])
