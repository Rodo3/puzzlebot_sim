"""
Real-robot launch — Puzzlebot differential drive (Jetson Orin).

Corre completamente en el PC del operador. La Jetson solo publica sensores;
todo el cómputo (odometría, SLAM, percepción, control) ocurre en el PC.

════════════════════════════════════════════════════════════════
 PASO 1 — EN LA JETSON (via SSH, una terminal por servicio)
════════════════════════════════════════════════════════════════

  # Mismo ROS_DOMAIN_ID en Jetson y PC (añadir al ~/.bashrc de ambas):
  export ROS_DOMAIN_ID=42

  # Sincronizar reloj (crítico para SLAM):
  sudo chronyc makestep

── Terminal 1: micro-ROS agent (MCU ↔ ROS 2) ────────────────
  ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0 -b 921600
  # Publica: /VelocityEncR, /VelocityEncL  (std_msgs/Float32)
  # Recibe:  /cmd_vel                       (geometry_msgs/Twist)

── Terminal 2: LiDAR (RPLIDAR A1) ────────────────────────────
  # SOLO si micro-ROS agent NO está corriendo (el MCU ya lee el LiDAR):
  cd ~/sllidar_ros2-main && source install/setup.bash
  ros2 launch sllidar_ros2 sllidar_a1_launch.py frame_id:=lidar_link
  # Publica: /scan  (sensor_msgs/LaserScan, ~10 Hz)

── Terminal 3: Cámara ────────────────────────────────────────
  cd ~/ros2_ws && source install/setup.bash
  ros2 run <camera_package> <camera_node>
  # Publica: /camera/image/compressed  (CompressedImage, JPEG ~30 Hz)

════════════════════════════════════════════════════════════════
 PASO 2 — EN EL PC (workspace local)
════════════════════════════════════════════════════════════════

  cd ~/Documents/puzzlebot_sim
  source install/setup.bash

  # Stack completo (default):
  ros2 launch puzzlebot_bringup real_robot.launch.py

  # Solo mapeo con teleop (sin controlador autónomo):
  ros2 launch puzzlebot_bringup real_robot.launch.py avoidance:=false aruco:=false

  # Sin ArUco / sin Kalman (odometría pura):
  ros2 launch puzzlebot_bringup real_robot.launch.py aruco:=false

  # Con viewer de cámara rectificada:
  ros2 launch puzzlebot_bringup real_robot.launch.py viewer:=true

  # LiDAR directo desde sllidar (sin micro-ROS agent):
  ros2 launch puzzlebot_bringup real_robot.launch.py lidar_topic:=/scan

════════════════════════════════════════════════════════════════
 ARGUMENTOS
════════════════════════════════════════════════════════════════

  slam        [true]   slam_node: construye /map desde /scan + /odom
  avoidance   [true]   obstacle_avoidance_node: frena ante obstáculos < 0.3 m
  aruco       [true]   aruco_node + kalman_filter_node: corrección de pose con marcadores
  viewer      [false]  image_viewer_node con corrección de distorsión integrada
  rviz        [true]   Abre RViz con la config del robot
  lidar_topic [/Lidar] Tópico LiDAR; usa /scan si sllidar corre directo en Jetson
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    bringup_pkg = get_package_share_directory('puzzlebot_bringup')
    desc_pkg    = get_package_share_directory('puzzlebot_description')

    # ── Rutas de configuración ────────────────────────────────────────────
    controller_cfg  = os.path.join(bringup_pkg, 'config', 'controller_params.yaml')
    robot_cfg       = os.path.join(bringup_pkg, 'config', 'robot_params.yaml')
    slam_cfg        = os.path.join(bringup_pkg, 'config', 'slam_params.yaml')
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
                          description='Enable SLAM: slam_node builds /map from /scan + /odom')
    arg_avoidance   = DeclareLaunchArgument('avoidance',   default_value='false',
                          description='Enable obstacle_avoidance_node (needs LiDAR)')
    arg_aruco       = DeclareLaunchArgument('aruco',       default_value='true',
                          description='Enable aruco_node + kalman_filter_node for ArUco pose corrections')
    arg_viewer      = DeclareLaunchArgument('viewer',      default_value='true',
                          description='Enable image_viewer_node with distortion correction')
    arg_rviz        = DeclareLaunchArgument('rviz',        default_value='true',
                          description='Open RViz2 for visualization')
    arg_lidar_topic = DeclareLaunchArgument('lidar_topic', default_value='/Lidar',
                          description='LiDAR source topic (/Lidar via micro-ROS, /scan via sllidar direct)')

    slam_en      = LaunchConfiguration('slam')
    avoidance_en = LaunchConfiguration('avoidance')
    aruco_en     = LaunchConfiguration('aruco')
    viewer_en    = LaunchConfiguration('viewer')
    rviz_en      = LaunchConfiguration('rviz')
    lidar_topic  = LaunchConfiguration('lidar_topic')

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
    camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_tf',
        arguments=['0.08', '0.00', '0.12',
                   '0.0',  '0.0',  '0.0',
                   'base_link', 'camera_link'],
        output='screen',
    )

    # El sllidar publica con frame_id='laser' por default.
    # Este TF conecta 'laser' con 'lidar_link' (que está en el URDF).
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
    # Siempre publica a /odom_raw con publish_tf=false.
    # El kalman_filter_node es el dueño de /odom y del TF odom→base_footprint.
    odometry = Node(
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

    # ── 4. Kalman Filter EKF (C++) ────────────────────────────────────────
    # Suscribe: /odom_raw (predicción) + /aruco/poses (corrección, opcional).
    # Publica:  /odom + TF odom→base_footprint.
    # Sin aruco activo degrada a dead-reckoning, lo cual es correcto.
    kalman = Node(
        package='puzzlebot_localization',
        executable='kalman_filter_node',
        name='kalman_filter_node',
        output='screen',
        parameters=[kalman_cfg, {'use_sim_time': False}],
    )

    # ── 5. ArUco pose estimation ──────────────────────────────────────────
    # Suscribe: /camera/image/compressed + carga camera_calibration.yaml propio.
    # Publica:  /aruco/poses (PoseArray) → consumido por kalman_filter_node.
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
        }],
        condition=IfCondition(aruco_en),
    )

    # ── 6. Camera viewer con corrección de distorsión ─────────────────────
    # Opcional (viewer:=true). Aplica calibración internamente — no necesita
    # calib_apply_node corriendo en paralelo.
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
        condition=IfCondition(viewer_en),
    )

    # ── 7. Scan restamper ────────────────────────────────────────────────
    # Lee el LiDAR de la Jetson (timestamp incorrecto) y publica
    # /scan_stamped con el reloj del PC y frame_id=lidar_link.
    # Usar un topic de salida distinto evita conflictos si el LiDAR
    # ya publica en /scan.
    scan_restamper = Node(
        package='puzzlebot_localization',
        executable='scan_restamper',
        name='scan_restamper',
        output='screen',
        parameters=[{
            'input_topic':  lidar_topic,   # /scan o /Lidar desde la Jetson
            'target_frame': 'lidar_link',  # frame correcto del URDF
        }],
    )

    # ── 8. SLAM mapping ───────────────────────────────────────────────────
    # Suscribe /scan_stamped (re-sellado por scan_restamper).
    slam = Node(
        package='puzzlebot_slam',
        executable='slam_node',
        name='slam_node',
        output='screen',
        parameters=[slam_cfg, {'use_sim_time': False}],
        remappings=[('/scan', '/scan_stamped')],
        condition=IfCondition(slam_en),
    )

    # ── 8. PD Controller ─────────────────────────────────────────────────
    # Lee /odom + /goal_pose, publica /cmd_vel_in.
    # Con avoidance activo: obstacle_avoidance_node filtra /cmd_vel_in → /cmd_vel.
    # Sin avoidance: pd_controller publica directo a /cmd_vel.
    pd_controller = Node(
        package='puzzlebot_controller',
        executable='pd_controller_node',
        name='pd_controller_node',
        output='screen',
        parameters=[controller_cfg, {'use_sim_time': False}],
        condition=IfCondition(avoidance_en),
    )

    pd_controller_direct = Node(
        package='puzzlebot_controller',
        executable='pd_controller_node',
        name='pd_controller_node',
        output='screen',
        parameters=[controller_cfg, {'use_sim_time': False}],
        remappings=[('/cmd_vel_in', '/cmd_vel')],
        condition=IfCondition(PythonExpression(["'", avoidance_en, "' == 'false'"])),
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
        arg_avoidance,
        arg_aruco,
        arg_viewer,
        arg_rviz,
        arg_lidar_topic,
        # Infraestructura base (siempre activos)
        rsp,
        camera_tf,
        laser_tf,
        scan_restamper,
        odometry,
        kalman,
        # Percepción (condicional)
        aruco,
        viewer,
        # Navegación
        slam,
        pd_controller,
        pd_controller_direct,
        obstacle_avoidance,
        # Visualización
        rviz,
    ])
