"""
Real-robot launch — Puzzlebot differential drive (Jetson Orin).

Runs entirely on the PC. The Jetson only publishes sensors; all computation
(odometry, SLAM, control, perception) happens locally.

════════════════════════════════════════════════════════════════
 PREREQUISITOS (hacer UNA sola vez antes de cada sesión)
════════════════════════════════════════════════════════════════

1. Misma red WiFi y mismo ROS_DOMAIN_ID en Jetson y PC:

     # En la Jetson y en el PC (debe ser el mismo número, p.ej. 42):
     export ROS_DOMAIN_ID=42

   Añadir al ~/.bashrc de ambas máquinas para no olvidarlo:
     echo "export ROS_DOMAIN_ID=42" >> ~/.bashrc

2. Sincronizar relojes (crítico para que el SLAM empareje /odom con /scan):

     # En la Jetson (via SSH):
     sudo chronyc makestep

     # Verificar que la diferencia es < 0.1 s:
     # PC:     date +%s%N
     # Jetson: date +%s%N

════════════════════════════════════════════════════════════════
 PASO 1 — EN LA JETSON (via SSH)
════════════════════════════════════════════════════════════════

Abrir 3 terminales SSH (o usar tmux):

── Terminal 1: micro-ROS agent (puente MCU ↔ ROS 2) ──────────
  # El device puede ser /dev/ttyUSB0 o /dev/ttyACM0 según el puerto:
  ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0 -b 921600

  # Verificar que los topics de encoders aparecen:
  ros2 topic list | grep velocity_enc
  # Esperado: /velocity_enc_r  y  /velocity_enc_l

  # Si no aparecen después de 10 s, verificar el puerto:
  ls /dev/tty*   # buscar ttyUSB0 o ttyACM0

── Terminal 2: LiDAR (RPLIDAR A1) ────────────────────────────
  ros2 launch rplidar_ros rplidar_a1_launch.py

  # Verificar que /scan publica:
  ros2 topic hz /scan
  # Esperado: ~10 Hz

  # Si el puerto es distinto (ver ls /dev/tty*):
  ros2 launch rplidar_ros rplidar_a1_launch.py serial_port:=/dev/ttyUSB1

── Terminal 3: Cámara ────────────────────────────────────────
  # La cámara la lanza un script Python propio de la Jetson (fuera de este repo).
  # Ejecutar ese script como lo indica el equipo / documentación del Puzzlebot.

  # Publica: /camera/image/compressed  (sensor_msgs/CompressedImage, JPEG)
  # Verificar:
  ros2 topic hz /camera/image/compressed
  # Esperado: ~30 Hz

  # Ver el feed en vivo desde el PC:
  ros2 run puzzlebot_perception image_viewer_node

  # NOTA: camera_node.py dentro de puzzlebot_perception es un placeholder
  # no utilizado — la cámara real la gestiona el script de la Jetson.

════════════════════════════════════════════════════════════════
 PASO 2 — EN TU PC (workspace local)
════════════════════════════════════════════════════════════════

── Terminal 1: Build y launch principal ──────────────────────
  cd ~/Documents/puzzlebot_sim
  colcon build --symlink-install --packages-select \\
      puzzlebot_bringup puzzlebot_localization puzzlebot_slam \\
      puzzlebot_control puzzlebot_controller puzzlebot_planning
  source install/setup.bash

  # Modo completo (odometría + SLAM + PD controller + obstacle avoidance):
  ros2 launch puzzlebot_bringup real_robot.launch.py

  # Solo odometría + SLAM, sin controlador activo (para mapeo con teleop):
  ros2 launch puzzlebot_bringup real_robot.launch.py avoidance:=false

  # Sin SLAM (solo odometría + controlador):
  ros2 launch puzzlebot_bringup real_robot.launch.py slam:=false

  # Con RViz para ver el mapa en tiempo real:
  ros2 launch puzzlebot_bringup real_robot.launch.py rviz:=true

── Terminal 2: Teleop (para mapeo manual) ────────────────────
  ros2 run teleop_twist_keyboard teleop_twist_keyboard \\
      --ros-args --remap cmd_vel:=/cmd_vel

  # Consejos para minimizar deriva del mapa:
  #   - Velocidad lineal < 0.15 m/s
  #   - Evitar rotaciones rápidas en el lugar
  #   - Mover en líneas rectas largas antes de girar

── Terminal 3: Verificación de topics ────────────────────────
  ros2 topic list
  # Esperado:
  #   /VelocityEncR, /VelocityEncL       ← desde Jetson (micro-ROS, remapeados internamente)
  #   /Lidar                             ← desde Jetson (LIDAR, remapeado a /scan internamente)
  #   /camera/image/compressed            ← desde Jetson (cámara, CompressedImage)
  #   /odom                              ← publicado por odometry_node (PC)
  #   /map                               ← publicado por slam_node (PC)
  #   /cmd_vel                           ← hacia Jetson (MCU)

  # Ver el mapa construyéndose:
  ros2 topic echo /map --no-arr

  # Ver pose actual del robot:
  ros2 topic echo /odom

════════════════════════════════════════════════════════════════
 TUNING EN TIEMPO REAL
════════════════════════════════════════════════════════════════

  # Ajustar ganancias del PD sin reiniciar:
  ros2 param set /pd_controller_node Kp_angular 2.0
  ros2 param set /pd_controller_node Kp_linear  1.0

  # Monitor de señales del controlador:
  ros2 run rqt_plot rqt_plot \\
      /controller/debug/data[0]:data[1]:data[2]:data[3]

  # Enviar goal al PD controller:
  ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \\
      '{header: {frame_id: "odom"}, pose: {position: {x: 1.0, y: 0.0}}}'

════════════════════════════════════════════════════════════════
 ARGUMENTOS DEL LAUNCH
════════════════════════════════════════════════════════════════

  slam      [true]   slam_node: construye /map desde /scan + /odom
  avoidance [true]   obstacle_avoidance_node: para el robot si /scan detecta obstáculo < 0.3 m
  rviz      [false]  Abre RViz con la config por defecto del robot
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

    controller_cfg = os.path.join(bringup_pkg, 'config', 'controller_params.yaml')
    robot_cfg      = os.path.join(bringup_pkg, 'config', 'robot_params.yaml')
    urdf_file      = os.path.join(desc_pkg,    'urdf',   'puzzlebot_gz.urdf')
    rviz_cfg       = os.path.join(desc_pkg,    'rviz',   'puzzlebot_rviz.rviz')

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    slam_cfg = os.path.join(bringup_pkg, 'config', 'slam_params.yaml')

    arg_rviz        = DeclareLaunchArgument('rviz',        default_value='true')
    arg_avoidance   = DeclareLaunchArgument('avoidance',   default_value='true')
    arg_slam        = DeclareLaunchArgument('slam',        default_value='true')
    arg_lidar_topic = DeclareLaunchArgument(
        'lidar_topic',
        default_value='/Lidar',
        description=(
            'LiDAR source topic. '
            'Use /Lidar when micro-ROS agent is running (default). '
            'Use /scan when sllidar_ros2 is launched directly on the Jetson.'
        ),
    )

    rviz_en      = LaunchConfiguration('rviz')
    avoidance_en = LaunchConfiguration('avoidance')
    slam_en      = LaunchConfiguration('slam')
    lidar_topic  = LaunchConfiguration('lidar_topic')

    # ── 1. robot_state_publisher ─────────────────────────────────────────
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': False,
        }],
        output='screen',
    )

    # ── 2. Wheel odometry (C++) — reads /VelocityEncR, /VelocityEncL ────
    # The micro-ROS agent on the Jetson publishes with capital letters.
    # Remappings bridge the firmware topic names to what odometry_node expects.
    # publish_tf=true: this node owns the odom→base_footprint transform.
    # When the EKF (kalman_filter_node) is active, set publish_tf=false here
    # and let the EKF broadcast TF instead.
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

    # ── 3. PD controller ─────────────────────────────────────────────────
    # Reads /odom + /goal_pose, publishes /cmd_vel_in.
    pd_controller = Node(
        package='puzzlebot_controller',
        executable='pd_controller_node',
        name='pd_controller_node',
        output='screen',
        parameters=[controller_cfg, {'use_sim_time': False}],
    )

    # ── 4. Obstacle avoidance ─────────────────────────────────────────────
    # Reads /cmd_vel_in + /scan, publishes /cmd_vel (final output to motors).
    # Disable with avoidance:=false if no LIDAR is connected during bench tests.
    obstacle_avoidance = Node(
        package='puzzlebot_planning',
        executable='obstacle_avoidance_node',
        name='obstacle_avoidance_node',
        output='screen',
        parameters=[controller_cfg, {'use_sim_time': False}],
        condition=IfCondition(avoidance_en),
    )

    # When avoidance is disabled, wire pd_controller directly to /cmd_vel.
    pd_controller_direct = Node(
        package='puzzlebot_controller',
        executable='pd_controller_node',
        name='pd_controller_node',
        output='screen',
        parameters=[controller_cfg, {'use_sim_time': False}],
        remappings=[('/cmd_vel_in', '/cmd_vel')],
        condition=IfCondition(PythonExpression(["'", avoidance_en, "' == 'false'"])),
    )

    # ── 5. SLAM mapping (optional, default on) ──────────────────────────
    # Reads /scan + /odom, publishes /map and TF map→odom.
    # lidar_topic remap lets us switch between two hardware modes:
    #   lidar_topic:=/Lidar  → micro-ROS agent is running (default)
    #   lidar_topic:=/scan   → sllidar_ros2 launched directly on the Jetson
    slam = Node(
        package='puzzlebot_slam',
        executable='slam_node',
        name='slam_node',
        output='screen',
        parameters=[slam_cfg, {'use_sim_time': False}],
        remappings=[
            ('/scan', lidar_topic),
        ],
        condition=IfCondition(slam_en),
    )

    # ── 6. RViz (optional) ───────────────────────────────────────────────
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_cfg],
        parameters=[{'use_sim_time': False}],
        condition=IfCondition(rviz_en),
        output='screen',
    )

    return LaunchDescription([
        arg_rviz,
        arg_avoidance,
        arg_slam,
        arg_lidar_topic,
        rsp,
        odometry,
        slam,
        pd_controller,
        obstacle_avoidance,
        pd_controller_direct,
        rviz,
    ])
