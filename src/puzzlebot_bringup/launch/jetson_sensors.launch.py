"""
Jetson-side sensor launch — corre en la Jetson via SSH.

Lanza los tres nodos de hardware que deben correr en la Jetson:
  1. LiDAR   — sllidar_ros2  → publica /scan
  2. Cámara  — v4l2_camera   → publica /image_raw + compresión → /camera/image/compressed
  3. micro-ROS agent         → bridge STM32 ↔ ROS 2 (encoders + cmd_vel)

El PC corre todo lo demás (real_robot.launch.py):
  SLAM, odometría, ArUco, navegación, bridge del dashboard.

════════════════════════════════════════════════════════════════
 USO
════════════════════════════════════════════════════════════════

  # En la Jetson (desde el workspace de ROS 2):
  ros2 launch puzzlebot_bringup jetson_sensors.launch.py

  # Argumentos opcionales:
  ros2 launch puzzlebot_bringup jetson_sensors.launch.py \\
    lidar_model:=a1 \\
    camera_device:=/dev/video0 \\
    microros_port:=/dev/ttyUSB0

  # En el PC (en otra terminal):
  ros2 launch puzzlebot_bringup real_robot.launch.py \\
    lidar_topic:=/scan slam:=true navigation:=true \\
    artifact_dir:=src/puzzlebot_voice_commands/artifacts_final

════════════════════════════════════════════════════════════════
 ARGUMENTOS
════════════════════════════════════════════════════════════════

  lidar_model     [a1]           Modelo RPLIDAR: a1, a2, a3, s2, s3
                                 Determina el launch file de sllidar_ros2.
  camera_device   [/dev/video0]  Device path de la cámara USB.
  microros_port   [/dev/ttyUSB0] Puerto serial para micro-ROS agent.
                                 Usa udp4 si la conexión es por red:
                                   microros_transport:=udp4
  microros_transport [serial]    serial o udp4.
  microros_udp_port  [8888]      Puerto UDP (solo si transport=udp4).

════════════════════════════════════════════════════════════════
 NOTAS
════════════════════════════════════════════════════════════════

  - ROS_DOMAIN_ID debe ser IGUAL en Jetson y PC.
  - Ambas máquinas deben estar en la misma red WiFi.
  - micro-ROS agent no es un nodo ROS 2 nativo, se lanza aparte
    como proceso del sistema. Ver comentario en generate_launch_description().
  - La compresión de la cámara requiere image_transport_plugins instalado.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():

    # ── Argumentos ──────────────────────────────────────────────────────────
    arg_lidar_model = DeclareLaunchArgument(
        'lidar_model', default_value='a1',
        description='Modelo RPLIDAR: a1, a2, a3, s2, s3')

    arg_camera_device = DeclareLaunchArgument(
        'camera_device', default_value='/dev/video0',
        description='Device path de la cámara USB')

    arg_microros_port = DeclareLaunchArgument(
        'microros_port', default_value='/dev/ttyUSB0',
        description='Puerto serial para micro-ROS agent (solo si transport=serial)')

    arg_microros_transport = DeclareLaunchArgument(
        'microros_transport', default_value='serial',
        description='Transport de micro-ROS: serial o udp4')

    arg_microros_udp_port = DeclareLaunchArgument(
        'microros_udp_port', default_value='8888',
        description='Puerto UDP para micro-ROS agent (solo si transport=udp4)')

    lidar_model        = LaunchConfiguration('lidar_model')
    camera_device      = LaunchConfiguration('camera_device')
    microros_port      = LaunchConfiguration('microros_port')
    microros_transport = LaunchConfiguration('microros_transport')
    microros_udp_port  = LaunchConfiguration('microros_udp_port')

    # ── 1. LiDAR — sllidar_ros2 ─────────────────────────────────────────────
    # sllidar_ros2 publica /scan (frame_id: laser) a ~10 Hz.
    # El launch file varía según el modelo del sensor.
    # scan_restamper en el PC reescribe frame y timestamp antes de SLAM.
    lidar_node = Node(
        package='sllidar_ros2',
        executable='sllidar_node',
        name='sllidar_node',
        parameters=[{
            'serial_port':      '/dev/ttyUSB1',   # ajustar si el puerto cambia
            'serial_baudrate':  115200,
            'frame_id':         'laser',
            'inverted':         False,
            'angle_compensate': True,
            'scan_mode':        'Standard',
        }],
        output='screen',
    )

    # ── 2. Cámara USB — v4l2_camera ─────────────────────────────────────────
    # Publica /image_raw y /camera_info.
    # El container de abajo agrega compresión vía image_transport
    # para publicar /camera/image/compressed (lo que espera aruco_node y el bridge).
    camera_node = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        name='v4l2_camera',
        namespace='camera',
        parameters=[{
            'video_device':   camera_device,
            'image_size':     [640, 480],
            'pixel_format':   'YUYV',
            'output_encoding': 'rgb8',
        }],
        remappings=[
            ('image_raw',   '/camera/image_raw'),
            ('camera_info', '/camera/camera_info'),
        ],
        output='screen',
    )

    # Republica /camera/image_raw → /camera/image/compressed
    # Requiere: image_transport y image_transport_plugins instalados.
    camera_compress = Node(
        package='image_transport',
        executable='republish',
        name='camera_compressor',
        arguments=['raw', 'compressed'],
        remappings=[
            ('in',              '/camera/image_raw'),
            ('out/compressed',  '/camera/image/compressed'),
        ],
        output='screen',
    )

    # ── 3. micro-ROS agent — serial ─────────────────────────────────────────
    # El micro-ROS agent NO es un nodo ROS 2 — es un proceso del sistema.
    # Puentes: STM32 → /VelocityEncR, /VelocityEncL
    #          /cmd_vel → STM32 (motores)
    #
    # SERIAL:
    microros_serial = ExecuteProcess(
        cmd=['ros2', 'run', 'micro_ros_agent', 'micro_ros_agent',
             'serial', '--dev', microros_port, '-b', '921600'],
        output='screen',
        condition=IfCondition(
            PythonExpression(["'", microros_transport, "' == 'serial'"])
        ),
    )

    # UDP4 (alternativa si la Jetson se conecta por red):
    microros_udp = ExecuteProcess(
        cmd=['ros2', 'run', 'micro_ros_agent', 'micro_ros_agent',
             'udp4', '--port', microros_udp_port],
        output='screen',
        condition=IfCondition(
            PythonExpression(["'", microros_transport, "' == 'udp4'"])
        ),
    )

    # ── Info de inicio ───────────────────────────────────────────────────────
    startup_msg = LogInfo(msg=[
        '\n',
        '══════════════════════════════════════════════\n',
        ' Jetson sensors iniciando\n',
        '   LiDAR  → /scan\n',
        '   Cámara → /camera/image/compressed\n',
        '   micro-ROS transport: ', microros_transport, '\n',
        ' Asegúrate de que ROS_DOMAIN_ID coincida con el PC\n',
        '══════════════════════════════════════════════',
    ])

    return LaunchDescription([
        arg_lidar_model,
        arg_camera_device,
        arg_microros_port,
        arg_microros_transport,
        arg_microros_udp_port,
        startup_msg,
        lidar_node,
        camera_node,
        camera_compress,
        microros_serial,
        microros_udp,
    ])
