"""
mission.launch.py — Capa de misión + voz para el robot REAL.

Arranca SOLO la capa lógica de la misión logística, asumiendo que la base ya
está corriendo aparte (localización, percepción de imagen, navegación A*):

    ros2 launch puzzlebot_bringup real_robot.launch.py navigation:=true
    ros2 launch puzzlebot_control mission.launch.py            # ← este

NODOS QUE ARRANCA:
  mission_manager_node   — FSM logística completa (objetivo del proyecto).
  voice_fsm_router_node  — traduce /voice/* → /mission_state_in (START/PAUSE).
  qr_reader_node         — QR del pallet: /qr/detected, /qr/client, /qr/pose.
  waypoint_navigator_node— /navigate_to_waypoint (nombre) → /goal_pose.

  yolo_node (perception) — logos del tráiler → /detections. NO se lanza por
                           defecto (launch_yolo:=false): el detector ONNX corre
                           standalone en la Jetson. Activar solo si no hay YOLO
                           externo, para no duplicar /detections.

  fork_mock_node         — MOCK del lifter (mock_fork:=true, default). Cierra
                           VERIFY_PICK publicando /fork/status sin hardware.
                           Cuando exista el actuador real, lanzar con
                           mock_fork:=false y correr el driver real en su lugar.

NO arranca el reconocedor de voz (voice_commands_node) ni el web_bridge: esos
corren donde está el micrófono (Jetson o dashboard) y publican /voice/*.

ARGUMENTOS:
  mission_number     [1]      1=conveyor (Misión 1), 2=rack (Misión 2)
  waypoints_file     [...]    YAML de waypoints (default: bringup/config/waypoints.yaml)
  mission_config     [...]    YAML de parámetros de misión (puzzlebot_control/config)
  mock_fork          [true]   Lanza el mock del lifter en vez del driver real
  mock_qr            [false]  Lanza el mock de QR/alignment en vez de qr_reader_node
  mock_yolo          [false]  Lanza el mock de logos (/detections) para bench sin cámara
  fork_lift_time     [2.0]    Segundos de la maniobra simulada del mock
  voice_threshold    [0.0]    Umbral de confianza para comandos de voz normales
  voice_cooldown     [3.0]    Cooldown (s) entre comandos de voz aceptados
  launch_yolo        [false]  Lanzar yolo_node del workspace (YOLO corre en Jetson)
  use_sim_time       [false]  Robot real → false

USO:
  ros2 launch puzzlebot_control mission.launch.py mission_number:=1
  # Disparar START por voz (o simularlo):
  ros2 topic pub --once /mission_state_in std_msgs/String "data: 'START'"
  # STOP global:
  ros2 topic pub --once /mission_state_in std_msgs/String "data: 'PAUSE'"
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup_pkg = get_package_share_directory('puzzlebot_bringup')
    control_pkg = get_package_share_directory('puzzlebot_control')

    default_waypoints = os.path.join(bringup_pkg, 'config', 'waypoints.yaml')
    default_aruco_map = os.path.join(bringup_pkg, 'config', 'aruco_map.yaml')
    default_mission   = os.path.join(control_pkg, 'config', 'mission_config.yaml')
    default_calib     = os.path.join(bringup_pkg, 'config', 'camera_calibration.yaml')
    default_extr      = os.path.join(bringup_pkg, 'config', 'camera_extrinsics.yaml')

    arg_mission_number = DeclareLaunchArgument(
        'mission_number', default_value='1',
        description='1=conveyor (Misión 1), 2=rack (Misión 2)')
    arg_waypoints = DeclareLaunchArgument(
        'waypoints_file', default_value=default_waypoints)
    arg_aruco_map = DeclareLaunchArgument(
        'aruco_map_file', default_value=default_aruco_map)
    arg_mission_cfg = DeclareLaunchArgument(
        'mission_config', default_value=default_mission)
    arg_mock_fork = DeclareLaunchArgument(
        'mock_fork', default_value='true',
        description='Lanzar el mock del lifter (false = usar driver real)')
    arg_mock_qr = DeclareLaunchArgument(
        'mock_qr', default_value='false',
        description='Lanzar el mock de QR/alignment en vez de qr_reader_node real')
    arg_qr_image_topic = DeclareLaunchArgument(
        'qr_image_topic', default_value='/camera/image/compressed',
        description='Tópico de imagen comprimida para el qr_reader_node')
    arg_qr_calib = DeclareLaunchArgument(
        'qr_camera_info_file', default_value=default_calib,
        description='YAML de calibración (K, D) para solvePnP del QR')
    arg_qr_extr = DeclareLaunchArgument(
        'qr_extrinsics_file', default_value=default_extr,
        description='YAML de extrínseca cámara→robot para la pose del QR')
    arg_qr_debug = DeclareLaunchArgument(
        'qr_publish_debug', default_value='true',
        description='Publicar /qr/debug_image con overlay de detección')
    arg_qr_hz = DeclareLaunchArgument(
        'qr_max_hz', default_value='20.0',
        description='Frecuencia máxima de procesamiento del qr_reader_node (Hz)')
    arg_mock_yolo = DeclareLaunchArgument(
        'mock_yolo', default_value='false',
        description='Lanzar el mock de logos (publica /detections del cliente del '
                    'QR). Para validar la segunda mitad en bench sin cámara. '
                    'Excluyente con launch_yolo y con el YOLO real de la Jetson.')
    arg_fork_lift_time = DeclareLaunchArgument(
        'fork_lift_time', default_value='2.0')
    arg_voice_threshold = DeclareLaunchArgument(
        'voice_threshold', default_value='0.0')
    arg_voice_cooldown = DeclareLaunchArgument(
        'voice_cooldown', default_value='3.0')
    arg_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='false')
    arg_launch_yolo = DeclareLaunchArgument(
        'launch_yolo', default_value='false',
        description='Lanzar el yolo_node del workspace. Por defecto FALSE: el '
                    'detector de logos corre standalone en la Jetson y ya '
                    'publica /detections. Activar solo si no hay YOLO externo.')

    mission_number = LaunchConfiguration('mission_number')
    waypoints_file = LaunchConfiguration('waypoints_file')
    aruco_map_file = LaunchConfiguration('aruco_map_file')
    mission_config = LaunchConfiguration('mission_config')
    mock_fork      = LaunchConfiguration('mock_fork')
    mock_qr        = LaunchConfiguration('mock_qr')
    qr_image_topic = LaunchConfiguration('qr_image_topic')
    qr_calib_file  = LaunchConfiguration('qr_camera_info_file')
    qr_extr_file   = LaunchConfiguration('qr_extrinsics_file')
    qr_pub_debug   = LaunchConfiguration('qr_publish_debug')
    qr_max_hz      = LaunchConfiguration('qr_max_hz')
    mock_yolo      = LaunchConfiguration('mock_yolo')
    fork_lift_time = LaunchConfiguration('fork_lift_time')
    voice_thresh   = LaunchConfiguration('voice_threshold')
    voice_cooldown = LaunchConfiguration('voice_cooldown')
    launch_yolo    = LaunchConfiguration('launch_yolo')
    use_sim_time   = LaunchConfiguration('use_sim_time')

    # ── FSM de misión ─────────────────────────────────────────────────────────
    mission_manager = Node(
        package='puzzlebot_control',
        executable='mission_manager_node',
        name='mission_manager_node',
        output='screen',
        parameters=[mission_config, {
            'use_sim_time':   use_sim_time,
            'mission_number': mission_number,
            'waypoints_file': waypoints_file,
            'aruco_map_file': aruco_map_file,
        }],
    )

    # ── Router de voz → FSM ─────────────────────────────────────────────────────
    voice_router = Node(
        package='puzzlebot_control',
        executable='voice_fsm_router_node',
        name='voice_fsm_router_node',
        output='screen',
        parameters=[{
            'use_sim_time':         use_sim_time,
            'confidence_threshold': voice_thresh,
            'cooldown_sec':         voice_cooldown,
        }],
    )

    # ── Percepción de misión ────────────────────────────────────────────────────
    # QR del pallet (qr_reader_node → /qr/detected, /qr/client, /qr/pose).
    # mock_qr:=false → cámara/QR reales; mock_qr:=true → mock que converge solo.
    qr_reader = Node(
        package='puzzlebot_perception',
        executable='qr_reader_node',
        name='qr_reader_node',
        output='screen',
        parameters=[{
            'use_sim_time':     use_sim_time,
            'image_topic':      qr_image_topic,
            'camera_info_file': qr_calib_file,
            'extrinsics_file':  qr_extr_file,
            'publish_debug':    qr_pub_debug,
            'max_processing_hz': qr_max_hz,
        }],
        condition=UnlessCondition(mock_qr),
    )

    qr_mock = Node(
        package='puzzlebot_control',
        executable='qr_mock_node',
        name='qr_mock_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(mock_qr),
    )

    # Mock de logos: publica /detections del cliente del QR para validar la
    # segunda mitad (SEARCH_TRAILER→MATCH→ALIGN_TO_DOCK→DROP→EXIT) en bench.
    yolo_mock = Node(
        package='puzzlebot_control',
        executable='yolo_mock_node',
        name='yolo_mock_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(mock_yolo),
    )

    # Logos del tráiler → /detections (Detection2DArray).
    # Por defecto NO se lanza: el detector ONNX corre standalone en la Jetson y
    # ya publica /detections. Lanzar dos nodos en /detections los duplicaría.
    # Activar solo con launch_yolo:=true si no hay YOLO externo.
    yolo = Node(
        package='puzzlebot_perception',
        executable='yolo_node',
        name='yolo_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(launch_yolo),
    )

    # Traductor nombre-de-waypoint → /goal_pose.
    waypoint_navigator = Node(
        package='puzzlebot_planning',
        executable='waypoint_navigator_node',
        name='waypoint_navigator_node',
        output='screen',
        parameters=[{
            'use_sim_time':   use_sim_time,
            'waypoints_file': waypoints_file,
            'frame_id':       'map',
        }],
    )

    # ── Mock del lifter (mock_fork:=true) ───────────────────────────────────────
    fork_mock = Node(
        package='puzzlebot_control',
        executable='fork_mock_node',
        name='fork_mock_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'lift_time':    fork_lift_time,
        }],
        condition=IfCondition(mock_fork),
    )

    return LaunchDescription([
        arg_mission_number,
        arg_waypoints,
        arg_aruco_map,
        arg_mission_cfg,
        arg_mock_fork,
        arg_mock_qr,
        arg_qr_image_topic,
        arg_qr_calib,
        arg_qr_extr,
        arg_qr_debug,
        arg_qr_hz,
        arg_mock_yolo,
        arg_fork_lift_time,
        arg_voice_threshold,
        arg_voice_cooldown,
        arg_sim_time,
        arg_launch_yolo,
        mission_manager,
        voice_router,
        qr_reader,
        qr_mock,
        yolo_mock,
        yolo,
        waypoint_navigator,
        fork_mock,
    ])
