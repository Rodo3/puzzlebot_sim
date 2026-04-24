"""
Main simulation launch file — Gazebo Classic + ROS2.

Usage
-----
  # Full system:
  ros2 launch puzzlebot_bringup simulation.launch.py

  # Localization only (Sprint 1-2 testing, no perception/navigation):
  ros2 launch puzzlebot_bringup simulation.launch.py slam:=false perception:=false navigation:=false

  # Headless (no GUI, useful for CI):
  ros2 launch puzzlebot_bringup simulation.launch.py gui:=false

Arguments
---------
  world       [empty.world]  Gazebo world file (full path or name from gazebo model path)
  gui         [true]         Show Gazebo GUI
  slam        [true]         Launch slam_node
  perception  [true]         Launch aruco_node and yolo_node
  navigation  [true]         Launch path_planner, obstacle_avoidance, steering_controller, state_machine
  rviz        [true]         Launch RViz
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    ExecuteProcess,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup_dir = get_package_share_directory('puzzlebot_bringup')
    desc_dir    = get_package_share_directory('puzzlebot_description')
    gazebo_ros  = get_package_share_directory('gazebo_ros')
    cfg         = os.path.join(bringup_dir, 'config')
    urdf_file   = os.path.join(desc_dir, 'urdf', 'puzzlebot.urdf')
    rviz_file   = os.path.join(desc_dir, 'rviz', 'puzzlebot_rviz.rviz')

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # ── Gazebo model/resource path so package:// URIs resolve in meshes ──
    gazebo_model_path = SetEnvironmentVariable(
        'GAZEBO_MODEL_PATH',
        os.path.join(desc_dir, '..', '..') + ':' +
        os.environ.get('GAZEBO_MODEL_PATH', '')
    )

    # ── Launch arguments ──────────────────────────────────────────────
    arg_world      = DeclareLaunchArgument('world',      default_value='empty.world')
    arg_gui        = DeclareLaunchArgument('gui',        default_value='true')
    arg_slam       = DeclareLaunchArgument('slam',       default_value='true')
    arg_perception = DeclareLaunchArgument('perception', default_value='true')
    arg_navigation = DeclareLaunchArgument('navigation', default_value='true')
    arg_rviz       = DeclareLaunchArgument('rviz',       default_value='true')

    world_f      = LaunchConfiguration('world')
    gui_en       = LaunchConfiguration('gui')
    slam_en      = LaunchConfiguration('slam')
    perception_en = LaunchConfiguration('perception')
    navigation_en = LaunchConfiguration('navigation')
    rviz_en      = LaunchConfiguration('rviz')

    # ── Gazebo Classic ────────────────────────────────────────────────
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': world_f,
            'gui':   gui_en,
        }.items(),
    )

    # ── Spawn robot into Gazebo ───────────────────────────────────────
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity',    'puzzlebot',
            '-topic',     'robot_description',
            '-x', '0.0', '-y', '0.0', '-z', '0.20',
        ],
        output='screen',
    )

    # ── robot_state_publisher ─────────────────────────────────────────
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }],
        output='screen',
    )

    # ── RViz ──────────────────────────────────────────────────────────
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_file],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(rviz_en),
        output='screen',
    )

    # ── Localization (always on when simulating) ───────────────────────
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_dir, 'launch', 'localization.launch.py')
        )
    )

    # ── SLAM ──────────────────────────────────────────────────────────
    slam_node = Node(
        package='puzzlebot_slam',
        executable='slam_node',
        name='slam_node',
        parameters=[os.path.join(cfg, 'slam_params.yaml')],
        condition=IfCondition(slam_en),
        output='screen',
    )

    # ── Perception ────────────────────────────────────────────────────
    aruco_node = Node(
        package='puzzlebot_perception',
        executable='aruco_node',
        name='aruco_node',
        condition=IfCondition(perception_en),
        output='screen',
    )

    yolo_node = Node(
        package='puzzlebot_perception',
        executable='yolo_node',
        name='yolo_node',
        parameters=[os.path.join(cfg, 'yolo_params.yaml')],
        condition=IfCondition(perception_en),
        output='screen',
    )

    # ── Planning & Control ────────────────────────────────────────────
    path_planner_node = Node(
        package='puzzlebot_planning',
        executable='path_planner_node',
        name='path_planner_node',
        condition=IfCondition(navigation_en),
        output='screen',
    )

    obstacle_avoidance_node = Node(
        package='puzzlebot_planning',
        executable='obstacle_avoidance_node',
        name='obstacle_avoidance_node',
        parameters=[os.path.join(cfg, 'controller_params.yaml')],
        condition=IfCondition(navigation_en),
        output='screen',
    )

    steering_controller_node = Node(
        package='puzzlebot_controller',
        executable='steering_controller_node',
        name='steering_controller_node',
        parameters=[os.path.join(cfg, 'controller_params.yaml')],
        condition=IfCondition(navigation_en),
        output='screen',
    )

    state_machine_node = Node(
        package='puzzlebot_control',
        executable='state_machine_node',
        name='state_machine_node',
        condition=IfCondition(navigation_en),
        output='screen',
    )

    return LaunchDescription([
        # Environment
        gazebo_model_path,
        # Args
        arg_world, arg_gui,
        arg_slam, arg_perception, arg_navigation, arg_rviz,
        # Core sim
        gazebo,
        robot_state_publisher,
        spawn_robot,
        rviz_node,
        # Localization always on
        localization,
        # Conditional
        slam_node,
        aruco_node,
        yolo_node,
        path_planner_node,
        obstacle_avoidance_node,
        steering_controller_node,
        state_machine_node,
    ])
