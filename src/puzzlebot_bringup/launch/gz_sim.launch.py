"""
Gazebo Fortress (ignition-gazebo 6) simulation launch.

Stack: ros-humble-ros-gz (Fortress bridge) — the official ROS 2 Humble pairing.
Binary: ign gazebo (gz_version=6).  Never mix with gz sim / Harmonic binaries.

Usage:
  # Flat plane — dead-reckoning only:
  ros2 launch puzzlebot_bringup gz_sim.launch.py

  # Maze world with MCL localisation (default):
  ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze

  # Maze world — build map from scratch with SLAM (teleop to explore):
  ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze mode:=mapping

  # Without RViz:
  ros2 launch puzzlebot_bringup gz_sim.launch.py rviz:=false

  # Headless:
  ros2 launch puzzlebot_bringup gz_sim.launch.py gui:=false

Teleop (needs its own TTY):
  ros2 run teleop_twist_keyboard teleop_twist_keyboard \
    --ros-args --remap cmd_vel:=/model/puzzlebot/cmd_vel

Arguments:
  world  [flat_plane]  'flat_plane' or 'maze'
  gui    [true]
  slam   [true]        Launch dead_reckoning (+ mcl or slam_node depending on mode)
  rviz   [true]
  mode   [mcl]         'mcl' = localise against maze_map.png
                       'mapping' = build OccupancyGrid from scratch (any world)
  odom_source [ground_truth]  For mode:=mapping only:
                       'ground_truth' = Gazebo pose (best maps in simulation)
                       'dead_reckoning' = wheel odometry (realistic drift test)
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


def generate_launch_description():
    desc_pkg    = get_package_share_directory('puzzlebot_description')
    slam_pkg    = get_package_share_directory('puzzlebot_slam')
    bringup_pkg = get_package_share_directory('puzzlebot_bringup')
    ros_gz_sim  = get_package_share_directory('ros_gz_sim')

    # ── Asset paths ─────────────────────────────────────────────────────
    # puzzlebot_gz.urdf  — URDF for robot_state_publisher (Fortress-compatible
    #                      kinematics, no Gazebo plugin tags needed here)
    urdf_file       = os.path.join(desc_pkg, 'urdf', 'puzzlebot_gz.urdf')
    # puzzlebot_gz.sdf   — SDF spawned into Gazebo (Fortress plugins, correct
    #                      inertials, lidar at z=0.20 in base_footprint)
    sdf_file        = os.path.join(desc_pkg, 'sdf',    'puzzlebot_gz.sdf')
    world_flat      = os.path.join(desc_pkg, 'worlds', 'flat_plane.sdf')
    world_maze      = os.path.join(desc_pkg, 'worlds', 'maze.sdf')
    rviz_flat       = os.path.join(desc_pkg,    'rviz',   'puzzlebot_rviz.rviz')
    rviz_maze       = os.path.join(desc_pkg,    'rviz',   'mcl_rviz.rviz')
    rviz_mapping    = os.path.join(desc_pkg,    'rviz',   'mapping_rviz.rviz')
    map_file        = os.path.join(slam_pkg,    'puzzlebot_slam', 'maze_map.png')
    slam_cfg        = os.path.join(bringup_pkg, 'config', 'slam_params.yaml')

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # ── IGN_GAZEBO_RESOURCE_PATH ─────────────────────────────────────────
    # Must point to the PARENT of puzzlebot_description's share dir so Gazebo
    # can resolve model://puzzlebot_description/meshes/... URIs in the SDF.
    # SetEnvironmentVariable must be the FIRST action — Gazebo server and GUI
    # inherit it from the launch process environment.
    desc_share_parent = os.path.dirname(desc_pkg)
    existing = os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')
    ign_resource_path = (desc_share_parent + ':' + existing) if existing else desc_share_parent

    set_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=ign_resource_path,
    )

    # ── Launch arguments ─────────────────────────────────────────────────
    arg_world = DeclareLaunchArgument('world', default_value='flat_plane',
                                      description="'flat_plane' or 'maze'")
    arg_gui   = DeclareLaunchArgument('gui',   default_value='true')
    arg_slam  = DeclareLaunchArgument('slam',  default_value='true')
    arg_rviz  = DeclareLaunchArgument('rviz',  default_value='true')
    # mode: 'mapping' = slam_node builds OccupancyGrid from scratch
    #       'mcl'     = mcl.py localises against the pre-built maze_map.png
    arg_mode  = DeclareLaunchArgument('mode',  default_value='mcl',
                                      description="'mapping' or 'mcl'")
    arg_odom_source = DeclareLaunchArgument(
        'odom_source',
        default_value='ground_truth',
        description="'ground_truth' or 'dead_reckoning' for mode:=mapping",
    )

    world_name = LaunchConfiguration('world')
    slam_en    = LaunchConfiguration('slam')
    rviz_en    = LaunchConfiguration('rviz')
    mode       = LaunchConfiguration('mode')
    odom_source = LaunchConfiguration('odom_source')

    # ── 1. Gazebo Fortress ───────────────────────────────────────────────
    # gz_version='6' → gz_sim.launch.py picks the 'ign gazebo' code path.
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': PythonExpression([
                "'-r ' + ('" + world_maze + "' if '",
                world_name,
                "' == 'maze' else '" + world_flat + "')"
            ]),
            'gz_version': '6',
        }.items(),
    )

    # ── 2. robot_state_publisher ─────────────────────────────────────────
    # Uses puzzlebot_gz.urdf which has correct link/joint names matching the SDF.
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
        output='screen',
    )

    # ── 3. ros_gz_bridge (Fortress: ignition.msgs types) ─────────────────
    #
    # The DiffDrive plugin in puzzlebot_gz.sdf listens on the default topic
    # /model/puzzlebot/cmd_vel (no explicit <topic> tag → Fortress default).
    # Joint-state topic includes the world name, so we need one bridge per world.
    #
    # NOTE: gz_bridge.yaml in the repo root uses gz.msgs.* (Harmonic) — that
    # file is kept as a reference for a future Harmonic/Jazzy migration only.
    # We use explicit argument-style bridging here (Fortress / ignition.msgs).

    bridge_flat = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_bridge',
        arguments=[
            # cmd_vel: ROS 2 teleop → Gazebo DiffDrive plugin
            '/model/puzzlebot/cmd_vel'
            '@geometry_msgs/msg/Twist@ignition.msgs.Twist',
            # odometry: Gazebo → ROS 2 (reference only; dead_reckoning is primary)
            '/model/puzzlebot/odometry'
            '@nav_msgs/msg/Odometry@ignition.msgs.Odometry',
            # clock: sim time for use_sim_time=True
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            # lidar
            '/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan',
            # joint states for dead_reckoning and robot_state_publisher
            '/world/flat_plane/model/puzzlebot/joint_state'
            '@sensor_msgs/msg/JointState[ignition.msgs.Model',
            # true dynamic poses for mapping with ground_truth_odom
            '/world/flat_plane/dynamic_pose/info'
            '@geometry_msgs/msg/PoseArray[ignition.msgs.Pose_V',
        ],
        parameters=[{
            'qos_overrides./model/puzzlebot.subscriber.reliability': 'reliable',
        }],
        condition=IfCondition(
            PythonExpression(["'", world_name, "' == 'flat_plane'"])
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
        parameters=[{
            'qos_overrides./model/puzzlebot.subscriber.reliability': 'reliable',
        }],
        condition=IfCondition(
            PythonExpression(["'", world_name, "' == 'maze'"])
        ),
        output='screen',
    )

    # Relay world-scoped joint_state → /joint_states so robot_state_publisher
    # can publish wheel TF (needed for RViz RobotModel display).
    joint_relay_flat = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='joint_relay',
        arguments=[
            '/world/flat_plane/model/puzzlebot/joint_state'
            '@sensor_msgs/msg/JointState[ignition.msgs.Model',
        ],
        remappings=[
            ('/world/flat_plane/model/puzzlebot/joint_state', '/joint_states'),
        ],
        condition=IfCondition(
            PythonExpression(["'", world_name, "' == 'flat_plane'"])
        ),
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
        remappings=[
            ('/world/maze/model/puzzlebot/joint_state', '/joint_states'),
        ],
        condition=IfCondition(
            PythonExpression(["'", world_name, "' == 'maze'"])
        ),
        output='screen',
    )

    # ── 4. Spawn robot (5 s delay) ───────────────────────────────────────
    # Uses 'ign service' (Fortress CLI).  The 5 s wait ensures Gazebo has
    # registered the /world/<name>/create service before we call it.
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
            additional_env={'IGN_GAZEBO_RESOURCE_PATH': ign_resource_path},
            output='screen',
        )],
        condition=IfCondition(
            PythonExpression(["'", world_name, "' == 'flat_plane'"])
        ),
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
            additional_env={'IGN_GAZEBO_RESOURCE_PATH': ign_resource_path},
            output='screen',
        )],
        condition=IfCondition(
            PythonExpression(["'", world_name, "' == 'maze'"])
        ),
    )

    # ── Static TF: Fortress scopes the lidar frame name as
    #   'puzzlebot/base_footprint/lidar' internally, but the SDF declares
    #   <frame_id>lidar_link</frame_id>.  Publish a zero-offset TF between
    #   the two names so RViz and MCL can find the scan in the right frame.
    lidar_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='lidar_frame_fix',
        arguments=['0', '0', '0', '0', '0', '0',
                   'lidar_link', 'puzzlebot/base_footprint/lidar'],
        output='screen',
    )

    # ── 5. Dead-reckoning odometry ───────────────────────────────────────
    # Remapped to the world-scoped joint_state topic from the bridge.
    dead_reckoning_flat = Node(
        package='puzzlebot_slam',
        executable='dead_reckoning',
        name='dead_reckoning',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'wheel_radius': 0.05,
            'wheel_separation': 0.19,
            'odom_frame': 'odom',
            'base_frame': 'base_footprint',
            'input_source': 'joint_states',
        }],
        remappings=[
            ('/joint_states',
             '/world/flat_plane/model/puzzlebot/joint_state'),
        ],
        condition=IfCondition(PythonExpression([
            "'", world_name, "' == 'flat_plane' and '", slam_en, "' == 'true' and ",
            "('", mode, "' != 'mapping' or '", odom_source, "' == 'dead_reckoning')"
        ])),
    )

    dead_reckoning_maze = Node(
        package='puzzlebot_slam',
        executable='dead_reckoning',
        name='dead_reckoning',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'wheel_radius': 0.05,
            'wheel_separation': 0.19,
            'odom_frame': 'odom',
            'base_frame': 'base_footprint',
            'input_source': 'joint_states',
        }],
        remappings=[
            ('/joint_states',
             '/world/maze/model/puzzlebot/joint_state'),
        ],
        condition=IfCondition(PythonExpression([
            "'", world_name, "' == 'maze' and '", slam_en, "' == 'true' and ",
            "('", mode, "' != 'mapping' or '", odom_source, "' == 'dead_reckoning')"
        ])),
    )

    # ── 5b. Ground-truth odometry for simulation mapping ────────────────
    # Mapping from scratch needs a stable pose estimate.  In Gazebo, use the
    # simulator's true dynamic pose by default so mapping quality is limited by
    # the scan model, not by wheel slip / encoder integration drift.
    ground_truth_flat = Node(
        package='puzzlebot_slam',
        executable='ground_truth_odom',
        name='ground_truth_odom',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'model_name': 'puzzlebot',
            'odom_frame': 'odom',
            'base_frame': 'base_footprint',
            'pose_topic': '/world/flat_plane/dynamic_pose/info',
        }],
        condition=IfCondition(PythonExpression([
            "'", world_name, "' == 'flat_plane' and '", slam_en, "' == 'true' and ",
            "'", mode, "' == 'mapping' and '", odom_source, "' == 'ground_truth'"
        ])),
    )

    ground_truth_maze = Node(
        package='puzzlebot_slam',
        executable='ground_truth_odom',
        name='ground_truth_odom',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'model_name': 'puzzlebot',
            'odom_frame': 'odom',
            'base_frame': 'base_footprint',
            'pose_topic': '/world/maze/dynamic_pose/info',
        }],
        condition=IfCondition(PythonExpression([
            "'", world_name, "' == 'maze' and '", slam_en, "' == 'true' and ",
            "'", mode, "' == 'mapping' and '", odom_source, "' == 'ground_truth'"
        ])),
    )

    # ── 6a. Mapping SLAM node (any world, mode=mapping) ─────────────────
    # Builds an OccupancyGrid on /map from LiDAR + /odom.
    # Active when slam:=true AND mode:=mapping.
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

    # ── 6b. MCL node (maze world only, mode=mcl) ─────────────────────────
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

    # ── 7. RViz — delayed 15 s so /clock is stable ───────────────────────
    # Starting RViz before Gazebo's clock settles causes "jump back in time"
    # warnings that reset TF and break all displays.
    rviz_flat_node = TimerAction(
        period=15.0,
        actions=[Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_flat],
            parameters=[{'use_sim_time': True}],
            output='screen',
        )],
        condition=IfCondition(PythonExpression([
            "'", rviz_en, "' == 'true' and '", world_name, "' == 'flat_plane'"
        ])),
    )

    rviz_maze_node = TimerAction(
        period=15.0,
        actions=[Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_maze],
            parameters=[{'use_sim_time': True}],
            output='screen',
        )],
        condition=IfCondition(PythonExpression([
            "'", rviz_en, "' == 'true' and '",
            world_name, "' == 'maze' and '", mode, "' == 'mcl'"
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

    return LaunchDescription([
        set_resource_path,   # MUST be first — sets env before any subprocess
        arg_world, arg_gui, arg_slam, arg_rviz, arg_mode, arg_odom_source,
        gz_sim,
        rsp,
        bridge_flat,
        bridge_maze,
        joint_relay_flat,
        joint_relay_maze,
        lidar_tf,
        spawn_flat,
        spawn_maze,
        dead_reckoning_flat,
        dead_reckoning_maze,
        ground_truth_flat,
        ground_truth_maze,
        slam_mapping,
        mcl,
        rviz_flat_node,
        rviz_maze_node,
        rviz_mapping_node,
    ])
