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


def generate_launch_description():
    desc_pkg    = get_package_share_directory('puzzlebot_description')
    slam_pkg    = get_package_share_directory('puzzlebot_slam')
    bringup_pkg = get_package_share_directory('puzzlebot_bringup')
    ros_gz_sim  = get_package_share_directory('ros_gz_sim')

    urdf_file    = os.path.join(desc_pkg, 'urdf', 'puzzlebot_gz.urdf')
    sdf_file     = os.path.join(desc_pkg, 'sdf',    'puzzlebot_gz.sdf')
    world_flat   = os.path.join(desc_pkg, 'worlds', 'flat_plane.sdf')
    world_maze   = os.path.join(desc_pkg, 'worlds', 'maze.sdf')
    rviz_flat    = os.path.join(desc_pkg, 'rviz',   'puzzlebot_rviz.rviz')
    rviz_maze    = os.path.join(desc_pkg, 'rviz',   'mcl_rviz.rviz')
    rviz_mapping = os.path.join(desc_pkg, 'rviz',   'mapping_rviz.rviz')
    map_file     = os.path.join(slam_pkg, 'puzzlebot_slam', 'maze_map.png')
    slam_cfg     = os.path.join(bringup_pkg, 'config', 'slam_params.yaml')

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    desc_share_parent = os.path.dirname(desc_pkg)
    existing = os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')
    ign_resource_path = (desc_share_parent + ':' + existing) if existing else desc_share_parent

    set_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=ign_resource_path,
    )

    arg_world = DeclareLaunchArgument('world', default_value='flat_plane',
                                      description="'flat_plane' or 'maze'")
    arg_gui   = DeclareLaunchArgument('gui',   default_value='true')
    arg_slam  = DeclareLaunchArgument('slam',  default_value='true')
    arg_rviz  = DeclareLaunchArgument('rviz',  default_value='true')
    arg_mode  = DeclareLaunchArgument('mode',  default_value='mcl',
                                      description="'mapping' or 'mcl'")
    arg_odom_source = DeclareLaunchArgument(
        'odom_source',
        default_value='ground_truth',
        description="'ground_truth' or 'dead_reckoning' for mode:=mapping",
    )

    world_name  = LaunchConfiguration('world')
    slam_en     = LaunchConfiguration('slam')
    rviz_en     = LaunchConfiguration('rviz')
    mode        = LaunchConfiguration('mode')
    odom_source = LaunchConfiguration('odom_source')

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

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
        output='screen',
    )

    # ── 3. ros_gz_bridge ─────────────────────────────────────────────────
    # Camera topics added: /camera/image_raw and /camera/camera_info
    # Fortress syntax: '[' means Gazebo→ROS only (subscribe from Gazebo)

    bridge_flat = Node(
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
            '/world/flat_plane/model/puzzlebot/joint_state'
            '@sensor_msgs/msg/JointState[ignition.msgs.Model',
            '/world/flat_plane/dynamic_pose/info'
            '@geometry_msgs/msg/PoseArray[ignition.msgs.Pose_V',
            # ── Camera bridge (Gazebo → ROS 2) ──────────────────────────
            '/camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
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
            # ── Camera bridge (Gazebo → ROS 2) ──────────────────────────
            '/camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
        ],
        parameters=[{
            'qos_overrides./model/puzzlebot.subscriber.reliability': 'reliable',
        }],
        condition=IfCondition(
            PythonExpression(["'", world_name, "' == 'maze'"])
        ),
        output='screen',
    )

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

    wheel_odom_flat = Node(
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
        remappings=[
            ('/joint_states', '/world/flat_plane/model/puzzlebot/joint_state'),
        ],
        condition=IfCondition(PythonExpression([
            "'", world_name, "' == 'flat_plane' and '", slam_en, "' == 'true' and ",
            "('", mode, "' != 'mapping' or '", odom_source, "' == 'dead_reckoning')"
        ])),
    )

    wheel_odom_maze = Node(
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
        remappings=[
            ('/joint_states', '/world/maze/model/puzzlebot/joint_state'),
        ],
        condition=IfCondition(PythonExpression([
            "'", world_name, "' == 'maze' and '", slam_en, "' == 'true' and ",
            "('", mode, "' != 'mapping' or '", odom_source, "' == 'dead_reckoning')"
        ])),
    )

    ground_truth_flat = Node(
        package='puzzlebot_localization',
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
        package='puzzlebot_localization',
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
    aruco_node = Node(
        package='puzzlebot_perception',
        executable='aruco_node',
        name='aruco_node',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    kalman_node = Node(
        package='puzzlebot_perception',
        executable='kalman_node',
        name='kalman_node',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

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
        set_resource_path,
        arg_world, arg_gui, arg_slam, arg_rviz, arg_mode, arg_odom_source,
        gz_sim,
        rsp,
        bridge_flat,
        bridge_maze,
        joint_relay_flat,
        joint_relay_maze,
        lidar_tf,
        camera_tf,        # ← nuevo: TF alias para camera_link
        spawn_flat,
        spawn_maze,
        wheel_odom_flat,
        wheel_odom_maze,
        ground_truth_flat,
        ground_truth_maze,
        slam_mapping,
        mcl,
        aruco_node,       # ← nuevo
        kalman_node,      # ← nuevo
        rviz_flat_node,
        rviz_maze_node,
        rviz_mapping_node,
    ])