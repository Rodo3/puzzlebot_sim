# puzzlebot_localization

Owns pose estimation: everything that produces or filters robot pose.

Put here:
- `odometry_node`: canonical C++ wheel-odometry source. It can read physical encoder
  topics or Gazebo `JointState` wheel velocities.
- `kalman_filter_node`: EKF/Kalman fusion from odometry plus external corrections.
- `ground_truth_odom`: Gazebo-only pose source for mapping validation.
- `dead_reckoning_debug`: Python debug odometry for quick comparisons.

Do not put here:
- Occupancy grid mapping.
- MCL against a map.
- Path planning or steering control.

Expected topic ownership:
- `odometry_node` publishes `/odom_raw` by default for the robot-localization chain.
- `kalman_filter_node` publishes `/odom`.
- In simulation mapping, `ground_truth_odom` may publish `/odom`.
- Only one node should publish `/odom` and `odom -> base_footprint` at a time.

`odometry_node` modes:
- Physical/default: `input_source:=encoders`, subscribes `velocity_enc_r` and
  `velocity_enc_l`, publishes `odom_raw`, and leaves TF disabled so
  `kalman_filter_node` owns `odom -> base_footprint`.
- Gazebo wheel-odometry test: `input_source:=joint_states`, remap
  `joint_states` to `/world/<world>/model/puzzlebot/joint_state`, set
  `odom_topic:=/odom`, and enable `publish_tf:=true`.
