# puzzlebot_localization

Owns pose estimation: everything that produces or filters robot pose.

Put here:
- `odometry_node`: canonical C++ wheel-odometry source for the physical robot.
- `kalman_filter_node`: EKF/Kalman fusion from odometry plus external corrections.
- `ground_truth_odom`: Gazebo-only pose source for mapping validation.
- `dead_reckoning_debug`: Python debug odometry for quick comparisons.

Do not put here:
- Occupancy grid mapping.
- MCL against a map.
- Path planning or steering control.

Expected topic ownership:
- `odometry_node` publishes `/odom_raw`.
- `kalman_filter_node` publishes `/odom`.
- In simulation mapping, `ground_truth_odom` may publish `/odom`.
- Only one node should publish `/odom` and `odom -> base_footprint` at a time.
