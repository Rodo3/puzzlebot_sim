# puzzlebot_slam

Owns lidar-map algorithms.

Put here:
- Online occupancy-grid mapping from `/scan` and `/odom`.
- Scan matching against the map.
- Keyframe logic.
- Monte Carlo localization against a known map.
- Map generation helpers tied to lidar maps.

Do not put here:
- Wheel odometry.
- Kalman/EKF pose fusion.
- Gazebo ground-truth odometry.
- Controllers or path planners.

Current structure:
- `slam_node.py`: ROS wrapper for mapping.
- `occupancy_grid_map.py`: log-odds grid and ray tracing.
- `odometry_buffer.py`: scan/odom timestamp synchronization for mapping.
- `scan_matcher.py`: future local scan matching implementation point.
- `keyframe_manager.py`: optional scan integration gate.
- `mcl.py`: localization against `maze_map.png`.
