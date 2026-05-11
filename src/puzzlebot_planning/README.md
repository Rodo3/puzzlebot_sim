# puzzlebot_planning

Owns path planning and local obstacle policy.

Put here:
- A* or other global planners over an occupancy grid.
- Reactive obstacle filtering around planned commands.
- Path and goal handling.

Do not put here:
- Wheel odometry.
- SLAM map updates.
- Steering/PID loops.

Expected role:
- Subscribe to `/map`, `/odom`, and goals.
- Publish `/planned_path`.
- Optionally filter velocity commands before they reach the robot.
