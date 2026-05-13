# puzzlebot_controller

Owns time-sensitive motion controllers.

Put here:
- C++ steering or trajectory-following controllers.
- Pure pursuit, PID, and command generation from planned paths.

Do not put here:
- Global path planning.
- Obstacle-map construction.
- State-machine/task orchestration.
- Low-level firmware drivers.

Expected role:
- Subscribe to `/planned_path` and `/odom`.
- Publish velocity commands for the obstacle-avoidance or command-mux layer.
