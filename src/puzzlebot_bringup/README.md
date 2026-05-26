# puzzlebot_bringup

Owns launch files and runtime configuration. The entry point for running the robot.

Put here:
- Launch profiles for Gazebo simulation, real robot, mapping, and localization.
- YAML parameter files shared across packages.
- Small smoke-test or mock utilities.

Do not put here:
- Algorithm implementations.
- Robot model files (those belong in `puzzlebot_description`).
- Long-running production nodes, unless they are simple bringup utilities.

Launch files choose between simulation and physical hardware by wiring different input sources — not by duplicating algorithm code.

---

## Launch Files

### `real_robot.launch.py` — Physical robot (main entry point)

```bash
# Full stack (default)
ros2 launch puzzlebot_bringup real_robot.launch.py

# Mapping only (drive with teleop, no autonomous controller)
ros2 launch puzzlebot_bringup real_robot.launch.py avoidance:=false aruco:=false

# No ArUco correction (pure wheel odometry through EKF)
ros2 launch puzzlebot_bringup real_robot.launch.py aruco:=false

# Enable live rectified camera viewer
ros2 launch puzzlebot_bringup real_robot.launch.py viewer:=true

# LiDAR from sllidar direct (not micro-ROS)
ros2 launch puzzlebot_bringup real_robot.launch.py lidar_topic:=/scan
```

| Argument | Default | Nodes affected |
|----------|---------|----------------|
| `slam` | `true` | `slam_node` |
| `avoidance` | `true` | `obstacle_avoidance_node` |
| `aruco` | `true` | `aruco_node` |
| `viewer` | `false` | `image_viewer_node` |
| `rviz` | `true` | `rviz2` |
| `lidar_topic` | `/Lidar` | `slam_node` remap |

Nodes always active: `robot_state_publisher`, `camera_tf` (static), `odometry_node`, `kalman_filter_node`, `pd_controller_node`.

---

### `gz_sim.launch.py` — Gazebo Fortress simulation

```bash
# Flat plane (basic testing)
ros2 launch puzzlebot_bringup gz_sim.launch.py

# Maze with MCL localization
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze

# Maze — build a new map
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze mode:=mapping

# Mapping with wheel odometry (no ground truth)
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze mode:=mapping odom_source:=dead_reckoning
```

| Argument | Default | Description |
|----------|---------|-------------|
| `world` | `flat_plane` | `flat_plane` or `maze` |
| `gui` | `true` | Show Gazebo GUI |
| `slam` | `true` | Enable `slam_node` |
| `rviz` | `true` | Open RViz2 |
| `mode` | `mcl` | `mcl` (localize) or `mapping` (build map) |
| `odom_source` | `ground_truth` | `ground_truth` or `dead_reckoning` |

---

## Configuration Files (`config/`)

| File | Used by | Description |
|------|---------|-------------|
| `robot_params.yaml` | `odometry_node` | Wheel geometry, frame names |
| `controller_params.yaml` | `pd_controller_node`, `obstacle_avoidance_node` | PD gains, obstacle distances |
| `slam_params.yaml` | `slam_node` | Grid size, log-odds params, scan matching |
| `kalman_params.yaml` | `kalman_filter_node` | EKF process/measurement noise |
| `camera_calibration.yaml` | `aruco_node`, `image_viewer_node`, `calib_apply_node` | Intrinsic matrix K, distortion D (RMS 0.96 px) |
| `camera_extrinsics.yaml` | `aruco_node`, `camera_tf` (static TF) | Camera mount pose relative to `base_link` |
| `aruco_map.yaml` | `aruco_node` | Known ArUco marker poses in map frame |

### Tuning `robot_params.yaml`

Measure `wheel_separation` physically on your unit — the default (0.19 m) may differ:

```yaml
wheel_radius:     0.05    # meters
wheel_separation: 0.19    # meters — MEASURE ON YOUR UNIT
```

### Tuning `camera_extrinsics.yaml`

If ArUco gives a consistent offset, adjust the physical mount values:

```yaml
camera_extrinsics:
  x:     0.08   # meters forward from robot center
  y:     0.00   # lateral offset (positive = left)
  z:     0.12   # height from ground
  roll:  0.0    # radians
  pitch: 0.0    # positive = camera tilted down
  yaw:   0.0    # positive = camera pointing left
```

Update both `camera_extrinsics.yaml` **and** the `camera_tf` node arguments in `real_robot.launch.py` to keep them in sync.
