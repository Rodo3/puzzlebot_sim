# Puzzlebot — ROS 2 Humble Workspace

ROS 2 Humble workspace for a differential-drive Puzzlebot with:
- **Gazebo Fortress** (ignition-gazebo 6) physics simulation
- **SLAM** occupancy-grid mapping + **MCL** Monte Carlo Localization
- **Camera calibration** pipeline (intrinsic distortion correction)
- **ArUco** marker detection with **EKF** pose fusion
- **PD controller** + obstacle avoidance
- Path planning (A\*)

---

## Workspace Structure

```
puzzlebot_sim/
├── src/
│   ├── puzzlebot_description/    # URDF, SDF, meshes, RViz configs, worlds
│   ├── puzzlebot_bringup/        # Launch files + YAML configs (entry point)
│   ├── puzzlebot_localization/   # C++: odometry_node, kalman_filter_node (EKF)
│   ├── puzzlebot_slam/           # Python: slam_node (log-odds grid), mcl (particles)
│   ├── puzzlebot_perception/     # Python: aruco_node, image_viewer_node, calib pipeline
│   ├── puzzlebot_controller/     # C++: pd_controller_node (steering)
│   ├── puzzlebot_planning/       # Python: path_planner_node, obstacle_avoidance_node
│   ├── puzzlebot_control/        # Python: state_machine_node
│   ├── puzzlebot_msgs/           # Custom ROS 2 message definitions
│   └── shared_utils/             # Shared Python helpers
├── docs/                         # Technical guides (SLAM, setup, workflow)
└── scripts/                      # Build and run helper scripts
```

---

## Quick Start

```bash
# Install ROS dependencies
rosdep install --from-paths src --ignore-src -r -y

# Build all packages
colcon build --symlink-install

# Source workspace (every new terminal)
source /opt/ros/humble/setup.bash
source install/setup.bash
```

---

## Simulation (Gazebo Fortress)

### Launch commands

```bash
# Flat plane — basic motion testing
ros2 launch puzzlebot_bringup gz_sim.launch.py

# Maze — MCL localization against pre-built map
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze

# Maze — build a new map from LiDAR scans
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze mode:=mapping

# Realistic mapping stress test (wheel odometry instead of Gazebo ground-truth)
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze mode:=mapping odom_source:=dead_reckoning

# Headless (no Gazebo GUI)
ros2 launch puzzlebot_bringup gz_sim.launch.py gui:=false
```

### Simulation architecture

| Node | Package | Purpose |
|------|---------|---------|
| `gz_sim` | `ros_gz_sim` | Gazebo Fortress physics + sensors |
| `robot_state_publisher` | `robot_state_publisher` | TF tree from URDF |
| `ros_gz_bridge` | `ros_gz_bridge` | ROS ↔ Gazebo message bridge |
| `odometry_node` | `puzzlebot_localization` | C++ differential-drive odometry |
| `ground_truth_odom` | `puzzlebot_localization` | Gazebo pose → `/odom` (mapping mode) |
| `slam_node` | `puzzlebot_slam` | Log-odds occupancy-grid mapping |
| `mcl` | `puzzlebot_slam` | Monte Carlo Localization (maze mode) |
| `rviz2` | `rviz2` | Visualization (15 s delayed start) |

### Teleop (in a separate terminal)

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args --remap cmd_vel:=/model/puzzlebot/cmd_vel
```

Keys: **i** forward · **,** backward · **j** left · **l** right · **k** stop

### What to watch in RViz

- **MCL**: particles converge near the robot and stay aligned with maze walls.
- **Mapping (ground truth)**: map errors point to LiDAR/model parameters, not odometry.
- **Mapping (dead reckoning)**: small yaw drift is expected; tune `wheel_separation` in `robot_params.yaml`.

---

## Physical Robot

Architecture: the Jetson Orin publishes raw sensor data. All computation runs on the operator PC. Both machines share the same WiFi and `ROS_DOMAIN_ID`.

### Step 1 — Sensors on the Jetson (via SSH)

```bash
# On both machines — add to ~/.bashrc:
export ROS_DOMAIN_ID=42

# Sync Jetson clock (critical for SLAM timestamp matching):
sudo chronyc makestep
```

**Terminal 1 — micro-ROS agent** (encoders + motor bridge):
```bash
ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0 -b 921600
```

| Topic published by MCU | Type | Rate |
|------------------------|------|------|
| `/VelocityEncR` | `std_msgs/Float32` | ~50 Hz |
| `/VelocityEncL` | `std_msgs/Float32` | ~50 Hz |
| `/Lidar` | `sensor_msgs/LaserScan` | ~10 Hz |

**Terminal 2 — Camera**:
```bash
cd ~/ros2_ws && source install/setup.bash
ros2 run <camera_package> <camera_node>
# Publishes: /camera/image/compressed  (CompressedImage, JPEG ~30 Hz)
```

> **Note:** If `sllidar_ros2` runs directly on the Jetson instead of micro-ROS, use `lidar_topic:=/scan` in the PC launch.

### Step 2 — PC stack

```bash
cd ~/Documents/puzzlebot_sim
source install/setup.bash

# Full stack (default): odometry + EKF + ArUco + SLAM + controller + RViz
ros2 launch puzzlebot_bringup real_robot.launch.py

# Mapping only — drive with teleop, no autonomous controller:
ros2 launch puzzlebot_bringup real_robot.launch.py avoidance:=false aruco:=false

# No ArUco (pure wheel odometry through EKF):
ros2 launch puzzlebot_bringup real_robot.launch.py aruco:=false

# Enable live camera viewer with distortion correction:
ros2 launch puzzlebot_bringup real_robot.launch.py viewer:=true

# LiDAR from sllidar direct (not micro-ROS agent):
ros2 launch puzzlebot_bringup real_robot.launch.py lidar_topic:=/scan
```

### Launch arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `slam` | `true` | `slam_node` builds `/map` from `/scan` + `/odom` |
| `avoidance` | `true` | `obstacle_avoidance_node` stops robot near obstacles |
| `aruco` | `true` | `aruco_node` + `kalman_filter_node` for ArUco pose fusion |
| `viewer` | `false` | `image_viewer_node` with distortion correction |
| `rviz` | `true` | Open RViz2 |
| `lidar_topic` | `/Lidar` | LiDAR source topic |

### Physical robot node graph

```
Jetson                          PC
──────                          ──
/VelocityEncR ─────────────→  odometry_node ──→ /odom_raw
/VelocityEncL ─────────────→                          │
                                                       ↓
/aruco/poses (from aruco) ──→  kalman_filter_node ──→ /odom + TF odom→base
                                                       │
/Lidar ────────────────────→  slam_node ────────────→ /map

/camera/image/compressed ──→  aruco_node ───────────→ /aruco/poses
                          └→  image_viewer_node (optional, rectified)

/odom + /goal_pose ────────→  pd_controller_node ──→ /cmd_vel_in
/scan  ────────────────────→  obstacle_avoidance ──→ /cmd_vel ──→ Jetson MCU
```

### Teleop for manual mapping

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
    --ros-args --remap cmd_vel:=/cmd_vel
```

Mapping tips: drive < 0.15 m/s · prefer long straight lines · avoid fast in-place rotations.

---

## Camera Calibration

One-time procedure to compute intrinsic parameters (K matrix + distortion D).

### Step 1 — Capture images (chessboard 9×6 internal corners, 2.6 cm squares)

```bash
# Auto-capture mode (default): captures automatically when board moves
ros2 run puzzlebot_perception calib_capture_node

# Manual mode: press SPACE to capture each image
ros2 run puzzlebot_perception calib_capture_node --ros-args -p auto_capture:=false
```

Images are saved to `~/calib_images/`. Target: 50 images with varied angles, distances, and positions in the frame.

### Step 2 — Compute calibration parameters

```bash
ros2 run puzzlebot_perception calib_compute_node
```

Reads all PNGs, detects the chessboard in each, and runs OpenCV calibration.
Output: `~/calib_images/camera_calibration.yaml`.

Quality guide:
- RMS < 0.5 px — excellent
- RMS 0.5–1.0 px — acceptable
- RMS > 1.0 px — retake images (the node lists the worst ones)

### Step 3 — Install calibration

```bash
cp ~/calib_images/camera_calibration.yaml \
   src/puzzlebot_bringup/config/camera_calibration.yaml
colcon build --packages-select puzzlebot_bringup
source install/setup.bash
```

### Step 4 — View rectified camera (optional verification)

```bash
ros2 run puzzlebot_perception image_viewer_node --ros-args -p rectify:=true
```

Straight lines in the real world should appear straight in the image.

---

## EKF + ArUco Localization

The `kalman_filter_node` (C++ EKF) fuses wheel odometry with ArUco marker pose measurements:

- **Prediction**: differential-drive kinematics from `/odom_raw`
- **Update**: pose corrections from `aruco_node` via `/aruco/poses`
- **Output**: filtered `/odom` + TF `odom → base_footprint`

ArUco markers must be registered in `src/puzzlebot_bringup/config/aruco_map.yaml`:

```yaml
aruco_markers:
  1: {x: 0.50, y: 2.10, z: 0.25, roll: 0.0, pitch: 0.0, yaw: 3.14159}
  2: {x: 2.80, y: 0.30, z: 0.25, roll: 0.0, pitch: 0.0, yaw: 0.0}
```

Camera mount position is defined in `src/puzzlebot_bringup/config/camera_extrinsics.yaml`.
Tune `x/y/z/roll/pitch/yaw` to match the physical mount on your unit.

---

## SLAM Mapping

```bash
# Recommended: clean map using Gazebo ground-truth pose
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze mode:=mapping

# Physical robot: build map while teleopating
ros2 launch puzzlebot_bringup real_robot.launch.py avoidance:=false aruco:=false slam:=true
```

Save a completed map (navigate to the package folder):
```bash
cd src/puzzlebot_slam/puzzlebot_slam
# The map PNG is updated live at maze_map.png
```

See [docs/slam_mapping.md](docs/slam_mapping.md) for the algorithm details.

---

## Configuration Files

All configuration lives in `src/puzzlebot_bringup/config/`:

| File | Description |
|------|-------------|
| `robot_params.yaml` | Wheel radius, wheel separation, odometry frame names |
| `controller_params.yaml` | PD gains, lookahead distance, obstacle stop distance |
| `slam_params.yaml` | Grid size, log-odds probabilities, scan matching |
| `kalman_params.yaml` | EKF process noise Q and measurement noise R |
| `camera_calibration.yaml` | Intrinsic matrix K and distortion coefficients D |
| `camera_extrinsics.yaml` | Camera mount pose relative to `base_link` |
| `aruco_map.yaml` | Known ArUco marker poses in the map frame |

---

## Build Reference

```bash
# Full build
colcon build --symlink-install

# Specific packages only
colcon build --packages-select puzzlebot_bringup puzzlebot_perception

# Clean build artifacts
rm -rf build/ install/ log/

# Source workspace
source install/setup.bash
```
