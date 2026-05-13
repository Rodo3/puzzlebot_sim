# Puzzlebot Simulation — ROS 2 Humble Workspace

ROS 2 Humble workspace for Puzzlebot with Gazebo Fortress (ignition-gazebo 6) physics simulation, SLAM mapping, MCL localization, and weekly homework assignments.

## Workspace Structure

```
puzzlebot_sim/
├── src/
│   ├── puzzlebot_description/      # URDF, SDF, meshes, RViz configs, worlds
│   ├── puzzlebot_bringup/          # Gazebo Fortress launch files
│   ├── puzzlebot_localization/     # Odometry, EKF, sim/debug pose sources
│   ├── puzzlebot_slam/             # Lidar mapping, scan matching, MCL
│   ├── homework_01_transforms/     # HW1: TF transforms + circular trajectory
│   ├── puzzlebot_tf_tools/         # Reusable TF utilities (shared)
│   └── shared_utils/               # General Python helpers (shared)
├── docs/                           # Setup, workflow, architecture guides
└── scripts/                        # Build and run helper scripts
```

## Quick Start

### Ubuntu 22.04 / WSL2

```bash
# Install ROS dependencies
rosdep install --from-paths src --ignore-src -r -y

# Install Gazebo Fortress (if not already installed)
sudo apt install ros-humble-ros-gz

# Build
make build

# Source the workspace (every new terminal)
source /opt/ros/humble/setup.bash
source install/setup.bash

# Launch simulation in RViz
make rviz
```

See [docs/setup.md](docs/setup.md) for detailed setup instructions.

## Running In Simulation

The workspace includes a complete Gazebo Fortress (ignition-gazebo 6) simulation stack with two worlds: flat plane and maze.

### Gazebo Fortress

Start the simulation (choose one):

```bash
# Flat plane world
ros2 launch puzzlebot_bringup gz_sim.launch.py

# Maze world with MCL localization against the known map
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze

# Maze world — build a map from lidar scans
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze mode:=mapping

# Mapping with wheel odometry instead of Gazebo ground truth
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze mode:=mapping odom_source:=dead_reckoning

# Without RViz visualization
ros2 launch puzzlebot_bringup gz_sim.launch.py rviz:=false

# Headless (no GUI)
ros2 launch puzzlebot_bringup gz_sim.launch.py gui:=false

# Disable SLAM/localization extras
ros2 launch puzzlebot_bringup gz_sim.launch.py slam:=false
```

### Simulation Teleop

In a **new terminal**, launch the keyboard teleop node:

```bash
# Source your workspace first
source /opt/ros/humble/setup.bash
source install/setup.bash

# Start teleop_twist_keyboard
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args --remap cmd_vel:=/model/puzzlebot/cmd_vel
```

Then use keyboard to control:
- **i** — forward
- **,** — backward
- **j** — turn left
- **l** — turn right
- **k** — stop
- **q** — quit

### Simulation Checks

Use these checks before changing SLAM or localization logic:

```bash
# 1. MCL sanity check: odometry prediction + lidar correction on known map
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze mode:=mcl

# 2. Clean mapping baseline: Gazebo ground-truth pose + lidar mapper
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze mode:=mapping

# 3. Realistic mapping stress test: wheel odometry + lidar mapper
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze mode:=mapping odom_source:=dead_reckoning
```

What to watch in RViz:
- In MCL, particles should converge near the robot and stay aligned with the maze walls.
- During in-place rotations, `base_footprint` should rotate without large sideways jumps.
- In mapping with wheel odometry, small yaw drift is expected until `wheel_separation` and encoder scale are tuned.
- In mapping with ground truth, map errors usually point to lidar/model/map parameters instead of odometry.

### Simulation Architecture

| Component | Purpose |
|---|---|
| **Gazebo Fortress** | Physics engine (ODE), sensor simulation |
| **robot_state_publisher** | Publishes TF tree from URDF |
| **ros_gz_bridge** | Bidirectional ROS ↔ Gazebo message bridge |
| **odometry_node** | C++ differential-drive odometry from Gazebo joint states when `odom_source:=dead_reckoning` |
| **dead_reckoning_debug** | Python debug odometry for comparisons only |
| **ground_truth_odom** | Gazebo pose → `/odom` for clean mapping in simulation |
| **slam_node** | Log-odds occupancy grid mapping from `/scan` + `/odom` |
| **MCL (maze only)** | Monte Carlo Localization for map-based pose estimation |
| **RViz** | Visualization (delayed 15s to stabilize clock) |

### SLAM Mapping

The live mapper publishes `/map` as an `OccupancyGrid` using a log-odds inverse
sensor model and Bresenham ray tracing. In Gazebo mapping mode, `odom_source`
defaults to `ground_truth`, so the map quality is not dominated by simulated
wheel slip or encoder integration error.

```bash
# Recommended: clean map in Gazebo using simulator pose
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze mode:=mapping

# Realistic drift test: use wheel odometry
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze mode:=mapping odom_source:=dead_reckoning
```

See [docs/slam_mapping.md](docs/slam_mapping.md) for the theory, implementation
details, and robot-real considerations.

## Running On The Physical Robot

Use the physical stack when the real robot is publishing hardware topics such as
`/velocity_enc_r`, `/velocity_enc_l`, and `/scan`. Do not use Gazebo bridge
topics or `use_sim_time:=true` on the robot.

### Physical Localization

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash

# Wheel odometry -> /odom_raw, Kalman -> /odom
ros2 launch puzzlebot_bringup localization.launch.py
```

Current status:
- `odometry_node` reads `/velocity_enc_r` and `/velocity_enc_l` by default.
- `odometry_node` publishes `/odom_raw`.
- `kalman_filter_node` publishes `/odom` and owns `odom -> base_footprint`.
- Kalman currently has limited correction value until external measurements,
  such as ArUcos, are implemented and calibrated.

### Physical SLAM

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash

# Uses physical odometry and physical lidar topics
ros2 launch puzzlebot_bringup slam.launch.py
```

Before trusting a physical SLAM run:
- Verify encoder sign: forward command should increase x in `/odom_raw`.
- Verify turn sign: left turn should increase yaw.
- Measure and tune `wheel_radius` and `wheel_separation` in
  `src/puzzlebot_bringup/config/robot_params.yaml`.
- Confirm TF contains a stable path from `odom` to `base_footprint` and from
  `base_footprint` to the lidar frame.
- Move slowly at first; wheel slip and fast rotations make the map accumulate yaw error.

### Map Generation (Maze World)

The maze map used by MCL is generated from the SDF world geometry:

```bash
cd src/puzzlebot_slam/puzzlebot_slam
python3 generate_maze_map.py
```

This creates `maze_map.png` (206×221 px) with free cells in white and obstacles in black, matching the walls and boxes in `worlds/maze.sdf`.

## Homework Organization

| Package | Assignment |
|---|---|
| `homework_01_transforms` | TF frames, circular trajectory, joint states |

### Adding a new homework package

**Python (rclpy):**
```bash
cd src
ros2 pkg create --build-type ament_python --dependencies rclpy homework_02_<topic>
```

**C++ (rclcpp):**
```bash
cd src
ros2 pkg create --build-type ament_cmake --dependencies rclcpp homework_02_<topic>
```

After creating the package, rebuild from the repo root:
```bash
cd ..
make build
```

## Build Commands

```bash
make build    # colcon build
make clean    # remove build/install/log
make source   # print source command
make rviz     # launch puzzlebot simulation
make help     # list all commands
```

## Contributing

- No direct push to `main`
- Open a PR with at least 1 review
- See [docs/workflow.md](docs/workflow.md) for branch naming and commit conventions
