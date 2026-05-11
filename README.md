# Puzzlebot Simulation — ROS 2 Humble Workspace

ROS 2 Humble workspace for Puzzlebot with Gazebo Fortress (ignition-gazebo 6) physics simulation, SLAM mapping, MCL localization, and weekly homework assignments.

## Workspace Structure

```
puzzlebot_sim/
├── src/
│   ├── puzzlebot_description/      # URDF, SDF, meshes, RViz configs, worlds
│   ├── puzzlebot_bringup/          # Gazebo Fortress launch files
│   ├── puzzlebot_slam/             # SLAM mapping + dead reckoning + MCL
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

## Gazebo Fortress Simulation

The workspace includes a complete Gazebo Fortress (ignition-gazebo 6) simulation stack with two worlds: flat plane and maze.

### Launch Gazebo Simulation

Start the simulation (choose one):

```bash
# Flat plane world — dead-reckoning odometry only
ros2 launch puzzlebot_bringup gz_sim.launch.py

# Maze world with MCL localization (recommended for testing)
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze

# Maze world — build a map from lidar scans (recommended mapping mode)
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze mode:=mapping

# Mapping with wheel odometry instead of Gazebo ground truth
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze mode:=mapping odom_source:=dead_reckoning

# Without RViz visualization
ros2 launch puzzlebot_bringup gz_sim.launch.py rviz:=false

# Headless (no GUI)
ros2 launch puzzlebot_bringup gz_sim.launch.py gui:=false

# Disable SLAM (dead_reckoning + MCL)
ros2 launch puzzlebot_bringup gz_sim.launch.py slam:=false
```

### Test Robot Movement with Teleop

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

### Simulation Architecture

| Component | Purpose |
|---|---|
| **Gazebo Fortress** | Physics engine (ODE), sensor simulation |
| **robot_state_publisher** | Publishes TF tree from URDF |
| **ros_gz_bridge** | Bidirectional ROS ↔ Gazebo message bridge |
| **dead_reckoning** | Differential-drive odometry from joint states |
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
