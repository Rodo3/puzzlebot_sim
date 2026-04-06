# Puzzlebot Simulation — ROS 2 Humble Workspace

ROS 2 Humble workspace for Puzzlebot kinematic simulation, TF transforms, and weekly homework assignments.

## Workspace Structure

```
puzzlebot_sim/
├── src/
│   ├── puzzlebot_description/      # URDF, meshes, RViz config
│   ├── puzzlebot_bringup/          # Launch files
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

# Build
make build

# Source the workspace (every new terminal)
source /opt/ros/humble/setup.bash
source install/setup.bash

# Launch simulation in RViz
make rviz
```

See [docs/setup.md](docs/setup.md) for detailed setup instructions.

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
