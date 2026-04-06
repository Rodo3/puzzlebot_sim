# Setup Guide

## Prerequisites

- Ubuntu 22.04 (native or WSL2)
- ROS 2 Humble Hawksbill
- Python 3.10+

## Install ROS 2 Humble

Follow the [official ROS 2 Humble installation guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html).

## Install workspace dependencies

```bash
sudo apt install python3-colcon-common-extensions python3-rosdep python3-numpy
rosdep update
pip3 install --upgrade transforms3d
```

> **Note:** Install `transforms3d` via pip, not apt (`python3-transforms3d`).
> The apt package is incompatible with NumPy ≥ 1.24.

## Clone and build

```bash
git clone <repo-url>
cd puzzlebot_sim
rosdep install --from-paths src --ignore-src -r -y
make build
source install/setup.bash
```

## WSL2 Notes

- Enable WSLg for GUI support (Windows 11 or WSL2 with X server)
- Alternatively install VcXsrv on Windows and set `DISPLAY=:0`
- RViz requires a display — ensure `DISPLAY` is set correctly

## Build and Run

```bash
# Build all packages
make build

# Source the workspace (every new terminal)
source /opt/ros/humble/setup.bash
source install/setup.bash

# Launch simulation (RViz + TF publisher)
make rviz
```

## Troubleshooting

| Problem | Solution |
|---|---|
| `colcon build` fails with missing deps | Run `rosdep install --from-paths src --ignore-src -r -y` |
| `option --uninstall not recognized` | Stale symlink-install state — run `make clean && make build` |
| `option --editable not recognized` | Do NOT use `colcon build --symlink-install` (setuptools ≥ 64 incompatible) |
| `np.float` AttributeError in transforms3d | Run `pip3 install --upgrade transforms3d` |
| RViz doesn't open | Check `DISPLAY` env variable is set |
| `package not found` after build | Run `source install/setup.bash` |
| Mesh not visible in RViz | Ensure `puzzlebot_description` built successfully |
| Wheels have no transform in RViz | The `joint_state_publisher` node is not running — check terminal for errors |
