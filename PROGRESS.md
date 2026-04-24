# Puzzlebot Sim — Project Progress
**Course:** Robots Autónomos — ITESM 2026
**Timeline:** 6 weeks remaining (April – May 2026)
**Mode:** Simulation-first (Gazebo / RViz). Hardware transfer in week 6 only if time allows.

---

## Current Status: Sprint 1 — Topic Plumbing & Sim Bringup

---

## Package Map

| Package | Type | Status | What it contains |
|---|---|---|---|
| `puzzlebot_description` | CMake | ✅ Exists | URDF, meshes, RViz config |
| `puzzlebot_msgs` | CMake | ✅ Created | `HealthIndex.msg`, `ParticleArray.msg` |
| `puzzlebot_bringup` | Python | ✅ Updated | Launch files, YAML configs |
| `puzzlebot_localization` | CMake (C++) | ✅ Created | `odometry_node`, `kalman_filter_node` |
| `puzzlebot_slam` | Python | ✅ Created | `slam_node` (MCL particle filter) |
| `puzzlebot_perception` | Python | ✅ Created | `camera_node`, `aruco_node`, `yolo_node` |
| `puzzlebot_planning` | Python | ✅ Created | `path_planner_node` (A*), `obstacle_avoidance_node` |
| `puzzlebot_control` | Python | ✅ Created | `state_machine_node` |
| `puzzlebot_controller` | CMake (C++) | ✅ Created | `steering_controller_node` (pure pursuit) |
| `shared_utils` | Python | ⬜ Empty | Placeholder for shared helpers |
| `homework_01_transforms` | Python | ⬜ Legacy | Circular motion homework — not used in project |

---

## Sprint Checklist

### Sprint 1 — Week 1: Topic Plumbing & Sim Bringup
- [x] All packages scaffolded with correct structure
- [x] Config YAMLs created (`robot_params`, `kalman_params`, `slam_params`, `yolo_params`, `controller_params`)
- [x] Launch files created (`localization.launch.py`, `slam.launch.py`, `simulation.launch.py`)
- [ ] **Build the workspace** — `colcon build --symlink-install`
- [ ] Verify all packages compile without errors
- [ ] Confirm topics are live: `ros2 topic list` after launching
- [ ] `odometry_node` publishes `/odom_raw`
- [ ] `kalman_filter_node` publishes `/odom`
- [ ] Pose visible in RViz (base_footprint TF moving)

### Sprint 2 — Week 2: Odometry + EKF
- [ ] Robot drives in simulation (send `/cmd_vel` manually)
- [ ] `odometry_node` integrates encoder ticks correctly
- [ ] `kalman_filter_node` fuses odometry with ArUco corrections
- [ ] Pose drift < 5 cm over 3 m straight-line run
- [ ] Tune `kalman_params.yaml` (Q and R matrices)

### Sprint 3 — Week 3: SLAM
- [ ] `slam_node` subscribes to `/scan` and `/odom`
- [ ] `/map` topic publishes an `OccupancyGrid`
- [ ] Map visible in RViz
- [ ] Corridor walls visible in map after one pass
- [ ] Tune `slam_params.yaml` (particle count, noise)

### Sprint 4 — Week 4: Perception
- [ ] `aruco_node` detects markers and publishes `/aruco/poses`
- [ ] ArUco pose feeds into `kalman_filter_node` correction step
- [ ] `yolo_node` runs in sim (PyTorch mode, `use_trt:=false`)
- [ ] Detections published on `/detections` at ≥ 10 Hz
- [ ] Verify detection visible in RViz or logged

### Sprint 5 — Week 5: Planning + Control
- [ ] `path_planner_node` receives `/goal_pose` and outputs `/planned_path`
- [ ] A* path visible in RViz
- [ ] `steering_controller_node` follows path (pure pursuit)
- [ ] `obstacle_avoidance_node` halts robot when obstacle within 0.3 m
- [ ] `state_machine_node` transitions IDLE → NAVIGATING → DONE
- [ ] Full point-to-point autonomous run in Gazebo

### Sprint 6 — Week 6: Full Integration + (Optional) Hardware
- [ ] Complete autonomous mission: start → map → navigate → detect → dock
- [ ] All nodes stable for > 5 minute run
- [ ] **Hardware (if available):** deploy with `use_sim:=false`, tune YAML params only

---

## How to Build & Run

```bash
# From workspace root
cd ~/Documents/puzzlebot_sim
colcon build --symlink-install
source install/setup.bash

# Localization only (Sprint 1-2):
ros2 launch puzzlebot_bringup simulation.launch.py slam:=false perception:=false navigation:=false

# Full stack:
ros2 launch puzzlebot_bringup simulation.launch.py

# Useful debug commands:
ros2 topic list
ros2 topic echo /odom
ros2 topic echo /map --no-arr   # check map is publishing
ros2 param set /slam_node n_particles 500
```

---

## Known Issues / Notes

- `slam_node` map-update uses best-weight particle; performance depends on initial particle spread — may need tuning in Sprint 3.
- `yolo_node` falls back to PyTorch (`ultralytics`) in simulation; TensorRT path only activates on Jetson with `use_trt:=true`.
- `obstacle_avoidance_node` sits between `/cmd_vel_in` (steering controller output) and `/cmd_vel` (firmware input) — ensure remapping is correct in launch.
- `homework_01_transforms` package is legacy homework, not used by any project launch file.
