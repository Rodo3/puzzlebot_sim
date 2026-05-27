# Claude Handoff — Puzzlebot Real-Robot Mapping & Localization

**Date:** 2026-05-27  
**Branch:** `feat/aruco-detection`  
**Platform:** Puzzlebot Differential Drive — Jetson Orin (sensors) + PC (all compute)

---

## Objective

Improve the real-robot mapping and localization stack for the Puzzlebot so that:

1. The robot can build a clean occupancy-grid map using wheel odometry + ArUco marker corrections.
2. The robot can then localize inside that saved map using MCL (Monte Carlo Localization).
3. Odometry drift is minimized enough to close loops reliably around the physical test track (3.76 × 4.86 m rectangle).
4. The ArUco detection pipeline correctly handles oblique viewing angles without discarding useful corrections.
5. The Kalman filter initializes from the first valid ArUco detection instead of a hardcoded pose.

---

## Current Architecture (as of handoff)

### Node graph (real robot)

```
Jetson:
  micro_ros_agent  →  /VelocityEncR, /VelocityEncL  (BEST_EFFORT)
  sllidar_ros2     →  /scan  (frame_id: laser)
  camera_node      →  /camera/image/compressed

PC:
  odometry_node        /VelocityEncR,L → /odom  +  TF odom→base_footprint
  scan_restamper       /scan           → /scan_stamped  (re-timestamps + frame_id=lidar_link)
  aruco_node           /camera/image/compressed → /aruco/pose  (PoseWithCovarianceStamped)
  aruco_map_odom       /aruco/pose + /odom → TF map→odom  (smoothed correction)
  slam_node            /scan_stamped + /odom → /map  (does NOT publish map→odom when aruco active)
  robot_state_publisher URDF → TF base_footprint→base_link→camera_link/lidar_link
```

### TF tree
```
map → odom (owned by aruco_map_odom when aruco:=true, by mcl when mcl:=true, by slam when both false)
      → base_footprint (owned by odometry_node, continuous wheel odometry)
           → base_link  (URDF)
                → camera_link  (URDF joint: x=0.152, z=0.044)
                → lidar_link   (URDF joint: z=0.15)
                     → laser   (static TF: identity, for sllidar driver compatibility)
```

**Key architectural decision made this session:** Wheel odometry owns the continuous `odom→base_footprint` TF. ArUco-derived absolute corrections go into `map→odom` via `aruco_map_odom` node. The Kalman filter (`kalman_filter_node`) was removed from the active launch pipeline — `aruco_map_odom` replaced it for the map→odom correction role. The `odometry_node` now publishes directly to `/odom` with `publish_tf: true`.

> **Note:** The user's commit "update" (10ad8c6) changed the launch file significantly — it removed the Kalman node from the pipeline, added `aruco_map_odom` as a separate node, added `invert_lidar` and `lidar_yaw_offset` args to `scan_restamper`, and changed `odometry_node` to publish TF directly. This means the `kalman_filter_node` changes made in this session (ArUco bootstrap initialization) are **built but not wired** in the current launch. The node exists but is not launched by default.

---

## Files Modified This Session

### `src/puzzlebot_bringup/config/kalman_params.yaml`
- Added `initial_x`, `initial_y`, `initial_theta` parameters (robot start pose in map frame).
- Added `init_from_aruco: true` — Kalman waits for first valid ArUco detection to set state.
- Default start pose: `(0.30, 0.30, π/2)` — 30 cm inside SW corner, facing north.

### `src/puzzlebot_localization/src/kalman_filter_node.cpp`
- Added `init_from_aruco` parameter declaration.
- Added `initialized_` and `init_from_aruco_` member variables.
- `odom_cb`: returns early (but updates `last_time_`) until `initialized_ = true`.
- `aruco_cb`: first call does a **direct reset** to the ArUco pose (gain=1, sets P from ArUco covariance). Subsequent calls do normal EKF fusion.
- Log message distinguishes "waiting for ArUco" vs "using static initial pose".
- **Status:** Built and working, but currently not wired in `real_robot.launch.py` (the user switched to `aruco_map_odom` architecture).

### `src/puzzlebot_perception/puzzlebot_perception/aruco_node.py`

#### Incidence angle filter (raised threshold 55° → 75°)
- Previously: detections with incidence > 55° were hard-rejected.
- Now: detections up to 75° are accepted but with **degraded covariance on the depth axis**.
- Rationale: a marker seen at 60° from the side still gives precise position along the axis parallel to the marker wall.

#### `transform_marker_to_robot_pose()` — returns 4-tuple
- Now returns `(x, y, yaw_robot, marker_yaw_in_map)` instead of `(x, y, yaw)`.
- `marker_yaw_in_map` is the yaw of the marker in the map frame, used to determine which world axis is the "depth axis" for that marker.

#### `_compute_covariance_per_axis()` — new method (replaces `_compute_covariance`)
- Computes separate `cov_x`, `cov_y`, `cov_yaw` for each detection.
- Depth axis uncertainty scales as `1/cos(incidence)` — capped at ×8 (cos_min = 0.125).
- North/south markers (yaw≈0 or π): Y is depth axis → `cov_y` degrades with oblique angle.
- East/west markers (yaw≈±π/2): X is depth axis → `cov_x` degrades with oblique angle.

#### `fuse_multiple_marker_poses()` — updated
- Now calls `_compute_covariance_per_axis()` per marker.
- Fuses covariances using weighted inverse-variance: `cov_x = W / Σ(w_i / cov_x_i)`.
- Log now shows `std_x` and `std_y` separately instead of a single `pos_std`.

#### All callers of `robot_pose` updated
- `candidates` dict, `filter_detections`, `_publish_debug_image` — all updated to use 4-tuple indexing.

### `src/puzzlebot_description/urdf/puzzlebot_gz.urdf`
- Added `camera_link` as a proper URDF link + `camera_joint` (fixed, base_link→camera_link at x=0.152, z=0.044).
- Added clarifying comment to `lidar_joint` about measuring physical X offset.
- Note: the `camera_tf` static TF in the launch file is now redundant with the URDF joint (both define the same transform). They are consistent in value.

### `src/puzzlebot_bringup/launch/real_robot.launch.py`
- User/linter modified this file after our session changes. Current state (see file):
  - Removed: `camera_tf` static TF (replaced by URDF joint).
  - Added: `camera_optical_tf` (`camera_link → camera_optical_frame`, for OpenCV convention).
  - Added: `aruco_map_odom` node (new architecture for map→odom correction).
  - Added: `invert_lidar` and `lidar_yaw_offset` launch args (passed to `scan_restamper`).
  - Changed: `odometry_node` now has `odom_topic: /odom` and `publish_tf: true` (no more Kalman in the chain).
  - Removed: `kalman_filter_node` from the launch (Kalman is built but not used).
  - Added: `slam_publishes_map_odom` logic — SLAM only owns map→odom TF when neither aruco nor mcl is active.
  - Removed: old STEP 1/STEP 2 documentation header.

### `src/puzzlebot_localization/scripts/aruco_map_odom` (NEW — user added)
- Python node that computes `map→odom` from ArUco absolute poses + current wheel odometry.
- Formula: `T_map_odom = T_map_base_measured * inv(T_odom_base_current)`
- Applies exponential smoothing (`correction_alpha: 0.35`) to avoid sudden TF jumps.
- Sanity checks: rejects poses outside map bounds, rejects jumps > 0.35 m or 0.70 rad.
- First detection is accepted directly (bootstrap initialization).
- Publishes TF `map→odom` at 20 Hz + `/map_to_odom` topic.

### `src/puzzlebot_bringup/config/slam_params.yaml`
- `scan_matching_enabled: false` — intentional for mapping session (avoids divergence on empty map).
- `p_free: 0.40`, `keyframe_min_translation: 0.10`, `keyframe_min_rotation: 0.0873` (5°).

### `src/puzzlebot_bringup/config/mcl_params.yaml` (NEW)
- Full MCL parameters with two-session workflow documentation.
- `num_particles: 300`, `top_k: 100`, `score_rays: 36`, `noise_xy/theta: 0.03`.

### `src/puzzlebot_testing/` (NEW PACKAGE)
- `puzzlebot_testing/odometry_validator.py` — passive odometry validation node.
- **Usage:** run alongside teleop; detects lap completion automatically.
- **Initialization:** waits for first valid `/aruco/pose` and uses that as the origin (not a hardcoded pose).
- **Lap detection:** two-phase logic — must exit `min_exit_radius` (1.5 m) before regress to `lap_close_radius` (0.4 m) counts as a lap close.
- **Reports per lap:** XY close error, yaw error, total distance, ArUco correction count + mean delta.
- **Final summary:** auto-diagnosis of `wheel_radius` and `wheel_separation` errors with suggested correction value.
- `setup.py`: includes `scripts/odometry_validator` wrapper for `ros2 run` compatibility.

---

## What Is Working

- **Wheel odometry** is robust (midpoint integration, calibrated `wheel_radius=0.0425`, `wheel_separation=0.172`).
- **ArUco detection and pose estimation** works reliably when markers are within ~1.8 m and <75° incidence.
- **`aruco_map_odom` node** correctly computes and smooths the `map→odom` TF from absolute ArUco measurements.
- **SLAM mapping** builds clean occupancy maps with the current `scan_matching_enabled: false` setting.
- **Odometry validator** correctly detects lap closures and measures drift when initialized from ArUco.
- User confirmed: "ya mejoro un buen el mapeo, si corrige gracias a kalman, y va haciendo un mapeo bastante bueno" — mapping quality is significantly better after ArUco corrections and architectural cleanup.

---

## What Has Not Worked / Known Issues

### Scan matching disabled
`scan_matching_enabled: false` in `slam_params.yaml`. This was intentional to avoid the "sunburst" divergence pattern that occurred when the scan matcher ran on an empty or sparse map. **Scan matching is still not enabled** — enabling it requires a warm map (enough scans integrated) before the matcher is useful. There is a `WARMUP_SCANS = 8` guard but it wasn't enough in practice with early divergence from incorrect frame handling (since fixed).

### Kalman node not in pipeline
The `kalman_filter_node` was improved this session (ArUco bootstrap init, per-axis covariance fusion) but the user switched to a different architecture (`aruco_map_odom` + direct wheel odometry). The Kalman node is **built and functional** but not wired in the launch. Its improvements could be reintegrated if the user wants EKF-style fusion instead of the current smoothed-correction approach.

### LiDAR X offset unknown
The URDF sets the LiDAR at `x=0.0` relative to `base_link`. The physical robot may have a forward offset. This was flagged as "measure and update" but no measurement was taken. A non-zero X offset causes the SLAM map to have systematic positional error between the robot center and LiDAR origin.

### `scan_restamper` new parameters not yet in CMakeLists
The user's version of `scan_restamper` now accepts `invert_angles` and `angle_offset_rad` parameters (passed from launch). These are wired in the launch but the actual script implementation needs to be verified to handle them — the version in the repo (`scripts/scan_restamper`) may not have these parameters yet.

---

## What Was Planned Next (SLAM Improvement)

The user asked: "dime que mas sugieres para mejorar el mapeo" (what do you suggest to improve mapping). The answer was interrupted. The planned analysis was:

### Option A — Enable scan matching progressively
- Keep `scan_matching_enabled: false` for the first ~15 scans (warmup).
- After warmup (map has walls), enable scan matching automatically.
- This would use `LocalScanMatcher` in `scan_matcher.py` which already has the two-phase rotation+translation search.
- Risk: still needs correct initial pose from ArUco bootstrap to avoid matching against wrong map area.

### Option B — ICP / better scan matcher
- The current scan matcher does a brute-force angular search (±12° coarse, ±2° fine) + translation search (±20 cm).
- A proper ICP (Iterative Closest Point) would be more accurate but more expensive in Python.
- Alternative: use `scan_to_scan` matching (ICP between consecutive scans) to refine odometry before map integration, independent of the occupancy grid.

### Option C — Keyframe-based loop closure
- When the robot returns to a known area (detected by ArUco), force a scan match against the existing map at that location.
- This would explicitly close loops using ArUco as the trigger.

### Option D — Tune existing parameters
Before implementing new algorithms, tune:
- `p_occ` / `p_free` ratio in `slam_params.yaml` — current 0.75/0.40 may be creating thick walls.
- `keyframe_min_translation` / `keyframe_min_rotation` — more keyframes = denser map but more noise.
- `max_range_factor` — currently 0.95; reducing to 0.85 would discard more long-range noisy readings.

### Option E — Use `map_to_odom` topic in SLAM
The `aruco_map_odom` node publishes `/map_to_odom` as a `TransformStamped` topic in addition to the TF. The SLAM node currently reads `/odom` for prediction but uses the TF tree for the actual map→odom transform. Explicitly subscribing to `/map_to_odom` in the slam node would make the correction more direct and timestamped.

---

## Build Status

All packages build cleanly as of handoff:
```
puzzlebot_bringup      ✓
puzzlebot_description  ✓
puzzlebot_localization ✓
puzzlebot_perception   ✓
puzzlebot_testing      ✓ (new package)
```

Run commands:
```bash
cd ~/Documents/puzzlebot_sim && source install/setup.bash
colcon build --packages-select puzzlebot_bringup puzzlebot_description \
  puzzlebot_localization puzzlebot_perception puzzlebot_testing \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
```

---

## Launch Commands

### Session 1 — Mapping
```bash
ros2 launch puzzlebot_bringup real_robot.launch.py \
  avoidance:=false viewer:=false lidar_topic:=/scan
```

### Session 2 — MCL Localization (after saving map)
```bash
ros2 launch puzzlebot_bringup real_robot.launch.py \
  slam:=false mcl:=true avoidance:=false viewer:=false lidar_topic:=/scan
```

### Odometry Validator
```bash
# Run alongside mapping session
ros2 run puzzlebot_testing odometry_validator
# Point robot at any ArUco marker first — it auto-initializes from first detection
```

---

## Physical Calibration Still Needed

| Parameter | File | Current Value | How to measure |
|-----------|------|--------------|----------------|
| `wheel_separation` | `robot_params.yaml` | 0.172 m | Command N full rotations, measure actual angle, apply ratio correction |
| `wheel_radius` | `robot_params.yaml` | 0.0425 m | Command known distance, measure actual, apply ratio |
| LiDAR X offset | `puzzlebot_gz.urdf` | 0.0 m | Measure from wheel axle center to LiDAR center along robot X axis |
| Camera extrinsics | `camera_extrinsics.yaml` | x=0.152, z=0.044 | Verify with ruler; must match `camera_joint` in URDF |
