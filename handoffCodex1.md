# Codex Handoff 1

## Objective

The user asked to improve the real-robot SLAM/localization pipeline for a Puzzlebot running on ROS 2. The main goals were:

- Reduce odometry drift on the physical robot.
- Fuse wheel odometry with absolute ArUco pose corrections.
- Make the `real_robot.launch.py` pipeline coherent for real hardware.
- Fix LiDAR orientation issues where walls appeared mirrored front/back or left/right.
- Improve occupancy-grid mapping quality, especially duplicated walls after repeated passes, turns, and wheel slip.
- Evaluate whether scan matching should be enabled and tune it conservatively.

The user explicitly does not yet have path following implemented. The focus has been mapping/localization, not navigation.

## Current State

The current intended architecture is:

- `odometry_node` owns continuous wheel odometry:
  - Topic: `/odom`
  - TF: `odom -> base_footprint`
- `aruco_node` estimates absolute robot pose from known ArUco markers:
  - Topic: `/aruco/pose`
  - Frame: `map`
- `aruco_map_odom` converts absolute ArUco pose plus current odometry into:
  - TF: `map -> odom`
  - Topic: `/map_to_odom`
- `scan_restamper` restamps and corrects LiDAR scans:
  - Input default: `/Lidar`
  - Output: `/scan_stamped`
  - Frame rewritten to `lidar_link`
  - Current real launch defaults:
    - `invert_lidar := false`
    - `lidar_yaw_offset := 3.14159265359`
- `slam_node` subscribes to `/scan_stamped`, `/odom`, and `/map_to_odom`, then publishes `/map`.
- `mcl` exists as a future/localization mode after saving a map:
  - Use with `slam:=false mcl:=true`.

The user reported a major improvement after the ArUco correction pipeline and LiDAR angle fixes:

- Robot motion direction became correct.
- ArUco corrections helped recover pose.
- Mapping became much better.
- Remaining problem: duplicated walls/features after multiple passes, slow turns, or wheel slip.

Latest reported issue:

- After the last attempted mapping/scan-matching changes, the robot visualization appeared to keep spinning or the odometry/map pose kept rotating.
- This likely came from feedback between scan matching and `map->odom` correction.

Important current observation from the files as of this handoff:

- `slam_node.py` still updates its internal `map->odom` estimate from scan matching in every scan via `_update_map_odom_tf(map_pose, odom_pose)`.
- `real_robot.launch.py` sets `publish_map_odom_tf` false when ArUco or MCL owns TF, but `slam_node.py` still updates the internal correction used by `_odom_to_map_pose()`.
- This means even if SLAM is not publishing the TF, scan matching can still feed back into the internal map-frame pose used for the next scan. That can explain the apparent continuous rotation.

## Files Worked On

### `src/puzzlebot_bringup/launch/real_robot.launch.py`

Purpose:

- Reworked the real-robot launch pipeline around one continuous odometry source and one global correction source.

Key current behavior:

- Launch arguments include:
  - `slam`
  - `mcl`
  - `avoidance`
  - `aruco`
  - `viewer`
  - `rviz`
  - `lidar_topic`
  - `invert_lidar`
  - `lidar_yaw_offset`
- `odometry_node` publishes `/odom` and TF.
- `aruco_map_odom` is enabled when `aruco:=true` and `mcl:=false`.
- `aruco_node` is enabled when `aruco:=true`.
- `scan_restamper` publishes `/scan_stamped`.
- `slam_node` remaps `/scan` to `/scan_stamped`.
- `publish_map_odom_tf` for `slam_node` is computed so SLAM does not publish `map->odom` when ArUco or MCL owns it.
- `mcl` is available as a separate localization mode.

Important launch defaults:

- `invert_lidar`: `false`
- `lidar_yaw_offset`: `3.14159265359`

These were selected after the user observed LiDAR points were inverted. The final working combination for LiDAR orientation appeared to be:

- Do not reverse scan order.
- Add pi radians yaw offset.

### `src/puzzlebot_localization/scripts/scan_restamper`

Purpose:

- Restamp incoming `LaserScan` messages with PC time.
- Rewrite `frame_id` to `lidar_link`.
- Optionally correct LiDAR orientation.

Changes/concepts:

- Added/used `invert_angles`.
- Added/used `angle_offset_rad`.
- Publishes to `/scan_stamped`.

Current behavior:

- If `invert_angles` is true:
  - Reverses ranges and intensities.
  - Mirrors `angle_min` and `angle_max`.
- If `angle_offset_rad` is nonzero:
  - Adds the offset to `angle_min` and `angle_max`.

What worked:

- `angle_offset_rad = pi` fixed front/back inversion.
- `invert_angles = false` was needed to avoid reintroducing left/right inversion.

### `src/puzzlebot_localization/scripts/aruco_map_odom`

Purpose:

- New node that publishes global correction:
  - `T_map_odom = T_map_base_from_aruco * inv(T_odom_base_current)`

Current behavior:

- Subscribes to `/odom`.
- Subscribes to `/aruco/pose`.
- Publishes TF `map -> odom`.
- Publishes `/map_to_odom`.
- Smooths corrections with `correction_alpha`.
- Rejects large jumps using:
  - `max_correction_step_m`
  - `max_correction_step_yaw`
- Rejects corrections if current odometry is too old.
- Rejects ArUco poses outside map bounds.

What worked:

- This significantly improved pose recovery.
- It moved the system away from correcting odometry directly and instead correctly put global corrections in `map -> odom`.

### `src/puzzlebot_localization/CMakeLists.txt`

Purpose:

- Installed the new script `aruco_map_odom`.

### `src/puzzlebot_perception/puzzlebot_perception/aruco_node.py`

Purpose:

- Estimate absolute robot pose from known ArUco markers.

Important current logic:

- Converts between ROS `camera_link` convention and OpenCV optical frame convention.
- Uses:
  - `T_map_marker`
  - `T_camera_optical_marker`
  - `T_base_camera_optical`
- Computes:
  - `T_map_base = T_map_marker * inv(T_camera_optical_marker) * inv(T_base_camera_optical)`
- Filters detections by:
  - known marker IDs
  - marker area
  - detection distance
  - incidence angle
  - position/yaw jump
  - map bounds
- Fuses multiple markers with distance weighting.

What worked:

- ArUco corrections became much more useful after camera optical frame handling and map-bound filtering.

### `src/puzzlebot_bringup/config/aruco_map.yaml`

Purpose:

- Defines marker map for absolute pose correction.

Notes:

- The physical map dimensions discussed were approximately:
  - width: `3.76 m`
  - height: `4.86 m`
- The user also mentioned `4.86 x 3.76`, but the current pipeline treats X as `3.76` and Y as `4.86`.

### `src/puzzlebot_bringup/config/camera_calibration.yaml`

Purpose:

- Camera intrinsics and distortion used by ArUco pose estimation.

### `src/puzzlebot_bringup/config/camera_extrinsics.yaml`

Purpose:

- Camera extrinsics from robot base to camera.

Notes:

- This is critical for ArUco pose. If this is off, the robot pose will be biased even with correct marker detections.

### `src/puzzlebot_bringup/config/slam_params.yaml`

Purpose:

- Controls occupancy-grid geometry, inverse sensor model, keyframes, and scan matching.

Current important values:

```yaml
map_resolution: 0.05
map_width_meters: 4.26
map_height_meters: 5.36
map_origin_x: -0.25
map_origin_y: -0.25
p_occ: 0.80
p_free: 0.45
l_clamp: 5.0
scan_step: 1
max_range_factor: 0.95
min_useful_range: 0.20
max_mapping_range: 5.5
use_keyframes: true
keyframe_min_translation: 0.10
keyframe_min_rotation: 0.0873
scan_matching_enabled: true
```

What worked:

- Rectangular map geometry with margins better matches the real track.
- Conservative `p_free` helped avoid erasing walls too aggressively.
- Keyframes reduced repeated noisy scan integration.

What did not fully work:

- Duplicates still appear after multiple passes and especially during turns/slip.
- Scan matching is enabled, but current architecture may allow scan matching to feed back into the internal map correction and cause unstable rotation.

### `src/puzzlebot_slam/puzzlebot_slam/occupancy_grid_map.py`

Purpose:

- Log-odds occupancy grid and LiDAR ray integration.

Current behavior:

- Supports rectangular maps via `width_pixels`, `height_pixels`, and `resolution`.
- Supports LiDAR extrinsic offsets:
  - `lidar_x`
  - `lidar_y`
  - `lidar_yaw`
- Limits mapping range with `max_mapping_range`.
- Treats max-range/no-hit beams as free-space updates.

What worked:

- Rectangular map and range limiting improved map consistency.

### `src/puzzlebot_slam/puzzlebot_slam/scan_matcher.py`

Purpose:

- Local scan-to-map matching.

Current behavior:

- Warmup scans: `12`
- Angular search:
  - Coarse: plus/minus `8 deg`, step `2 deg`
  - Fine: plus/minus `1.5 deg`, step `0.5 deg`
- Translation search:
  - plus/minus `0.05 m`, step `0.05 m`
- Uses every third valid ray.
- Uses the current map's positive log-odds cells as score.
- Respects `max_mapping_range`.
- Uses LiDAR extrinsic offsets from the occupancy grid.

What partially worked:

- Helped improve map alignment somewhat.

What did not fully work:

- It is still too easy for scan matching to choose a plausible but wrong yaw in symmetric or partially built map regions.
- During wheel slip/slow turns, scan matching can amplify errors if it is allowed to affect the running map-frame pose.

### `src/puzzlebot_slam/puzzlebot_slam/slam_node.py`

Purpose:

- Main mapping node.

Current behavior:

- Uses `/odom` buffered by timestamp.
- Receives `/map_to_odom`.
- Converts odom pose to map pose using internal `map->odom`.
- Runs scan matching from that initial map pose.
- Rejects scan matching corrections larger than:
  - `0.18 m`
  - `10 deg`
- Calls `_update_map_odom_tf(map_pose, odom_pose)` after each scan.
- Integrates scans only when keyframe thresholds pass.
- Publishes `/map`.
- Publishes `map->odom` only if `publish_map_odom_tf` is true.

Important caveat:

- Even when `publish_map_odom_tf` is false, `_update_map_odom_tf()` still changes the internal correction used by `_odom_to_map_pose()`.
- This is likely unsafe when ArUco is supposed to own `map->odom`.

### `src/puzzlebot_localization/src/odometry_node.cpp`

Purpose:

- Wheel odometry from encoders.

Notes:

- The intended design is for this node to remain local and continuous.
- It should not be corrected by ArUco directly.
- Global correction should remain in `map -> odom`.

### `src/puzzlebot_localization/src/kalman_filter_node.cpp`

Purpose:

- Older/alternative localization fusion node.

Current status:

- The real robot pipeline moved away from relying on this as the main source.
- The user still calls the improvements "Kalman" in conversation, but the current real-robot architecture is more accurately:
  - wheel odometry for `odom -> base`
  - ArUco map correction for `map -> odom`
  - SLAM map integration using the corrected map pose

## What Worked

- Separating continuous odometry from global correction:
  - `odom -> base_footprint` from wheel odometry.
  - `map -> odom` from ArUco.
- Correcting camera optical frame handling in ArUco pose estimation.
- Adding map-bound filtering to ArUco corrections.
- Adding `aruco_map_odom`.
- Fixing LiDAR orientation with:
  - `invert_lidar := false`
  - `lidar_yaw_offset := pi`
- Rectangular occupancy grid matching the real map dimensions.
- Keyframe-based scan integration.
- Conservative inverse sensor model:
  - stronger occupancy hits
  - conservative free-space clearing
- Scan matching improved mapping in some cases.

## What Did Not Work Or Was Risky

### 1. Scan matcher influencing `map->odom` every scan

This is the biggest architectural risk.

Symptom:

- The robot appeared to keep rotating/spinning in RViz after the latest scan matching/integration changes.

Likely cause:

- `slam_node` updates internal `map->odom` from scan matching every scan.
- ArUco also provides `map->odom`.
- Even if SLAM does not publish TF, it uses the internally updated correction for the next scan's initial pose.
- This creates a feedback loop:
  - map pose estimate changes
  - scan matcher sees a new initial pose
  - matcher changes yaw again
  - internal correction changes again

Recommended fix:

- When ArUco or MCL owns `map->odom`, scan matching should not update `map->odom`, even internally.
- Scan matching should only refine the pose used for deciding whether/how to integrate the scan, or it should be disabled until the map is stable.

### 2. Integration gate experiment

I attempted an integration gate concept:

- Track latest odometry linear/angular speed.
- Skip map integration during high angular velocity.
- Skip some scans after large `/map_to_odom` corrections.

The idea was valid for reducing duplicated walls during slip/turns, but the first implementation interacted badly with the existing scan matcher feedback and the robot appeared to spin.

This should be retried only after fixing the scan matcher feedback issue.

### 3. Mapping during turns and slip

The remaining duplicate walls are consistent with:

- Wheel slip during turns.
- Delayed or sparse ArUco corrections.
- Scan matching in an underconstrained or symmetric map.
- Integrating scans while the robot is rotating too fast.
- Using a scan timestamp/pose that is close but still slightly mismatched.

## What I Was Going To Do Next

Recommended next steps, in order:

### Step 1: Make `map->odom` ownership explicit

Add a parameter to `slam_node`, for example:

```yaml
scan_match_updates_map_odom: false
```

Then change `slam_node.py` so `_update_map_odom_tf(map_pose, odom_pose)` is only called when both conditions are true:

```python
if self._publish_map_odom_tf and self._scan_match_updates_map_odom:
    self._update_map_odom_tf(map_pose, odom_pose)
```

For the real robot with ArUco enabled, this should be false.

This should stop scan matching from creating a `map->odom` yaw feedback loop.

### Step 2: Move the integration gate before scan matching

If the robot is currently rotating too fast or a recent ArUco correction just happened, skip both:

- scan matching
- map integration

Do not run scan matching on scans that will not be integrated anyway.

Suggested parameters:

```yaml
enable_integration_gate: true
max_integrate_linear_speed: 0.25
max_integrate_angular_speed: 0.35
map_correction_cooldown_scans: 2
correction_cooldown_min_translation: 0.08
correction_cooldown_min_yaw: 0.0524
```

Tune:

- If duplicates remain during turns, lower `max_integrate_angular_speed` to `0.25`.
- If too many gaps appear, raise it to `0.45`.

### Step 3: Make scan matching less aggressive

Recommended scan matcher direction:

- Keep translation search very small or disable it initially.
- Prefer yaw-only matching until the map is stable.
- Require a stronger confidence criterion before accepting corrections.
- Compare best score vs initial pose score and accept only if improvement is meaningful.

Possible acceptance rule:

- Accept match only if:
  - correction distance less than `0.10 m`
  - correction yaw less than `5 deg`
  - best score improves over initial pose score by a ratio or absolute margin

### Step 4: Add diagnostic logging

Add throttled logs for:

- current odometry speed
- whether integration was skipped
- scan match correction distance/yaw
- scan match accepted/rejected
- source of `map->odom` correction

This will make it easier to prove whether duplicates come from:

- odometry drift
- scan matching
- ArUco correction jumps
- LiDAR timestamping
- wrong LiDAR extrinsics

### Step 5: Validate TF publishers live

When debugging, run:

```bash
ros2 run tf2_ros tf2_echo map odom
ros2 run tf2_ros tf2_echo odom base_footprint
ros2 topic echo /map_to_odom --once
ros2 topic info /tf -v
```

Expected:

- Only one strong owner should publish `map -> odom`:
  - ArUco during mapping with markers, or
  - MCL during localization on a saved map, or
  - SLAM only if both ArUco and MCL are disabled.
- `odom -> base_footprint` should be owned by `odometry_node`.

### Step 6: Save a map and switch to MCL for localization

Once a map is acceptable:

```bash
ros2 run nav2_map_server map_saver_cli -f ~/puzzlebot_map
```

Then use localization mode:

```bash
ros2 launch puzzlebot_bringup real_robot.launch.py slam:=false mcl:=true avoidance:=false viewer:=false lidar_topic:=/scan
```

This should be more robust for repeated runs than continuing to update the map forever.

## Suggested Immediate Patch

The first patch I would apply next is:

1. Add `scan_match_updates_map_odom` parameter to `slam_node.py`.
2. Set it to false in `slam_params.yaml`.
3. Prevent `_update_map_odom_tf()` from being called unless SLAM is explicitly the owner of `map->odom`.

This is lower risk than retuning scan matching first because it removes the suspected feedback loop.

## Commands Used For Verification

At different points, these were used or recommended:

```bash
python3 -m py_compile src/puzzlebot_slam/puzzlebot_slam/slam_node.py
colcon build --packages-select puzzlebot_slam puzzlebot_bringup
ros2 launch puzzlebot_bringup real_robot.launch.py --show-args
ros2 topic echo /scan_stamped --once
ros2 run tf2_ros tf2_echo map odom
ros2 topic echo /map_to_odom --once
```

## Operational Notes

- Use `source install/setup.bash` after each build.
- For the real LiDAR, the current working launch options should be:

```bash
ros2 launch puzzlebot_bringup real_robot.launch.py lidar_topic:=/scan invert_lidar:=false lidar_yaw_offset:=3.14159265359
```

- If using the micro-ROS `/Lidar` topic instead, keep the same LiDAR orientation parameters unless the scan direction changes.
- If RViz shows the map moving with the robot, check `map -> odom` ownership first.
- If walls are mirrored, check `scan_restamper` first, not SLAM.
- If pose jumps only when ArUco is visible, inspect `aruco_node` pose output and `aruco_map_odom` rejection logs.
- If duplicates appear mostly while turning, add/restore an integration gate after fixing scan matcher feedback.

