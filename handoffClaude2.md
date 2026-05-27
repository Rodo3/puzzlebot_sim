# Claude Handoff 2 — SLAM Drift Analysis & Improvement Proposals

**Date:** 2026-05-27
**Branch:** `feat/aruco-detection`
**Platform:** Puzzlebot Differential Drive — Jetson Orin (sensors) + PC (all compute)
**Previous handoff:** `handoffClaude1.md`

---

## Objective

The user reported two problems after the previous session's changes:

1. **Angular drift in the SLAM map** — when building the occupancy grid with pure odometry, the map develops a rotational drift over time. Walls that should be straight become curved or duplicated. The robot appears to "spin" continuously in RViz even when stationary.
2. **Teleop is noticeably slower** than before the recent SLAM changes (integration gate + scan matching guard added in the previous session).

The user attempted to fix the spinning/drift issue by:
- Preventing `scan_matching_enabled` from updating `map→odom` (`scan_match_updates_map_odom: false`)
- Adding an integration gate that skips map writes during fast rotation (`enable_integration_gate: true`, `max_integrate_angular_speed: 0.35 rad/s`)
- Adding a cooldown counter after ArUco correction jumps

Neither change significantly improved map quality. The user asked for a **full analysis of the codebase** and concrete proposals to fix the drift, before writing any code.

---

## Current State (at start of this session)

### What was working
- Wheel odometry (midpoint integration, C++): solid, tested
- ArUco detection pipeline: functional at ≤1.8 m, ≤75° incidence
- `aruco_map_odom` node: computes smoothed `map→odom` from ArUco absolute pose + wheel odom
- SLAM node builds a recognizable map of the test track (3.76 × 4.86 m)
- The two-session workflow (SLAM mapping → save → MCL localization) is architecturally correct

### What was broken / problematic
- Map has angular drift accumulating over a full traversal of the track
- Robot appears to rotate in RViz (confirmed was scan matcher feedback loop)
- Teleop response felt slower after adding the integration gate Python logic
- `scan_matching_enabled: true` in `slam_params.yaml` despite the comment saying it should be false for mapping

---

## Files Analyzed This Session (Read-Only — No Code Changes Made)

| File | Why analyzed |
|------|-------------|
| `src/puzzlebot_slam/puzzlebot_slam/slam_node.py` | Main SLAM orchestration — integration gate, scan matching wiring, map→odom TF logic |
| `src/puzzlebot_slam/puzzlebot_slam/scan_matcher.py` | Two-phase coarse/fine rotation + translation search, scoring function |
| `src/puzzlebot_bringup/config/slam_params.yaml` | Active parameter values — discovered `scan_matching_enabled: true` was still on |
| `src/puzzlebot_localization/src/odometry_node.cpp` | Wheel odometry implementation — midpoint integration, wheel_separation usage |
| `src/puzzlebot_localization/src/kalman_filter_node.cpp` | EKF node — predictor (odom) + corrector (ArUco), already implemented but not in launch |
| `src/puzzlebot_perception/puzzlebot_perception/aruco_node.py` | ArUco pipeline — detection, per-axis covariance, pose fusion |
| `src/puzzlebot_bringup/config/aruco_map.yaml` | 5-marker layout, physical dimensions of the track |
| `src/puzzlebot_bringup/launch/real_robot.launch.py` | Node graph, TF ownership, launch arguments |
| `src/puzzlebot_bringup/config/kalman_params.yaml` | Kalman tuning, start pose, `init_from_aruco` flag |

---

## Root Cause Analysis

### 1. Angular drift — primary cause: `wheel_separation` systematic error

The `odometry_node` computes angular velocity as:
```
omega = (vR - vL) / wheel_separation
```

If the physically-measured `wheel_separation` differs from the configured `0.172 m` by even 5 mm, every wheel-driven rotation accumulates a proportional angular error. For a full 360° turn:
- Error = `Δl / l` × 360° — e.g., 5 mm error on 172 mm → 10.5° per full turn

Over a complete lap of the track (~3 180° turns), this adds ~31° of unrecoverable yaw drift. No scan matcher or ArUco correction can compensate once the map is built with rotated wall segments.

**This parameter has been noted as inconsistent across files:**
- `robot_params.yaml` / `odometry_node` param: `0.172 m`
- Comments in `handoffClaude1.md`: mentions `0.18` and `0.19` as alternative values seen elsewhere
- No physical calibration has been performed yet (see "Physical Calibration Still Needed" in handoffClaude1.md)

### 2. Angular drift — secondary cause: scan_to_map feedback loop

With `scan_matching_enabled: true` and `scan_match_updates_map_odom: false` (current state):
- The scan matcher corrects the `map_pose` used for ray integration
- But `map_pose` is derived from `odom_pose` + the stored `map_odom_*` correction
- If the map already has rotated wall segments (from odometry drift), the matcher finds a best-fit yaw that aligns the scan to the *wrong* (drifted) map
- This "confirmed" pose is then integrated into the map, reinforcing the drift
- Next scan matches against an even more drifted map → progressive error amplification

The scoring function in `scan_matcher.py` maximizes grid occupancy hits. When the map is symmetric (corridor walls), small angular errors can produce the same score as the correct pose — the matcher cannot distinguish them.

### 3. Teleop slowdown — cause: Python SLAM scan callback blocking

The scan callback `_scan_cb` now executes (with `scan_matching_enabled: true`):
1. OdometryBuffer lookup (fast)
2. `_should_integrate_scan()` (fast)
3. `LocalScanMatcher.match()` — ~47 pose evaluations, each hitting the NumPy grid twice

At 10 Hz LiDAR with Python GIL, this can consume 30–50% of a CPU core. The `rclpy.spin()` loop shares the same thread as all callbacks. The `/cmd_vel` subscription processing is delayed by the scan callback duration.

Additionally, `enable_integration_gate: true` with `max_integrate_linear_speed: 0.25 m/s` means the gate fires frequently during normal driving, producing `get_logger().info()` calls at every skipped scan — logging itself adds overhead in Python ROS nodes.

### 4. Two publishers on `odom → base_footprint`

Confirmed by reading both files:
- `odometry_node.cpp` line 201–213: publishes TF when `publish_tf: true` (set in launch)
- `kalman_filter_node.cpp` line 222–229: also publishes `odom → base_footprint`

The Kalman node is **not in the current launch**, so there is currently no conflict. But if someone re-adds the Kalman node without removing `publish_tf: true` from `odometry_node`, both will fight over the transform. This is a latent bug.

---

## What Has NOT Worked (history from user + this analysis)

| Attempt | Why it failed |
|---------|--------------|
| `scan_match_updates_map_odom: false` | Prevented TF feedback loop, but didn't fix drift because the matcher still influences which pose is integrated into the map |
| `enable_integration_gate: true` | Skipped map writes during rotation, reducing some artifacts, but the drift was already accumulated from straight-line segments |
| `max_integrate_angular_speed: 0.35` | Gate fires too frequently, causing sluggish RViz map updates + slow scan callback |
| Integration cooldown after ArUco jumps | Correct in theory but ArUco is rarely visible during mapping (robot moves away from markers quickly) |

---

## Proposals Generated (Not Yet Implemented)

These are ordered by expected impact. **No code was written this session** — the user asked for analysis first.

---

### 🔴 Proposal A — Physical calibration of `wheel_separation` (highest impact, no code)

**The single most impactful fix.** All algorithmic improvements are bounded by the quality of the odometry model.

**Procedure:**
1. Place the robot on a flat surface. Mark the starting orientation with tape.
2. Send `omega = 2π` commands via `/cmd_vel` (one full 360° rotation) at slow speed (~0.3 rad/s).
3. Measure actual rotation angle with a protractor or ArUco pose comparison.
4. If actual = A° and commanded = 360°:
   ```
   wheel_separation_corrected = wheel_separation_current × (360 / A)
   ```
5. Repeat 3× and average. Update `wheel_separation` in `robot_params.yaml`.
6. Do the same for `wheel_radius` using straight-line distance calibration.

**Expected result:** Angular drift reduced from ~30°/lap to <5°/lap.

**Files to edit:**
- `src/puzzlebot_bringup/config/robot_params.yaml`

---

### 🔴 Proposal B — Disable scan matching during mapping (immediate fix, 1 line)

`slam_params.yaml` currently has `scan_matching_enabled: true`. This should be `false` for the mapping session. Scan matching on a progressively-building map creates the reinforcement loop described in Root Cause §2.

**The comment in the file already says this should be false for mapping** but the value was left at `true`.

**File to edit:**
```
src/puzzlebot_bringup/config/slam_params.yaml
  scan_matching_enabled: false   # was: true
```

**Expected result:** Stops the secondary drift amplification. Map quality depends solely on odometry quality.

---

### 🟡 Proposal C — Activate `kalman_filter_node` in the pipeline (medium complexity)

The `kalman_filter_node` is already implemented, built, and tested. It fuses wheel odometry (prediction) with ArUco measurements (correction via EKF update step). This is the standard approach for mobile robots with external landmarks.

Currently, the pipeline is:
```
odometry_node → /odom → slam_node
aruco_map_odom → map→odom TF (smoothed, not EKF)
```

The proposed pipeline:
```
odometry_node → /odom_raw → kalman_filter_node → /odom → slam_node
aruco_node    → /aruco/pose → kalman_filter_node (EKF correction)
kalman_filter_node → odom→base_footprint TF
```

**Advantages over current `aruco_map_odom`:**
- EKF properly weights corrections by measurement uncertainty (per-axis covariance from `aruco_node`)
- Prediction step (odom) fills in between ArUco sightings with correct uncertainty growth
- The SLAM node receives an already-corrected `/odom` — map is built in the corrected frame
- Eliminates the `aruco_map_odom` α-smoothing which introduces lag

**Changes needed:**
1. `real_robot.launch.py`:
   - Re-add `kalman_filter_node` node
   - Change `odometry_node` → `odom_topic: /odom_raw`, `publish_tf: false`
   - Change `kalman_filter_node` to be the TF publisher for `odom → base_footprint`
   - Remove or repurpose `aruco_map_odom` (or keep for map→odom via `/map_to_odom` topic)
2. `kalman_params.yaml`: already ready, no changes needed

**Risk:** The `kalman_filter_node` uses a fixed R matrix for odom noise. If the robot slips or the wheel velocity is noisy, the EKF can diverge. The current `aruco_map_odom` approach is more robust to momentary noise because it only updates when ArUco is detected.

---

### 🟡 Proposal D — Scan-to-scan matching for local yaw correction (medium complexity)

Instead of matching each scan against the growing occupancy map (scan-to-map), match consecutive scans against each other (scan-to-scan). This gives a local odometry correction without the feedback loop problem.

**How it would work:**
- Store the previous valid scan
- For each new scan: run the rotation phase of the current scan matcher against the *previous scan's ray endpoints* projected into a local frame
- The yaw difference is a relative correction to the odometry delta
- Apply this as `theta_corrected += (scan_match_dyaw - odom_dyaw) × alpha`

**Advantages:**
- No circular dependency on the map being correct
- Works even with an empty map (warmup not needed)
- Pure yaw correction — does not fight with translation odometry

**Disadvantages:**
- Does not fix absolute pose (still drifts globally)
- Scan-to-scan matching is noisier than scan-to-map in long corridors
- Needs careful handling of static scenes (robot stationary → score meaningless)

**Files to modify:**
- `src/puzzlebot_slam/puzzlebot_slam/scan_matcher.py` — add `match_scan_to_scan()` method
- `src/puzzlebot_slam/puzzlebot_slam/slam_node.py` — call scan-to-scan in `_scan_cb`, apply correction to `_map_odom_yaw`

---

### 🟢 Proposal E — Two-session workflow improvements (low complexity, immediate quality boost)

The current SLAM workflow is: **map while driving freely** → save → localize with MCL.

A better workflow with the existing infrastructure:
1. **Pre-mapping pass**: drive slowly (< 0.15 m/s) in straight lines along each wall, pausing in front of each ArUco marker. This gives the map clean wall segments and multiple ArUco anchor resets.
2. **Enable ArUco anchoring during mapping**: when the robot is within 0.8 m of a known marker (detected), do a hard reset of the `map_odom` correction to the exact ArUco measurement instead of the smoothed correction. Use `correction_alpha = 1.0` for stationary or slow-moving robot.
3. **Reduce `keyframe_min_translation` to 0.05 m during straight sections** — more keyframes = more wall coverage per meter.

**No code changes required.** Just:
- Modify `aruco_map_odom` `correction_alpha: 0.35` → `1.0` temporarily, or add a distance-based alpha (close to marker → alpha=1.0, moving away → alpha=0.2)

---

### 🟢 Proposal F — Reduce scan callback overhead to fix teleop latency

**Changes to make teleop feel responsive again:**

1. In `slam_params.yaml`:
   ```yaml
   scan_matching_enabled: false   # already proposed in B
   enable_integration_gate: false # remove overhead
   ```
2. In `slam_node.py` timer:
   ```python
   # Change map publish from 0.5s to 1.0s — halves Python overhead
   self.create_timer(1.0, self._publish_map)
   ```
3. Suppress info logging in `_should_integrate_scan()`:
   - Change `get_logger().info(...)` to `get_logger().debug(...)` for the gate skip messages
   - These fire every skipped scan (up to 10 Hz) and Python logging is expensive

**Files to edit:**
- `src/puzzlebot_bringup/config/slam_params.yaml`
- `src/puzzlebot_slam/puzzlebot_slam/slam_node.py` lines 284–306 (log level change)

---

## Recommended Action Plan

**Immediate (before next robot session):**
1. Set `scan_matching_enabled: false` in `slam_params.yaml` (Proposal B)
2. Change gate/logging to `debug` level in `slam_node.py` (Proposal F)

**During next robot session:**
3. Perform wheel calibration (Proposal A) — measure actual vs commanded rotation
4. Update `wheel_separation` and `wheel_radius` in `robot_params.yaml`

**After calibration:**
5. Decide between:
   - Path 1: Keep `aruco_map_odom` architecture, do Proposal E (better mapping discipline)
   - Path 2: Integrate `kalman_filter_node` (Proposal C) for proper EKF fusion

**Optional (if still drifting after A+B+C):**
6. Implement scan-to-scan yaw correction (Proposal D)

---

## Files to Change (Next Session)

| File | Change | Proposal |
|------|--------|----------|
| `src/puzzlebot_bringup/config/slam_params.yaml` | `scan_matching_enabled: false`, `enable_integration_gate: false` | B, F |
| `src/puzzlebot_slam/puzzlebot_slam/slam_node.py` | `logger.info` → `logger.debug` in gate skip messages; map timer 0.5 → 1.0s | F |
| `src/puzzlebot_bringup/config/robot_params.yaml` | Update `wheel_separation` and `wheel_radius` after physical calibration | A |
| `src/puzzlebot_bringup/launch/real_robot.launch.py` | Re-add `kalman_filter_node`; change `odometry_node` to publish `/odom_raw`, `publish_tf: false` | C |
| `src/puzzlebot_bringup/config/kalman_params.yaml` | No changes needed — already correct | C |

---

## Known Latent Bugs (Not Urgent but Document)

1. **Double TF publisher:** `odometry_node` (`publish_tf: true`) and `kalman_filter_node` both publish `odom → base_footprint`. Currently no conflict (Kalman not in launch), but dangerous if both are launched.
2. **`scan_restamper` parameter mismatch:** Launch passes `invert_angles` and `angle_offset_rad` but the actual script may not handle these — needs verification.
3. **`wheel_separation` inconsistency:** Three different values appear across the codebase (0.172, 0.18, 0.19). Only `robot_params.yaml` matters at runtime; others are in comments/documentation.
4. **`kalman_filter_node` subscribes to `/odom_raw`** (hardcoded, line 86 of `kalman_filter_node.cpp`) but `odometry_node` publishes to `/odom` in the current launch. If Kalman is re-added, `odometry_node` must be changed to publish to `/odom_raw` first.

---

## Build Status

No code changes made this session. Build state is identical to `handoffClaude1.md`:
```
puzzlebot_bringup      ✓
puzzlebot_description  ✓
puzzlebot_localization ✓ (kalman_filter_node built but not in launch)
puzzlebot_perception   ✓
puzzlebot_slam         ✓
puzzlebot_testing      ✓
```

---

## Key Parameters Reference

```yaml
# slam_params.yaml — current values vs recommended
scan_matching_enabled:        true    # → should be false for mapping
scan_match_updates_map_odom:  false   # correct
enable_integration_gate:      true    # → false (causes latency)
max_integrate_linear_speed:   0.25    # only relevant if gate enabled
max_integrate_angular_speed:  0.35    # only relevant if gate enabled
p_occ:                        0.80    # ok
p_free:                       0.45    # ok
keyframe_min_translation:     0.10    # ok
keyframe_min_rotation:        0.0873  # 5° — ok

# robot_params.yaml — needs physical calibration
wheel_radius:                 0.0425  # m — calibrate with straight line test
wheel_separation:             0.172   # m — CRITICAL: calibrate with rotation test
```
