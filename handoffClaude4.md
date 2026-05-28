# Claude Handoff 4 — Real-Robot Kalman EKF Deployment & Tuning Attempts

**Date:** 2026-05-27
**Branch:** `slam_simulation_debug`
**Platform:** Puzzlebot Differential Drive — Jetson Orin (sensors) + PC (all compute)
**Previous handoff:** `handoffClaude3.md`

---

## Objective

Deploy the Kalman EKF pipeline (validated in simulation in handoffClaude3) on the **physical robot** and diagnose issues encountered:

1. `/map` not appearing in RViz despite slam_node running
2. Robot not moving with `teleop_twist_keyboard` when full stack is active
3. Very large uncertainty ellipse in RViz (σ >> simulation values)
4. Robot odometry spinning in circles during turns

---

## Bugs Fixed This Session

### Fix 1 — `slam_publishes_map_odom` condition wrong for `kalman:=true`

**File:** `src/puzzlebot_bringup/launch/real_robot.launch.py`

**Symptom:** RViz showed "No map received" despite `ros2 topic echo /map --once` returning valid data (`width=86, height=108`). The map existed but RViz couldn't locate it because the `map→odom` TF was not being published by anyone.

**Root cause:** With `kalman:=true aruco:=true`:
- `aruco_map_odom` was disabled (correct — Kalman handles ArUco internally)
- `slam_node` had `publish_map_odom_tf: false` because the condition was `aruco==false AND kalman==false`
- Nobody published `map→odom` → RViz had no transform to place the map

**Fix:** Updated `slam_publishes_map_odom` condition:
```python
# BEFORE — wrong: slam disabled when aruco:=true regardless of kalman
"'", aruco_en, "' == 'false' and '", mcl_en, "' == 'false' and '", kalman_en, "' == 'false'"

# AFTER — correct: slam disabled only when aruco_map_odom is actually running
"('", aruco_en, "' == 'false' or '", kalman_en, "' == 'true') and '", mcl_en, "' == 'false'"
```

**TF ownership table (corrected):**

| Mode | aruco_map_odom | slam_node publishes map→odom |
|------|---------------|------------------------------|
| `kalman:=false aruco:=true` | ✅ active | ❌ no |
| `kalman:=true aruco:=any` | ❌ disabled | ✅ yes ← was broken |
| `kalman:=false aruco:=false` | ❌ disabled | ✅ yes |
| `mcl:=true` | ❌ disabled | ❌ no (mcl owns it) |

---

### Fix 2 — ArUco detection range expanded

**File:** `src/puzzlebot_bringup/launch/real_robot.launch.py`

`max_detection_distance: 1.8 → 2.5 m` and `max_incidence_angle_deg: 65 → 75°`

**Note:** These were later **reverted by the user** to `1.8 m / 65°`. Current values in the file are the original ones. See "Pending Work" for the recommended values.

---

## Issues Encountered & Attempted Fixes (Reverted by User)

### Issue A — Large uncertainty ellipse on real robot

**Observation:** RViz showed σ_x ≈ 50+ cm uncertainty ellipse (vs ~1–4 cm in simulation). Map walls were recognizable but robot position was poorly localized.

**Root cause analysis:**
- `max_detection_distance: 1.8 m` is too small — robot moves >1.8 m from all markers frequently
- ArUco corrections only at ~1–2 Hz (vs 10 Hz oracle in simulation)
- 10 seconds without ArUco → P_xx += 0.01×10 = 0.1 m² → σ ≈ 32 cm
- 30 seconds without ArUco → σ ≈ 55 cm

**Changes attempted (reverted):**

| Parameter | Tried | Effect | Status |
|-----------|-------|--------|--------|
| `max_detection_distance` | 1.8→2.5 m | More frequent corrections | **Reverted** |
| `max_incidence_angle_deg` | 65→75° | Detects markers at more angles | **Reverted** |
| `process_noise_x/y` | 0.01→0.02 | Larger K → stronger corrections | **Reverted** |
| `process_noise_theta` | 0.005→0.05 | Too aggressive: σ_θ grew 10× | **Reverted** |

---

### Issue B — Robot odometry spinning in circles during turns

**Observation:** When the user made a turn with `teleop_twist_keyboard`, the odometry in RViz showed the robot spinning continuously.

**Root cause 1 — `wheel_separation` wrong value:**
The official manual gives the geometric/nominal wheel separation (0.18 m). The **effective** separation (wheel contact points on ground, accounting for bearing play, tire compression, ICR shift) is always larger. With 0.18 m:
- `ω_odom = (vR - vL) / 0.18` overestimates angular velocity
- Small encoder velocity differences during straight driving → large angular drift in odometry

Tested: `0.18` → spinning. `0.19` → tilted map but recognizable. User kept 0.18 (from manual). **Current value: 0.18.**

**Root cause 2 — R_theta too small (ArUco yaw overcorrection):**
In `kalman_filter_node.cpp`:
```cpp
// aruco_cb — R construction
Mat3 R = {
    cov[0],  0.0,    0.0,
    0.0,     cov[7], 0.0,
    0.0,     0.0,    cov[35] > 1e-9 ? cov[35] : R_[8]  // ← issue
};
```
`aruco_node` reports `cov[35]` (yaw variance) ≈ 2.25e-4 rad² (σ ≈ 0.86°), based on its internal `sigma_yaw = 0.015`. But at 2+ m distance and oblique angles on the real camera, actual yaw uncertainty is ±5–15°. The Kalman trusts the ArUco yaw too much → K_theta ≈ 1 → sudden yaw jumps on each detection → appears as spinning in RViz.

**Code fix attempted (reverted by user):**
```cpp
// Proposed fix — enforce minimum R floor from meas_noise_theta
double r_x     = std::max(cov[0],                          R_[0]);
double r_y     = std::max(cov[7],                          R_[4]);
double r_theta = std::max(cov[35] > 1e-9 ? cov[35] : R_[8], R_[8]);
```
With `meas_noise_theta: 0.07 rad²` (σ_min ≈ 15°), K_theta drops from ~0.98 to ~0.12 → smooth yaw corrections. **This fix was reverted by the user. The current code still uses raw `cov[35]` without floor.**

---

### Issue C — Robot not moving with full stack running

**Observation:** With teleop + micro_ros_agent only → robot moves. With full stack → robot doesn't move.

**Root cause (probable):** DDS multicast saturation — with many ROS 2 nodes running, discovery traffic floods the Jetson's network. The micro_ros_agent may drop `/cmd_vel` packets. A `fastdds_puzzlebot.xml` profile was created to reduce multicast traffic.

**Status:** Not fully resolved. Workaround: run with minimal nodes first to test cmd_vel, then add others. The DDS profile file was created but not tested with the Jetson IP configured.

---

## Current State of Files

### `src/puzzlebot_bringup/launch/real_robot.launch.py`
- ✅ `kalman:=true/false` argument support (from handoffClaude3)
- ✅ `slam_publishes_map_odom` condition fixed (this session)
- ✅ `odometry_direct`, `odometry_raw`, `kalman_filter_node` nodes wired
- ⚠️ `max_detection_distance: 1.8 m` (user reverted — see Pending Work)
- ⚠️ `max_incidence_angle_deg: 65.0°` (user reverted — see Pending Work)

### `src/puzzlebot_bringup/config/kalman_params.yaml`
- ✅ `init_from_aruco: true` (bootstrap from first ArUco)
- ✅ `initial_x/y/theta: 0.0` with documentation (ignored when `init_from_aruco: true`)
- ⚠️ **All noise parameters reverted to original values:**
  ```yaml
  process_noise_x:     0.01
  process_noise_y:     0.01
  process_noise_theta: 0.005
  meas_noise_x:        0.05
  meas_noise_y:        0.05
  meas_noise_theta:    0.01
  ```

### `src/puzzlebot_localization/src/kalman_filter_node.cpp`
- ✅ Q·dt fix (from handoffClaude3) — still in place
- ⚠️ R_theta floor fix **reverted** — `meas_noise_theta` is still only a fallback, not a floor

### `src/puzzlebot_bringup/config/robot_params.yaml`
```yaml
wheel_radius:      0.05   # not physically calibrated
wheel_separation:  0.18   # official manual — causes angular drift in practice
```

### `fastdds_puzzlebot.xml` (NEW — root of workspace)
DDS profile to reduce multicast discovery traffic. Requires editing `<address>` with real Jetson IP before use. Not yet tested.

---

## What Is Working

- ✅ `slam_node` publishes `/map` and `map→odom` TF correctly with `kalman:=true aruco:=true`
- ✅ Kalman EKF boots from first ArUco detection (`init_from_aruco: true`)
- ✅ Map of the physical arena builds recognizably (walls visible, shape correct)
- ✅ `/odom` at 50 Hz from Kalman pipeline
- ✅ ArUco corrections reach the Kalman when camera faces a marker

---

## What Is NOT Working / Known Issues

### 1. `wheel_separation` not calibrated — primary drift source
The official manual value (0.18 m) causes angular over-estimation. The empirical value 0.19 m produced a slightly tilted but coherent map. Neither value is the physically correct effective separation.

**Pending calibration procedure:**
```bash
# 1. Mark starting orientation on floor
# 2. Command 5 full rotations at slow speed (0.3 rad/s = 104.7 s for 5 turns)
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{angular: {z: 0.3}}" --rate 10
# 3. Measure actual rotation with protractor
# 4. Apply correction:
#    ws_corrected = ws_current × (5 × 360°) / actual_angle_degrees
```

**Files to edit after calibration:**
- `src/puzzlebot_bringup/config/robot_params.yaml` → `wheel_separation: <measured_value>`

### 2. ArUco yaw corrections too aggressive (K_theta ≈ 1)
The `meas_noise_theta: 0.01` in `kalman_params.yaml` **is not applied as a floor** — it only activates when `cov[35]` from ArUco is zero. The actual yaw uncertainty at 1.5–2.5 m and oblique angles is ±5–15°, not the ±0.86° claimed by `aruco_node`.

**Recommended pending fix in `kalman_filter_node.cpp`:**
```cpp
// In aruco_cb, replace current R construction:
Mat3 R = {
    cov[0],  0.0,    0.0,
    0.0,     cov[7], 0.0,
    0.0,     0.0,    cov[35] > 1e-9 ? cov[35] : R_[8]
};

// With floor-enforced version:
double r_x     = std::max(cov[0],                          R_[0]);
double r_y     = std::max(cov[7],                          R_[4]);
double r_theta = std::max(cov[35] > 1e-9 ? cov[35] : R_[8], R_[8]);
Mat3 R = {r_x, 0.0, 0.0,  0.0, r_y, 0.0,  0.0, 0.0, r_theta};
```
Then set `meas_noise_theta: 0.07` (σ_min ≈ 15°) in `kalman_params.yaml` to prevent yaw from jumping on every ArUco detection.

### 3. ArUco detection range possibly too small
`max_detection_distance: 1.8 m` was reverted after a failed experiment. However, the center of the arena (1.88, 2.43 m) is >1.9 m from all 5 markers. This means the robot in the center of the arena has no ArUco coverage.

**Recommended values:**
```python
# In real_robot.launch.py, aruco_node parameters:
'max_detection_distance': 2.5,   # was 1.8
'max_incidence_angle_deg': 75.0, # was 65.0
```
These are conservative enough to avoid noisy detections but expand coverage to the full arena.

### 4. cmd_vel not reaching robot with full stack (DDS saturation)
When running: SLAM + ArUco + Kalman + RViz + scan_restamper simultaneously, `/cmd_vel` from teleop may not reach the Jetson micro_ros_agent reliably. Workaround: launch with reduced node set first.

**DDS profile file:** `fastdds_puzzlebot.xml` (workspace root) — fill in Jetson IP and export before launching:
```bash
export FASTRTPS_DEFAULT_PROFILES_FILE=~/Documents/puzzlebot_sim/fastdds_puzzlebot.xml
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

---

## Parameter Tuning Reference

### Tuning `process_noise_theta` (how fast angular uncertainty grows)
| Value | σ_θ after 5s no ArUco | Effect |
|-------|----------------------|--------|
| 0.005 | 0.16 rad = 9° | Very slow growth — trusts odometry heavily |
| 0.01 | 0.22 rad = 13° | **Current** — balanced |
| 0.05 | 0.50 rad = 28° | Too fast — was tried and reverted |

### Tuning `meas_noise_theta` (minimum ArUco yaw trust — only if floor fix applied)
| Value | σ_min_theta | K_theta (typical) | Effect |
|-------|------------|-------------------|--------|
| 0.01 | 5.7° | ~0.98 | ArUco fully trusted — **current, can spin** |
| 0.05 | 12.9° | ~0.30 | Moderate smoothing |
| 0.07 | 15.2° | ~0.12 | Stable, recommended after floor fix |
| 0.15 | 22.3° | ~0.05 | Very conservative — slow convergence |

### `wheel_separation` empirical behavior
| Value | Observed behavior |
|-------|------------------|
| 0.18 | Spinning in odometry (overestimates angular velocity) |
| 0.19 | Tilted map (~5–10°) but coherent — empirically best so far |
| 0.20 | Would underestimate turns — map would curve opposite direction |

---

## Build Status

```
puzzlebot_bringup      ✓  (slam_publishes_map_odom fix)
puzzlebot_localization ✓  (Q·dt fix from handoffClaude3 — R_theta floor reverted)
puzzlebot_description  ✓
puzzlebot_perception   ✓
puzzlebot_slam         ✓
```

```bash
cd ~/Documents/puzzlebot_sim && source install/setup.bash
colcon build --packages-select puzzlebot_bringup puzzlebot_localization \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
```

---

## Launch Commands (current working state)

```bash
# Estrategia B — Kalman EKF + ArUco (principal):
ros2 launch puzzlebot_bringup real_robot.launch.py \
  kalman:=true aruco:=true lidar_topic:=/scan

# Protocolo de arranque:
# 1. Colocar robot mirando a cualquier ArUco marker (< 1.5 m)
# 2. Lanzar el sistema
# 3. Esperar: "✅ Pose inicial desde ArUco: x=... y=... theta=..."
# 4. Mover robot

# Mapeo clásico (sin Kalman, modo probado anteriormente):
ros2 launch puzzlebot_bringup real_robot.launch.py \
  aruco:=true lidar_topic:=/scan

# Debug cmd_vel (stack mínimo):
ros2 launch puzzlebot_bringup real_robot.launch.py \
  kalman:=true aruco:=false slam:=false rviz:=false lidar_topic:=/scan
```

---

## Recommended Next Steps (Priority Order)

1. **[Alta] Calibrar `wheel_separation`** — sin esto, toda corrección ArUco compensa un error sistemático que seguirá volviendo. Hacer el test de 5 rotaciones con el robot sobre la pista real.

2. **[Alta] Aplicar R_theta floor fix** en `kalman_filter_node.cpp` — el spinning durante giros es consecuencia directa de K_theta ≈ 1. Esta es la corrección de código más impactante pendiente.

3. **[Media] Volver a ampliar detección ArUco** — `max_detection_distance: 2.5 m` y `max_incidence_angle_deg: 75°`. Se revirtieron junto con otros cambios pero son independientes y necesarios.

4. **[Media] Resolver cmd_vel con stack completo** — configurar el DDS profile con la IP del Jetson y probar. Si no resuelve, identificar el nodo específico que satura (probar deshabilitando RViz, ArUco, SLAM de uno en uno).

5. **[Baja] `wheel_radius` calibración** — medir distancia real en línea recta vs comandada. Afecta velocidad lineal pero no el drift angular.
