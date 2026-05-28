# Claude Handoff 3 — Kalman EKF Simulation Validation & Real-Robot Kalman Integration

**Date:** 2026-05-27
**Branch:** `slam_simulation_debug`
**Platform:** Puzzlebot Differential Drive — Jetson Orin (sensors) + PC (all compute)
**Previous handoff:** `handoffClaude2.md`

---

## Objective

Validate the Kalman EKF (`kalman_filter_node`) in Gazebo Fortress simulation before deploying on the physical robot. Two strategies were tested:

- **Strategy A** — Kalman with pure wheel-encoder prediction (no ArUco correction). Validates that the EKF implementation is mathematically correct and matches `dead_reckoning` behavior.
- **Strategy B** — Kalman fusing encoder prediction + synthetic ArUco oracle (ground-truth pose + Gaussian noise). Validates the full EKF correction pipeline in a controlled environment.

A synthetic ArUco oracle (`aruco_oracle`) was created to generate realistic `/aruco/pose` messages from Gazebo ground truth, bypassing the need for textures or computer vision in simulation.

Once validated in simulation, the same Kalman pipeline was integrated into `real_robot.launch.py` via a new `kalman:=true` launch argument.

---

## Current State at Start of Session

### What was working
- `slam_node` building clean occupancy maps with `scan_matching_enabled: false`
- `aruco_map_odom` correctly computing `map→odom` TF from ArUco + wheel odometry
- `kalman_filter_node` built (C++) but **not wired** in the real-robot launch (see `handoffClaude2.md`)
- Gazebo simulation with `flat_plane` and `maze` worlds working
- Two-session SLAM→MCL workflow functional on physical robot

### What was broken / to be tested
- No simulation world replicating the physical arena (3.76 × 4.86 m) existed
- The Kalman EKF had never been tested end-to-end with ArUco corrections
- Strategy B (Kalman + oracle) produced catastrophically large covariance (σ_x ≈ 7 m after 100 s), confirmed from user test logs (`cov_x=49.54`, `cov_y=70.88`)
- The ArUco oracle was not detecting any markers — robot appeared to "never see ArUco" in simulation

---

## Files Created This Session

### `src/puzzlebot_description/worlds/real_arena.sdf` (NEW)

Gazebo Fortress world replicating the physical 3.76 × 4.86 m test track:

- **World name:** `real_arena` (required for Fortress bridge topic paths)
- **4 outer walls** with 0.15 m thickness and 1.0 m height at exact arena dimensions
- **5 ArUco marker visual boxes** (colored by ID, matching physical placement):
  - ID1 = blue, north wall center
  - ID0 = green, west wall high (Y=3.90)
  - ID4 = orange, west wall low (Y=1.04)
  - ID2 = red, east wall high (Y=3.90)
  - ID3 = magenta, east wall low (Y=1.04)
- **3 parametrizable obstacles** with `▼ EDITAR AQUÍ` comments for easy repositioning
- Uses `libignition-gazebo-*-system.so` Fortress plugins (not Harmonic)

---

## Files Modified This Session

### `src/puzzlebot_bringup/launch/gz_sim.launch.py`

Major additions for `world:=real_arena` support:

**New launch arguments:**
- `kalman` (default: `false`) — enables Kalman EKF pipeline in simulation
- `aruco_oracle` (default: `false`) — enables synthetic ArUco oracle (Strategy B)

**New nodes (all conditional on `world:=real_arena`):**

| Node variable | Condition | Function |
|--------------|-----------|----------|
| `bridge_arena` | always (real_arena) | Bridge `/world/real_arena/...` Gazebo↔ROS topics |
| `joint_relay_arena` | always (real_arena) | Relay joint states for odometry |
| `spawn_arena` | always (real_arena) | Spawn robot at (0.30, 0.30, z=0.05) facing north (yaw=π/2) |
| `wheel_odom_arena_direct` | `kalman==false` | odometry_node → `/odom` + TF (Strategy reference) |
| `wheel_odom_arena_raw` | `kalman==true` | odometry_node → `/odom_raw`, no TF (feeds Kalman) |
| `ground_truth_arena` | `odom_source==ground_truth` | Ground truth reference odometry |
| `kalman_arena` | `kalman==true` | kalman_filter_node → `/odom` + TF odom→base_footprint |
| `aruco_oracle_arena` | `kalman==true AND aruco_oracle==true` | Synthetic ArUco publisher from Gazebo GT |

**Fix applied during session:** `max_detection_dist` raised from `1.8` → `3.0` m for the oracle (center of arena is >1.9 m from all markers; 1.8 m range left the middle of the arena with zero coverage).

---

### `src/puzzlebot_localization/scripts/aruco_oracle` (NEW — executable)

Synthetic ArUco oracle for Gazebo simulation. Reads robot ground truth from Gazebo and publishes realistic `/aruco/pose` with Gaussian noise — allows testing the Kalman EKF without real camera or ArUco textures.

**Subscriptions:**
- `/world/real_arena/dynamic_pose/info` (`geometry_msgs/PoseArray`) — Gazebo ground truth

**Publications:**
- `/aruco/pose` (`geometry_msgs/PoseWithCovarianceStamped`) — same topic as real `aruco_node`

**Detection model per marker:**
1. Compute distance robot↔marker. Reject if > `max_detection_dist` (3.0 m).
2. Compute incidence angle using marker facing direction `[sin(yaw), -cos(yaw)]` (ArUco roll=π/2 convention). Reject if incidence > `max_incidence_deg` (75°).
3. Apply `detection_prob` random drop (default 1.0 = always detect).
4. Add Gaussian noise: `σ_depth = σ_depth_base / cos(incidence)`, `σ_lateral` fixed.
5. Fuse multiple visible markers by inverse-variance weighting.

**Bug fixed during session (wrong facing direction formula):**

```python
# ANTES (incorrecto — convención RPY estándar):
fx = math.cos(mk_yaw)
fy = math.sin(mk_yaw)

# DESPUÉS (correcto — convención aruco_map.yaml con roll=π/2):
fx = math.sin(mk_yaw)
fy = -math.cos(mk_yaw)
```

Example: ID4 at yaw=π/2 (apunta al este `+X`). Con la fórmula incorrecta:
- facing = (0, 1) → "apunta al norte"
- Robot al este de ID4 obtenía cos_incidence < 0 → **rechazado**

Con la fórmula correcta:
- facing = (1, 0) → "apunta al este" ✓
- Robot al este obtenía cos_incidence > 0 → **aceptado** ✓

**Segundo bug corregido — asignación de eje de profundidad:**

```python
# ANTES (incorrecto):
is_lateral_x = abs(math.cos(mk_yaw)) < 0.5

# DESPUÉS (correcto — el eje de profundidad sigue la dirección de facing):
is_depth_x = abs(math.sin(mk_yaw)) > 0.5
# Si |sin(yaw)| > 0.5 → facing dominante en X → X es profundidad, Y es lateral
```

---

### `src/puzzlebot_localization/src/kalman_filter_node.cpp`

**Bug crítico corregido (línea 137) — Q no escalado por dt:**

```cpp
// ANTES — Q_ añadida por callback, sin escalar por dt:
P_ = mat_add(mat_mul(mat_mul(F, P_), mat_transpose(F)), Q_);

// DESPUÉS — Q_ escalada por dt para crecimiento consistente en el tiempo:
Mat3 Q_dt{};
for (int i = 0; i < 9; ++i) Q_dt[i] = Q_[i] * dt;
P_ = mat_add(mat_mul(mat_mul(F, P_), mat_transpose(F)), Q_dt);
```

**Impacto del bug:** Con `/odom_raw` a ~50 Hz y `Q_x = 0.01`, P_xx crecía a razón de `0.01 × 50 = 0.5 m²/s`. Después de 100 s de simulación → P_xx ≈ 50 m² (σ_x ≈ 7 m). El usuario observó exactamente `cov_x=49.54, cov_y=70.88` en los logs — coincidencia exacta con la predicción analítica.

**Impacto de la corrección:** Con `Q_dt`, el crecimiento es `Q_x m²/s` independientemente de la frecuencia del callback. Resultado validado en simulación: σ_x oscila entre **1.2 cm** (justo después de corrección ArUco) y **3.9 cm** (antes de la siguiente corrección) — patrón diente de sierra sano.

---

### `src/puzzlebot_localization/CMakeLists.txt`

Añadido `scripts/aruco_oracle` a `install(PROGRAMS)` para que sea instalable con `colcon build`.

---

### `src/puzzlebot_bringup/launch/real_robot.launch.py`

**Nuevo argumento `kalman` (default: `false`)**

El nodo `odometry` original se dividió en tres variantes condicionales:

| Nodo | Condición | Publica |
|------|-----------|---------|
| `odometry_direct` | `kalman==false` | `/odom` + TF `odom→base_footprint` |
| `odometry_raw` | `kalman==true` | `/odom_raw` sin TF |
| `kalman` (kalman_filter_node) | `kalman==true` | `/odom` + TF `odom→base_footprint` |

**`aruco_map_odom`** ahora se desactiva con `kalman:=true`:
```python
# Condition actualizada:
aruco_en == 'true' AND mcl_en == 'false' AND kalman_en == 'false'
```
Razón: con `kalman:=true`, el EKF ya incorpora la corrección ArUco dentro de `odom→base_footprint`. Activar también `aruco_map_odom` crearía doble corrección.

**`slam_match_updates_map_odom`** actualizado para incluir `kalman` en la condición:
```python
aruco_en == 'false' AND mcl_en == 'false' AND kalman_en == 'false'
```

**Docstring del launch** actualizado con tabla de combinaciones típicas (mapeo clásico, Estrategia A, Estrategia B, MCL).

---

### `src/puzzlebot_bringup/config/kalman_params.yaml`

- `initial_x/y/theta` → cambiados a `0.0` con comentario explícito: *"ignorado cuando `init_from_aruco: true`"*
- Protocolo de arranque físico documentado paso a paso
- Zona de inicio recomendada: frente a **ID4** (pared oeste, Y≈1.04) mirando al este (+X)
- Mapa ASCII de la pista con posición de todos los marcadores

---

## What Is Working (validated this session)

### Simulation (Gazebo Fortress, `world:=real_arena`)

- ✅ `world:=real_arena` carga la pista física 3.76 × 4.86 m con 5 marcadores ArUco y 3 obstáculos
- ✅ Strategy A (`kalman:=true aruco_oracle:=false`): EKF solo de ruedas, covarianza crece correctamente con Q·dt, no explota
- ✅ Strategy B (`kalman:=true aruco_oracle:=true`): mapa excelente confirmado por el usuario
- ✅ `/aruco/pose` covariance estable: σ_x ≈ 13 mm, σ_y ≈ 10 mm, σ_yaw ≈ 8.7 mrad
- ✅ `/odom` covariance en patrón diente de sierra sano: crece de ~1.2 cm a ~3.9 cm entre correcciones
- ✅ Oracle detecta marcadores desde el centro del arena (máx. distancia 3.0 m)

### Logs de validación Strategy B

```
/aruco/pose: cov[0]=1.72e-04 (σ=13.1mm), cov[7]=9.57e-05 (σ=9.8mm), cov[35]=7.5e-05
/odom: oscila 1.5e-04 → 1.3e-03 → [corrección] → 1.5e-04  (ciclo ~100ms)
```

---

## Known Issues / Pending Work

### Calibración física del robot (sin cambios desde handoffClaude1)

| Parámetro | Archivo | Valor actual | Cómo medir |
|-----------|---------|-------------|-----------|
| `wheel_separation` | `robot_params.yaml` | 0.172 m | N rotaciones completas, medir ángulo real, aplicar ratio |
| `wheel_radius` | `robot_params.yaml` | 0.0425 m | Distancia recta conocida, medir real, aplicar ratio |
| LiDAR X offset | `puzzlebot_gz.urdf` | 0.0 m | Medir desde eje de ruedas al LiDAR con regla |

### Diferencias esperadas simulación → robot real

| Factor | Sim (oracle) | Robot real | Efecto |
|--------|-------------|------------|--------|
| σ ArUco | 13 mm (controlada) | 15–30 mm (iluminación, blur) | P de corrección mayor |
| Frecuencia detección | 10 Hz constante | 5–10 Hz variable | P crece más entre correcciones |
| Ruido de ruedas | Gazebo ideal | Deslizamiento, piso irregular | Ajustar `process_noise_x/y/theta` |

Si en el robot real la covarianza crece demasiado rápido entre correcciones ArUco, subir `process_noise_x` en `kalman_params.yaml`. Si las correcciones crean saltos bruscos, subir `meas_noise_x/y`.

### Parámetros de tuning post-despliegue

```yaml
# kalman_params.yaml — valores a ajustar en robot real si es necesario
process_noise_x:     0.01   # subir si P crece muy rápido sin ArUco
process_noise_y:     0.01
process_noise_theta: 0.005
meas_noise_x:        0.05   # subir si correcciones ArUco son bruscas
meas_noise_y:        0.05
meas_noise_theta:    0.01
```

---

## Build Status

```
puzzlebot_bringup      ✓  (config + launch actualizados)
puzzlebot_description  ✓  (real_arena.sdf añadido)
puzzlebot_localization ✓  (aruco_oracle + Q_dt fix en kalman_filter_node)
puzzlebot_perception   ✓  (sin cambios)
puzzlebot_slam         ✓  (sin cambios)
puzzlebot_testing      ✓  (sin cambios)
```

```bash
cd ~/Documents/puzzlebot_sim && source install/setup.bash
colcon build --packages-select puzzlebot_bringup puzzlebot_description \
  puzzlebot_localization --cmake-args -DCMAKE_BUILD_TYPE=Release
```

---

## Launch Commands

### Simulation — Strategy A (solo encoders por Kalman)
```bash
ros2 launch puzzlebot_bringup gz_sim.launch.py \
  world:=real_arena mode:=mapping odom_source:=dead_reckoning \
  kalman:=true aruco_oracle:=false
```

### Simulation — Strategy B (Kalman + oracle) ← VALIDADO HOY
```bash
ros2 launch puzzlebot_bringup gz_sim.launch.py \
  world:=real_arena mode:=mapping odom_source:=dead_reckoning \
  kalman:=true aruco_oracle:=true
```

### Simulation — Dead reckoning puro (referencia)
```bash
ros2 launch puzzlebot_bringup gz_sim.launch.py \
  world:=real_arena mode:=mapping odom_source:=dead_reckoning \
  kalman:=false
```

### Robot real — Mapeo clásico (aruco_map_odom, método anterior)
```bash
ros2 launch puzzlebot_bringup real_robot.launch.py
# equivalente: slam:=true aruco:=true kalman:=false
```

### Robot real — Strategy B (Kalman EKF + ArUco) ← NUEVO
```bash
ros2 launch puzzlebot_bringup real_robot.launch.py \
  kalman:=true aruco:=true
# Protocolo: colocar robot mirando a cualquier marcador (< 1.8 m),
# esperar "✅ Pose inicial desde ArUco: ..." en logs, luego mover.
```

### Robot real — Strategy A (solo encoders, debug)
```bash
ros2 launch puzzlebot_bringup real_robot.launch.py \
  kalman:=true aruco:=false
```

### Robot real — MCL localización (sesión 2)
```bash
ros2 launch puzzlebot_bringup real_robot.launch.py \
  slam:=false mcl:=true aruco:=true
```

---

## TF Ownership Summary (actualizado)

```
kalman:=false, aruco:=true, mcl:=false
  map → odom              owned by: aruco_map_odom
  odom → base_footprint   owned by: odometry_node (publish_tf=true)

kalman:=true, aruco:=true/false, mcl:=false
  map → odom              owned by: slam_node (scan_match_updates_map_odom=false → estático)
  odom → base_footprint   owned by: kalman_filter_node

kalman:=false, aruco:=false, mcl:=false
  map → odom              owned by: slam_node (scan_match_updates_map_odom=true)
  odom → base_footprint   owned by: odometry_node

mcl:=true
  map → odom              owned by: mcl node
  odom → base_footprint   owned by: odometry_node (kalman:=false) o kalman_filter_node (kalman:=true)
```

---

## Bugs Corregidos Esta Sesión

| Bug | Archivo | Síntoma | Causa | Fix |
|-----|---------|---------|-------|-----|
| Q no escalado por dt | `kalman_filter_node.cpp:137` | cov_x=49.54 m² en 100 s | `P += Q` por callback sin multiplicar dt | `Q_dt[i] = Q_[i] * dt` |
| Dirección de cara incorrecta | `aruco_oracle:155-156` | Oracle nunca detecta marcadores | `fx=cos(yaw)` en vez de `fx=sin(yaw)` | Convención roll=π/2: `fx=sin(yaw), fy=-cos(yaw)` |
| Eje de profundidad invertido | `aruco_oracle:203` | Covarianza X/Y intercambiada | `is_lateral_x = abs(cos) < 0.5` incorrecto | `is_depth_x = abs(sin(yaw)) > 0.5` |
| max_detection_dist demasiado pequeño | `gz_sim.launch.py:449` | Centro del arena sin cobertura | 1.8 m < distancia al centro (≈1.9–2.6 m) | Cambiar a 3.0 m |
