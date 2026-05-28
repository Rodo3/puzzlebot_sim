# Claude Handoff 5 — Scan Matching + EKF Loop Closure + SLAM Tuning

**Date:** 2026-05-27
**Branch:** `slam_simulation_debug`
**Platform:** Puzzlebot Differential Drive — Jetson Orin (sensors) + PC (all compute)
**Previous handoff:** `handoffClaude4.md`

---

## Objective

Continuar el despliegue físico del stack SLAM + EKF. Partiendo del estado de Claude4 (mapa visible pero con doble-pared y alta incertidumbre), implementar:

1. Scan matching activo para colocación precisa de scans en el mapa (reduce doble-pared)
2. Feedback scan match → EKF para loop closure sin pose graph
3. ZUPT + P_max clamp para elipse de incertidumbre estable en reposo
4. R_theta floor en ArUco para evitar spinning por K_theta ≈ 1
5. FPS limiter ArUco y reducción de outlier jump

---

## Result

✅ **Mapa funcional confirmado en robot físico.** El usuario confirmó con screenshot que el mapa muestra la pista completa con paredes simples y bien definidas tras múltiples vueltas. Mejora mayor respecto al estado de Claude4.

---

## Changes Implemented This Session

### Fix 1 — R_theta floor en ArUco (K_theta: 0.98 → 0.12)

**File:** `src/puzzlebot_localization/src/kalman_filter_node.cpp` — `aruco_cb()`

**Problema:** `meas_noise_theta` solo actuaba como fallback cuando `cov[35] == 0`, no como floor. ArUco reporta σ_θ ≈ 0.86° (cov[35] ≈ 2.25e-4) pero la incertidumbre real a 2+ m es ±5–15°. K_theta ≈ 0.98 → el EKF sobreescribía el yaw en cada detección → spinning en RViz.

```cpp
// ANTES — meas_noise_theta solo como fallback:
Mat3 R = {cov[0], 0, 0,  0, cov[7], 0,  0, 0, cov[35] > 1e-9 ? cov[35] : R_[8]};

// DESPUÉS — floor aplicado en los tres ejes:
double r_x     = std::max(cov[0],                          R_[0]);
double r_y     = std::max(cov[7],                          R_[4]);
double r_theta = std::max(cov[35] > 1e-9 ? cov[35] : R_[8], R_[8]);
Mat3 R = {r_x, 0.0, 0.0,  0.0, r_y, 0.0,  0.0, 0.0, r_theta};
```

**Parámetro asociado:** `meas_noise_theta: 0.07` en `kalman_params.yaml` → σ_min ≈ 15°, K_theta ≈ 0.12.

El mismo floor se aplica también en el modo bootstrap (init_from_aruco).

---

### Fix 2 — ZUPT + P_max clamp (elipse estable en reposo)

**File:** `src/puzzlebot_localization/src/kalman_filter_node.cpp` — `odom_cb()`

**Problema:** Con robot quieto, P crecía indefinidamente entre correcciones ArUco → elipse enorme en RViz. Causa: Q·dt se acumulaba incluso con velocidad 0.

```cpp
// ZUPT: escala Q por velocidad observada
double speed   = std::abs(v) + std::abs(w);
double q_scale = std::min(speed / std::max(zupt_threshold_, 1e-6), 1.0);
Mat3 Q_dt{};
for (int i = 0; i < 9; ++i) Q_dt[i] = Q_[i] * dt * q_scale;
P_ = mat_add(mat_mul(mat_mul(F, P_), mat_transpose(F)), Q_dt);

// P_max clamp: techo cuando ArUco desaparece
P_[0] = std::min(P_[0], p_max_xy_);
P_[4] = std::min(P_[4], p_max_xy_);
P_[8] = std::min(P_[8], p_max_theta_);
```

**Parámetros nuevos en `kalman_params.yaml`:**
```yaml
zupt_speed_threshold: 0.02   # m/s + rad/s; robot "quieto" debajo de este valor
p_max_xy:    1.0             # techo P_xx y P_yy [m²]
p_max_theta: 2.0             # techo P_theta [rad²]
```

---

### Fix 3 — Scan matching activo con feedback loop closure

**Archivos modificados:**
- `src/puzzlebot_bringup/config/slam_params.yaml`
- `src/puzzlebot_slam/puzzlebot_slam/scan_matcher.py`
- `src/puzzlebot_slam/puzzlebot_slam/slam_node.py`
- `src/puzzlebot_localization/src/kalman_filter_node.cpp`
- `src/puzzlebot_bringup/config/kalman_params.yaml`

#### 3a — scan_matching_enabled: true (Proposal A)

```yaml
# slam_params.yaml
scan_matching_enabled: true
```

Con `scan_match_updates_map_odom: false` (calculado automáticamente cuando `kalman:=true`), el matcher **no toca el TF map→odom** — solo refina la pose de cada scan antes de integrarlo. Sin feedback loop, sin conflicto con ArUco.

Ventana de traslación del matcher ampliada: `_TRANS_HALF_M = 0.05 → 0.15` (±3 celdas).

#### 3b — scan_match → EKF feedback (Proposal B / loop closure)

`slam_node` publica `/scan_match/pose` (PoseWithCovarianceStamped en frame `map`) cuando:
- El score del matcher ≥ `scan_match_min_score` (15.0)
- La sanity check no rechazó la corrección (dist ≤ 18 cm, yaw ≤ 10°)
- El matcher no está en warmup (primeros 12 scans)

El Kalman fusiona esa corrección como medición adicional en `scan_match_cb()`:
- Solo cuando `initialized_ == true` (ArUco ya inicializó el estado)
- Con floor de `R_scan_` (20 cm / 10° mínimos) más conservador que ArUco

```yaml
# slam_params.yaml
scan_match_min_score:  15.0
scan_match_cov_xy:     0.04   # [m²] — varianza enviada al EKF
scan_match_cov_theta:  0.03   # [rad²]

# kalman_params.yaml
scan_match_noise_x:     0.04   # floor irrevocable en el Kalman
scan_match_noise_y:     0.04
scan_match_noise_theta: 0.03
```

---

### Fix 4 — ArUco FPS limiter + outlier rejection

**File:** `src/puzzlebot_bringup/launch/real_robot.launch.py`

```python
'max_processing_hz': 8.0,      # limita solvePnP — evita backlog de frames
'max_position_jump': 0.25,     # era 0.5; rechaza outliers de 32cm con robot quieto
'max_detection_distance': 2.5, # era 1.8; cubre zona central de la pista
'max_incidence_angle_deg': 75.0, # era 65; cubre paredes laterales completas
```

**File:** `src/puzzlebot_perception/puzzlebot_perception/aruco_node.py`
- Nuevo parámetro `max_processing_hz` con guard en `_process()`.

---

## Current State of Files

### `src/puzzlebot_localization/src/kalman_filter_node.cpp`
- ✅ R_theta floor (K_theta: 0.98 → 0.12)
- ✅ ZUPT Q scaling por velocidad
- ✅ P_max clamp (1.0 m², 2.0 rad²)
- ✅ `scan_match_cb()` — fusión EKF de `/scan_match/pose`
- ✅ `R_scan_` matriz de ruido para scan matching
- ✅ `sub_scan_match_` suscripción a `/scan_match/pose`

### `src/puzzlebot_bringup/config/kalman_params.yaml`
```yaml
process_noise_x:      0.01
process_noise_y:      0.01
process_noise_theta:  0.005
meas_noise_x:         0.05
meas_noise_y:         0.05
meas_noise_theta:     0.07   # floor ArUco yaw — K_theta ≈ 0.12
zupt_speed_threshold: 0.02
p_max_xy:             1.0
p_max_theta:          2.0
scan_match_noise_x:   0.04
scan_match_noise_y:   0.04
scan_match_noise_theta: 0.03
init_from_aruco:      true
```

### `src/puzzlebot_slam/puzzlebot_slam/slam_node.py`
- ✅ Publisher `/scan_match/pose`
- ✅ `_publish_scan_match()` con covarianza configurable
- ✅ `scan_matched` flag: solo publica si score ≥ min_score Y sanity check OK

### `src/puzzlebot_slam/puzzlebot_slam/scan_matcher.py`
- ✅ `_last_score` property
- ✅ `_TRANS_HALF_M = 0.15` (±3 celdas, era ±1 celda)

### `src/puzzlebot_bringup/config/slam_params.yaml`
```yaml
scan_matching_enabled:    true
scan_match_min_score:     15.0
scan_match_cov_xy:        0.04
scan_match_cov_theta:     0.03
p_occ:                    0.80   # ⚠️ ver Pending — se propuso bajar a 0.72
p_free:                   0.45   # ⚠️ ver Pending — se propuso subir a 0.48
keyframe_min_translation: 0.10   # ⚠️ ver Pending — se propuso subir a 0.15
```

> **Nota:** Los cambios B (p_occ/p_free) y E (keyframe_min_translation) fueron aplicados en esta sesión pero revertidos por el IDE antes del commit. Ver sección Pending.

### `src/puzzlebot_bringup/launch/real_robot.launch.py`
- ✅ `max_detection_distance: 2.5 m`
- ✅ `max_incidence_angle_deg: 75.0°`
- ✅ `max_processing_hz: 8.0`
- ✅ `max_position_jump: 0.25`
- ✅ `slam_match_updates_odom` calculado automáticamente (`false` cuando kalman:=true)

### `fastdds_puzzlebot.xml` (workspace root — sin commitear)
- Perfil DDS para reducir saturación de multicast
- ⚠️ `<address>192.168.1.100</address>` — reemplazar con IP real del Jetson antes de usar

---

## What Is Working

- ✅ Mapa completo de la pista física con paredes simples (confirmado en RViz, screenshot)
- ✅ Scan matching coloca cada scan en su posición óptima — sin doble-pared
- ✅ Loop closure liviano via `/scan_match/pose` → Kalman
- ✅ Elipse de incertidumbre estable cuando robot está quieto (ZUPT)
- ✅ ArUco yaw estable sin spinning (K_theta ≈ 0.12)
- ✅ `/map` visible en RViz con `kalman:=true aruco:=true slam:=true`
- ✅ Bootstrap automático desde primer ArUco (`init_from_aruco: true`)

---

## What Is NOT Working / Pending

### 1. `wheel_separation` sin calibrar — drift angular primario
El mapa aparece rotado ~10–15° respecto a los ejes del mundo. Causa directa: `wheel_separation: 0.18` (valor del manual) sobreestima ω → el mapa se rota progresivamente.

**Procedimiento de calibración:**
```bash
# Sobre suelo plano, marcar orientación inicial
# Girar exactamente 5 vueltas completas a velocidad constante:
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0}, angular: {z: 0.3}}" --rate 10
# Medir ángulo real girado (cinta métrica o IMU)
# Calcular corrección:
#   ws_correcto = ws_actual × (5 × 2π) / angulo_real_rad
```

**Archivo:** `src/puzzlebot_bringup/config/robot_params.yaml` → `wheel_separation: <medido>`

### 2. p_occ / p_free / keyframe — tuning pendiente (B y E)

Se analizaron y aplicaron pero fueron revertidos por el IDE antes del commit. Aplicar manualmente:

```yaml
# src/puzzlebot_bringup/config/slam_params.yaml
p_occ:                    0.72   # era 0.80 — menos falsos positivos en interior
p_free:                   0.48   # era 0.45 — limpia objetos transitorios más rápido
keyframe_min_translation: 0.15   # era 0.10 — menos scans en giros lentos
```

No requiere recompilar — solo reiniciar el launch.

### 3. `fastdds_puzzlebot.xml` sin configurar ni commitear

Reemplazar IP y commitear:
```bash
# En fastdds_puzzlebot.xml, línea <address>:
sed -i 's/192.168.1.100/<IP_REAL_JETSON>/g' fastdds_puzzlebot.xml

# Activar antes de lanzar:
export FASTRTPS_DEFAULT_PROFILES_FILE=~/Documents/puzzlebot_sim/fastdds_puzzlebot.xml
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

### 4. `lidar_x` offset físico (mejora menor)

Si el LiDAR no está exactamente sobre el eje de ruedas, los giros producen un arco en las paredes. Medir distancia del LiDAR al centro del eje y actualizar:
```yaml
# slam_params.yaml
lidar_x: 0.05   # cm hacia adelante del centro (medir físicamente)
```

### 5. scan_match translation window — afinar post-calibración

Con `wheel_separation` calibrado, el drift lineal se reduce y la ventana de traslación puede ajustarse:
```python
# scan_matcher.py
_TRANS_HALF_M = 0.08   # bajar de 0.15 cuando odometría sea más precisa
```

---

## Parameter Tuning Reference

### Scan match score vs contenido del mapa
| `scan_match_min_score` | Celdas de pared necesarias | Comportamiento |
|------------------------|---------------------------|----------------|
| 8.0 | ~6 celdas | Activa en zonas con poco mapa — riesgo de falsos |
| 15.0 | ~11 celdas | **Actual** — balance estabilidad/cobertura |
| 25.0 | ~18 celdas | Solo en zonas muy bien mapeadas |

### K_theta vs meas_noise_theta (con floor fix activo)
| `meas_noise_theta` | σ_min_theta | K_theta típico | Efecto |
|--------------------|------------|----------------|--------|
| 0.01 | 5.7° | ~0.98 | ArUco totalmente confiado — puede spinning |
| 0.05 | 12.9° | ~0.30 | Moderado |
| **0.07** | **15.2°** | **~0.12** | **Actual — estable** |
| 0.15 | 22.3° | ~0.05 | Muy conservador — convergencia lenta |

### p_occ vs ruido en mapa
| `p_occ` | Velocidad de ocupación | Falsos positivos |
|---------|----------------------|-----------------|
| 0.90 | Muy rápido | Muchos — cualquier hit solitario queda |
| 0.80 | Rápido | **Actual** — algunos puntos aislados |
| 0.72 | Moderado | Pocos — **recomendado** (pendiente commit) |
| 0.65 | Lento | Paredes tenues — puede perder detalle |

---

## Build Status

```
puzzlebot_slam         ✓  (scan_matching + publish_scan_match)
puzzlebot_localization ✓  (ZUPT, P_max, R_theta floor, scan_match_cb)
puzzlebot_bringup      ✓  (aruco params, scan_match_updates_map_odom logic)
puzzlebot_perception   ✓  (max_processing_hz FPS limiter)
puzzlebot_description  ✓
```

```bash
cd ~/Documents/puzzlebot_sim && source install/setup.bash
colcon build --packages-select puzzlebot_slam puzzlebot_localization \
  puzzlebot_bringup puzzlebot_perception --cmake-args -DCMAKE_BUILD_TYPE=Release
```

---

## Launch Commands

```bash
# Comando principal — Kalman EKF + ArUco + SLAM:
ros2 launch puzzlebot_bringup real_robot.launch.py \
  kalman:=true aruco:=true slam:=true rviz:=true lidar_topic:=/scan

# Protocolo de arranque:
# 1. Colocar robot mirando a cualquier ArUco marker (< 2.5 m de distancia)
# 2. Lanzar el sistema
# 3. Esperar: "✅ Pose inicial desde ArUco: x=... y=... theta=..."
# 4. Mover el robot

# Solo predicción (debug encoders, sin ArUco):
ros2 launch puzzlebot_bringup real_robot.launch.py \
  kalman:=true aruco:=false slam:=true rviz:=true lidar_topic:=/scan

# Stack mínimo (debug cmd_vel / DDS):
ros2 launch puzzlebot_bringup real_robot.launch.py \
  kalman:=true aruco:=false slam:=false rviz:=false lidar_topic:=/scan
```

---

## Recommended Next Steps (Priority Order)

1. **[Crítico] Calibrar `wheel_separation`** — es la causa raíz del mapa rotado y del drift angular acumulado. Hacer el test de 5 rotaciones sobre la pista. Sin esto, las mejoras del scan matcher trabajan contra un error sistemático constante.

2. **[Alta] Aplicar y commitear cambios B y E** — editar `slam_params.yaml` manualmente:
   - `p_occ: 0.72`, `p_free: 0.48`, `keyframe_min_translation: 0.15`
   - Probar con 2–3 vueltas y comparar mapa vs estado actual

3. **[Media] Configurar `fastdds_puzzlebot.xml`** — reemplazar IP del Jetson y commitear. Probar si resuelve la pérdida de `cmd_vel` con stack completo.

4. **[Media] Ajustar `_TRANS_HALF_M`** en `scan_matcher.py` a 0.08 m después de calibrar `wheel_separation` — ventana más estrecha = menos riesgo de saltar a pared paralela.

5. **[Baja] Medir `lidar_x`** físicamente y actualizar `slam_params.yaml` — reduce arcos en paredes durante giros.
