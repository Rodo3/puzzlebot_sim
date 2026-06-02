# Claude Handoff 7 — Mapa estático en RViz + Localización EKF+ScanMatch sobre mapa conocido

**Date:** 2026-05-30
**Branch:** `mi_rama_de_pruebas`
**Platform:** Puzzlebot Differential Drive — Jetson Orin (sensors) + PC (all compute)
**Previous handoff:** `handoffClaude6.md`

---

## Objective

1. Corregir la visualización del mapa estático en RViz (mapa no aparecía por QoS incompatible)
2. Crear modo de localización sobre mapa conocido más rápido que MCL
3. Agregar scan matching contra mapa conocido como corrección continua al EKF
4. Bajar velocidad de navegación para reducir incertidumbre acumulada

---

## Result

✅ **Mapa estático visible en RViz** — corregido QoS mismatch (TRANSIENT_LOCAL publisher vs VOLATILE subscriber).
✅ **Modo `use_map:=true`** — localización EKF+ArUco sobre mapa conocido, sin MCL, convergencia instantánea.
✅ **Scan matching contra mapa conocido** — `slam_node` en modo `localization_only` publica `/scan_match/pose` al EKF sin modificar el mapa.
✅ **Velocidad reducida** — `max_linear_vel: 0.20 m/s`, `max_angular_vel: 1.00 rad/s`.
⚠️ **Scan matching causa perturbaciones en navegación** — cuando el matcher corrige bruscamente la pose estimada, el robot se desorienta brevemente y puede estrellarse. Pendiente: hacer el scan matching condicional a la incertidumbre del EKF (ver sección Pending).

---

## Changes Implemented This Session

### Change 1 — Corrección QoS en RViz: mapa no aparecía

**Problema raíz:** `map_server_node` publica `/map` con `TRANSIENT_LOCAL`, pero ambos archivos RViz tenían `Durability: Volatile` en el display Map. En ROS 2 Humble, publisher TRANSIENT_LOCAL + subscriber VOLATILE = fallo silencioso.

**Archivos modificados:**
- `src/puzzlebot_description/rviz/mcl_rviz.rviz`
  - Display Map: tópico `/mcl/map` → `/map`
  - QoS ya era `Transient Local` — mantenido
  - Fixed Frame: ya era `map` — mantenido
  - **Agregado:** herramientas `SetInitialPose` (tecla P) y `SetGoal` (tecla G)
  - **Agregado:** display `Planned Path` escuchando `/planned_path`

- `src/puzzlebot_description/rviz/puzzlebot_rviz.rviz`
  - Sin cambios — sigue con Fixed Frame `odom` para sesión de mapeo (correcto)

### Change 2 — `real_robot.launch.py`: RViz correcto por modo

**Antes:** siempre abría `puzzlebot_rviz.rviz` (Fixed Frame `odom`), incorrecto durante MCL/navegación.

**Ahora:** dos nodos RViz mutuamente excluyentes:
```python
rviz_slam  # mcl:=false AND use_map:=false → puzzlebot_rviz.rviz (mapeo)
rviz_mcl   # mcl:=true  OR  use_map:=true  → mcl_rviz.rviz (navegación, Fixed Frame: map)
```

### Change 3 — `map_server_node.py`: thresholds correctos + zonas unknown

**Antes:** píxeles grises (valor 127) → `free (0)`. Toda zona sin explorar aparecía como suelo libre.

**Ahora:** conversión compatible con `nav2_map_server`:
```
pixel > 164  (≥ 0.65×255) → free (0)      — suelo blanco
pixel ≤ 50   (≤ 0.196×255) → occupied (100) — paredes negras
pixel 51–164              → unknown (-1)   — zonas grises sin explorar
```

**Nuevos parámetros** (configurables en `mcl_params.yaml`):
```yaml
occupied_thresh: 0.65   # normalizado: pixel/255 >= umbral → libre
free_thresh:     0.196  # normalizado: pixel/255 <= umbral → ocupado
```

### Change 4 — Nuevo argumento `use_map:=true`

Activa el stack completo de localización sobre mapa conocido sin MCL:

| Nodo | Rol |
|---|---|
| `map_server_node` | Publica el PNG como `/map` (TRANSIENT_LOCAL) |
| `aruco_map_odom` | Publica `map→odom` desde detecciones ArUco absolutas |
| `slam_node [LOCALIZATION-ONLY]` | Scan matching contra mapa pre-cargado → `/scan_match/pose` al EKF |

**Comando:**
```bash
ros2 launch puzzlebot_bringup real_robot.launch.py \
  slam:=false mcl:=false kalman:=true aruco:=true use_map:=true \
  rviz:=true lidar_topic:=/scan navigation:=true
```

### Change 5 — `slam_node.py`: modo `localization_only`

**Archivo:** `src/puzzlebot_slam/puzzlebot_slam/slam_node.py`

Nuevo parámetro `localization_only: true`:
- Al arrancar: carga el PNG en el grid de log-odds via `load_from_png()`
- Fuerza `scan_matching_enabled: true`
- Hace scan matching contra el mapa inmutable en cada scan
- Publica `/scan_match/pose` al EKF cuando score ≥ `scan_match_min_score`
- **NO integra nuevos scans** — mapa no se modifica
- **NO guarda PNG al Ctrl+C** — evita sobreescribir el mapa bueno
- `publish_map_odom_tf: false` — `aruco_map_odom` es el dueño de `map→odom`

Log al arrancar:
```
slam_node ready [LOCALIZATION-ONLY] — 86×108 px @ 0.050 m/px ...
[localization_only] Mapa cargado desde: ...slam_map_FECHA.pgm
```

### Change 6 — `OccupancyGridMap.load_from_png()`

**Archivo:** `src/puzzlebot_slam/puzzlebot_slam/occupancy_grid_map.py`

Nuevo método que invierte `to_png()` para reconstruir el grid de log-odds:
```
pixel 255 → log-odds = -l_clamp  (libre)
pixel 127 → log-odds =  0.0      (desconocido)
pixel   0 → log-odds = +l_clamp  (ocupado)
```
Incluye resize automático si el PNG tiene dimensiones distintas al grid configurado.

### Change 7 — `mcl_params.yaml`: centralización de rutas de mapa

**Antes:** ruta del mapa hardcodeada en `real_robot.launch.py`.

**Ahora:** un solo archivo para cambiar el mapa:
```yaml
map_server_node:
  ros__parameters:
    map_path: '/home/jesus/Documents/puzzlebot_sim/slam_map_FECHA.pgm'

slam_node:
  ros__parameters:
    localization_map_path: '/home/jesus/Documents/puzzlebot_sim/slam_map_FECHA.pgm'

mcl:
  ros__parameters:
    map_path: '/home/jesus/Documents/puzzlebot_sim/slam_map_FECHA.pgm'
```

> ⚠️ `map_server_node.map_path` y `slam_node.localization_map_path` tienen rutas independientes actualmente (quedaron así tras edición del usuario). Si se usa un mapa nuevo, actualizar **las tres entradas** y luego `colcon build --packages-select puzzlebot_bringup`.

### Change 8 — Velocidad de navegación reducida

**Archivo:** `src/puzzlebot_bringup/config/controller_params.yaml`

```yaml
# ANTES → AHORA
lookahead_distance:  0.30 → 0.30   # sin cambio (el usuario revirtió a 0.30)
max_linear_vel:      0.30 → 0.20   # m/s
max_angular_vel:     1.50 → 1.00   # rad/s
```

Reducción de velocidad disminuye el slip de encoders y el intervalo sin corrección ArUco entre markers.

---

## TF Chain en modo `use_map:=true`

```
map ──aruco_map_odom──▶ odom ──kalman_filter_node──▶ base_footprint
                               (EKF: encoders + ArUco + scan_match)
```

Correcciones al EKF por frecuencia típica:
- **ArUco:** solo cuando hay marker visible (~1–8 Hz)
- **Scan match:** cada scan donde score ≥ 15 (~2–5 Hz, continuo mientras haya paredes visibles)

---

## What Is Working

- ✅ Mapa estático visible en RViz inmediatamente al arrancar (QoS correcto)
- ✅ `mcl_rviz.rviz` tiene Fixed Frame `map`, herramientas P y G, display Path
- ✅ `use_map:=true` levanta el stack completo de localización sin MCL
- ✅ `slam_node [LOCALIZATION-ONLY]` carga mapa PNG y hace scan matching continuo
- ✅ EKF fusiona encoders + ArUco + scan match simultáneamente
- ✅ Mapa inmutable: Ctrl+C no sobreescribe el PNG guardado
- ✅ Velocidad reducida a 0.20 m/s / 1.00 rad/s

---

## What Is NOT Working / Pending

### 1. [Crítico] Scan matching causa perturbaciones en navegación

**Síntoma observado:** cuando el scan matcher hace una corrección brusca de pose, el robot pierde el seguimiento de la ruta brevemente y puede estrellarse contra una pared.

**Causa:** el scan matching siempre está activo, incluso cuando el EKF ya tiene baja incertidumbre (ArUco visible, encoders con poco drift). Una corrección del matcher en ese caso introduce una perturbación innecesaria en la pose estimada.

**Solución propuesta (NO implementada):** hacer el scan matching condicional a la traza de P (covarianza del EKF):
```python
# Idea: en slam_node._scan_cb(), antes de publicar /scan_match/pose,
# suscribirse a /odom (que incluye la covarianza del EKF) y solo publicar
# la corrección si tr(P_xy) > umbral (por ejemplo 0.05 m²).
# Cuando ArUco está activo y P es pequeña, ignorar el scan match.
# Cuando el robot pierde ArUco y P crece, activar el scan match.
```

**Alternativa más simple:** deshabilitar scan matching en `use_map:=true` hasta implementar el control por covarianza, y solo usar ArUco+EKF. Cambiar en `mcl_params.yaml`:
```yaml
slam_node:
  ros__parameters:
    localization_map_path: '...'
    scan_match_min_score: 999.0   # efectivamente deshabilita la publicación
```

O directamente no lanzar `slam_loc` desactivando scan matching en el launch (requiere un argumento nuevo `scan_match:=true/false`).

### 2. [Alta] Ruta del mapa duplicada e inconsistente en `mcl_params.yaml`

`map_server_node.map_path` apunta a `slam_map_20260528_224124.png` mientras `slam_node.localization_map_path` y `mcl.map_path` apuntan a `slam_map_20260529_235356.png`. Dos nodos usan mapas diferentes.

**Acción:** decidir cuál es el mapa bueno y unificar las tres entradas en `mcl_params.yaml`. Luego `colcon build --packages-select puzzlebot_bringup`.

### 3. [Alta] Mapa sigue siendo incompleto

Los mapas disponibles tienen paredes parciales (esquinas sin cerrar, ruido en perímetro superior). El scan matcher trabaja mejor con un mapa completo. Ver protocolo de mapeo limpio en `handoffClaude6.md`.

### 4. [Heredado] `wheel_separation` sin calibrar

Drift angular sistemático. Ver procedimiento completo en `handoffClaude5.md` sección Pending #1.

### 5. [Heredado] `fastdds_puzzlebot.xml` sin IP real

Ver `handoffClaude5.md` sección Pending #3.

---

## Modos de operación disponibles

### Sesión de mapeo
```bash
ros2 launch puzzlebot_bringup real_robot.launch.py \
  kalman:=true aruco:=true slam:=true rviz:=true lidar_topic:=/scan
# Ctrl+C → guarda slam_map_FECHA.png automáticamente
```

### Sesión de navegación con MCL (convergencia lenta)
```bash
ros2 launch puzzlebot_bringup real_robot.launch.py \
  slam:=false mcl:=true aruco:=true rviz:=true \
  lidar_topic:=/scan navigation:=true
```

### Sesión de navegación con EKF+ArUco+ScanMatch (convergencia instantánea)
```bash
ros2 launch puzzlebot_bringup real_robot.launch.py \
  slam:=false mcl:=false kalman:=true aruco:=true use_map:=true \
  rviz:=true lidar_topic:=/scan navigation:=true
# En RViz: P → clic en posición del robot, G → clic en destino
```

> ⚠️ Si el scan matching causa inestabilidad en la navegación, ver Pending #1 para desactivarlo temporalmente.

---

## Archivos modificados esta sesión

| Archivo | Cambio |
|---|---|
| `src/puzzlebot_bringup/launch/real_robot.launch.py` | `use_map` arg, `slam_loc` nodo, RViz dual, condiciones actualizadas |
| `src/puzzlebot_bringup/config/mcl_params.yaml` | Secciones `map_server_node` y `slam_node`, rutas de mapa |
| `src/puzzlebot_bringup/config/controller_params.yaml` | `max_linear_vel: 0.20`, `max_angular_vel: 1.00` |
| `src/puzzlebot_slam/puzzlebot_slam/map_server_node.py` | Thresholds correctos, parámetros `occupied_thresh`/`free_thresh` |
| `src/puzzlebot_slam/puzzlebot_slam/slam_node.py` | `localization_only`, `load_from_png` call, skip integración, skip save |
| `src/puzzlebot_slam/puzzlebot_slam/occupancy_grid_map.py` | Método `load_from_png()` |
| `src/puzzlebot_description/rviz/mcl_rviz.rviz` | Tópico `/map`, herramientas P/G, display Path |

---

## Build Status

```
puzzlebot_slam    ✓  (localization_only + load_from_png + map_server thresholds)
puzzlebot_bringup ✓  (use_map arg, slam_loc node, dual RViz, mcl_params actualizado)
```

```bash
cd ~/Documents/puzzlebot_sim
colcon build --packages-select puzzlebot_slam puzzlebot_bringup
source install/setup.bash
```

---

## Recommended Next Steps (Priority Order)

1. **[Crítico] Decidir si el scan matching en navegación se deja activo o se desactiva.** Si el robot se sigue estrellando, subir `scan_match_min_score` a `999.0` en `mcl_params.yaml` para efectivamente desactivarlo, y validar si la navegación mejora solo con ArUco+EKF.

2. **[Crítico] Unificar las rutas del mapa en `mcl_params.yaml`.** Las tres entradas (`map_server_node.map_path`, `slam_node.localization_map_path`, `mcl.map_path`) deben apuntar al mismo archivo.

3. **[Crítico] Obtener mapa completo de la pista.** El protocolo de mapeo limpio está en `handoffClaude6.md`. Con mapa completo, el scan matching funciona mejor y la navegación es más robusta.

4. **[Alta] Implementar scan matching condicional a covarianza del EKF.** La idea está descrita en Pending #1. Requiere suscribir `/odom` (que trae la covarianza) en `slam_node` o crear un mecanismo de activación externa.

5. **[Heredado] Calibrar `wheel_separation`.** Ver `handoffClaude5.md` Pending #1.
