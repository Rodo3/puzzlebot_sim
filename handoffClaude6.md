# Claude Handoff 6 — Guardado de mapa PNG + Stack de navegación con waypoints

**Date:** 2026-05-29
**Branch:** `mi_rama_de_pruebas`
**Platform:** Puzzlebot Differential Drive — Jetson Orin (sensors) + PC (all compute)
**Previous handoff:** `handoffClaude5.md`

---

## Objective

1. Guardar automáticamente el mapa SLAM como PNG al hacer Ctrl+C
2. Activar navegación autónoma con waypoints (tecla G en RViz) en robot físico
3. Cargar un mapa PNG/PGM guardado en RViz sin necesidad de mapear de nuevo

---

## Result

✅ **Guardado automático de mapa PNG** al detener `slam_node` (Ctrl+C).
✅ **Stack de navegación A\*** disponible en robot físico con `navigation:=true`.
✅ **`map_server_node`** creado — publica mapa PNG/PGM estático en `/map` al usar `mcl:=true`.
⚠️ **Mapeo con ruido** — los mapas guardados en esta sesión tienen paredes incompletas. Pendiente obtener un mapa limpio completo.

---

## Changes Implemented This Session

### Change 1 — Guardado automático de mapa al Ctrl+C

**Files:**
- `src/puzzlebot_slam/puzzlebot_slam/occupancy_grid_map.py` — nuevo método `to_png()`
- `src/puzzlebot_slam/puzzlebot_slam/slam_node.py` — `_save_map_on_exit()` en bloque `finally`

**Comportamiento:**
- Al hacer Ctrl+C en cualquier sesión de mapeo, se guarda automáticamente `slam_map_YYYYMMDD_HHMMSS.png` en el directorio de trabajo (donde se ejecutó el launch)
- Para guardar en ruta fija: `export SLAM_MAP_DIR=~/maps` antes de lanzar

**Convención del PNG (compatible con ROS map_server):**
```
255 = libre     (log-odds < -0.5)
127 = desconocido (log-odds ≈ 0)
  0 = ocupado   (log-odds > 0.5)
```

Imagen volteada verticalmente (`np.flipud`) para que el origen bottom-left del OccupancyGrid corresponda al píxel inferior izquierdo del PNG.

**Mapas guardados en esta sesión** (en `/home/jesus/Documents/puzzlebot_sim/`):
- `slam_map_20260528_192751.png` — mapeo incompleto, rayos divergentes
- `slam_map_20260528_193256.png` — mapeo incompleto, rayos divergentes
- `slam_map_20260528_193926.png` — mapeo incompleto
- `slam_map_20260528_195141.png` — mapeo incompleto
- `slam_map_20260528_223027.pgm` — mejor mapa disponible (paredes parciales)
- `slam_map_20260528_224124.png` — paredes parciales

> ⚠️ Ninguno de los mapas tiene el área completa. Ver sección Pending.

---

### Change 2 — `map_server_node` (nuevo nodo)

**File:** `src/puzzlebot_slam/puzzlebot_slam/map_server_node.py`

Nodo que carga un PNG o PGM y lo publica como `nav_msgs/OccupancyGrid` en `/map` con QoS TRANSIENT_LOCAL. Reemplaza `nav2_map_server` para este proyecto.

**Parámetros:**
```yaml
map_path:       ruta al PNG o PGM
map_resolution: 0.05   # m/px
map_origin_x:  -0.25   # esquina inferior izquierda
map_origin_y:  -0.25
map_frame:     'map'
```

**Conversión de valores:**
- `< 50`   → 100 (ocupado)
- `>= 50`  → 0   (libre — incluye desconocido para mostrar el suelo completo en RViz)

**Registrado en:** `src/puzzlebot_slam/setup.py` como `map_server_node`

**Lanzado automáticamente** cuando `mcl:=true` en `real_robot.launch.py` con parámetros hardcoded al mapa `slam_map_20260528_223027.pgm`.

> ⚠️ La ruta del mapa está hardcodeada en `real_robot.launch.py`. Actualizar cuando haya un mapa mejor — ver sección Pending.

---

### Change 3 — `real_robot.launch.py`: integración `map_server_node`

**File:** `src/puzzlebot_bringup/launch/real_robot.launch.py`

Agregado nodo `map_server` con condición `IfCondition(mcl_en)`:

```python
map_server = Node(
    package='puzzlebot_slam',
    executable='map_server_node',
    name='map_server_node',
    parameters=[{
        'map_path':       '/home/jesus/Documents/puzzlebot_sim/slam_map_20260528_223027.pgm',
        'map_resolution':  0.05,
        'map_origin_x':   -0.25,
        'map_origin_y':   -0.25,
        'map_frame':      'map',
    }],
    condition=IfCondition(mcl_en),
)
```

También se agrega `navigation_stack` (ya existía en la rama) pasando `avoidance` como argumento.

---

### Change 4 — `mcl_params.yaml`: ruta del mapa actualizada

```yaml
map_path:     '/home/jesus/Documents/puzzlebot_sim/slam_map_20260528_223027.pgm'
map_origin_x: -0.25
map_origin_y: -0.25
```

> ⚠️ Actualizar `map_path` en este archivo Y en `real_robot.launch.py` cuando haya un mapa completo.

---

## Flujo de trabajo confirmado

### Sesión de mapeo
```bash
ros2 launch puzzlebot_bringup real_robot.launch.py \
  kalman:=true aruco:=true slam:=true rviz:=true lidar_topic:=/scan
# Mapear todo el área despacio
# Ctrl+C → se guarda slam_map_FECHA.png automáticamente
```

### Sesión de navegación con waypoints (un solo comando)
```bash
ros2 launch puzzlebot_bringup real_robot.launch.py \
  slam:=false mcl:=true aruco:=true rviz:=true \
  lidar_topic:=/scan navigation:=true
```

En RViz:
1. Fixed Frame = `map`
2. **P** → clic donde está el robot → arrastra en dirección que apunta (inicializa MCL)
3. **G** → clic en el destino → robot navega solo

### Mapeo + navegación en misma sesión (dos terminales)
```bash
# Terminal 1 — mapeo:
ros2 launch puzzlebot_bringup real_robot.launch.py \
  kalman:=true aruco:=true slam:=true rviz:=true lidar_topic:=/scan

# Terminal 2 — cuando mapa esté listo:
ros2 launch puzzlebot_bringup navigation.launch.py \
  use_sim_time:=false cmd_vel_topic:=/cmd_vel
```

---

## What Is Working

- ✅ `slam_node` guarda PNG automáticamente al Ctrl+C
- ✅ `map_server_node` carga PNG/PGM y publica `/map` con QoS TRANSIENT_LOCAL
- ✅ Stack de navegación A\* disponible en robot físico via `navigation:=true`
- ✅ `path_planner_node` recibe `/goal_pose` desde RViz (tecla G) y calcula ruta A\*
- ✅ `steering_controller` sigue la ruta publicada en `/planned_path`
- ✅ `obstacle_avoidance` protege al robot en navegación autónoma
- ✅ Navegación confirmada funcionando en simulación

---

## What Is NOT Working / Pending

### 1. [Crítico] Obtener mapa completo de la pista

Todos los mapas guardados en esta sesión tienen paredes incompletas. El patrón de "rayos en estrella" del mapeo malo sugiere drift del Kalman/ArUco en los primeros segundos antes de que el scan matcher tenga suficiente mapa para trabajar.

**Estrategia recomendada para mapeo limpio:**
- Iniciar con robot quieto mirando a un ArUco hasta ver "✅ Pose inicial desde ArUco"
- Mover muy despacio (vel lineal ≤ 0.1 m/s) los primeros 30 segundos
- Recorrer el perímetro completo antes de entrar al interior
- Evitar giros bruscos

### 2. [Crítico] Actualizar ruta del mapa cuando haya uno completo

Cuando se obtenga un mapa bueno, actualizar en **dos lugares**:

```python
# src/puzzlebot_bringup/launch/real_robot.launch.py — map_server Node:
'map_path': '/home/jesus/Documents/puzzlebot_sim/slam_map_NUEVA_FECHA.pgm'
```

```yaml
# src/puzzlebot_bringup/config/mcl_params.yaml:
map_path: '/home/jesus/Documents/puzzlebot_sim/slam_map_NUEVA_FECHA.pgm'
```

Luego: `colcon build --packages-select puzzlebot_bringup && source install/setup.bash`

### 3. [Media] Validar navegación en robot físico

La navegación A\* está confirmada en simulación pero aún no probada en robot físico con mapa real completo. Los parámetros a ajustar si hay problemas:

```python
# navigation.launch.py — path_planner_node:
'inflation_radius':   0.15,   # subir a 0.20 si el robot roza paredes
'occupied_threshold': 50,     # bajar a 40 si el planner rechaza goals válidos

# controller_params.yaml:
# goal_tolerance:  0.10 m    # subir si el robot oscila cerca del goal
# max_linear_vel:  0.30 m/s  # bajar si el robot va muy rápido en el real
```

### 4. [Baja] Mover ruta del mapa a parámetro configurable

Actualmente la ruta está hardcodeada en `real_robot.launch.py`. Moverla a `mcl_params.yaml` y leerla desde ahí en el launch sería más limpio. Requiere que el launch lea el YAML y extraiga solo `map_path`.

---

## Build Status

```
puzzlebot_slam    ✓  (map_server_node + to_png + _save_map_on_exit)
puzzlebot_bringup ✓  (map_server integrado en real_robot.launch.py)
```

```bash
cd ~/Documents/puzzlebot_sim
colcon build --packages-select puzzlebot_slam puzzlebot_bringup
source install/setup.bash
```

---

## Recommended Next Steps (Priority Order)

1. **[Crítico] Hacer mapeo completo de la pista** con el protocolo de arranque lento descrito arriba. El mapa resultante es el prerequisito para todo lo demás.

2. **[Crítico] Actualizar `map_path`** en `real_robot.launch.py` y `mcl_params.yaml` con el nuevo mapa.

3. **[Alta] Probar navegación con waypoints en robot físico** — lanzar sesión de navegación, inicializar MCL con P, mandar goal con G, observar comportamiento.

4. **[Media] Ajustar `inflation_radius` y `max_linear_vel`** según comportamiento observado en robot físico.

5. **[Heredado de Claude5] Calibrar `wheel_separation`** — sigue siendo la causa raíz del drift angular acumulado en el mapeo.
