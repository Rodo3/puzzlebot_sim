# Puzzlebot Sim — Estado del Proyecto para Agentes

> Leer este archivo PRIMERO antes de tocar cualquier código.
> Refleja el estado real del repositorio a mayo 2026.

---

## Contexto rápido

Robot diferencial (Puzzlebot) con lidar 2D, Jetson Orin 2 GB.
Stack: **ROS 2 Humble + Gazebo Fortress (ignition-gazebo 6)**.
Objetivo final: navegación autónoma completa en Gazebo → transferir al robot real.

---

## Estado actual por sprint

| Sprint | Objetivo | Estado |
|--------|----------|--------|
| 1 — Sim bringup | Gazebo Fortress + bridge + robot visible + teleop | ✅ Completo |
| 2 — Odometría | Dead reckoning publica `/odom` y TF `odom→base_footprint` | ✅ Completo |
| 3 — SLAM | `slam_node` genera `/map`; `mcl` localiza contra mapa PNG | ✅ Validado en Gazebo |
| 4 — Percepción | ArUco + YOLO en simulación | ⏳ Bloqueado (sin modelos en Gazebo) |
| 5 — Planificación | A* + obstacle avoidance + steering controller | ⏳ Pendiente |
| 6 — Integración final | Loop completo autónomo en Gazebo → robot real | ⏳ Pendiente |

---

## Qué está funcionando hoy

### Simulación Gazebo Fortress

```bash
# Flat plane (dead reckoning):
ros2 launch puzzlebot_bringup gz_sim.launch.py

# Maze — localización MCL contra mapa prebuilt:
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze

# Maze — construir mapa desde cero con SLAM (necesita teleop para explorar):
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze mode:=mapping

# Maze — mapping usando odometría por ruedas para probar deriva:
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze mode:=mapping odom_source:=dead_reckoning

# Teleop (terminal separada):
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args --remap cmd_vel:=/model/puzzlebot/cmd_vel
```

### Nodos activos en cada modo

| Nodo | flat_plane | maze+mcl | maze+mapping |
|------|-----------|----------|--------------|
| `robot_state_publisher` | ✅ | ✅ | ✅ |
| `gz_bridge` (Fortress, ignition.msgs) | ✅ | ✅ | ✅ |
| `dead_reckoning` → `/odom` | ✅ | ✅ | solo si `odom_source:=dead_reckoning` |
| `ground_truth_odom` → `/odom` | — | — | ✅ default (`odom_source:=ground_truth`) |
| `mcl` → `/mcl/pose`, `/mcl/map`, TF `map→odom` | — | ✅ | — |
| `slam_node` → `/map`, TF `map→odom` | — | — | ✅ |

---

## Arquitectura de paquetes

```
src/
├── puzzlebot_description/    # URDF, SDF, meshes, worlds, RViz configs
│   ├── urdf/puzzlebot_gz.urdf        # Para robot_state_publisher (cinemática)
│   ├── sdf/puzzlebot_gz.sdf          # Spawneado en Gazebo (plugins Fortress)
│   ├── worlds/flat_plane.sdf
│   ├── worlds/maze.sdf               # 7 paredes + 8 cajas 1×1×1 m
│   └── rviz/                         # puzzlebot_rviz.rviz, mcl_rviz.rviz
│
├── puzzlebot_bringup/        # Launch files y configuración
│   ├── launch/gz_sim.launch.py       # ← LAUNCH PRINCIPAL (Fortress)
│   ├── launch/simulation.launch.py   # Gazebo Classic (obsoleto, no usar)
│   ├── launch/slam.launch.py         # Stub sin paquete puzzlebot_localization
│   ├── launch/localization.launch.py # Stub sin paquete puzzlebot_localization
│   └── config/
│       ├── robot_params.yaml         # wheel_radius: 0.05, wheelbase: 0.18
│       ├── slam_params.yaml          # 500×500 px, 25 m, origen (-12.5, -12.5)
│       ├── kalman_params.yaml        # Q, R matrices (EKF pendiente)
│       ├── controller_params.yaml    # lookahead, velocidades, PID
│       └── yolo_params.yaml          # engine_path, conf threshold
│
├── puzzlebot_slam/           # Algoritmos SLAM/localización
│   └── puzzlebot_slam/
│       ├── dead_reckoning.py         # ✅ Funcional (sim + real robot)
│       ├── mcl.py                    # ✅ MCL contra maze_map.png
│       ├── slam_node.py              # ✅ Orquestador ROS del mapper
│       ├── occupancy_grid_map.py     # ✅ Log-odds + Bresenham + OccupancyGrid
│       ├── odometry_buffer.py        # ✅ Sincronización /odom ↔ /scan
│       ├── scan_matcher.py           # ⚠️ Hook; aún pass-through
│       ├── keyframe_manager.py       # ✅ Gate opcional de integración
│       ├── slam_math.py              # ✅ Helpers geométricos
│       ├── slam_types.py             # ✅ Pose2D
│       ├── ground_truth_odom.py      # ✅ Pose real de Gazebo → /odom para mapping
│       ├── maze_map.png              # 206×221 px, origen (-5.54, -8.10)
│       └── generate_maze_map.py      # Regenera maze_map.png desde maze.sdf
│
├── puzzlebot_control/
│   └── state_machine_node.py         # ⚠️ Skeleton — no conectado a nada
│
├── puzzlebot_planning/
│   ├── path_planner_node.py          # ✅ A* sobre OccupancyGrid (no lanzado aún)
│   └── obstacle_avoidance_node.py    # ✅ Filtro reactivo /scan → /cmd_vel
│
└── puzzlebot_perception/
    ├── aruco_node.py                 # ⚠️ Skeleton — sin cámara en Gazebo aún
    ├── camera_node.py                # ⚠️ Skeleton
    └── yolo_node.py                  # ⚠️ Skeleton — sin modelo .engine
```

---

## Lo que FALTA implementar (por orden de prioridad)

### 1. SLAM mapping en simulación [VALIDADO]

El `slam_node.py` construye `/map` con occupancy grid mapping log-odds. Usa un
buffer de poses por timestamp para integrar cada `/scan` con la pose de `/odom`
correspondiente. En Gazebo, `gz_sim.launch.py` usa `ground_truth_odom` por default
cuando `mode:=mapping`, evitando que la deriva de wheel odometry deforme el mapa.
El nodo ya está dividido internamente en `OdometryBuffer`, `OccupancyGridMap`,
`KeyframeManager` y `LocalScanMatcher`; el matcher todavía es pass-through.

```bash
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze mode:=mapping
# Teleop para explorar el maze mientras se observa /map en RViz
```

Criterio observado: el OccupancyGrid en `/map` muestra paredes exteriores rectas,
cajas internas cerradas y espacio libre coherente con `maze.sdf`.

Documento técnico: `docs/slam_mapping.md`.

### 2. steering_controller_node.py [BLOQUEADOR para navegación autónoma]

**No existe**. `simulation.launch.py` lo referencia de un paquete `puzzlebot_controller`
que no existe en el repo. Sin esto el robot no puede seguir el path del A*.

Implementar en `src/puzzlebot_control/puzzlebot_control/steering_controller_node.py`:
- Subscribe: `/planned_path` (nav_msgs/Path), `/odom` (nav_msgs/Odometry)
- Publica: `/cmd_vel_in` (geometry_msgs/Twist)  ← la obstacle_avoidance lo filtra
- Algoritmo recomendado: Pure Pursuit (parámetro `lookahead_distance` ya en YAML)
- Parámetros en `config/controller_params.yaml`: `lookahead_distance: 0.30`,
  `max_linear_vel: 0.30`, `max_angular_vel: 1.50`

### 3. Conectar path_planner + obstacle_avoidance al gz_sim.launch.py

Ambos nodos están implementados pero no están en ningún launch activo.
Añadirlos a `gz_sim.launch.py` con argumentos condicionales `navigation:=true`.

`path_planner_node.py` escucha `/map` — funciona tanto con `slam_node` (mapping)
como con `mcl` (que publica `/mcl/map`). **Ojo**: con MCL el planner debe suscribirse
a `/mcl/map`, no a `/map`. Hay que alinear este topic.

### 4. kalman_filter_node [DESPUÉS de percepción]

El EKF no tiene sentido sin correcciones externas (ArUco). No implementar hasta
tener ArUco en el maze world. El `dead_reckoning.py` es suficiente mientras tanto.

### 5. ArUco en maze world [DESPUÉS del steering controller]

Requiere:
- Añadir una cámara al SDF del robot (`puzzlebot_gz.sdf`)
- Añadir marcadores ArUco al `maze.sdf` en posiciones conocidas
- Bridgear el topic de imagen en `gz_bridge` (ignition.msgs.Image)
- Activar `aruco_node.py`

### 6. YOLO [AL FINAL]

Requiere modelo `.engine` entrenado. Hasta entonces el `yolo_node.py` es skeleton.

---

## Reglas críticas de este stack

> Violar estas reglas rompe la simulación de formas difíciles de debuggear.

1. **Gazebo Fortress ONLY** — no mezclar con Harmonic. Binario: `ign gazebo` (no `gz sim`).
   Plugins: `libignition-gazebo-*-system.so` / `ignition::gazebo::systems::*`.
   Bridge messages: `ignition.msgs.*` (no `gz.msgs.*`).

2. **`gz_bridge.yaml` en el root del repo es Harmonic** — no integrarlo. Es solo referencia.

3. **`gz_version: '6'`** en el include de `ros_gz_sim` — sin esto usa el binario Harmonic.

4. **`IGN_GAZEBO_RESOURCE_PATH`** debe ser el PADRE del share dir de `puzzlebot_description`
   para que `model://puzzlebot_description/meshes/` resuelva correctamente.

5. **`use_sim_time: True`** en todos los nodos de Gazebo. En robot real: `False`.

6. **No hay paquete `puzzlebot_localization`** — el code path de `localization.launch.py`
   y `slam.launch.py` está roto (referencia a paquete inexistente). Usar `gz_sim.launch.py`.

7. **`robot_params.yaml` tiene `wheelbase: 0.18`** pero el SDF usa `wheel_separation: 0.19`.
   El `dead_reckoning.py` usa el parámetro del launch (`wheel_separation: 0.19`). Hay
   inconsistencia — al pasar al robot real medir la separación real y unificar.

---

## Topic map (lo que existe hoy)

```
Gazebo Fortress
  └─→ /model/puzzlebot/cmd_vel      ← teleop / steering_controller (futuro)
  └─→ /model/puzzlebot/odometry     (referencia, no usado por dead_reckoning)
  └─→ /clock                        → use_sim_time
  └─→ /scan                         → mcl, slam_node, obstacle_avoidance
  └─→ /world/*/model/puzzlebot/joint_state
  └─→ /world/*/dynamic_pose/info    → ground_truth_odom (mode=mapping default)
                  │
                  ▼
         dead_reckoning
                  │
                  └─→ /odom   →  mcl (moves particles)
                  └─→ TF odom→base_footprint

mcl (mode=mcl, world=maze):
  └─→ /mcl/particles  (PoseArray — visualización)
  └─→ /mcl/pose       (PoseStamped — mejor estimado)
  └─→ /mcl/map        (OccupancyGrid latched — mapa PNG como grid)
  └─→ TF map→odom     (corrección de localización)

slam_node (mode=mapping):
  └─→ /odom           (desde ground_truth_odom por default, o dead_reckoning si se solicita)
  └─→ /map            (OccupancyGrid latched — construido en tiempo real)
  └─→ TF map→odom     (identity — el mapa crece desde la pose inicial)
```

---

## Parámetros importantes del maze

| Parámetro | Valor | Dónde se usa |
|-----------|-------|--------------|
| maze_map.png tamaño | 206×221 px | mcl.py |
| map_origin_x (MCL) | -5.54 m | gz_sim.launch.py → mcl params |
| map_origin_y (MCL) | -8.10 m | gz_sim.launch.py → mcl params |
| map_resolution | 0.05 m/px | mcl + slam_node |
| slam map origen | (-12.5, -12.5) | slam_params.yaml |
| slam map tamaño | 500×500 px = 25×25 m | slam_params.yaml |
| slam p_occ / p_free | 0.75 / 0.45 | slam_params.yaml |
| slam pose buffer | 3.0 s, max age 0.20 s | slam_params.yaml |
| keyframes / scan matching | desactivados por default | slam_params.yaml |
| wheel_separation | 0.19 m | gz_sim.launch.py → dead_reckoning |
| wheel_radius | 0.05 m | gz_sim.launch.py → dead_reckoning |

---

## Pasos para transferir al robot real

1. Crear `src/puzzlebot_bringup/launch/real_robot.launch.py`:
   - `dead_reckoning` con `input_source: encoders` (topics `/velocity_enc_r`, `/velocity_enc_l`)
   - `use_sim_time: False` en TODOS los nodos
   - Sin Gazebo, sin bridge, sin RSP basado en SDF

2. Medir `wheelbase` real físicamente y actualizar `robot_params.yaml`.

3. Calibrar zona muerta de motores — añadir parámetro `motor_deadband` al steering controller.

4. Para SLAM real, recordar que no existe `ground_truth_odom`: el mapa dependerá
   completamente de `/odom` por encoders/IMU. Mapear lento, con giros suaves, y
   validar primero en un entorno pequeño.

5. Ajustar `hit_sigma` del MCL de 0.20 → ~0.35 para compensar ruido real del RPLIDAR.

6. Tunear PID/Pure Pursuit en el robot con trayectoria simple (línea recta → cuadrado).
