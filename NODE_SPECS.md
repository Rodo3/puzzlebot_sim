# Puzzlebot — Especificaciones de Nodos (Robot Real)

> Arquitectura: Jetson Orin publica sensores vía micro-ROS; todo el cómputo corre en el PC del operador.

---

## Working Tree de Nodos

```
[Jetson Orin — micro-ROS]
   /VelocityEncR  (BEST_EFFORT, ~20 Hz)
   /VelocityEncL  (BEST_EFFORT, ~20 Hz)
   /Lidar o /scan (BEST_EFFORT, ~10 Hz)
   /camera/image/compressed (BEST_EFFORT, ~30 Hz)
   /cmd_vel       (escucha)
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│                         PC del Operador                          │
│                                                                  │
│  [scan_restamper]  ──/scan_stamped──▶  [slam_node]              │
│       ↑ /Lidar o /scan (BEST_EFFORT)       │                     │
│                                            │ /map (TRANSIENT)    │
│  [odometry_node]  ──/odom_raw──▶  [kalman_filter_node]          │
│       ↑ /VelocityEncR,L                    │ /odom + TF          │
│         (SensorDataQoS)                    │ odom→base_footprint │
│                                            ▼                     │
│                              [aruco_node]  ──/aruco/pose──▶  ┐  │
│                                    ↑                          │  │
│                   /camera/image/compressed                    │  │
│                                                               │  │
│              [aruco_map_odom]  ◀──/aruco/pose + /odom         │  │
│                    │ TF map→odom                              │  │
│                    ▼                                          │  │
│              [slam_node]  ──/map──▶  [path_planner_node]  ◀──┘  │
│                                           │ /planned_path        │
│                                           ▼                      │
│                              [steering_controller_node]          │
│                                    │ /cmd_vel_steering           │
│                                    ▼                             │
│                         [bug_navigation_node]                    │
│                                    │ /cmd_vel_in                 │
│                                    ▼                             │
│                         [obstacle_avoidance_node]               │
│                                    │ /cmd_vel ──────────────────▶│
│                                    ↑ /scan_stamped               │
│                                    ↑ /odom (covarianza EKF)      │
│                                                                  │
│  [state_machine_node]  ──/goal_pose──▶ path_planner_node        │
│  [map_server_node]     ──/map──▶ path_planner_node + RViz       │
│  [mcl]                 ──TF map→odom                            │
└──────────────────────────────────────────────────────────────────┘
```

---

## Tabla de Nodos

| Nodo | Paquete | Frecuencia | Trigger | Topics que publica | Topics que suscribe |
|------|---------|------------|---------|-------------------|---------------------|
| `odometry_node` | puzzlebot_localization (C++) | **20 Hz** (timer 50 ms) | wall_timer | `/odom` o `/odom_raw`, TF `odom→base_footprint` | `/VelocityEncR`, `/VelocityEncL` |
| `kalman_filter_node` | puzzlebot_localization (C++) | **event-driven** | mensaje entrante | `/odom`, TF `odom→base_footprint` | `/odom_raw`, `/aruco/pose`, `/scan_match/pose` |
| `scan_restamper` | puzzlebot_localization (Python) | **pass-through** | mensaje entrante | `/scan_stamped` (RELIABLE, depth=10) | `/Lidar` o `/scan` (BEST_EFFORT) |
| `aruco_node` | puzzlebot_perception (Python) | **≤ 8 Hz** (throttled) | mensaje entrante | `/aruco/pose`, `/aruco/markers` | `/camera/image/compressed` (SensorDataQoS) |
| `aruco_map_odom` | puzzlebot_localization (Python) | **20 Hz** (timer) | wall_timer | TF `map→odom`, `/map_to_odom` | `/aruco/pose`, `/odom` |
| `slam_node` | puzzlebot_slam (Python) | TF: **10 Hz** / mapa: **1 Hz** | timers + evento scan | `/map` (TRANSIENT_LOCAL), `/scan_match/pose`, TF `map→odom` | `/scan_stamped` (SensorDataQoS), `/odom` |
| `mcl` | puzzlebot_slam (Python) | **event-driven** (scan) | mensaje entrante | `/mcl/pose`, `/mcl/particles`, `/mcl/map`, TF `map→odom` | `/scan_stamped`, `/odom` |
| `map_server_node` | puzzlebot_slam (Python) | latched (una vez) | arranque | `/map` (TRANSIENT_LOCAL) | — |
| `path_planner_node` | puzzlebot_planning (Python) | **event-driven** (goal/pose) | mensaje entrante | `/planned_path` (depth=1) | `/map` (TRANSIENT_LOCAL), `/odom`, `/goal_pose` |
| `steering_controller_node` | puzzlebot_controller (C++) | **20 Hz** (timer 50 ms) | wall_timer | `/cmd_vel_steering` | `/odom`, `/planned_path` |
| `bug_navigation_node` | puzzlebot_planning (Python) | **event-driven** | mensaje entrante | `/cmd_vel_in`, blobs en `/map` | `/cmd_vel_steering`, `/scan_stamped`, `/odom` |
| `obstacle_avoidance_node` | puzzlebot_planning (Python) | **event-driven** | mensaje entrante | `/cmd_vel` | `/scan_stamped` (SensorDataQoS), `/cmd_vel_in`, `/odom` |
| `state_machine_node` | puzzlebot_control (Python) | estado: **1 Hz** | wall_timer + evento | `/mission_state`, `/goal_pose_out` | `/goal_pose`, `/planned_path`, `/detections` |
| `robot_state_publisher` | ros2 built-in | event-driven | mensaje entrante | TF `base_footprint→*` | `/joint_states` |

---

## Frecuencias de Sensores (Jetson → PC)

| Fuente | Topic | QoS | Frecuencia estimada |
|--------|-------|-----|---------------------|
| Encoders rueda derecha | `/VelocityEncR` | BEST_EFFORT | ~20 Hz |
| Encoders rueda izquierda | `/VelocityEncL` | BEST_EFFORT | ~20 Hz |
| LiDAR (micro-ROS) | `/Lidar` | BEST_EFFORT | ~10 Hz |
| LiDAR (sllidar directo) | `/scan` | BEST_EFFORT | ~10 Hz |
| Cámara comprimida | `/camera/image/compressed` | BEST_EFFORT | ~30 Hz |

---

## Latencia y Buffers

### Reglas generales
- **Sensores de la Jetson → BEST_EFFORT**: evita que mensajes atrasados llenen el buffer. Usar siempre `SensorDataQoS` o `BEST_EFFORT` en los suscriptores de encoders, LiDAR y cámara.
- **Datos de mapa → TRANSIENT_LOCAL + RELIABLE + depth=1**: garantiza que nodos que arrancan tarde reciban el último mapa sin esperar una republución.
- **Comandos de velocidad → depth=10, RELIABLE**: cola pequeña para que el robot no ejecute comandos obsoletos si hubo lag.

### Parámetros críticos de latencia

| Parámetro | Valor | Dónde | Efecto |
|-----------|-------|-------|--------|
| `max_processing_hz` (aruco_node) | 8 Hz | aruco_node | Limita solvePnP para no acumular backlog de frames de cámara |
| `max_scan_pose_age` (slam_node) | 0.20 s | slam_params.yaml | Descarta scans con pose más vieja de 200 ms → evita doble-pared |
| `pose_buffer_sec` (slam_node) | 3.0 s | slam_params.yaml | Ventana de búsqueda de pose cercana en tiempo al scan |
| `cov_timeout_sec` (obstacle_avoidance) | 2.0 s | controller_params.yaml | Si `/odom` no llega en 2 s → parada de emergencia |
| Timer odometry | 50 ms | odometry_node.cpp | Publicación de odom a 20 Hz aunque encoders lleguen irregulares |
| Timer steering | 50 ms | steering_controller_node.cpp | Control de velocidad a 20 Hz |
| Timer slam TF | 100 ms | slam_node.py | Broadcast `map→odom` a 10 Hz |
| Timer slam mapa | 1000 ms | slam_node.py | Publicación `/map` a 1 Hz (costoso) |
| Timer aruco_map_odom | 50 ms | aruco_map_odom | Corrección `map→odom` a 20 Hz |
| `correction_alpha` (aruco_map_odom) | 0.35 | real_robot.launch.py | Suavizado exponencial de la corrección; evita saltos bruscos en TF |

### No saturar buffers — checklist

1. **Cámara**: `aruco_node` procesa máx. 8 Hz aunque la cámara llegue a 30 Hz. El resto de frames se descarta en el callback (throttle por tiempo, no por cola).
2. **LiDAR**: `scan_restamper` es pass-through sin buffer extra; re-sella el timestamp al reloj del PC para corregir deriva de reloj de la Jetson.
3. **SLAM**: keyframes descartan scans si el robot se movió < 10 cm o < 5°, reduciendo carga de integración.
4. **Path planner**: publica `/planned_path` con `depth=1`; el steering siempre lee la trayectoria más reciente.
5. **Obstacle avoidance**: parada si covarianza EKF > 0.8 m² (`cov_stop_threshold`), evitando maniobras con localización perdida.
6. **Kalman**: gating de Mahalanobis (umbral 3.5σ) rechaza correcciones de scan match espurias antes de actualizar el estado.

---

## Modos de operación y nodos activos

| Modo | Nodos activos adicionales | Dueño de `map→odom` |
|------|--------------------------|---------------------|
| Mapeo clásico (`slam:=true aruco:=true kalman:=false`) | slam_node, aruco_node, aruco_map_odom, odometry_node (directo) | aruco_map_odom |
| Mapeo EKF+ArUco (`slam:=true kalman:=true aruco:=true`) | slam_node, aruco_node, kalman_filter_node, odometry_node (raw) | slam_node |
| Localización MCL (`slam:=false mcl:=true`) | mcl, map_server_node, aruco_node, odometry_node | mcl |
| Localización EKF+mapa (`use_map:=true kalman:=true aruco:=true`) | slam_loc, map_server_node, aruco_node, kalman_filter_node, aruco_map_odom | aruco_map_odom |
| Navegación autónoma (`navigation:=true`) | + path_planner_node, steering_controller_node, bug_navigation_node, obstacle_avoidance_node | según modo base |
