# puzzlebot_sim — CLAUDE.md

## Resumen del repositorio
Workspace ROS 2 Humble para el robot diferencial Puzzlebot (Jetson + LiDAR).  
Contiene: simulación Gazebo, SLAM, localización, planificación, percepción, reconocimiento de voz y un dashboard web en tiempo real con **control bidireccional**.

**IMPORTANTE — reglas de seguridad:**
- Nunca hagas `git push` o `git commit` automáticamente desde Claude Code.
- Nunca borres archivos sin confirmar con el usuario.
- El bridge puede publicar `/cmd_vel`, `/goal_pose`, `/navigate_to_waypoint` y `/slam/reset` desde el dashboard — esto es intencional y controlado.
- **NUNCA** publicar `/initialpose` desde el bridge o el frontend.
- El bridge NO hace planeación, navegación ni evasión — solo retransmite comandos del usuario.

---

## Partes del repositorio

### ROS 2 (`src/`)
| Paquete | Tipo | Rol |
|---|---|---|
| `puzzlebot_bringup` | Python | Launch files para simulación y robot físico |
| `puzzlebot_control` | Python | State machine de misión logística (QR → logos → entrega) |
| `puzzlebot_controller` | C++ | Pure-pursuit steering |
| `puzzlebot_description` | CMake | URDF, SDF, meshes, RViz |
| `puzzlebot_localization` | C++ | Odometría + Kalman filter + scan_restamper |
| `puzzlebot_logo_detector` | Python | Detección de logos de clientes (Pepsi/Amazon/Walmart) con YOLO11n ONNX |
| `puzzlebot_msgs` | CMake/rosidl | Mensajes custom |
| `puzzlebot_perception` | Python | Cámara, ArUco, YOLO, QR codes |
| `puzzlebot_planning` | Python | A* planner + obstacle avoidance + waypoint navigator |
| `puzzlebot_slam` | Python | SLAM log-odds + MCL + map_server |
| `puzzlebot_voice_commands` | Python | Reconocimiento de voz offline (MFCC + HMM) |
| `puzzlebot_web_bridge` | Python | Bridge ROS 2 ↔ WebSocket (bidireccional) |
| `shared_utils` | Python | Utilidades compartidas |

### Dashboard web (`web_dashboard/`)
Frontend React + Vite. **Visualización + control del robot.**  
Paneles: SLAM Map (click-to-goal), LiDAR, Cámara, Teleop (D-pad con toggle Robot/Elevador), Misión (strip siempre visible), Waypoints, Modo, Voz, Logs.  
Tabs: **Modo | Waypoints | Voz** (3 tabs). Panel de Misión siempre visible fuera de tabs.  
Ver [web_dashboard/CLAUDE.md](web_dashboard/CLAUDE.md) para detalles del frontend.

### Mock (`mock/`)
Paquete temporal `puzzlebot_mock` para probar el dashboard **sin el robot físico**.  
Publica datos simulados (odometría, LiDAR, mapa, cámara). **Borrar cuando ya no se necesite.**

```bash
# Build (desde workspace root):
colcon build --base-paths src mock --packages-select puzzlebot_mock puzzlebot_web_bridge puzzlebot_voice_commands

# Lanzar todo:
ros2 launch puzzlebot_mock mock_test.launch.py \
  artifact_dir:=src/puzzlebot_voice_commands/artifacts_final
```

### Documentación (`docs/`)
- `architecture.md` — arquitectura del sistema ROS
- `slam_mapping.md` — teoría e implementación del SLAM
- `setup.md` — configuración de Ubuntu 22.04
- `workflow.md` — convenciones Git
- `web_dashboard_architecture.md` — arquitectura del dashboard

---

## Arquitectura del flujo de datos

```
/velocity_enc_r, /velocity_enc_l
  → odometry_node (C++)
  → /odom_raw
  → kalman_filter_node (C++)
  → /odom
  → slam_node (Python) + /scan
  → /map (nav_msgs/OccupancyGrid)

/odom, /scan, /map, /cmd_vel, /cmd_vel_in, /voice/*
  → puzzlebot_web_bridge (bridge_node.py)   ← ROS → WebSocket
  → WebSocket JSON (ws://0.0.0.0:8000/ws)
  → web_dashboard (React + Vite)

web_dashboard (botones/clic en mapa)
  → WebSocket JSON                          ← WebSocket → ROS
  → puzzlebot_web_bridge (bridge_node.py)
  → /cmd_vel | /goal_pose | /navigate_to_waypoint | /slam/reset
```

---

## Protocolo de comandos WebSocket (dashboard → bridge)

```json
{ "type": "cmd_vel",              "linear_x": 0.2, "angular_z": 0.5 }
{ "type": "goal_pose",            "x": 1.5, "y": 2.3, "theta": 0.0 }
{ "type": "navigate_to_waypoint", "name": "centro" }
{ "type": "slam_reset" }
{ "type": "mission_start",        "mission": "1" }
{ "type": "mission_start",        "mission": "2" }
{ "type": "mission_stop" }
{ "type": "elevator",             "action": "up" }
```

> **Misión 2**: el nodo de misión queda en estado `WAITING_FOR_GOAL` esperando
> un `goal_pose` del dashboard (click en mapa) para navegar al punto de inicio.

---

## Tópicos core

| Tópico | Tipo | Fuente |
|---|---|---|
| `/odom` | nav_msgs/Odometry | kalman_filter_node |
| `/scan` | sensor_msgs/LaserScan | LiDAR hardware |
| `/map` | nav_msgs/OccupancyGrid | slam_node |
| `/cmd_vel` | geometry_msgs/Twist | teleop dashboard / avoidance |
| `/goal_pose` | geometry_msgs/PoseStamped | dashboard (clic en mapa) / waypoint_navigator |
| `/slam/reset` | std_msgs/Bool | dashboard (botón "Iniciar Mapeo") |

## Tópicos opcionales

| Tópico | Tipo | Fuente |
|---|---|---|
| `/cmd_vel_in` | geometry_msgs/Twist | planificación (antes de evasión) |
| `/navigate_to_waypoint` | std_msgs/String | dashboard (WaypointPanel) / state_machine_node |
| `/mission_start` | std_msgs/String | dashboard → state_machine_node |
| `/mission_state` | std_msgs/String | state_machine_node → dashboard |
| `/qr/detections` | std_msgs/String | qr_node (JSON: `[{data, corners, area_px, center}]`) |
| `/logo_detection/result` | std_msgs/String | logo_detector_node (JSON: `[{class_name, confidence, bbox}]`) |
| `/forklift/command` | std_msgs/String | state_machine_node → montacargas (stub) |
| `/mission/markers` | visualization_msgs/MarkerArray | state_machine_node → RViz (punto+etiqueta donde se vio cada QR/logo) |
| `/voice/command` | std_msgs/String | voice_commands_node |
| `/voice/confidence` | std_msgs/Float32 | voice_commands_node |
| `/voice/status` | std_msgs/String | voice_commands_node |
| `/voice/ranked_predictions` | std_msgs/String | voice_commands_node |
| `/voice/inference_time_ms` | std_msgs/Float32 | voice_commands_node |

---

## Comandos comunes

### Build ROS 2
```bash
cd ~/puzzlebot_sim
colcon build
source install/setup.bash
```

Build solo el bridge:
```bash
colcon build --packages-select puzzlebot_web_bridge
source install/setup.bash
```

### Simulación Gazebo — mapeo con dashboard
```bash
# Terminal 1: Gazebo + SLAM + bridge (web_bridge activo por defecto)
ros2 launch puzzlebot_bringup gz_sim.launch.py \
  world:=flat_plane mode:=mapping slam:=true

# Terminal 2: Dashboard
cd web_dashboard && npm run dev -- --host 0.0.0.0
```

### Simulación Gazebo — navegación autónoma + dashboard
```bash
ros2 launch puzzlebot_bringup gz_sim.launch.py \
  world:=flat_plane mode:=mapping slam:=true \
  navigation:=true web_bridge:=true
```
> `navigation:=true` activa automáticamente `scan_restamper` para que los nodos de navegación
> reciban `/scan_stamped` correctamente desde Gazebo.

### Ejecutar el bridge standalone (robot físico)
```bash
ros2 run puzzlebot_web_bridge bridge_node
# En Gazebo, pasar cmd_vel_out_topic diferente:
ros2 run puzzlebot_web_bridge bridge_node \
  --ros-args -p cmd_vel_out_topic:=/model/puzzlebot/cmd_vel
```

### Ejecutar el frontend
```bash
cd web_dashboard
cp .env.example .env   # ajustar VITE_WS_URL si el bridge está en otra máquina
npm install
npm run dev -- --host 0.0.0.0
```

Acceso local: `http://localhost:5173`  
Acceso desde otra laptop en la misma red: `http://<IP_MAQUINA>:5173`

### SLAM en robot físico
```bash
ros2 launch puzzlebot_bringup slam.launch.py
```

---

## Reglas para Claude Code
1. No ejecutar `git push` ni `git commit` automáticamente.
2. No borrar archivos sin confirmación.
3. El bridge SÍ puede publicar a `/cmd_vel`, `/goal_pose`, `/navigate_to_waypoint`, `/slam/reset`, `/mission_start`, `/forklift/command` — son comandos explícitos del usuario desde el dashboard.
4. El bridge NUNCA publica a `/initialpose`.
5. Al agregar dependencias Python al bridge, actualizar `package.xml` y `setup.py`.
6. Al agregar dependencias npm al frontend, usar solo lo estrictamente necesario.
7. `cmd_vel_out_topic` del bridge debe ser `/model/puzzlebot/cmd_vel` en Gazebo y `/cmd_vel` en robot físico.
8. La lógica de misión vive en `state_machine_node.py` (`puzzlebot_control`). No duplicarla en el bridge ni en el frontend.
9. `state_machine_node.py` reemplaza funcionalmente al esqueleto anterior — no lanzar ambos en simultáneo.
