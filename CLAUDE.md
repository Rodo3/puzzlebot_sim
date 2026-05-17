# puzzlebot_sim — CLAUDE.md

## Resumen del repositorio
Workspace ROS 2 Humble para el robot diferencial Puzzlebot (Jetson + LiDAR).  
Contiene: simulación Gazebo, SLAM, localización, planificación, percepción, reconocimiento de voz y un dashboard web en tiempo real.

**IMPORTANTE — reglas de seguridad:**
- Nunca publiques comandos de control desde el bridge o el frontend: `/cmd_vel`, `/goal_pose`, `/initialpose`.
- Nunca hagas `git push` o `git commit` automáticamente desde Claude Code.
- Nunca borres archivos sin confirmar con el usuario.

---

## Partes del repositorio

### ROS 2 (`src/`)
| Paquete | Tipo | Rol |
|---|---|---|
| `puzzlebot_bringup` | Python | Launch files para simulación y robot físico |
| `puzzlebot_control` | Python | State machine de misión |
| `puzzlebot_controller` | C++ | Pure-pursuit steering |
| `puzzlebot_description` | CMake | URDF, SDF, meshes, RViz |
| `puzzlebot_localization` | C++ | Odometría + Kalman filter |
| `puzzlebot_msgs` | CMake/rosidl | Mensajes custom |
| `puzzlebot_perception` | Python | Cámara, ArUco, YOLO |
| `puzzlebot_planning` | Python | A* planner + obstacle avoidance |
| `puzzlebot_slam` | Python | SLAM log-odds + MCL |
| `puzzlebot_voice_commands` | Python | Reconocimiento de voz offline (MFCC + HMM) |
| `puzzlebot_web_bridge` | Python | **NUEVO** — Bridge ROS 2 → WebSocket |
| `shared_utils` | Python | Utilidades compartidas |

### Dashboard web (`web_dashboard/`)
Frontend React + Vite. Solo visualización. Se conecta al bridge vía WebSocket.  
Ver [web_dashboard/CLAUDE.md](web_dashboard/CLAUDE.md) para detalles del frontend.

### Documentación (`docs/`)
- `architecture.md` — arquitectura del sistema ROS
- `slam_mapping.md` — teoría e implementación del SLAM
- `setup.md` — configuración de Ubuntu 22.04
- `workflow.md` — convenciones Git
- `web_dashboard_architecture.md` — **NUEVO** — arquitectura del dashboard

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
  → puzzlebot_web_bridge (bridge_node.py)
  → WebSocket JSON (ws://0.0.0.0:8000/ws)
  → web_dashboard (React + Vite)
```

---

## Tópicos core

| Tópico | Tipo | Fuente |
|---|---|---|
| `/odom` | nav_msgs/Odometry | kalman_filter_node |
| `/scan` | sensor_msgs/LaserScan | LiDAR hardware |
| `/map` | nav_msgs/OccupancyGrid | slam_node |
| `/cmd_vel` | geometry_msgs/Twist | steering / avoidance |

## Tópicos opcionales

| Tópico | Tipo | Fuente |
|---|---|---|
| `/cmd_vel_in` | geometry_msgs/Twist | planificación (antes de evasión) |
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

### Ejecutar el bridge
```bash
ros2 run puzzlebot_web_bridge bridge_node
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

### Simulación Gazebo
```bash
ros2 launch puzzlebot_bringup gz_sim.launch.py
```

### SLAM en robot físico
```bash
ros2 launch puzzlebot_bringup slam.launch.py
```

---

## Reglas para Claude Code
1. No ejecutar `git push` ni `git commit` automáticamente.
2. No borrar archivos sin confirmación.
3. No publicar a tópicos de control desde ningún componente nuevo.
4. El bridge es solo lectura de tópicos.
5. El frontend es solo visualización.
6. Al agregar dependencias Python al bridge, actualizar `package.xml` y `setup.py`.
7. Al agregar dependencias npm al frontend, usar solo lo estrictamente necesario.
