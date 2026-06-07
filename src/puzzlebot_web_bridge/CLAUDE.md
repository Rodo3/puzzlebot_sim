# puzzlebot_web_bridge — CLAUDE.md

## Propósito
Paquete ROS 2 Python (ament_python) que actúa como **puente bidireccional** entre los tópicos del Puzzlebot y el dashboard web.

- **ROS → WebSocket**: suscribe tópicos, serializa a JSON, broadcast a clientes.
- **WebSocket → ROS**: recibe comandos JSON del browser, los publica en tópicos de control.
- **POST /audio**: inferencia de voz KMeans+HMM desde el micrófono del browser.

## Archivos principales

| Archivo | Rol |
|---|---|
| `bridge_node.py` | Nodo ROS 2. Parámetros, suscriptores, publicadores de control, `_handle_command`. |
| `websocket_server.py` | FastAPI + uvicorn en hilo daemon. `/ws`, `/health`, `/audio`. Maneja mensajes entrantes del browser. |
| `serializers.py` | Funciones puras: `odom_to_json`, `scan_to_json`, `map_to_json`, `twist_to_json`, `voice_to_json`. |
| `rate_limiter.py` | `RateLimiter(max_hz)` — throttle de publicación al WebSocket. |
| `topic_config.py` | Nombres de tópico y rate limits. **Editar aquí para cambiar defaults.** |

## Tópicos que escucha (ROS → WebSocket)

### Core
- `/odom` — nav_msgs/Odometry → `robot_state`
- `/scan` — sensor_msgs/LaserScan → `scan`
- `/map` — nav_msgs/OccupancyGrid → `map`
- `/cmd_vel` — geometry_msgs/Twist → `velocity_command` (source: `cmd_vel`)

### Opcionales
- `/cmd_vel_in`, `/voice/command`, `/voice/confidence`, `/voice/status`, `/voice/ranked_predictions`, `/voice/inference_time_ms`, `/camera/image/compressed`
- `/mission_state` — std_msgs/String → `mission_state` (estado actual de la máquina de misión, event-driven)
- `/qr/detections` — std_msgs/String → `qr_detections` (JSON, event-driven)
- `/logo_detection/result` — std_msgs/String → `logo_detection` (JSON, event-driven)

## Tópicos que publica (WebSocket → ROS)

| Tópico | Tipo | Comando dashboard | Notas |
|---|---|---|---|
| `cmd_vel_out_topic` | geometry_msgs/Twist | `"type":"cmd_vel"` | Default `/cmd_vel`; en Gazebo: `/model/puzzlebot/cmd_vel` |
| `/goal_pose` | geometry_msgs/PoseStamped | `"type":"goal_pose"` | QoS TRANSIENT_LOCAL (latched) |
| `/navigate_to_waypoint` | std_msgs/String | `"type":"navigate_to_waypoint"` | Nombre del waypoint |
| `/slam/reset` | std_msgs/Bool | `"type":"slam_reset"` | True → slam_node limpia el mapa |
| `/mission_start` | std_msgs/String | `"type":"mission_start"` | `"1"`, `"2"`, o `"stop"` |

## Parámetro crítico: `cmd_vel_out_topic`

```bash
# Robot físico (default)
ros2 run puzzlebot_web_bridge bridge_node

# Gazebo (DiffDrive plugin)
ros2 run puzzlebot_web_bridge bridge_node \
  --ros-args -p cmd_vel_out_topic:=/model/puzzlebot/cmd_vel
```

En `gz_sim.launch.py` este parámetro ya está configurado correctamente para Gazebo.

## Protocolo de comandos entrantes (JSON del browser)

```json
{ "type": "cmd_vel",              "linear_x": 0.2, "angular_z": 0.5 }
{ "type": "goal_pose",            "x": 1.5, "y": 2.3, "theta": 0.0, "frame_id": "map" }
{ "type": "navigate_to_waypoint", "name": "centro" }
{ "type": "slam_reset" }
{ "type": "mission_start",        "mission": "1" }
{ "type": "mission_start",        "mission": "2" }
{ "type": "mission_stop" }
{ "type": "elevator",             "action": "up" }
```

> `mission_start "2"` pone al `state_machine_node` en estado `WAITING_FOR_GOAL`.
> El siguiente `goal_pose` que llegue del dashboard (click en mapa) será el punto de inicio de la Misión 2.

## Mensajes salientes al browser (JSON)

| `type` | Tópico ROS | Descripción |
|---|---|---|
| `mission_state` | `/mission_state` | Estado actual: `IDLE/WAITING_FOR_GOAL/SCANNING_QR/FORKLIFT_UP/NAVIGATING_TO_DOCKS/SCANNING_LOGOS/FORKLIFT_DOWN/DONE/ERROR` |
| `qr_detections` | `/qr/detections` | JSON array con QR detectados `[{data, corners, area_px, center}]` |
| `logo_detection` | `/logo_detection/result` | JSON array de logos `[{class_name, confidence, bbox}]` (para overlay en cámara) |

## Endpoints HTTP/WebSocket
- `ws://0.0.0.0:8000/ws` — canal bidireccional con el dashboard
- `GET http://0.0.0.0:8000/health` — liveness check
- `POST http://0.0.0.0:8000/audio` — inferencia de voz WAV

## Parámetro artifact_dir
Ruta a `artifacts_final/` con los modelos KMeans+HMM. Si está vacío, `/audio` responde 503.

## Rate limits (WebSocket → browser)
| Tópico | Hz |
|---|---|
| /odom | 10 |
| /cmd_vel | 10 |
| /scan | 5 |
| /map | 1 |
| /camera | 10 |
| /voice/* | sin límite |

## Dependencias Python
```bash
pip install fastapi "uvicorn[standard]" websockets "numpy>=1.25" scipy librosa
```

## Restricciones
- NUNCA publicar `/initialpose`.
- El bridge NO hace planeación ni navegación — solo retransmite comandos del usuario.
- No fallar si los tópicos opcionales no existen.
- El bridge NO interpreta el estado de misión — solo retransmite `/mission_state` al dashboard y `/mission_start` al nodo.
