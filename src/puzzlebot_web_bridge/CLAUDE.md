# puzzlebot_web_bridge — CLAUDE.md

## Propósito
Paquete ROS 2 Python (ament_python) que actúa como puente entre los tópicos del Puzzlebot y el dashboard web.  
Suscribe tópicos ROS 2, serializa los mensajes a JSON y los transmite por WebSocket a todos los clientes conectados.

## Archivos principales

| Archivo | Rol |
|---|---|
| `bridge_node.py` | Nodo ROS 2. Declara parámetros, crea suscriptores, arranca el servidor. |
| `websocket_server.py` | FastAPI + uvicorn en hilo daemon. Expone `/ws` y `/health`. |
| `serializers.py` | Funciones puras: `odom_to_json`, `scan_to_json`, `map_to_json`, `twist_to_json`, `voice_to_json`. |
| `rate_limiter.py` | `RateLimiter(max_hz)` — decide si se debe enviar según tiempo transcurrido. |
| `topic_config.py` | Nombres de tópico por defecto y rate limits (Hz). Editar aquí para cambiar defaults. |

## Tópicos que escucha

### Core (siempre presentes en robot físico)
- `/odom` — nav_msgs/Odometry → tipo `robot_state`
- `/scan` — sensor_msgs/LaserScan → tipo `scan`
- `/map` — nav_msgs/OccupancyGrid → tipo `map`
- `/cmd_vel` — geometry_msgs/Twist → tipo `velocity_command` (source: `cmd_vel`)

### Opcionales (silenciados si no existen)
- `/cmd_vel_in` — geometry_msgs/Twist → tipo `velocity_command` (source: `cmd_vel_in`)
- `/voice/command` — std_msgs/String
- `/voice/confidence` — std_msgs/Float32
- `/voice/status` — std_msgs/String
- `/voice/ranked_predictions` — std_msgs/String (JSON serializado)
- `/voice/inference_time_ms` — std_msgs/Float32

## Formato JSON enviado al frontend
Ver [web_dashboard_architecture.md](../../docs/web_dashboard_architecture.md) para el formato completo de cada tipo.

Tipos de mensaje: `robot_state`, `scan`, `map`, `velocity_command`, `voice_command`.

## Rate limits recomendados
| Tópico | Hz |
|---|---|
| /odom | 10 |
| /cmd_vel | 10 |
| /cmd_vel_in | 10 |
| /scan | 5 |
| /map | 1 |
| /voice/* | sin límite (por evento) |

## Cómo correr el nodo

```bash
# Desde el workspace
cd ~/puzzlebot_sim
colcon build --packages-select puzzlebot_web_bridge
source install/setup.bash
ros2 run puzzlebot_web_bridge bridge_node
```

Con parámetros personalizados:
```bash
ros2 run puzzlebot_web_bridge bridge_node \
  --ros-args -p websocket_port:=8080 -p odom_topic:=/odom_filtered
```

## Endpoint WebSocket
- `ws://0.0.0.0:8000/ws` — flujo de datos
- `http://0.0.0.0:8000/health` — liveness check (`{"status":"ok","clients":N}`)

## Lo que el bridge NO debe hacer
- **NUNCA publicar** a `/cmd_vel`, `/goal_pose`, `/initialpose` ni ningún tópico de control.
- No recibir comandos desde el frontend para mover el robot.
- No hacer ningún tipo de planeación, navegación ni evasión.
- No fallar si los tópicos opcionales no existen — ROS 2 simplemente no recibe mensajes.

## Dependencias Python (instalar si no están presentes)
```bash
pip install fastapi uvicorn[standard] websockets
```
