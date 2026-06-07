# Web Dashboard Architecture

## Overview

```
ROS 2 topics (robot físico)
    │
    ▼
puzzlebot_web_bridge  (src/puzzlebot_web_bridge/)
    │  FastAPI + uvicorn + WebSocket
    │  ws://0.0.0.0:8000/ws
    │  http://0.0.0.0:8000/health
    │
    ▼
web_dashboard  (web_dashboard/)
    React 18 + Vite 5
    http://0.0.0.0:5173  (dev)
```

**El flujo es bidireccional**:
- **ROS → bridge → frontend**: datos del robot (odom, scan, map, cámara, `mission_state`,
  `qr_detections`, `logo_detection`, voz).
- **frontend → bridge → ROS**: comandos del usuario (`cmd_vel`, `goal_pose`,
  `navigate_to_waypoint`, `slam_reset`, `mission_start`/`mission_stop`, `elevator`).

> **Seguridad:** el bridge NUNCA publica `/initialpose` y no hace planeación ni
> navegación — solo retransmite comandos explícitos del usuario. La lógica de la
> misión vive en `state_machine_node`, no en el bridge ni en el frontend.

---

## Componentes del bridge

### bridge_node.py
Nodo ROS 2. Declara parámetros, crea suscriptores, arrancar el WebSocket server.

### websocket_server.py
FastAPI app con un endpoint `/ws` (WebSocket) y `/health` (HTTP).  
Corre en un hilo daemon con su propio event loop asyncio.  
`broadcast_sync()` permite llamarlo desde callbacks ROS (hilo diferente).

### serializers.py
Funciones puras de conversión ROS msg → dict JSON-safe.  
Filtra NaN/inf del LiDAR. Convierte quaternion a yaw.

### rate_limiter.py
`RateLimiter(max_hz).should_send()` — decisión de throttle sin dependencias externas.

### topic_config.py
Única fuente de verdad para nombres de tópico por defecto y rate limits.

---

## Formato JSON de mensajes WebSocket

### robot_state (desde /odom)
```json
{
  "type": "robot_state",
  "timestamp": 1710000000.25,
  "pose": { "x": 1.25, "y": 0.74, "theta": 1.57 },
  "odom_twist": { "linear_x": 0.16, "angular_z": 0.03 }
}
```

### map (desde /map)
```json
{
  "type": "map",
  "timestamp": 1710000001.10,
  "width": 500,
  "height": 500,
  "resolution": 0.05,
  "origin": { "x": -12.5, "y": -12.5 },
  "data": [0, -1, 100, 0]
}
```
`data` es un array plano row-major. Valores: 0=libre, 100=ocupado, -1=desconocido.

### scan (desde /scan)
```json
{
  "type": "scan",
  "timestamp": 1710000002.20,
  "angle_min": -3.14,
  "angle_max": 3.14,
  "angle_increment": 0.01,
  "range_min": 0.12,
  "range_max": 3.5,
  "min_distance": 0.47,
  "ranges": [1.2, 1.15, null, 0.80]
}
```
`null` en `ranges` indica valor inf/NaN filtrado por el bridge.

### velocity_command (desde /cmd_vel y /cmd_vel_in)
```json
{
  "type": "velocity_command",
  "timestamp": 1710000003.30,
  "source": "cmd_vel",
  "linear_x": 0.20,
  "angular_z": 0.00
}
```
`source` puede ser `"cmd_vel"` o `"cmd_vel_in"`.

### voice_command (desde /voice/*)
```json
{
  "type": "voice_command",
  "timestamp": 1710000004.40,
  "command": "izquierda",
  "confidence": 0.94,
  "status": "command_detected",
  "inference_time_ms": 0.21,
  "ranked_predictions": [
    { "command": "izquierda", "score": 0.94 },
    { "command": "derecha",   "score": 0.04 },
    { "command": "alto",      "score": 0.01 }
  ]
}
```
Se envía al llegar `/voice/command`. Los otros campos se acumulan desde sus tópicos y se incluyen con el último valor conocido.

### mission_state (desde /mission_state)
```json
{ "type": "mission_state", "state": "SCANNING_QR", "timestamp": 1710000005.5 }
```
Estado actual de la máquina de misión. Alimenta el badge del `MissionPanel`.
Valores: `IDLE / WAITING_FOR_GOAL / GOING_TO_START / SCANNING_QR / FORKLIFT_UP /
NAVIGATING_TO_DOCKS / SCANNING_LOGOS / FORKLIFT_DOWN / DONE / ERROR`.

### qr_detections (desde /qr/detections) — overlay de cámara
```json
{ "type": "qr_detections", "timestamp": 1710000006.0,
  "detections": [{ "data": "wolmar",
                   "corners": [[x0,y0],[x1,y1],[x2,y2],[x3,y3]],
                   "area_px": 5120.0,
                   "center": { "x": 318.0, "y": 240.0, "nx": -0.006, "ny": 0.0 } }] }
```

### logo_detection (desde /logo_detection/result) — overlay de cámara
```json
{ "type": "logo_detection", "timestamp": 1710000006.2,
  "detections": [{ "class_name": "Walmart", "confidence": 0.91,
                   "bbox": { "x1": 120.0, "y1": 80.0, "x2": 210.0, "y2": 170.0 } }] }
```
`corners` (QR) y `bbox` (logo) están en píxeles del frame original → el `CameraPanel`
los dibuja directamente sobre el stream crudo.

### Comandos enviados (frontend → bridge → ROS)
```json
{ "type": "cmd_vel",       "linear_x": 0.2, "angular_z": 0.5 }
{ "type": "goal_pose",     "x": 1.5, "y": 2.3, "theta": 0.0 }
{ "type": "navigate_to_waypoint", "name": "centro" }
{ "type": "slam_reset" }
{ "type": "mission_start", "mission": "1" }   // o "2"
{ "type": "mission_stop" }
{ "type": "elevator",      "action": "up" }   // up | down | stop
```

---

## Rate limits

| Tópico     | Hz máx | Justificación |
|------------|--------|---------------|
| /odom      | 10     | Pose suficientemente fluida sin saturar |
| /cmd_vel   | 10     | Sincrono con odometría |
| /cmd_vel_in| 10     | Sincrono con cmd_vel |
| /scan      | 5      | Canvas update a ~5fps es suficiente |
| /map       | 1      | El mapa es pesado; cambia lentamente |
| /voice/*   | ∞      | Por evento, baja frecuencia natural |
| /mission_state | ∞  | Por evento + heartbeat 2 Hz |
| /qr/detections | ∞  | Por evento (overlay cámara) |
| /logo_detection/result | ∞ | Por evento (overlay cámara) |

---

## Parámetros ROS del bridge

```bash
ros2 run puzzlebot_web_bridge bridge_node \
  --ros-args \
  -p websocket_host:=0.0.0.0 \
  -p websocket_port:=8000 \
  -p odom_topic:=/odom \
  -p scan_topic:=/scan \
  -p map_topic:=/map \
  -p cmd_vel_topic:=/cmd_vel \
  -p cmd_vel_in_topic:=/cmd_vel_in \
  -p voice_command_topic:=/voice/command \
  -p voice_confidence_topic:=/voice/confidence \
  -p voice_status_topic:=/voice/status \
  -p voice_ranked_predictions_topic:=/voice/ranked_predictions \
  -p voice_inference_time_topic:=/voice/inference_time_ms
```

---

## Instrucciones de despliegue

### Desarrollo (recomendado para laboratorio)
```bash
# Terminal 1 — ROS 2 stack
ros2 launch puzzlebot_bringup slam.launch.py

# Terminal 2 — Bridge
ros2 run puzzlebot_web_bridge bridge_node

# Terminal 3 — Frontend
cd web_dashboard
npm run dev -- --host 0.0.0.0
```

### Producción (para demo sin herramientas de dev)
```bash
cd web_dashboard
npm run build      # genera dist/
cd dist
python3 -m http.server 8080   # servir en :8080
```

Apuntar el navegador a `http://<IP_MAQUINA>:8080`.  
Asegurarse de que `VITE_WS_URL` esté configurado con la IP correcta antes del build.

---

## Consideraciones de red

- El bridge y el frontend pueden correr en la misma máquina o en máquinas diferentes.
- Si están separados, el VITE_WS_URL debe apuntar a la IP de la máquina con el bridge.
- Puerto 8000 (bridge) y 5173 (dev frontend) deben ser accesibles en la red del laboratorio.
- Para producción, compilar con `npm run build` y servir `dist/` estáticamente.
