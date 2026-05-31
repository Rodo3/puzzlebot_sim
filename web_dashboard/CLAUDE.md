# web_dashboard — CLAUDE.md

## Propósito
Frontend React + Vite para visualización **y control** en tiempo real del Puzzlebot físico.  
Se conecta al `puzzlebot_web_bridge` vía WebSocket — recibe datos del robot Y envía comandos de control.

## Componentes principales

| Componente | Fuente de datos | Descripción |
|---|---|---|
| `App.jsx` | — | Raíz. Gestiona estado global, conexión WS, `sendCommand`. |
| `StatusPanel.jsx` | estado app | Indicadores de conexión y tópicos activos. |
| `SlamMap.jsx` | `map` + `robot_state` | Canvas 2D con OccupancyGrid, robot, trayectoria. En modo `navigation`: clic → envía `goal_pose`. Muestra marker verde en el goal activo. |
| `LidarView.jsx` | `scan` | Canvas 2D polar con puntos del LiDAR. |
| `CameraPanel.jsx` | `camera_frame` | Stream de imágenes JPEG en base64. |
| `ModePanel.jsx` | estado app | Botones "Iniciar Mapeo" (reset SLAM) / "Navegar" (activa click-to-goal). |
| `TeleopPanel.jsx` | — | D-pad (↑↓←→■) + sliders de velocidad. Envía `cmd_vel` al presionar. |
| `WaypointPanel.jsx` | — | Dropdown con 11 waypoints nombrados. Envía `navigate_to_waypoint`. Solo activo en modo `navigation`. |
| `VelocityPanel.jsx` | `velocity_command` | Muestra `/cmd_vel` y `/cmd_vel_in`, detecta obstacle stop. |
| `VoiceCommandPanel.jsx` | `voice_command` | Último comando, confianza, ranking, historial. |
| `LogsPanel.jsx` | eventos internos | Log de eventos frontend. |

## Servicios y utilidades

| Archivo | Rol |
|---|---|
| `services/websocketClient.js` | Conexión WS con auto-reconexión. Métodos: `close()`, `send(data)`. |
| `utils/geometry.js` | Conversión mundo→canvas, dibujo de flecha de orientación. |
| `utils/mapUtils.js` | Render de OccupancyGrid en ImageData, conversión coords. |

## Estado en App.jsx

```
connected       bool
lastUpdate      float (Unix timestamp)
robotState      { type, timestamp, pose: {x,y,theta}, odom_twist: {linear_x, angular_z} }
scanData        { type, timestamp, ranges[], ... }
mapData         { type, timestamp, width, height, resolution, origin: {x,y}, data[] }
cmdVel          { type, timestamp, source:'cmd_vel', linear_x, angular_z }
cmdVelIn        { type, timestamp, source:'cmd_vel_in', linear_x, angular_z }
voiceData       { type, timestamp, command, confidence, status, ... }
trajectory      Array<{x,y}>  (máx 500 puntos)
voiceHistory    Array<string> (máx 20 comandos)
logs            Array<{time, msg}>
mode            'mapping' | 'navigation'
goalMarker      {x, y} | null  — posición del goal actual en coords de mapa
```

## Protocolo de mensajes WebSocket salientes (dashboard → bridge)

```json
{ "type": "cmd_vel",              "linear_x": 0.2, "angular_z": 0.5 }
{ "type": "goal_pose",            "x": 1.5, "y": 2.3, "theta": 0.0 }
{ "type": "navigate_to_waypoint", "name": "centro" }
{ "type": "slam_reset" }
```

## Conversión de coordenadas en SlamMap (click-to-goal)

```
canvas pixel (canvasX, canvasY)
  → col  = canvasX / cellSize
  → row  = mapHeight - canvasY / cellSize   ← flip Y (canvas top = map south)
  → wx   = origin.x + col * resolution
  → wy   = origin.y + row * resolution
```

## Waypoints hardcodeados en WaypointPanel

Los 11 waypoints vienen de `src/puzzlebot_bringup/config/waypoints.yaml`.  
Si se añaden/cambian waypoints en el YAML, actualizar también `WaypointPanel.jsx`.

## Variables de entorno

```env
VITE_WS_URL=ws://localhost:8000/ws
# Para bridge en otra máquina:
# VITE_WS_URL=ws://192.168.x.x:8000/ws
```

## Cómo correr en desarrollo

```bash
cd web_dashboard
npm install
npm run dev -- --host 0.0.0.0
# Abre: http://localhost:5173
```

## Dependencias
Solo React 18 + Vite + @vitejs/plugin-react. Sin librerías de UI externas.

## Reglas de seguridad
- El frontend NO importa paquetes de ROS.
- Teleop solo envía cuando el usuario tiene el botón presionado; suelta → stop automático.
- Los botones de control se deshabilitan cuando `connected === false`.
- Nunca enviar `initialpose` desde el frontend.
