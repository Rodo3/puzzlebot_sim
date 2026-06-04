# web_dashboard — CLAUDE.md

## Propósito
Frontend React + Vite para **visualización y control** en tiempo real del Puzzlebot.
Se conecta al `puzzlebot_web_bridge` vía WebSocket — recibe datos del robot Y envía comandos de control.
Node.js 18+ requerido (Vite 5 no soporta Node 12).

## Layout

```
┌─ Header (44px) ──────────────────────────────────────────────────────┐
│ PUZZLEBOT LIVE DASHBOARD | Connected | topic dots | [SIM] [MAPPING] [DOM STATE] │
├─ Main area ────────────────────────┬─────────────────────────────────┤
│ SLAM Map (flex:1)                  │ col-right (360px fijo)           │
│  - OccupancyGrid canvas            │  ┌── sensors-row (PINNED) ─────┐ │
│  - Robot azul + flecha amarilla    │  │ LidarView | CameraPanel     │ │
│  - Trayectoria cyan                │  └─────────────────────────────┘ │
│  - Goal marker verde               │  ┌── col-right-scroll ─────────┐ │
│  - AUG/BASE toggle                 │  │ TeleopPanel                 │ │
│  - Zoom/pan/click-to-goal          │  │ Tabs: Modo/Waypoints/Voz/   │ │
│                                    │  │       Elevador              │ │
│                                    │  └─────────────────────────────┘ │
├────────────────────────────────────┴─────────────────────────────────┤
│ ▶ MÉTRICAS  dist Xm  vel Xm/s  replans N  stops N  [↺] [⛶]         │
│   (collapsible bar — expande hasta 280px con scroll)                 │
│   ⛶ abre overlay fullscreen (Opción B)                               │
├──────────────────────────────────────────────────────────────────────┤
│ Footer: VelocityPanel | LogsPanel                                    │
└──────────────────────────────────────────────────────────────────────┘
```

**IMPORTANTE**: `LidarView` y `CameraPanel` están fuera del scroll de la columna derecha — siempre visibles. Solo `TeleopPanel` y las tabs hacen scroll.

## Componentes

| Componente | Fuente de datos | Descripción |
|---|---|---|
| `App.jsx` | — | Raíz. Estado global, WebSocket, métricas, layout. |
| `SlamMap.jsx` | `map`/`augmented_map` + `robot_state` | Canvas 2D OccupancyGrid. AUG/BASE toggle. Click-to-goal en nav mode. Zoom/pan. |
| `LidarView.jsx` | `scan` | Canvas 2D polar LiDAR. **Pinned** — no scrollea. |
| `CameraPanel.jsx` | `camera_frame` | Stream JPEG base64. **Pinned** — no scrollea. |
| `TeleopPanel.jsx` | — | D-pad + sliders 10Hz. Global pointerup → stop. |
| `ModePanel.jsx` | estado app | Mapping/Nav toggle, reset SLAM, carga mapas guardados. |
| `WaypointPanel.jsx` | — | 11 waypoints de `waypoints.yaml`. Solo activo en nav mode. |
| `VoiceCommandPanel.jsx` | `voice_command` | Último comando, confianza, ranking, historial. |
| `VelocityPanel.jsx` | `velocity_command` + `navState` | Pipeline: Steering → Pre-avoidance → Final. Badge DOM state. |
| `ElevatorPanel.jsx` | — | Stub elevador (backend pendiente). |
| `LogsPanel.jsx` | eventos internos | Log de eventos frontend. |
| `MetricsPanel.jsx` | props desde App.jsx | Contadores + 3 gráficas SVG + export CSV + PDF report. |

## Servicios y utilidades

| Archivo | Rol |
|---|---|
| `services/websocketClient.js` | Conexión WS con auto-reconexión exponencial. `close()`, `send(data)`. |
| `utils/geometry.js` | `drawCircle`, `drawArrow` para canvas. |
| `utils/mapUtils.js` | `renderGridToImageData`, `worldToCell`, `cellToCanvas`. |

## Estado en App.jsx

```
// Conexión
connected            bool
lastUpdate           float (Unix timestamp)

// Robot
robotState           { pose:{x,y,theta}, odom_twist:{linear_x,angular_z}, timestamp }
scanData             { ranges[], angle_min, angle_max, timestamp }
mapData              { width, height, resolution, origin:{x,y}, data[], timestamp }
augMapData           { same as mapData } — /augmented_map (con obstáculos dinámicos)
cmdVel               { source:'cmd_vel',          linear_x, angular_z }
cmdVelIn             { source:'cmd_vel_in',        linear_x, angular_z }
cmdVelSteering       { source:'cmd_vel_steering',  linear_x, angular_z }
navState             string — FSM del dynamic_obstacle_manager
cameraData           { data: base64 JPEG }
voiceData            { command, confidence, status, ranked_predictions }

// UI
trajectory           Array<{x,y}> máx 500
voiceHistory         Array<string> máx 20
logs                 Array<{time, msg}> máx 50
mode                 'mapping' | 'navigation'
goalMarker           {x,y} | null
activeTab            'mode'|'waypoints'|'voice'|'elevator'
availableMaps        string[]
mapSource            'live' | 'static'
metricsOpen          bool — bar expandida/colapsada
metricsFullscreen    bool — overlay fullscreen activo

// Métricas (recolectadas en handleMessage)
velHistory           [{time, linear, angular}] máx 400, muestreado a ~5Hz
lidarHist            [{time, min}]             máx 400, muestreado a ~5Hz
domStateLog          [{time, state}]           máx 500, cada transición
sessionStats         { startTime, distanceTraveled, obstacleStops, replanCount, maxLinearVel, goalsSent }
```

## Recolección de métricas en App.jsx

- **distanceTraveled**: acumulado en `robot_state` comparando pose con `prevPoseRef` (delta < 0.5m para evitar teleports)
- **velHistory**: en `velocity_command` source=`cmd_vel`, throttled a `METRICS_MIN_DT = 0.15s`
- **lidarHist**: en `scan`, min de `ranges` filtrado (> 0.05m), throttled a `METRICS_MIN_DT`
- **domStateLog**: en `nav_state`, solo en transiciones (cuando cambia `navStateRef.current`)
- **replanCount**: incrementa cuando FSM → `REPLAN` o `BRAKE_FOR_REPLAN`
- **obstacleStops**: incrementa cuando FSM → `SAFE_STOP`
- **goalsSent**: incrementa en cada `handleGoalPose`

## MetricsPanel — estructura interna

Contadores de sesión + 3 gráficas SVG puras (sin librerías externas):
1. **LineChart** — velocidad linear+angular de `/cmd_vel` sobre tiempo
2. **LineChart** — distancia mínima LiDAR (zona roja < 0.30m)
3. **StateTimeline** — barra horizontal coloreada por estado FSM

Exports:
- **CSV**: todas las series + resumen de sesión, descarga via blob URL
- **PDF**: abre nueva ventana con HTML formateado → `window.print()` → Save as PDF

## Panel de métricas — ubicación en el layout

**Opción A (default)**: barra colapsable entre main-area y footer.
- Colapsada: 34px con mini stats siempre visibles
- Expandida: hasta 280px con scroll interno
- Animación: `grid-template-rows: 0fr ↔ 1fr` (simétrica en ambas direcciones)
- Botón ⛶ → activa Opción B

**Opción B**: overlay `position:fixed` que cubre todo el viewport.
- `metricsFullscreen = true`
- Botón "⊡ Barra" regresa a A, "✕ Cerrar" cierra
- Contenido centrado, max-width 1100px

## DOM FSM states (dynamic_obstacle_manager)

```
NORMAL | BRAKE_FOR_REPLAN | REPLAN | FOLLOW_NEW_PATH | RECOVERY_REVERSE | RECOVERY_TURN | SAFE_STOP
```
Color coding en header pill y MetricsPanel:
- NORMAL → gris
- FOLLOW_NEW_PATH → cyan
- BRAKE_FOR_REPLAN / REPLAN → amarillo
- RECOVERY_* / SAFE_STOP → rojo

## Mensajes WebSocket recibidos (bridge → dashboard)

| `type` | Tópico ROS | Descripción |
|---|---|---|
| `robot_state` | `/odom` + `/slam/robot_pose_in_map` | pose + twist |
| `scan` | `/scan` | LiDAR ranges[] |
| `map` | `/map` | OccupancyGrid base |
| `augmented_map` | `/augmented_map` | OccupancyGrid con obstáculos dinámicos |
| `nav_state` | `/dom/state` | FSM string |
| `velocity_command` | `/cmd_vel`, `/cmd_vel_in`, `/cmd_vel_steering` | Twist con campo `source` |
| `voice_command` | `/voice/*` | Resultado de inferencia |
| `camera_frame` | `/camera/image/compressed` | JPEG base64 |
| `available_maps` | respuesta a `list_maps` | string[] de archivos |

## Mensajes WebSocket enviados (dashboard → bridge)

```json
{ "type": "cmd_vel",              "linear_x": 0.2, "angular_z": 0.5 }
{ "type": "goal_pose",            "x": 1.5, "y": 2.3, "theta": 0.0 }
{ "type": "navigate_to_waypoint", "name": "centro" }
{ "type": "slam_reset" }
{ "type": "list_maps" }
{ "type": "load_map",             "filename": "slam_map.png" }
{ "type": "use_slam_map" }
{ "type": "elevator",             "action": "up" }
```

## Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `VITE_WS_URL` | `ws://localhost:8000/ws` | URL del bridge |
| `VITE_ROBOT_ENV` | `sim` | `sim` o `real` — badge en header |

Presets: `.env.sim` (Gazebo local) y `.env.real` (robot físico, editar `BRIDGE_IP`).

## Cómo correr

```bash
cd web_dashboard
cp .env.sim .env      # o .env.real para robot físico
npm install           # solo la primera vez
npm run dev           # requiere Node 18+
```

## Reglas críticas
- No importar paquetes de ROS en el frontend.
- Teleop envía solo mientras el botón está presionado; `pointerup` global → stop.
- Botones de control deshabilitados cuando `connected === false`.
- Nunca enviar `initialpose` desde el frontend.
- Gráficas SVG sin librerías externas (React + Vite únicos deps).
- Al cambiar waypoints en `waypoints.yaml`, actualizar también `WaypointPanel.jsx`.
