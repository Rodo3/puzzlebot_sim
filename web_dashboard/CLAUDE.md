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
│  - OccupancyGrid canvas            │  ┌── PINNED (siempre visible) ─┐ │
│  - Robot azul + flecha amarilla    │  │ MissionPanel (strip)        │ │
│  - Trayectoria cyan                │  │ CameraPanel (grande,        │ │
│  - Goal marker verde               │  │   overlay QR + logos)       │ │
│  - AUG/BASE toggle                 │  └─────────────────────────────┘ │
│  - Zoom/pan/click-to-goal          │  ┌── col-right-scroll ─────────┐ │
│                                    │  │ LidarView (mini)            │ │
│                                    │  │ TeleopPanel                 │ │
│                                    │  │  toggle: [🤖 Robot][↕ Elev] │ │
│                                    │  │ Tabs: Modo / Waypoints / Voz│ │
│                                    │  └─────────────────────────────┘ │
├────────────────────────────────────┴─────────────────────────────────┤
│ ▶ MÉTRICAS  dist Xm  vel Xm/s  replans N  stops N  [↺] [⛶]         │
│   (collapsible bar — expande hasta 280px con scroll)                 │
│   ⛶ abre overlay fullscreen (Opción B)                               │
├──────────────────────────────────────────────────────────────────────┤
│ Footer: VelocityPanel | LogsPanel                                    │
└──────────────────────────────────────────────────────────────────────┘
```

**IMPORTANTE**:
- `MissionPanel` y `CameraPanel` están fijos arriba de la columna derecha — siempre visibles (foco visual del reto).
- `CameraPanel` dibuja overlays de detección (cajas de QR + logos) sobre el stream crudo, con toggle `[▣ Anotada / ▢ Raw]`.
- `LidarView` se movió al área scrolleable (mini).
- `ElevatorPanel.jsx` ya no se usa como tab — su lógica se integró en `TeleopPanel` (toggle).
- Tab-bar: **3 tabs** — `Modo | Waypoints | Voz`. El tab `Elevador` fue eliminado.

## Componentes

| Componente | Fuente de datos | Descripción |
|---|---|---|
| `App.jsx` | — | Raíz. Estado global, WebSocket, métricas, layout. |
| `SlamMap.jsx` | `map`/`augmented_map` + `robot_state` | Canvas 2D OccupancyGrid. AUG/BASE toggle. Click-to-goal en nav mode. Zoom/pan. |
| `LidarView.jsx` | `scan` | Canvas 2D polar LiDAR. En el área scrolleable (mini, max 220px). |
| `CameraPanel.jsx` | `camera_frame` + `qr_detections` + `logo_detection` | Stream JPEG base64 con overlay canvas (cajas QR + logos). Toggle `[▣ Anotada / ▢ Raw]`. **Pinned** — no scrollea. |
| `TeleopPanel.jsx` | — | D-pad + sliders 10Hz. Toggle `[🤖 Robot] / [↕ Elevador]` en cabecera. Robot mode → `/cmd_vel`. Elevator mode → `/forklift/command`. |
| `MissionPanel.jsx` | `missionState` | Strip siempre visible. Botones M1/M2/Stop. Badge de estado de misión. Misión 2 espera click en mapa (estado `WAITING_FOR_GOAL`). |
| `ModePanel.jsx` | estado app | Mapping/Nav toggle, reset SLAM, carga mapas guardados. |
| `WaypointPanel.jsx` | — | Waypoints de `waypoints.yaml`. Solo activo en nav mode. |
| `VoiceCommandPanel.jsx` | `voice_command` | Último comando, confianza, ranking, historial. |
| `VelocityPanel.jsx` | `velocity_command` + `navState` | Pipeline: Steering → Pre-avoidance → Final. Badge DOM state. |
| `ElevatorPanel.jsx` | — | **Obsoleto como tab**. Lógica movida a `TeleopPanel`. Archivo conservado pero no renderizado. |
| `LogsPanel.jsx` | eventos internos | Log de eventos frontend. |
| `MetricsPanel.jsx` | props desde App.jsx | Contadores + 3 gráficas SVG + export CSV + PDF report. |

## MissionPanel — comportamiento detallado

Strip siempre visible (fuera de tabs, entre TeleopPanel y tab-bar):

```
Estado normal:  [ ▶ Misión 1 ]  [ ▶ Misión 2 ]  badge: IDLE
Con misión:     badge: SCANNING_QR              [ ■ Detener ]
M2 esperando:   badge: 📍 Haz click en el mapa...  [ ■ Detener ]
```

- **Misión 1**: envía `{"type": "mission_start", "mission": "1"}`. No requiere nada más.
- **Misión 2**: envía `{"type": "mission_start", "mission": "2"}`. El badge cambia a
  `WAITING_FOR_GOAL`. El siguiente click en el mapa (SlamMap) activa `goal_pose` normal
  — el `state_machine_node` lo captura como punto de inicio. El panel lo indica visualmente.
- **Detener**: envía `{"type": "mission_stop"}`. Siempre disponible cuando hay misión activa.

## TeleopPanel — toggle Robot / Elevador

Pill toggle en la cabecera del panel:
```
[ 🤖 Robot ]  [ ↕ Elevador ]
```

- **Modo Robot** (default): d-pad y sliders funcionan igual que antes → `cmd_vel`.
- **Modo Elevador**: ↑ → `{type:"elevator", action:"up"}`, ↓ → `{type:"elevator", action:"down"}`, ■ → `{type:"elevator", action:"stop"}`. Flechas ←/→ deshabilitadas. Sliders ocultos.

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
qrDetections         Array<{data, corners, area_px, center}>  — overlay cámara
logoDetections       Array<{class_name, confidence, bbox}>    — overlay cámara
voiceData            { command, confidence, status, ranked_predictions }
missionState         string — estado de la misión: IDLE/WAITING_FOR_GOAL/GOING_TO_START/
                              SCANNING_QR/FORKLIFT_UP/NAVIGATING_TO_DOCKS/
                              SCANNING_LOGOS/FORKLIFT_DOWN/DONE/ERROR

// UI
trajectory           Array<{x,y}> máx 500
voiceHistory         Array<string> máx 20
logs                 Array<{time, msg}> máx 50
mode                 'mapping' | 'navigation'
goalMarker           {x,y} | null
activeTab            'mode' | 'waypoints' | 'voice'   ← sin 'elevator'
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
| `qr_detections` | `/qr/detections` | `[{data, corners, area_px, center}]` — overlay cámara |
| `logo_detection` | `/logo_detection/result` | `[{class_name, confidence, bbox}]` — overlay cámara |
| `available_maps` | respuesta a `list_maps` | string[] de archivos |
| `mission_state` | `/mission_state` | Estado de la máquina de misión (string) |

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
{ "type": "mission_start",        "mission": "1" }
{ "type": "mission_start",        "mission": "2" }
{ "type": "mission_stop" }
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
- `MissionPanel` y `TeleopPanel` son los únicos que envían comandos de misión/elevador.
- `ElevatorPanel.jsx` no se renderiza — no eliminarlo del repo, solo está desconectado.
