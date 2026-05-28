# web_dashboard — CLAUDE.md

## Propósito
Frontend React + Vite para visualización en tiempo real del Puzzlebot físico.  
Se conecta al `puzzlebot_web_bridge` vía WebSocket y muestra pose, mapa SLAM, LiDAR, velocidad y voz.

**Regla de oro: el frontend NO controla el robot. Solo visualiza. No envía mensajes al WebSocket.**

## Componentes principales

| Componente | Fuente de datos | Descripción |
|---|---|---|
| `App.jsx` | — | Raíz. Gestiona estado global y conexión WS. |
| `StatusPanel.jsx` | estado app | Indicadores de conexión y tópicos activos. |
| `SlamMap.jsx` | `map` + `robot_state` | Canvas 2D con OccupancyGrid, robot y trayectoria. |
| `LidarView.jsx` | `scan` | Canvas 2D polar con puntos del LiDAR. |
| `VelocityPanel.jsx` | `velocity_command` | Muestra `/cmd_vel` y `/cmd_vel_in`, detecta obstacle stop. |
| `VoiceCommandPanel.jsx` | `voice_command` | Último comando, confianza, ranking, historial. |
| `LogsPanel.jsx` | eventos internos | Log de eventos frontend (no logs de ROS). |

## Servicios y utilidades

| Archivo | Rol |
|---|---|
| `services/websocketClient.js` | Conexión WS con auto-reconexión exponencial. |
| `utils/geometry.js` | Conversión mundo→canvas, dibujo de flecha de orientación. |
| `utils/mapUtils.js` | Render de OccupancyGrid en ImageData, conversión coords con origin y resolution. |

## Estructura del estado en App.jsx

```
connected       bool
lastUpdate      float (Unix timestamp)
robotState      { type, timestamp, pose: {x,y,theta}, odom_twist: {linear_x, angular_z} }
scanData        { type, timestamp, angle_min, angle_max, angle_increment, range_min, range_max, min_distance, ranges[] }
mapData         { type, timestamp, width, height, resolution, origin: {x,y}, data[] }
cmdVel          { type, timestamp, source:'cmd_vel', linear_x, angular_z }
cmdVelIn        { type, timestamp, source:'cmd_vel_in', linear_x, angular_z }
voiceData       { type, timestamp, command, confidence, status, inference_time_ms, ranked_predictions[] }
trajectory      Array<{x,y}>  (máx 500 puntos)
voiceHistory    Array<string> (máx 20 comandos)
logs            Array<{time, msg}>
```

## Formato de mensajes WebSocket (JSON entrante)

Tipos posibles: `robot_state`, `scan`, `map`, `velocity_command`, `voice_command`.  
Ver [docs/web_dashboard_architecture.md](../docs/web_dashboard_architecture.md) para el esquema completo de cada tipo.

## Variables de entorno

Copiar `.env.example` a `.env` y ajustar la URL:

```env
VITE_WS_URL=ws://localhost:8000/ws
# Para robot en otra máquina:
# VITE_WS_URL=ws://192.168.x.x:8000/ws
```

## Cómo correr en desarrollo

```bash
cd web_dashboard
npm install
npm run dev -- --host 0.0.0.0
# Abre: http://localhost:5173
# Desde otra laptop en la red: http://<IP_MAQUINA>:5173
```

## Build de producción

```bash
npm run build          # genera web_dashboard/dist/
npm run preview        # verifica el build localmente en :4173
```

Para servir `dist/` en el robot o desde el bridge (no implementado en v1):
```bash
# Opción simple: Python static server
cd dist && python3 -m http.server 8080
```

## Dependencias
Solo React 18 + Vite + @vitejs/plugin-react. Sin librerías de UI externas.  
Canvas 2D nativo para SlamMap y LidarView.

## Regla de seguridad
El frontend NO debe importar paquetes de ROS, ni publicar a ningún tópico de control.
