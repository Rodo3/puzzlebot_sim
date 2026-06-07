# Handoff — rama `Dashboard-Jesus`

> Contexto para retomar el trabajo (incluido al cambiar de modelo Opus → Sonnet).
> Estado al **2026-06-06**.

## Qué es esta rama
`Dashboard-Jesus` = **base robusta de `master`** (localización mejorada) + **todo el
trabajo de dashboard/bridge/voz/misiones** que estaba en `rama_pruebas_dashboard`,
reaplicado encima archivo por archivo (NO fue git merge automático).

- Construida en un git worktree: `C:\Users\rpzda\Documents\ProyectoFinal\puzzlebot_merge`
- Ya **commiteada y pusheada** a `origin/Dashboard-Jesus`.
- `master` y `rama_pruebas_dashboard` quedaron **intactos**.

## Decisiones clave del merge
- **Localización y planning → siempre master** (kalman, scan_restamper, scan_matcher,
  path_planner, obstacle_avoidance, bug_navigation, navigation.launch, real_slam_nav).
- **Sobrevive de nuestra rama:** dashboard, bridge, voz (HMM), misiones/QR/logo,
  navegación dinámica, montacargas (`/forklift/command`).
- **Bridge `scan_topic` → `/scan_stamped`** por defecto. Master corre `scan_restamper`
  en sim y en real, así que es el topic universal del scan ya con timestamp/frame correctos.
- **`real_robot.launch.py` de master**: nuevo flag **`dashboard_features:=true`** que lanza
  `state_machine_node` + `qr_node` + `logo_detector_node` + `waypoint_navigator_node`.
- **`waypoint_navigator_node` reincorporado** (master lo había borrado). Traduce
  `/navigate_to_waypoint` (nombre) → `/goal_pose` para el path_planner A* de master.
  Sin él, el botón "ir a waypoint" del dashboard y la navegación de las misiones se
  quedarían sin consumidor. Es aditivo, no toca la robustez de planning.
  *(Si se quiere respetar "solo master" a rajatabla, este es el único nodo a discutir.)*

## Sensores de la Jetson (botones del dashboard)
- Se inician **por separado**, un botón por sensor. **No hay launch combinado**
  (`jetson_sensors.launch.py` fue eliminado a propósito).
- Flujo: dashboard → WebSocket `{type:"launch_sensor", sensor:"lidar|camera|microros"}`
  → bridge ejecuta **SSH toggle** (`sshpass -tt` a `puzzlebot@10.42.0.1`). Un clic abre y
  mantiene vivo el túnel; otro clic lo mata (≈ Ctrl+C). El estado vuelve al dashboard como
  `{type:"sensor_status", sensor, status}`.
- Código: handler `launch_sensor` y `SENSOR_SSH_CMDS` en
  `src/puzzlebot_web_bridge/puzzlebot_web_bridge/bridge_node.py` (~línea 63).
  `web_dashboard/src/components/SensorPanel.jsx` es controlado por `sensor_status`.
- Params SSH del bridge: `robot_ssh_host` (10.42.0.1), `robot_ssh_user` (puzzlebot),
  `robot_ssh_password` (default Puzzlebot72 — se puede sobreescribir con `-p`).

## ⏳ Pendiente
1. **Comandos SSH exactos** por sensor (lidar/cámara/micro-ROS) → reemplazar los
   PLACEHOLDER en `SENSOR_SSH_CMDS`.
2. Instalar **`sshpass`** en la PC del bridge (`sudo apt install sshpass`).
3. `npm install` en este `web_dashboard` (worktree nuevo, sin node_modules).
4. `colcon build` + prueba en el robot real (la sintaxis Python ya está validada,
   pero el grafo de nodos solo se prueba en hardware).
5. Decidir si `Dashboard-Jesus` se mergea a `master` vía PR.

## Cómo correr (robot real)
```bash
# Jetson: cada sensor se inicia desde su botón en el dashboard (SSH automático)
# PC:
ros2 launch puzzlebot_bringup real_robot.launch.py \
  slam:=true aruco:=true navigation:=true dashboard_features:=true
# PC (bridge, manual):
ros2 run puzzlebot_web_bridge bridge_node
# PC (dashboard):
cd web_dashboard && npm run dev -- --host 0.0.0.0
```

## Reglas de trabajo (no negociables)
- El bridge **NUNCA** publica a `/initialpose`.
- **No** `git commit` ni `git push` sin confirmación explícita del usuario.
- La contraseña SSH no se hardcodea fuera de los params del bridge (el usuario aceptó
  el default por conveniencia local).
- Trabajar por fases y mostrar el plan + dudas **antes** de implementar.
- El build/runtime real es en WSL2 / Ubuntu 22.04 / ROS 2 Humble (no en Windows).
