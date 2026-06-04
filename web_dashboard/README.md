# web_dashboard

React + Vite frontend for real-time Puzzlebot **visualization and control**.
Connects to `puzzlebot_web_bridge` over WebSocket — receives robot data AND
sends control commands (teleop, goal poses, waypoints, SLAM reset).

## Requirements

- Node.js 18+ and npm (no ROS installation needed)
- `puzzlebot_web_bridge` running and reachable on the network

## Quick start

```bash
cd web_dashboard

# Simulation (Gazebo, bridge on same machine)
cp .env.sim .env
npm install
npm run dev

# Physical robot (bridge on robot PC)
cp .env.real .env
# Edit .env: replace BRIDGE_IP with the robot PC's IP
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Environment configuration

| File | Use case |
|---|---|
| `.env.sim`  | Gazebo simulation — bridge on localhost |
| `.env.real` | Physical robot — edit `BRIDGE_IP` before use |
| `.env.example` | Template with all available variables |

Variables:

| Variable | Default | Description |
|---|---|---|
| `VITE_WS_URL` | `ws://localhost:8000/ws` | WebSocket URL of the bridge |
| `VITE_ROBOT_ENV` | `sim` | `sim` or `real` — shown as badge in header |

---

## Dashboard features

### Header
- **Connected / Disconnected** badge
- **Topic dots** — green when data is arriving: `odom`, `scan`, `map`, `aug` (augmented map), `vel`, `cam`, `voice`
- **[SIM] / [REAL]** environment badge
- **MAPPING / NAV** mode pill
- **DOM FSM state** pill (navigation mode only): `NORMAL` / `FOLLOWING` / `REPLANNING` / `RECOVERY` / `SAFE STOP`

### SLAM Map (left panel)
- OccupancyGrid rendered on a 2D canvas
- Robot position (blue circle + yellow arrow for heading)
- Trajectory trace (cyan)
- Goal marker (green circle + X)
- **AUG / BASE toggle** — switches between the live SLAM map and the augmented
  map (`/augmented_map`) that includes injected dynamic obstacles. The toggle
  appears automatically once `/augmented_map` starts arriving.
- Zoom (scroll wheel, zoom-to-cursor), pan (drag), reset view (⌂)
- **Click-to-goal** in navigation mode: click anywhere on the map to send a
  `goal_pose` command

### Right column
- **LiDAR** polar view (`/scan`, 5 Hz)
- **Camera** JPEG stream (`/camera/image/compressed`, 10 Hz)
- **Teleop D-pad** — arrow buttons + stop. Sends `cmd_vel` at 10 Hz while held.
  Global `pointerup` ensures stop on release even outside the button.
- **Tabs:**
  - **Modo** — switch Mapping / Navigation, reset SLAM, load saved maps
  - **Waypoints** — 11 predefined waypoints (from `waypoints.yaml`), active in
    navigation mode only
  - **Voz** — voice command history, confidence scores, model ranking
  - **Elevador** — elevator stub (backend pending)

### Footer
- **Velocity panel** — shows the full velocity pipeline:
  - `Steering` → steering_controller output (`/cmd_vel_steering`, shown in nav mode)
  - `Pre-avoidance` → after dynamic_obstacle_manager (`/cmd_vel_in`)
  - `Final /cmd_vel` → after obstacle_avoidance (what the robot actually executes)
  - DOM FSM state badge (color-coded)
  - `OBSTACLE STOP` / `CMD MODIFIED` badge when avoidance intervenes
- **Logs** — frontend event log (connections, mode changes, DOM transitions, goal sends)

---

## Commands sent to the bridge (WebSocket → ROS)

```json
{ "type": "cmd_vel",              "linear_x": 0.2, "angular_z": 0.5 }
{ "type": "goal_pose",            "x": 1.5, "y": 2.3, "theta": 0.0 }
{ "type": "navigate_to_waypoint", "name": "centro" }
{ "type": "slam_reset" }
{ "type": "list_maps" }
{ "type": "load_map",             "filename": "slam_map_20260529.png" }
{ "type": "use_slam_map" }
{ "type": "elevator",             "action": "up" }
```

---

## Messages received from the bridge (ROS → WebSocket)

| `type` | Source topic | Rate |
|---|---|---|
| `robot_state` | `/odom` + `/slam/robot_pose_in_map` | 10 Hz |
| `scan` | `/scan` | 5 Hz |
| `map` | `/map` | 1 Hz |
| `augmented_map` | `/augmented_map` | 1 Hz |
| `nav_state` | `/dom/state` | 5 Hz |
| `velocity_command` (source: `cmd_vel`) | `/cmd_vel` | 10 Hz |
| `velocity_command` (source: `cmd_vel_in`) | `/cmd_vel_in` | 10 Hz |
| `velocity_command` (source: `cmd_vel_steering`) | `/cmd_vel_steering` | 10 Hz |
| `voice_command` | `/voice/*` | event-driven |
| `camera_frame` | `/camera/image/compressed` | 10 Hz |
| `available_maps` | bridge response to `list_maps` | on demand |

---

## Switching to physical robot

The only things that change for the real robot:

1. **Copy `.env.real`** → `.env` and set `BRIDGE_IP` to the robot PC's IP.
2. **Launch the bridge with** `cmd_vel_out_topic:=/cmd_vel` (default, already correct
   for the real robot — the `gz_sim.launch.py` overrides this for Gazebo).
3. **Use** `real_slam_nav.launch.py` instead of `gz_slam_nav.launch.py`.

The dashboard code itself requires no changes.

---

## Viewing from a different machine

Find the bridge PC's IP:
```bash
ip addr show | grep "inet " | grep -v 127
```

Open port 8000:
```bash
sudo ufw allow 8000
```

Set `VITE_WS_URL=ws://<BRIDGE_IP>:8000/ws` in `.env`, then:
```bash
npm run dev -- --host 0.0.0.0
```

---

## Project structure

```
web_dashboard/
  .env.sim              — simulation preset (cp to .env)
  .env.real             — real robot preset (cp to .env, edit BRIDGE_IP)
  .env.example          — template with all variables
  src/
    App.jsx             — global state, WebSocket routing, header, layout
    styles.css          — dark theme, all component styles
    services/
      websocketClient.js  — WS connection with exponential auto-reconnect
    components/
      SlamMap.jsx         — 2D canvas: map, robot, trajectory, click-to-goal, aug toggle
      LidarView.jsx       — polar LiDAR canvas
      CameraPanel.jsx     — JPEG camera stream
      TeleopPanel.jsx     — D-pad + velocity sliders
      ModePanel.jsx       — mapping/nav toggle, saved map loader
      WaypointPanel.jsx   — 11 named waypoints
      VelocityPanel.jsx   — velocity pipeline + DOM state + obstacle badges
      VoiceCommandPanel.jsx — voice command history
      ElevatorPanel.jsx   — elevator stub
      LogsPanel.jsx       — frontend event log
    utils/
      geometry.js         — world→canvas coordinate helpers
      mapUtils.js         — OccupancyGrid rendering
```

---

## Production build

```bash
npm run build     # output → web_dashboard/dist/
cd dist
python3 -m http.server 8080
```

`VITE_WS_URL` and `VITE_ROBOT_ENV` are baked in at build time — set them in `.env` before building.
