# web_dashboard

React + Vite frontend for real-time Puzzlebot **visualization and control**.
Connects to `puzzlebot_web_bridge` over WebSocket — receives robot data AND
sends control commands (teleop, goal poses, waypoints, SLAM reset).

## Requirements

- **Node.js 18+** and npm (Vite 5 requires Node 18 — install via `nvm install 18`)
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
| `.env.example` | Template with all variables |

| Variable | Default | Description |
|---|---|---|
| `VITE_WS_URL` | `ws://localhost:8000/ws` | WebSocket URL of the bridge |
| `VITE_ROBOT_ENV` | `sim` | `sim` or `real` — shown as badge in header |

---

## Layout

```
┌─ Header ─────────────────────────────────────────────────┐
│ Title | Connected | topic dots | [SIM/REAL] [MODE] [DOM] │
├─ Main area ──────────────────────┬───────────────────────┤
│ SLAM Map (resizable)             │ LiDAR | Camera (fixed)│
│                                  ├───────────────────────┤
│                                  │ Teleop       ↑        │
│                                  │ Tabs:        │ scroll  │
│                                  │  Modo        │        │
│                                  │  Waypoints   ↓        │
│                                  │  Voz                  │
│                                  │  Elevador             │
├──────────────────────────────────┴───────────────────────┤
│ ▶ MÉTRICAS  dist · vel · replans · stops  [↺] [⛶]       │
├──────────────────────────────────────────────────────────┤
│ Footer: Velocity pipeline | System Logs                  │
└──────────────────────────────────────────────────────────┘
```

**LiDAR and Camera are pinned** — they stay visible regardless of scroll.
Only Teleop + Tabs scroll within the right column.

---

## Dashboard features

### Header
- **Connected / Disconnected** badge
- **Topic dots** — green when data is arriving: `odom`, `scan`, `map`, `aug`, `vel`, `cam`, `voice`
- **[SIM] / [REAL]** environment badge
- **MAPPING / NAV** mode pill
- **DOM FSM state** pill (navigation mode only) — color-coded:
  - Grey: `NORMAL`
  - Cyan: `FOLLOWING`
  - Yellow: `REPLANNING` / `BRAKING`
  - Red: `RECOVERY` / `SAFE STOP`

### SLAM Map (left panel)
- OccupancyGrid rendered on a 2D canvas
- Robot position (blue circle + yellow heading arrow)
- Trajectory trace (cyan)
- Goal marker (green circle + X)
- **AUG / BASE toggle** — appears when `/augmented_map` is available; switches
  between the base SLAM map and the obstacle-injected map used by the path planner
- Zoom (scroll wheel, zoom-to-cursor), pan (drag), reset view (⌂)
- **Click-to-goal** in navigation mode

### Right column
- **LiDAR** polar canvas — always visible (pinned above scroll area)
- **Camera** JPEG stream — always visible (pinned above scroll area)
- **Teleop D-pad** — sends `cmd_vel` at 10 Hz while held; global `pointerup` = auto-stop
- **Tabs** (scroll independently):
  - **Modo** — Mapping / Navigation toggle, SLAM reset, saved map loader
  - **Waypoints** — 11 predefined waypoints, active in navigation mode only
  - **Voz** — voice command history, confidence scores, model ranking
  - **Elevador** — elevator stub (backend pending)

### Metrics bar (collapsible)
Sits between the main area and the footer. Always shows mini stats even when collapsed:

```
▶ MÉTRICAS   dist 2.3m   vel max 0.14m/s   replans 1   stops 0   [↺] [⛶]
```

Click to expand (up to 280px with scroll) and see:
- **Session counter cards**: duration, distance, max speed, replan count, obstacle stops, goals sent
- **Velocity chart**: linear + angular from `/cmd_vel` over time (SVG, no external libs)
- **LiDAR min distance chart**: with red danger zone below 0.30 m
- **DOM FSM state timeline**: color-coded horizontal bar per state transition
- **↓ CSV** button: downloads all time series + session summary
- **⎙ PDF** button: opens formatted print report → Save as PDF via browser

**⛶ button** opens fullscreen overlay (Option B): covers the entire viewport
for maximum chart readability. "⊡ Barra" returns to the collapsible bar.

### Footer
- **Velocity panel**: full pipeline — Steering → Pre-avoidance (`/cmd_vel_in`) → Final (`/cmd_vel`)
  + DOM state badge + obstacle stop / cmd modified detection
- **System logs**: frontend event log (connections, mode changes, DOM transitions, goals)

---

## Commands sent to the bridge

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

## Switching to physical robot

1. `cp .env.real .env` and set `BRIDGE_IP` to the robot PC's IP
2. Use `real_slam_nav.launch.py` instead of `gz_slam_nav.launch.py`
3. Dashboard code requires **no changes**

---

## Viewing from a different machine

```bash
# On bridge PC — find IP
ip addr show | grep "inet " | grep -v 127
sudo ufw allow 8000

# On dashboard PC — set URL, then run
# Edit .env: VITE_WS_URL=ws://<BRIDGE_IP>:8000/ws
npm run dev -- --host 0.0.0.0
```

---

## Project structure

```
web_dashboard/
  .env.sim / .env.real / .env.example
  src/
    App.jsx                   — global state, WebSocket, metrics collection, layout
    styles.css                — dark theme, all component styles
    services/
      websocketClient.js      — WS with exponential auto-reconnect
    components/
      SlamMap.jsx             — 2D map canvas, AUG/BASE toggle, click-to-goal
      LidarView.jsx           — polar LiDAR canvas (pinned)
      CameraPanel.jsx         — JPEG camera stream (pinned)
      TeleopPanel.jsx         — D-pad + velocity sliders
      ModePanel.jsx           — mapping/nav toggle, saved map loader
      WaypointPanel.jsx       — 11 named waypoints
      VelocityPanel.jsx       — velocity pipeline + DOM state badges
      VoiceCommandPanel.jsx   — voice command history
      ElevatorPanel.jsx       — elevator stub
      LogsPanel.jsx           — frontend event log
      MetricsPanel.jsx        — session counters, SVG charts, CSV/PDF export
    utils/
      geometry.js             — canvas drawing helpers
      mapUtils.js             — OccupancyGrid rendering
```

---

## Production build

```bash
npm run build     # output → web_dashboard/dist/
cd dist && python3 -m http.server 8080
```

`VITE_WS_URL` and `VITE_ROBOT_ENV` are baked in at build time.
