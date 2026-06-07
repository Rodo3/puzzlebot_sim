# web_dashboard

React + Vite frontend for real-time Puzzlebot visualization.
Connects to `puzzlebot_web_bridge` over WebSocket and displays
pose, SLAM map, LiDAR, velocity, voice commands, and system logs.

**Visualization only — the frontend never sends commands to the robot.**

## Requirements

- Node.js 18+ and npm (no ROS installation needed)
- `puzzlebot_web_bridge` running and reachable on the network

## Quick start (same machine as the bridge)

```bash
cd web_dashboard
cp .env.example .env        # VITE_WS_URL=ws://localhost:8000/ws
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Viewing the dashboard from a different PC

This is the recommended setup for lab demos: one PC runs Gazebo +
ROS 2 + the bridge, another PC (or laptop) shows the dashboard.

### Step 1 — Find the IP of the PC running the bridge

On the bridge PC (Linux/WSL2):

```bash
ip addr show | grep "inet " | grep -v 127
# Example output: inet 192.168.1.45/24 ...
```

On the bridge PC (Windows with WSL2), run inside WSL2:

```bash
hostname -I
# Example: 172.28.144.1
```

Make sure port 8000 is open on that machine:

```bash
sudo ufw allow 8000
```

Verify the bridge is reachable from the dashboard PC:

```bash
curl http://192.168.1.45:8000/health
# Expected: {"status":"ok","clients":0}
```

### Step 2 — Configure the dashboard PC

```bash
cd web_dashboard
cp .env.example .env
```

Edit `.env` and replace `localhost` with the bridge PC's IP:

```
VITE_WS_URL=ws://192.168.1.45:8000/ws
```

### Step 3 — Run the frontend

```bash
npm install        # only needed once
npm run dev -- --host 0.0.0.0
```

Open `http://localhost:5173` in the browser on the dashboard PC.

To open from a third device on the same network, use the dashboard
PC's IP instead:

```
http://<IP_DASHBOARD_PC>:5173
```

---

## What you should see when connected

- Header badge turns **Connected**
- Status panel dots turn green as each topic starts publishing
- SLAM map appears within a few seconds (rate-limited to 1 Hz)
- Robot marker moves on the map as `/odom` arrives
- LiDAR points update at 5 Hz
- Velocity panel shows `/cmd_vel` values in real time

## Production build (optional)

Build a static bundle and serve it without Vite:

```bash
npm run build               # output goes to web_dashboard/dist/
cd dist
python3 -m http.server 8080
```

Remember to set `VITE_WS_URL` to the correct bridge IP before building,
since the URL is baked into the bundle at build time.

## Current structure

```
src/
  App.jsx                   — global state, WebSocket routing
  services/
    websocketClient.js      — connection with exponential auto-reconnect
  components/
    StatusPanel.jsx         — connection and topic indicators
    SlamMap.jsx             — Canvas 2D: OccupancyGrid + robot + trajectory
    LidarView.jsx           — Canvas 2D: polar LiDAR point cloud
    VelocityPanel.jsx       — cmd_vel vs cmd_vel_in, obstacle-stop detection
    VoiceCommandPanel.jsx   — last command, confidence, history
    LogsPanel.jsx           — frontend event log
  utils/
    geometry.js             — world-to-canvas coordinate conversion
    mapUtils.js             — OccupancyGrid rendering helpers
```
