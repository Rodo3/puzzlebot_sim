# puzzlebot_web_bridge

ROS 2 Python package that acts as a **bidirectional bridge** between Puzzlebot
topics and the web dashboard.

- **ROS → WebSocket**: subscribes to topics, serializes to JSON, broadcasts to clients.
- **WebSocket → ROS**: receives JSON commands from the browser, publishes control messages.
- **POST /audio**: on-demand voice inference (KMeans + HMM) from the browser microphone.

## Dependencies

```bash
pip install fastapi "uvicorn[standard]" websockets "numpy>=1.25" scipy librosa
```

## Build

```bash
cd ~/puzzlebot_sim
colcon build --packages-select puzzlebot_web_bridge
source install/setup.bash
```

## Run

```bash
# Physical robot (default)
ros2 run puzzlebot_web_bridge bridge_node

# Gazebo — DiffDrive uses a different cmd_vel topic
ros2 run puzzlebot_web_bridge bridge_node \
  --ros-args -p cmd_vel_out_topic:=/model/puzzlebot/cmd_vel
```

WebSocket starts on `ws://0.0.0.0:8000/ws`. Health check:
```bash
curl http://localhost:8000/health
# {"status":"ok","clients":0}
```

---

## Topics subscribed (ROS → WebSocket)

### Core
| Topic | Type | WS message type | Rate |
|---|---|---|---|
| `/odom` | nav_msgs/Odometry | `robot_state` | 10 Hz |
| `/scan` | sensor_msgs/LaserScan | `scan` | 5 Hz |
| `/map` | nav_msgs/OccupancyGrid | `map` | 1 Hz |
| `/cmd_vel` | geometry_msgs/Twist | `velocity_command` (source: `cmd_vel`) | 10 Hz |
| `/slam/robot_pose_in_map` | geometry_msgs/PoseStamped | merged into `robot_state` | 10 Hz |

### Optional (silently ignored if absent)
| Topic | Type | WS message type | Rate |
|---|---|---|---|
| `/cmd_vel_in` | geometry_msgs/Twist | `velocity_command` (source: `cmd_vel_in`) | 10 Hz |
| `/cmd_vel_steering` | geometry_msgs/Twist | `velocity_command` (source: `cmd_vel_steering`) | 10 Hz |
| `/dom/state` | std_msgs/String | `nav_state` | 5 Hz |
| `/augmented_map` | nav_msgs/OccupancyGrid | `augmented_map` | 1 Hz |
| `/camera/image/compressed` | sensor_msgs/CompressedImage | `camera_frame` | 10 Hz |
| `/mission_state` | std_msgs/String | `mission_state` | event |
| `/qr/detections` | std_msgs/String | `qr_detections` | event |
| `/logo_detection/result` | std_msgs/String | `logo_detection` | event |
| `/voice/command` | std_msgs/String | accumulated into `voice_command` | event |
| `/voice/confidence` | std_msgs/Float32 | accumulated into `voice_command` | event |
| `/voice/status` | std_msgs/String | accumulated into `voice_command` | event |
| `/voice/ranked_predictions` | std_msgs/String | triggers `voice_command` send | event |
| `/voice/inference_time_ms` | std_msgs/Float32 | accumulated into `voice_command` | event |

**`/dom/state`** is published by `dynamic_obstacle_manager` with FSM states:
`NORMAL | BRAKE_FOR_REPLAN | REPLAN | FOLLOW_NEW_PATH | RECOVERY_REVERSE | RECOVERY_TURN | SAFE_STOP`

**`/augmented_map`** is the base SLAM map with dynamic obstacles injected as occupied cells.
Path planner uses this instead of `/map` when `obstacle_manager:=dynamic`.

---

## Topics published (WebSocket → ROS)

| Topic | Type | Dashboard command |
|---|---|---|
| `cmd_vel_out_topic` (default `/cmd_vel`) | geometry_msgs/Twist | `"type":"cmd_vel"` |
| `/cmd_vel_teleop` | geometry_msgs/Twist | `"type":"cmd_vel"` (also, for priority) |
| `/goal_pose` | geometry_msgs/PoseStamped | `"type":"goal_pose"` |
| `/navigate_to_waypoint` | std_msgs/String | `"type":"navigate_to_waypoint"` |
| `/slam/reset` | std_msgs/Bool | `"type":"slam_reset"` |
| `/slam/load_map` | std_msgs/String | `"type":"load_map"` |

**SAFETY**: `/initialpose` is never published. The bridge does no planning.

---

## Incoming command protocol (JSON from browser)

```json
{ "type": "cmd_vel",              "linear_x": 0.2, "angular_z": 0.5 }
{ "type": "goal_pose",            "x": 1.5, "y": 2.3, "theta": 0.0 }
{ "type": "navigate_to_waypoint", "name": "centro" }
{ "type": "slam_reset" }
{ "type": "list_maps" }
{ "type": "load_map",             "filename": "slam_map_20260529.png" }
{ "type": "use_slam_map" }
{ "type": "elevator",             "action": "up" }
{ "type": "mission_start",        "mission": "1" }
{ "type": "mission_start",        "mission": "2" }
{ "type": "mission_stop" }
```

`mission_start` publica en `/mission_start` (std_msgs/String).
`mission_stop` publica `"stop"` en `/mission_start`.

---

## Key parameters

| Parameter | Default | Notes |
|---|---|---|
| `cmd_vel_out_topic` | `/cmd_vel` | Override to `/model/puzzlebot/cmd_vel` for Gazebo |
| `websocket_host` | `0.0.0.0` | Listen address |
| `websocket_port` | `8000` | Listen port |
| `maps_dir` | `''` (cwd) | Directory scanned for `.pgm`/`.png` maps |
| `artifact_dir` | `''` | Path to voice model artifacts; empty = voice inference disabled |

The launch files (`gz_slam_nav.launch.py`, `real_slam_nav.launch.py`) set
`cmd_vel_out_topic` correctly for each environment.

---

## Package structure

| File | Role |
|---|---|
| `bridge_node.py` | ROS 2 node — parameters, subscribers, control publishers, `_handle_command` |
| `websocket_server.py` | FastAPI + uvicorn on daemon thread — `/ws`, `/health`, `/audio` |
| `serializers.py` | Pure functions: ROS msg → JSON dict |
| `rate_limiter.py` | `RateLimiter(max_hz)` — throttle per topic |
| `topic_config.py` | Default topic names and rate limits — edit here to change defaults |
