# puzzlebot_web_bridge

ROS 2 Python package that subscribes to Puzzlebot topics and streams
JSON over WebSocket to the web dashboard. Read-only — never publishes
to any control topic.

## Dependencies

Install once in the ROS environment:

```bash
pip install fastapi "uvicorn[standard]" websockets
```

## Build

```bash
cd ~/puzzlebot_sim
colcon build --packages-select puzzlebot_web_bridge
source install/setup.bash
```

## Run

```bash
ros2 run puzzlebot_web_bridge bridge_node
```

The WebSocket server starts on `ws://0.0.0.0:8000/ws` and accepts
connections from any machine on the network — no extra configuration
needed.

Verify it is running:

```bash
curl http://localhost:8000/health
# {"status":"ok","clients":0}
```

## Topics

Core (always expected):

| Topic      | Type                      |
|------------|---------------------------|
| `/odom`    | nav_msgs/Odometry         |
| `/scan`    | sensor_msgs/LaserScan     |
| `/map`     | nav_msgs/OccupancyGrid    |
| `/cmd_vel` | geometry_msgs/Twist       |

Optional (bridge ignores them silently if they don't exist):

| Topic                       | Type                |
|-----------------------------|---------------------|
| `/cmd_vel_in`               | geometry_msgs/Twist |
| `/voice/command`            | std_msgs/String     |
| `/voice/confidence`         | std_msgs/Float32    |
| `/voice/status`             | std_msgs/String     |
| `/voice/ranked_predictions` | std_msgs/String     |
| `/voice/inference_time_ms`  | std_msgs/Float32    |

## Rate limits

| Topic      | Max Hz |
|------------|--------|
| /odom      | 10     |
| /cmd_vel   | 10     |
| /cmd_vel_in| 10     |
| /scan      | 5      |
| /map       | 1      |
| /voice/*   | unlimited (event-driven) |

## Overriding parameters

```bash
ros2 run puzzlebot_web_bridge bridge_node \
  --ros-args -p websocket_port:=8080 -p odom_topic:=/odom_filtered
```

Full parameter list is in `puzzlebot_web_bridge/topic_config.py`.

## Current structure

- `bridge_node.py`: ROS 2 node, subscribers, parameter declarations.
- `websocket_server.py`: FastAPI + uvicorn WebSocket server on a daemon thread.
- `serializers.py`: ROS msg → JSON dict conversion (odom, scan, map, twist, voice).
- `rate_limiter.py`: simple Hz-based throttle.
- `topic_config.py`: default topic names and rate limits.
