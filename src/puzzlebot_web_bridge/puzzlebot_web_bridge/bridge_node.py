"""ROS 2 node that bridges topics to a WebSocket server.

Subscribes to core and optional topics, serializes messages, and broadcasts
JSON to all connected WebSocket clients. Missing optional topics are silently
ignored — the bridge will not crash if they don't exist.

Also exposes POST /audio for voice inference when the microphone is on the
dashboard machine. Audio is received as WAV bytes, inference runs locally
(KMeans + HMM), and results are published to /voice/* topics AND broadcast
to WebSocket clients.

BIDIRECTIONAL CONTROL: The bridge also accepts JSON commands from the dashboard
to publish control messages to ROS:
  {"type": "cmd_vel",              "linear_x": 0.2, "angular_z": 0.5}
  {"type": "goal_pose",            "x": 1.5, "y": 2.3, "theta": 0.0}
  {"type": "navigate_to_waypoint", "name": "centro"}
  {"type": "slam_reset"}

SAFETY: /initialpose is intentionally never published. Outgoing cmd_vel topic
is configurable via cmd_vel_out_topic (default /cmd_vel; use
/model/puzzlebot/cmd_vel for Gazebo).
"""

import glob
import io
import json
import logging
import math
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, OccupancyGrid
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, LaserScan
from std_msgs.msg import Bool, Float32, String

from .rate_limiter import RateLimiter
from .serializers import (
    camera_to_json,
    map_to_json,
    odom_to_json,
    scan_to_json,
    twist_to_json,
    voice_to_json,
)
from .topic_config import (
    DEFAULT_TOPICS,
    RATE_LIMITS_HZ,
    WEBSOCKET_HOST_DEFAULT,
    WEBSOCKET_PORT_DEFAULT,
)
from .websocket_server import WebSocketServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BridgeNode(Node):
    def __init__(self):
        super().__init__('puzzlebot_web_bridge')

        # Declare parameters with defaults from topic_config.
        self.declare_parameter('odom_topic',                    DEFAULT_TOPICS['odom'])
        self.declare_parameter('scan_topic',                    DEFAULT_TOPICS['scan'])
        self.declare_parameter('map_topic',                     DEFAULT_TOPICS['map'])
        self.declare_parameter('cmd_vel_topic',                 DEFAULT_TOPICS['cmd_vel'])
        self.declare_parameter('cmd_vel_in_topic',              DEFAULT_TOPICS['cmd_vel_in'])
        self.declare_parameter('voice_command_topic',           DEFAULT_TOPICS['voice_command'])
        self.declare_parameter('voice_confidence_topic',        DEFAULT_TOPICS['voice_confidence'])
        self.declare_parameter('voice_status_topic',            DEFAULT_TOPICS['voice_status'])
        self.declare_parameter('voice_ranked_predictions_topic', DEFAULT_TOPICS['voice_ranked_predictions'])
        self.declare_parameter('voice_inference_time_topic',    DEFAULT_TOPICS['voice_inference_time'])
        self.declare_parameter('camera_topic',                  DEFAULT_TOPICS['camera'])
        self.declare_parameter('websocket_host',                WEBSOCKET_HOST_DEFAULT)
        self.declare_parameter('websocket_port',                WEBSOCKET_PORT_DEFAULT)
        # artifact_dir: path to trained voice models. Empty string disables voice inference.
        self.declare_parameter('artifact_dir', '')
        # maps_dir: directory to scan for saved maps (.pgm / .png).
        # Defaults to current working directory (usually the workspace root).
        self.declare_parameter('maps_dir', '')
        # cmd_vel_out_topic: topic to publish teleop commands.
        # Default /cmd_vel for real robot; use /model/puzzlebot/cmd_vel for Gazebo.
        self.declare_parameter('cmd_vel_out_topic', DEFAULT_TOPICS['cmd_vel_out'])

        # Rate limiters.
        self._rl = {
            key: RateLimiter(hz)
            for key, hz in RATE_LIMITS_HZ.items()
        }

        # Voice state accumulator (assembled before sending, from ROS topics).
        self._voice: dict = {
            'command': None,
            'confidence': None,
            'status': None,
            'inference_time_ms': None,
            'ranked_predictions': None,
        }

        # Voice inference engine (optional — loaded if artifact_dir is set).
        self._inference_engine = None
        artifact_dir = self.get_parameter('artifact_dir').get_parameter_value().string_value
        if artifact_dir:
            self._load_voice_engine(artifact_dir)

        # Voice publishers (used when inference runs in the bridge via POST /audio).
        self._pub_voice_command = self.create_publisher(
            String, DEFAULT_TOPICS['voice_command'], 10)
        self._pub_voice_confidence = self.create_publisher(
            Float32, DEFAULT_TOPICS['voice_confidence'], 10)
        self._pub_voice_status = self.create_publisher(
            String, DEFAULT_TOPICS['voice_status'], 10)
        self._pub_voice_ranked = self.create_publisher(
            String, DEFAULT_TOPICS['voice_ranked_predictions'], 10)
        self._pub_voice_time = self.create_publisher(
            Float32, DEFAULT_TOPICS['voice_inference_time'], 10)

        # ── Control publishers (outgoing: dashboard → ROS) ─────────────────
        cmd_vel_out = self.get_parameter('cmd_vel_out_topic').get_parameter_value().string_value
        self._pub_cmd_vel_out    = self.create_publisher(Twist, cmd_vel_out, 10)
        # Teleop-priority signal: obstacle_avoidance suppresses navigation while this arrives
        self._pub_cmd_vel_teleop = self.create_publisher(Twist, '/cmd_vel_teleop', 10)

        _latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._pub_goal_pose  = self.create_publisher(PoseStamped, DEFAULT_TOPICS['goal_pose'], _latched_qos)
        self._pub_nav_wp     = self.create_publisher(String, DEFAULT_TOPICS['navigate_to_waypoint'], 10)
        self._pub_slam_reset = self.create_publisher(Bool, DEFAULT_TOPICS['slam_reset'], 10)
        self._pub_load_map   = self.create_publisher(String, '/slam/load_map', 10)

        self.get_logger().info(f'Control publishers ready — cmd_vel_out: {cmd_vel_out}')

        # Cache for map-frame robot pose from slam_node (overrides odom frame pose).
        self._map_pose: dict | None = None

        # Voice command executor state
        self.declare_parameter('voice_linear_speed',  0.20)
        self.declare_parameter('voice_angular_speed', 0.50)
        self.declare_parameter('voice_cmd_duration',  1.00)   # seconds per command
        self._voice_lin   = self.get_parameter('voice_linear_speed').value
        self._voice_ang   = self.get_parameter('voice_angular_speed').value
        self._voice_dur   = self.get_parameter('voice_cmd_duration').value
        self._voice_timer = None   # threading.Timer to stop after duration

        # Start WebSocket server.
        host = self.get_parameter('websocket_host').get_parameter_value().string_value
        port = self.get_parameter('websocket_port').get_parameter_value().integer_value
        self._ws = WebSocketServer(host=host, port=port)

        # Register audio handler only if inference engine is ready.
        if self._inference_engine is not None:
            self._ws.set_audio_handler(self._handle_audio_bytes)

        # Register command handler for incoming dashboard control messages.
        self._ws.set_command_handler(self._handle_command)

        self._ws.start()

        _sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Core subscribers.
        self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').get_parameter_value().string_value,
            self._odom_cb, _sensor_qos)

        self.create_subscription(
            LaserScan,
            self.get_parameter('scan_topic').get_parameter_value().string_value,
            self._scan_cb, _sensor_qos)

        self.create_subscription(
            OccupancyGrid,
            self.get_parameter('map_topic').get_parameter_value().string_value,
            self._map_cb, 1)

        self.create_subscription(
            Twist,
            self.get_parameter('cmd_vel_topic').get_parameter_value().string_value,
            lambda msg: self._twist_cb(msg, 'cmd_vel'), 10)

        # Map-frame robot pose from slam_node — overrides odom-frame pose for drawing.
        self.create_subscription(
            PoseStamped, '/slam/robot_pose_in_map', self._slam_pose_cb, 10)

        # Execute voice commands as robot actions.
        self.create_subscription(
            String, DEFAULT_TOPICS['voice_command'], self._voice_exec_cb, 10)

        # Optional subscribers — tolerate missing topics.
        self.create_subscription(
            Twist,
            self.get_parameter('cmd_vel_in_topic').get_parameter_value().string_value,
            lambda msg: self._twist_cb(msg, 'cmd_vel_in'), 10)

        self.create_subscription(
            String,
            self.get_parameter('voice_command_topic').get_parameter_value().string_value,
            self._voice_command_cb, 10)

        self.create_subscription(
            Float32,
            self.get_parameter('voice_confidence_topic').get_parameter_value().string_value,
            self._voice_confidence_cb, 10)

        self.create_subscription(
            String,
            self.get_parameter('voice_status_topic').get_parameter_value().string_value,
            self._voice_status_cb, 10)

        self.create_subscription(
            String,
            self.get_parameter('voice_ranked_predictions_topic').get_parameter_value().string_value,
            self._voice_ranked_cb, 10)

        self.create_subscription(
            Float32,
            self.get_parameter('voice_inference_time_topic').get_parameter_value().string_value,
            self._voice_inference_cb, 10)

        self.create_subscription(
            CompressedImage,
            self.get_parameter('camera_topic').get_parameter_value().string_value,
            self._camera_cb, _sensor_qos)

        self.get_logger().info(f'puzzlebot_web_bridge ready — WebSocket at ws://{host}:{port}/ws')

    # ------------------------------------------------------------------ #
    #  Core callbacks
    # ------------------------------------------------------------------ #

    def _slam_pose_cb(self, msg: PoseStamped):
        import math as _math
        q = msg.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._map_pose = {
            'x': round(msg.pose.position.x, 4),
            'y': round(msg.pose.position.y, 4),
            'theta': round(_math.atan2(siny, cosy), 4),
        }

    def _voice_exec_cb(self, msg: String):
        import threading
        cmd = msg.data.strip().lower()

        # Cancel any in-progress timed command
        if self._voice_timer is not None:
            self._voice_timer.cancel()
            self._voice_timer = None

        twist = Twist()
        if cmd == 'avanzar':
            twist.linear.x = self._voice_lin
        elif cmd == 'retroceder':
            twist.linear.x = -self._voice_lin
        elif cmd == 'izquierda':
            twist.angular.z = self._voice_ang
        elif cmd == 'derecha':
            twist.angular.z = -self._voice_ang
        elif cmd == 'alto':
            pass  # zero twist = stop, publish once and return
        elif cmd == 'inicio':
            self._pub_nav_wp.publish(String(data='inicio'))
            self.get_logger().info('voice → navigate_to_waypoint: inicio')
            return
        else:
            self.get_logger().warn(f'voice_exec: unknown command "{cmd}"')
            return

        self._pub_cmd_vel_out.publish(twist)
        self._pub_cmd_vel_teleop.publish(twist)
        self.get_logger().info(
            f'voice → cmd_vel: {cmd}  '
            f'(lin={twist.linear.x:.2f} ang={twist.angular.z:.2f})')

        if cmd != 'alto':
            def _stop():
                stop = Twist()
                self._pub_cmd_vel_out.publish(stop)
                self._pub_cmd_vel_teleop.publish(stop)
            self._voice_timer = threading.Timer(self._voice_dur, _stop)
            self._voice_timer.daemon = True
            self._voice_timer.start()

    def _odom_cb(self, msg: Odometry):
        if self._rl['odom'].should_send():
            payload = odom_to_json(msg)
            # Override pose with map-frame pose from slam_node when available
            if self._map_pose is not None:
                payload['pose'] = self._map_pose
            self._ws.broadcast_sync(payload)

    def _scan_cb(self, msg: LaserScan):
        if self._rl['scan'].should_send():
            self._ws.broadcast_sync(scan_to_json(msg))

    def _map_cb(self, msg: OccupancyGrid):
        if self._rl['map'].should_send():
            self._ws.broadcast_sync(map_to_json(msg))

    def _camera_cb(self, msg: CompressedImage):
        if self._rl['camera'].should_send():
            self._ws.broadcast_sync(camera_to_json(msg))

    def _twist_cb(self, msg: Twist, source: str):
        key = 'cmd_vel' if source == 'cmd_vel' else 'cmd_vel_in'
        if self._rl[key].should_send():
            self._ws.broadcast_sync(twist_to_json(msg, source))

    # ------------------------------------------------------------------ #
    #  Voice callbacks — accumulate fields, send on command arrival
    # ------------------------------------------------------------------ #

    def _voice_command_cb(self, msg: String):
        self._voice['command'] = msg.data
        self._ws.broadcast_sync(self._build_voice_payload())

    def _voice_confidence_cb(self, msg: Float32):
        self._voice['confidence'] = msg.data

    def _voice_status_cb(self, msg: String):
        self._voice['status'] = msg.data

    def _voice_ranked_cb(self, msg: String):
        self._voice['ranked_predictions'] = msg.data

    def _voice_inference_cb(self, msg: Float32):
        self._voice['inference_time_ms'] = msg.data

    def _build_voice_payload(self) -> dict:
        return voice_to_json(
            command=self._voice['command'],
            confidence=self._voice['confidence'],
            status=self._voice['status'],
            inference_time_ms=self._voice['inference_time_ms'],
            ranked_predictions_raw=self._voice['ranked_predictions'],
        )

    # ------------------------------------------------------------------ #
    #  Voice inference from dashboard microphone (POST /audio)
    # ------------------------------------------------------------------ #

    def _load_voice_engine(self, artifact_dir: str) -> None:
        try:
            from puzzlebot_voice_commands.voice_inference import VoiceInferenceEngine
            self._inference_engine = VoiceInferenceEngine.load(artifact_dir)
            self.get_logger().info(f'Voice inference engine loaded from: {artifact_dir}')
        except ModuleNotFoundError as e:
            self.get_logger().warn(f'puzzlebot_voice_commands not importable: {e}')
        except FileNotFoundError as e:
            self.get_logger().warn(f'Voice models not found: {e}')
        except Exception as e:
            self.get_logger().warn(f'Voice inference failed to load ({type(e).__name__}): {e}')

    def _handle_audio_bytes(self, wav_bytes: bytes) -> None:
        """Decode WAV bytes, run inference, publish results to ROS + WebSocket."""
        try:
            from scipy.io import wavfile

            self._pub_voice_status.publish(String(data='processing'))

            buf = io.BytesIO(wav_bytes)
            sample_rate, data = wavfile.read(buf)
            self.get_logger().info(
                f'Audio received: {len(wav_bytes)} bytes, '
                f'sr={sample_rate} Hz, samples={len(data)}, dtype={data.dtype}'
            )

            # Normalize to float32 [-1, 1]
            if data.dtype == np.int16:
                signal = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                signal = data.astype(np.float32) / 2147483648.0
            else:
                signal = data.astype(np.float32)

            if signal.ndim == 2:
                signal = signal.mean(axis=1)

            result = self._inference_engine.infer(signal, sample_rate)

            ranked_json = json.dumps({
                'kmeans': result.ranked_kmeans[:3],
                'hmm': result.ranked_hmm[:3],
            })

            # Pre-load the accumulator with fresh data so that when _voice_command_cb
            # fires (triggered by our own publication below), _build_voice_payload()
            # returns the new result instead of stale data from the previous command.
            self._voice['command']           = result.command
            self._voice['confidence']        = result.confidence
            self._voice['status']            = 'idle'
            self._voice['inference_time_ms'] = result.inference_time_ms
            self._voice['ranked_predictions'] = ranked_json

            # Broadcast directly to WebSocket — don't rely on the ROS roundtrip.
            self._ws.broadcast_sync(self._build_voice_payload())

            # Also publish to ROS topics for other nodes that may listen.
            self._pub_voice_ranked.publish(String(data=ranked_json))
            self._pub_voice_time.publish(Float32(data=result.inference_time_ms))
            self._pub_voice_confidence.publish(Float32(data=result.confidence))
            self._pub_voice_status.publish(String(data='idle'))
            self._pub_voice_command.publish(String(data=result.command))

            self.get_logger().info(
                f'Voice [POST /audio]: {result.command.upper()}  '
                f'conf={result.confidence:.4f}  {result.inference_time_ms:.1f}ms'
            )

        except Exception as exc:
            self.get_logger().error(f'Audio inference error: {exc}')
            self._pub_voice_status.publish(String(data='idle'))

    # ------------------------------------------------------------------ #
    #  Incoming commands from dashboard (WebSocket → ROS)
    # ------------------------------------------------------------------ #

    def _handle_command(self, data: dict) -> None:
        """Route a JSON command received from the browser to the appropriate ROS publisher."""
        msg_type = data.get('type')
        try:
            if msg_type == 'cmd_vel':
                msg = Twist()
                msg.linear.x  = float(data.get('linear_x', 0.0))
                msg.angular.z = float(data.get('angular_z', 0.0))
                self._pub_cmd_vel_out.publish(msg)
                self._pub_cmd_vel_teleop.publish(msg)

            elif msg_type == 'goal_pose':
                msg = PoseStamped()
                msg.header.stamp    = self.get_clock().now().to_msg()
                msg.header.frame_id = str(data.get('frame_id', 'map'))
                msg.pose.position.x = float(data.get('x', 0.0))
                msg.pose.position.y = float(data.get('y', 0.0))
                theta = float(data.get('theta', 0.0))
                msg.pose.orientation.z = math.sin(theta / 2.0)
                msg.pose.orientation.w = math.cos(theta / 2.0)
                self._pub_goal_pose.publish(msg)
                self.get_logger().info(
                    f'goal_pose → ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})'
                    f' θ={math.degrees(theta):.1f}°')

            elif msg_type == 'navigate_to_waypoint':
                name = str(data.get('name', '')).strip()
                if name:
                    self._pub_nav_wp.publish(String(data=name))
                    self.get_logger().info(f'navigate_to_waypoint → {name}')

            elif msg_type == 'slam_reset':
                self._pub_slam_reset.publish(Bool(data=True))
                self.get_logger().info('SLAM reset requested from dashboard')

            elif msg_type == 'list_maps':
                maps_dir = self.get_parameter('maps_dir').get_parameter_value().string_value
                if not maps_dir:
                    maps_dir = os.getcwd()
                raw = (glob.glob(os.path.join(maps_dir, '*.pgm')) +
                       glob.glob(os.path.join(maps_dir, '*.png')))
                # Deduplicate by stem — prefer .png over .pgm for same base name
                stems: dict = {}
                for f in sorted(raw):
                    stem = os.path.splitext(os.path.basename(f))[0]
                    ext  = os.path.splitext(f)[1].lower()
                    if stem not in stems or ext == '.png':
                        stems[stem] = os.path.basename(f)
                files = sorted(stems.values())
                self._ws.broadcast_sync({'type': 'available_maps', 'maps': files})
                self.get_logger().info(f'list_maps → {len(files)} maps in {maps_dir}')

            elif msg_type == 'load_map':
                filename = str(data.get('filename', '')).strip()
                maps_dir = self.get_parameter('maps_dir').get_parameter_value().string_value
                if not maps_dir:
                    maps_dir = os.getcwd()
                full_path = os.path.join(maps_dir, filename)
                if os.path.isfile(full_path):
                    self._pub_load_map.publish(String(data=full_path))
                    self.get_logger().info(f'load_map → {full_path}')
                else:
                    self.get_logger().warn(f'load_map: file not found: {full_path}')

            elif msg_type == 'use_slam_map':
                # Switch back to live SLAM by resetting the map.
                self._pub_slam_reset.publish(Bool(data=True))
                self.get_logger().info('use_slam_map → SLAM reset to live mode')

            elif msg_type == 'elevator':
                action = str(data.get('action', '')).strip()
                self.get_logger().info(f'elevator command: {action} (backend pending)')

            else:
                self.get_logger().debug(f'Unknown command type: {msg_type}')

        except Exception as exc:
            self.get_logger().error(f'Error handling command {msg_type}: {exc}')

    def destroy_node(self):
        self._ws.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
