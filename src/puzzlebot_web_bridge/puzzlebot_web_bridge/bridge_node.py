"""ROS 2 node that bridges topics to a WebSocket server.

Subscribes to core and optional topics, serializes messages, and broadcasts
JSON to all connected WebSocket clients. Missing optional topics are silently
ignored — the bridge will not crash if they don't exist.

Also exposes POST /audio for voice inference when the microphone is on the
dashboard machine. Audio is received as WAV bytes, inference runs locally
(KMeans + HMM), and results are published to /voice/* topics AND broadcast
to WebSocket clients.

SAFETY: This node NEVER publishes to any control topic (/cmd_vel, /goal_pose,
/initialpose, etc.). Publishing to /voice/* is intentional and safe.
"""

import io
import json
import logging
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import CompressedImage, LaserScan
from std_msgs.msg import Float32, String

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

        # Start WebSocket server.
        host = self.get_parameter('websocket_host').get_parameter_value().string_value
        port = self.get_parameter('websocket_port').get_parameter_value().integer_value
        self._ws = WebSocketServer(host=host, port=port)

        # Register audio handler only if inference engine is ready.
        if self._inference_engine is not None:
            self._ws.set_audio_handler(self._handle_audio_bytes)

        self._ws.start()

        # Core subscribers.
        self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').get_parameter_value().string_value,
            self._odom_cb, 10)

        self.create_subscription(
            LaserScan,
            self.get_parameter('scan_topic').get_parameter_value().string_value,
            self._scan_cb, 10)

        self.create_subscription(
            OccupancyGrid,
            self.get_parameter('map_topic').get_parameter_value().string_value,
            self._map_cb, 1)

        self.create_subscription(
            Twist,
            self.get_parameter('cmd_vel_topic').get_parameter_value().string_value,
            lambda msg: self._twist_cb(msg, 'cmd_vel'), 10)

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

        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
        _cam_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            CompressedImage,
            self.get_parameter('camera_topic').get_parameter_value().string_value,
            self._camera_cb, _cam_qos)

        self.get_logger().info(f'puzzlebot_web_bridge ready — WebSocket at ws://{host}:{port}/ws')

    # ------------------------------------------------------------------ #
    #  Core callbacks
    # ------------------------------------------------------------------ #

    def _odom_cb(self, msg: Odometry):
        if self._rl['odom'].should_send():
            self._ws.broadcast_sync(odom_to_json(msg))

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

            # Publish to ROS topics — _voice_command_cb will handle the WebSocket broadcast.
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
