"""ROS 2 node that bridges topics to a WebSocket server.

Subscribes to core and optional topics, serializes messages, and broadcasts
JSON to all connected WebSocket clients. Missing optional topics are silently
ignored — the bridge will not crash if they don't exist.

SAFETY: This node NEVER publishes to any control topic.
"""

import logging

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

        # Rate limiters.
        self._rl = {
            key: RateLimiter(hz)
            for key, hz in RATE_LIMITS_HZ.items()
        }

        # Voice state accumulator (assembled before sending).
        self._voice: dict = {
            'command': None,
            'confidence': None,
            'status': None,
            'inference_time_ms': None,
            'ranked_predictions': None,
        }

        # Start WebSocket server.
        host = self.get_parameter('websocket_host').get_parameter_value().string_value
        port = self.get_parameter('websocket_port').get_parameter_value().integer_value
        self._ws = WebSocketServer(host=host, port=port)
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

        self.get_logger().info('puzzlebot_web_bridge ready — WebSocket at ws://%s:%d/ws', host, port)

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
