"""
Static map server — publishes a PNG/PGM map as nav_msgs/OccupancyGrid on /map.

Replaces nav2_map_server for this project. Loads a grayscale image where:
  pixel >= occupied_thresh*255  → free (0)
  pixel <= free_thresh*255      → occupied (100)
  otherwise                     → unknown (-1)

Default thresholds (nav2_map_server compatible):
  occupied_thresh: 0.65   (pixel >= 166 → free)
  free_thresh:     0.196  (pixel <= 50  → occupied)

Parameters:
  map_path        str    path to PNG or PGM file
  map_resolution  float  0.05  metres per pixel
  map_origin_x    float  world X at bottom-left corner of the image
  map_origin_y    float  world Y at bottom-left corner of the image
  map_frame       str    'map'
  occupied_thresh float  0.65   normalised intensity threshold — above → free
  free_thresh     float  0.196  normalised intensity threshold — below → occupied
"""

import numpy as np
import rclpy
from nav_msgs.msg import MapMetaData, OccupancyGrid
from PIL import Image
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


class MapServerNode(Node):
    def __init__(self):
        super().__init__('map_server_node')

        self.declare_parameter('map_path',       '')
        self.declare_parameter('map_resolution',  0.05)
        self.declare_parameter('map_origin_x',   -0.25)
        self.declare_parameter('map_origin_y',   -0.25)
        self.declare_parameter('map_frame',      'map')

        map_path   = self.get_parameter('map_path').value
        resolution = float(self.get_parameter('map_resolution').value)
        origin_x   = float(self.get_parameter('map_origin_x').value)
        origin_y   = float(self.get_parameter('map_origin_y').value)
        map_frame  = self.get_parameter('map_frame').value

        if not map_path:
            self.get_logger().error('map_path parameter is empty — set it in mcl_params.yaml')
            return

        self.declare_parameter('occupied_thresh',  0.65)
        self.declare_parameter('free_thresh',      0.196)

        occ_thresh  = float(self.get_parameter('occupied_thresh').value)
        free_thresh = float(self.get_parameter('free_thresh').value)

        img = Image.open(map_path).convert('L')
        arr = np.array(img, dtype=np.uint8)

        # PNG origin is top-left, OccupancyGrid origin is bottom-left → flip
        arr = np.flipud(arr)
        h, w = arr.shape

        flat = arr.flatten().astype(np.float32) / 255.0

        data = np.full(h * w, -1, dtype=np.int8)          # default: unknown
        data[flat >= occ_thresh]  = 0                      # free  (bright)
        data[flat <= free_thresh] = 100                    # occupied (dark)

        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._pub = self.create_publisher(OccupancyGrid, '/map', map_qos)
        self._msg = self._build_msg(data, w, h, resolution, origin_x, origin_y, map_frame)

        # Publish once immediately, then keep latched via TRANSIENT_LOCAL
        self._pub.publish(self._msg)
        self.get_logger().info(
            f'Map loaded: {map_path} — {w}×{h} px @ {resolution} m/px, '
            f'origin=({origin_x}, {origin_y})')

    def _build_msg(self, data, w, h, resolution, origin_x, origin_y, frame):
        msg = OccupancyGrid()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = frame
        msg.info            = MapMetaData()
        msg.info.resolution = resolution
        msg.info.width      = w
        msg.info.height     = h
        msg.info.origin.position.x  = origin_x
        msg.info.origin.position.y  = origin_y
        msg.info.origin.orientation.w = 1.0
        msg.data = data.tolist()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = MapServerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
