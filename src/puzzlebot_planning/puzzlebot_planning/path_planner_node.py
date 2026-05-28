"""
A* path planner node for Puzzlebot.

Mejoras respecto a la versión anterior:
- Inflación morfológica de obstáculos antes de planear.
- Pose actual del robot desde TF (map→base_footprint) en lugar de solo /initialpose.
- Replanning automático cuando llega un mapa nuevo (si hay un goal activo).
- Emergency-stop: no publica ruta si el robot ya está en una celda ocupada.
- Parámetros configurables desde YAML.
"""
import heapq
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.ndimage import binary_dilation

try:
    import tf2_ros
    from tf2_ros import Buffer, TransformListener
    _HAS_TF2 = True
except ImportError:
    _HAS_TF2 = False


# ─────────────────────────────────────────────────────────────────────────────
# A* algorithm (pure function, no ROS deps)
# ─────────────────────────────────────────────────────────────────────────────

def _astar(grid: np.ndarray, start: tuple, goal: tuple):
    """
    A* search on a binary grid (0 = free, 1 = blocked).
    Returns list of (row, col) from start to goal, or None if no path exists.
    Uses 8-connected neighbors.
    """
    rows, cols = grid.shape
    if grid[start] or grid[goal]:
        return None

    open_set = []
    heapq.heappush(open_set, (0.0, start))
    came_from = {}
    g = {start: 0.0}

    def h(n):
        return math.hypot(n[0] - goal[0], n[1] - goal[1])

    NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return list(reversed(path))

        r, c = current
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc]:
                continue
            step = math.hypot(dr, dc)
            ng = g[current] + step
            nb = (nr, nc)
            if ng < g.get(nb, float('inf')):
                came_from[nb] = current
                g[nb] = ng
                heapq.heappush(open_set, (ng + h(nb), nb))
    return None


def _inflate_grid(grid: np.ndarray, radius_cells: int) -> np.ndarray:
    """
    Binary dilation of occupied cells by radius_cells.
    Input:  grid with values 0 (free) or 1 (occupied).
    Output: same shape, inflated occupied mask.
    """
    if radius_cells <= 0:
        return grid
    struct = np.ones((2 * radius_cells + 1, 2 * radius_cells + 1), dtype=bool)
    occupied = grid.astype(bool)
    inflated = binary_dilation(occupied, structure=struct)
    return inflated.astype(np.int8)


# ─────────────────────────────────────────────────────────────────────────────
# ROS 2 node
# ─────────────────────────────────────────────────────────────────────────────

class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('inflation_radius', 0.15)     # metres
        self.declare_parameter('occupied_threshold', 50)     # 0-100
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('use_tf_pose', True)          # prefer TF over /initialpose
        self.declare_parameter('replan_on_new_map', True)    # replan when map updates

        self.inflation_radius_m  = self.get_parameter('inflation_radius').value
        self.occ_thresh          = self.get_parameter('occupied_threshold').value
        self.map_frame           = self.get_parameter('map_frame').value
        self.base_frame          = self.get_parameter('base_frame').value
        self.use_tf_pose         = self.get_parameter('use_tf_pose').value
        self.replan_on_new_map   = self.get_parameter('replan_on_new_map').value

        # ── State ─────────────────────────────────────────────────────────
        self.map_      = None
        self.robot_x_  = 0.0
        self.robot_y_  = 0.0
        self.goal_msg_ = None   # last received goal; used for replanning

        # ── TF2 (optional — gracefully degrades to /initialpose) ──────────
        if _HAS_TF2 and self.use_tf_pose:
            self._tf_buffer   = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
        else:
            self._tf_buffer = None

        # ── QoS for map subscription (TRANSIENT_LOCAL to get latched maps) ─
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ── Subscriptions ─────────────────────────────────────────────────
        self.sub_map_  = self.create_subscription(
            OccupancyGrid, '/map', self.map_cb, map_qos)
        self.sub_pose_ = self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self.pose_cb, 10)
        self.sub_goal_ = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_cb, 10)

        # ── Publisher ─────────────────────────────────────────────────────
        self.pub_path_ = self.create_publisher(Path, '/planned_path', 1)

        self.get_logger().info(
            f'path_planner_node started (A*) — '
            f'inflation={self.inflation_radius_m} m, '
            f'use_tf_pose={self.use_tf_pose}, '
            f'replan_on_new_map={self.replan_on_new_map}'
        )

    # ── Callbacks ─────────────────────────────────────────────────────────

    def map_cb(self, msg: OccupancyGrid):
        self.map_ = msg
        if self.replan_on_new_map and self.goal_msg_ is not None:
            self._update_robot_pose_from_tf()
            self._plan(self.goal_msg_)

    def pose_cb(self, msg: PoseWithCovarianceStamped):
        # /initialpose from RViz "2D Pose Estimate" tool — fallback when TF unavailable
        self.robot_x_ = msg.pose.pose.position.x
        self.robot_y_ = msg.pose.pose.position.y

    def goal_cb(self, msg: PoseStamped):
        self.goal_msg_ = msg
        if self.map_ is None:
            self.get_logger().warn('No map received yet — ignoring goal')
            return
        self._update_robot_pose_from_tf()
        self._plan(msg)

    # ── TF pose lookup ────────────────────────────────────────────────────

    def _update_robot_pose_from_tf(self):
        if self._tf_buffer is None:
            return
        try:
            tf = self._tf_buffer.lookup_transform(
                self.map_frame, self.base_frame,
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1))
            self.robot_x_ = tf.transform.translation.x
            self.robot_y_ = tf.transform.translation.y
        except Exception:
            # TF not yet available — keep last known pose from /initialpose
            pass

    # ── Planning ──────────────────────────────────────────────────────────

    def _plan(self, goal_msg: PoseStamped):
        m   = self.map_
        res = m.info.resolution
        ox  = m.info.origin.position.x
        oy  = m.info.origin.position.y
        W   = m.info.width
        H   = m.info.height

        # Build binary occupancy grid: 1 = obstacle, 0 = free
        raw = np.array(m.data, dtype=np.int8).reshape(H, W)
        binary = np.zeros((H, W), dtype=np.int8)
        binary[raw > self.occ_thresh] = 1   # occupied
        # Unknown cells (-1) left as free (0) — planner assumes open space

        # Inflate obstacles by robot radius
        radius_cells = int(math.ceil(self.inflation_radius_m / res))
        inflated = _inflate_grid(binary, radius_cells)

        def world_to_cell(wx, wy):
            c = int((wx - ox) / res)
            r = int((wy - oy) / res)
            return (r, c)

        def clamp_cell(cell):
            return (max(0, min(H - 1, cell[0])),
                    max(0, min(W - 1, cell[1])))

        start = clamp_cell(world_to_cell(self.robot_x_, self.robot_y_))
        goal  = clamp_cell(world_to_cell(
            goal_msg.pose.position.x, goal_msg.pose.position.y))

        # Verify start cell is not inside an inflated obstacle
        if inflated[start]:
            self.get_logger().warn(
                f'Start cell {start} is inside inflated obstacle — '
                'robot may be too close to a wall. Shrinking inflation by 50%.')
            radius_cells = max(0, radius_cells // 2)
            inflated = _inflate_grid(binary, radius_cells)

        if inflated[goal]:
            self.get_logger().warn(
                f'Goal cell {goal} is inside inflated obstacle — goal may be inside a wall.')

        cell_path = _astar(inflated, start, goal)

        if cell_path is None:
            self.get_logger().warn(
                f'A* found no path from {start} to {goal}. '
                'Check that goal is reachable and not inside an obstacle.')
            return

        # Convert cell path to nav_msgs/Path
        now = self.get_clock().now().to_msg()
        path = Path()
        path.header.stamp    = now
        path.header.frame_id = self.map_frame

        for (r, c) in cell_path:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x  = ox + (c + 0.5) * res
            ps.pose.position.y  = oy + (r + 0.5) * res
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)

        self.pub_path_.publish(path)
        self.get_logger().info(
            f'Path: {len(path.poses)} waypoints | '
            f'start=({self.robot_x_:.2f},{self.robot_y_:.2f}) '
            f'→ goal=({goal_msg.pose.position.x:.2f},{goal_msg.pose.position.y:.2f}) | '
            f'inflation={radius_cells} cells ({radius_cells * res:.2f} m)'
        )


def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()
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
