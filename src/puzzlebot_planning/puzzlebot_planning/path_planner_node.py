import heapq
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import numpy as np


def astar(grid, start, goal):
    """A* on an occupancy grid. Returns list of (row, col) cells or None."""
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g = {start: 0.0}
    h = lambda n: math.hypot(n[0]-goal[0], n[1]-goal[1])

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
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = r+dr, c+dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] > 50:   # occupied cell
                continue
            step = math.hypot(dr, dc)
            ng = g[current] + step
            nb = (nr, nc)
            if ng < g.get(nb, float('inf')):
                came_from[nb] = current
                g[nb] = ng
                heapq.heappush(open_set, (ng + h(nb), nb))
    return None


class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')
        self.declare_parameter('inflation_radius', 0.10)  # metres

        self.map_      = None
        self.robot_x_  = 0.0
        self.robot_y_  = 0.0

        self.sub_map_  = self.create_subscription(OccupancyGrid, '/map',
                          self.map_cb, 1)
        self.sub_pose_ = self.create_subscription(PoseWithCovarianceStamped,
                          '/initialpose', self.pose_cb, 10)
        self.sub_goal_ = self.create_subscription(PoseStamped, '/goal_pose',
                          self.goal_cb, 10)
        self.pub_path_ = self.create_publisher(Path, '/planned_path', 1)

        self.get_logger().info('path_planner_node started (A*)')

    def map_cb(self, msg: OccupancyGrid):
        self.map_ = msg

    def pose_cb(self, msg: PoseWithCovarianceStamped):
        self.robot_x_ = msg.pose.pose.position.x
        self.robot_y_ = msg.pose.pose.position.y

    def goal_cb(self, msg: PoseStamped):
        if self.map_ is None:
            self.get_logger().warn('No map received yet — ignoring goal')
            return
        self._plan(msg)

    def _plan(self, goal_msg: PoseStamped):
        m  = self.map_
        res = m.info.resolution
        ox  = m.info.origin.position.x
        oy  = m.info.origin.position.y
        W   = m.info.width
        H   = m.info.height

        grid = np.array(m.data, dtype=np.int8).reshape(H, W)
        # Unknown cells (-1) treated as free for planning
        grid[grid < 0] = 0

        def world_to_cell(wx, wy):
            c = int((wx - ox) / res)
            r = int((wy - oy) / res)
            return (r, c)

        start = world_to_cell(self.robot_x_, self.robot_y_)
        goal  = world_to_cell(goal_msg.pose.position.x, goal_msg.pose.position.y)

        # Clamp to grid bounds
        def clamp_cell(cell):
            return (max(0, min(H-1, cell[0])), max(0, min(W-1, cell[1])))

        start = clamp_cell(start)
        goal  = clamp_cell(goal)

        cell_path = astar(grid, start, goal)
        if cell_path is None:
            self.get_logger().warn(f'A* found no path from {start} to {goal}')
            return

        path = Path()
        path.header.stamp    = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'
        for (r, c) in cell_path:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = ox + (c + 0.5) * res
            ps.pose.position.y = oy + (r + 0.5) * res
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)

        self.pub_path_.publish(path)
        self.get_logger().info(f'Published path with {len(path.poses)} waypoints')


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
