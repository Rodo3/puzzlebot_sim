"""
path_planner_node.py — Planeador de rutas con algoritmo A* + costmap de distancia.

POSICIÓN EN EL PIPELINE:
  slam_node → /map ──┐
  /goal_pose ────────┤→ [ESTE NODO] → /planned_path → steering_controller_node
  TF map→base ───────┘

FUNCIÓN:
  Recibe el mapa de ocupación (/map) y un destino (/goal_pose).
  Pasos antes de planear:
    1. Cierre morfológico — sella huecos pequeños en paredes del mapa SLAM.
    2. Inflación de obstáculos — expande cada obstáculo por inflation_radius.
    3. Costmap de distancia — penaliza celdas cerca de paredes; A* prefiere
       el centro del corredor en lugar de rozar paredes.
    4. Validación del goal — si cae dentro de un obstáculo, lo reprojecta
       automáticamente a la celda libre más cercana segura.

  La pose actual del robot se obtiene preferentemente del árbol TF
  (map→base_footprint). Si TF no está disponible, usa /initialpose de respaldo.

TOPICS SUSCRITOS:
  /map          (nav_msgs/OccupancyGrid)                  — mapa construido por slam_node
  /goal_pose    (geometry_msgs/PoseStamped)               — destino desde RViz o waypoint_navigator
  /initialpose  (geometry_msgs/PoseWithCovarianceStamped) — pose manual de respaldo

TOPICS PUBLICADOS:
  /planned_path (nav_msgs/Path) — secuencia de poses desde robot hasta goal

PARÁMETROS CLAVE:
  inflation_radius          [0.22 m]  — radio de inflación de obstáculos
  cost_weight               [2.0]     — peso del costmap (0 = A* clásico, 3 = evita paredes fuerte)
  unknown_as_occupied       [True]    — celdas desconocidas tratadas como obstáculos
  allow_goal_reprojection   [True]    — reprojectar goal inseguro automáticamente
  wall_closing_radius       [2 celdas] — radio de cierre morfológico previo a inflación
"""
import heapq
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.ndimage import binary_closing, binary_dilation, distance_transform_edt
from std_msgs.msg import Bool

try:
    import tf2_ros
    from tf2_ros import Buffer, TransformListener
    _HAS_TF2 = True
except ImportError:
    _HAS_TF2 = False


# ─────────────────────────────────────────────────────────────────────────────
# Algoritmo A* con costmap de distancia (función pura, sin dependencias de ROS)
# ─────────────────────────────────────────────────────────────────────────────

def _astar(
    binary: np.ndarray,
    cost_map: np.ndarray,
    start: tuple,
    goal: tuple,
    cost_weight: float,
) -> list | None:
    """
    A* en un grid binario (0 = libre, 1 = bloqueado) con costo adicional por celda.

    Cada paso paga:
      costo = distancia_euclidiana + cost_weight × cost_map[vecino]

    cost_map tiene valores altos cerca de paredes y 0 en espacio abierto,
    por lo que el planner prefiere el centro de los corredores en lugar de rozar paredes.

    Retorna lista de (fila, columna) de start a goal, o None si no hay camino.
    Usa conectividad de 8 vecinos.
    """
    rows, cols = binary.shape
    if binary[start] or binary[goal]:
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
            if binary[nr, nc]:
                continue
            step = math.hypot(dr, dc) + cost_weight * float(cost_map[nr, nc])
            ng = g[current] + step
            nb = (nr, nc)
            if ng < g.get(nb, float('inf')):
                came_from[nb] = current
                g[nb] = ng
                heapq.heappush(open_set, (ng + h(nb), nb))
    return None


def _inflate_grid(grid: np.ndarray, radius_cells: int) -> np.ndarray:
    if radius_cells <= 0:
        return grid
    struct = np.ones((2 * radius_cells + 1, 2 * radius_cells + 1), dtype=bool)
    inflated = binary_dilation(grid.astype(bool), structure=struct)
    return inflated.astype(np.int8)


def _build_cost_map(inflated: np.ndarray, resolution: float, cost_weight: float,
                    max_cost: float) -> np.ndarray:
    """
    Build a per-cell cost map from the inflated obstacle grid.

    Free cells receive a cost inversely proportional to their distance to the
    nearest obstacle.  Cells deep in open space get cost ≈ 0; cells just
    outside the inflation zone get cost ≈ max_cost.

    Formula (per cell):  cost = max_cost / (1 + dist_m / reference_dist)
      where reference_dist = resolution (one cell width).
    Cells inside the inflated zone keep cost = max_cost (they are blocked anyway).
    """
    # Distance in cells → convert to metres
    dist_cells = distance_transform_edt(~inflated.astype(bool))
    dist_m = dist_cells * resolution

    with np.errstate(divide='ignore', invalid='ignore'):
        raw = max_cost / (1.0 + dist_m / max(resolution, 1e-6))

    # Inside inflated zone: set to max_cost (will be blocked)
    raw[inflated.astype(bool)] = max_cost
    return raw.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# ROS 2 node
# ─────────────────────────────────────────────────────────────────────────────

class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')

        # ── Core parameters ───────────────────────────────────────────────────
        self.declare_parameter('inflation_radius',    0.22)   # metres
        self.declare_parameter('occupied_threshold',  50)     # 0-100
        self.declare_parameter('map_frame',           'map')
        self.declare_parameter('base_frame',          'base_footprint')
        self.declare_parameter('use_tf_pose',         True)
        self.declare_parameter('replan_on_new_map',   True)
        # Treat unknown cells (-1) as occupied so the planner never routes
        # through unmapped areas.  Set to false only for full maps.
        self.declare_parameter('unknown_as_occupied', True)

        # Apply morphological closing to the obstacle grid BEFORE inflating.
        # This fills small holes and gaps in walls caused by an incomplete map
        # (broken internal structures, partial walls from SLAM noise).
        # wall_closing_radius: half-size of the closing element in cells.
        #   1 cell = map resolution (0.05 m at 5 cm/px) → fills gaps ≤ 5 cm.
        #   2 cells → fills gaps ≤ 10 cm (use for noisier maps).
        #   Set 0 to disable.
        self.declare_parameter('wall_closing_radius', 2)

        # ── Costmap parameters ────────────────────────────────────────────────
        # cost_weight: how much the A* penalises proximity to walls.
        #   0.0 → classic A* (ignores distance to walls, hugs obstacles)
        #   1.0 → moderate wall avoidance (good starting point)
        #   3.0 → strong wall avoidance (routes through corridor centres)
        # max_obstacle_cost: maximum per-cell cost assigned to cells just
        #   outside the inflation zone.  Higher → sharper penalty gradient.
        self.declare_parameter('use_cost_map',       True)
        self.declare_parameter('cost_weight',        2.0)
        self.declare_parameter('max_obstacle_cost',  5.0)

        # ── Goal validation parameters ────────────────────────────────────────
        # min_goal_obstacle_distance: goals closer than this to a wall are
        #   considered unsafe and reprojected to the nearest safe cell.
        # allow_goal_reprojection: if false, unsafe goals are simply rejected.
        # goal_reprojection_radius: search radius for reprojection [metres].
        self.declare_parameter('min_goal_obstacle_distance', 0.22)   # metres
        self.declare_parameter('allow_goal_reprojection',    True)
        self.declare_parameter('goal_reprojection_radius',   0.50)   # metres

        # ── Path safety check parameter ────────────────────────────────────────
        # min_path_obstacle_distance: warn if any waypoint is closer to a wall.
        self.declare_parameter('min_path_obstacle_distance', 0.15)   # metres

        self.inflation_radius_m      = self.get_parameter('inflation_radius').value
        self.occ_thresh              = self.get_parameter('occupied_threshold').value
        self.map_frame               = self.get_parameter('map_frame').value
        self.base_frame              = self.get_parameter('base_frame').value
        self.use_tf_pose             = self.get_parameter('use_tf_pose').value
        self.replan_on_new_map       = self.get_parameter('replan_on_new_map').value
        self.wall_closing_radius     = int(self.get_parameter('wall_closing_radius').value)
        self.unknown_as_occupied     = self.get_parameter('unknown_as_occupied').value
        self.use_cost_map            = self.get_parameter('use_cost_map').value
        self.cost_weight             = self.get_parameter('cost_weight').value
        self.max_obstacle_cost       = self.get_parameter('max_obstacle_cost').value
        self.min_goal_dist           = self.get_parameter('min_goal_obstacle_distance').value
        self.allow_reprojection      = self.get_parameter('allow_goal_reprojection').value
        self.reproject_radius        = self.get_parameter('goal_reprojection_radius').value
        self.min_path_dist           = self.get_parameter('min_path_obstacle_distance').value

        # ── State ─────────────────────────────────────────────────────────────
        self.map_      = None
        self.robot_x_  = 0.0
        self.robot_y_  = 0.0
        self.goal_msg_ = None

        # ── TF2 ───────────────────────────────────────────────────────────────
        if _HAS_TF2 and self.use_tf_pose:
            self._tf_buffer   = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
        else:
            self._tf_buffer = None

        # ── QoS ───────────────────────────────────────────────────────────────
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ── Pub / Sub ─────────────────────────────────────────────────────────
        self.sub_map_  = self.create_subscription(OccupancyGrid, '/map', self.map_cb, map_qos)
        self.sub_pose_ = self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self.pose_cb, 10)
        self.goal_reached = self.create_subscription(Bool, '/goal_reached', self.goal_reached_cb,1)
        self.sub_goal_ = self.create_subscription(PoseStamped, '/goal_pose', self.goal_cb, 10)
        self.pub_path_ = self.create_publisher(Path, '/planned_path', 1)

        self.get_logger().info(
            f'path_planner_node ready — '
            f'inflation={self.inflation_radius_m} m  '
            f'wall_closing={self.wall_closing_radius} cells  '
            f'cost_weight={self.cost_weight}  '
            f'max_cost={self.max_obstacle_cost}  '
            f'min_goal_dist={self.min_goal_dist} m  '
            f'unknown_as_occupied={self.unknown_as_occupied}')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def map_cb(self, msg: OccupancyGrid):
        self.map_ = msg
        if self.replan_on_new_map and self.goal_msg_ is not None:
            self._update_robot_pose_from_tf()
            self.get_logger().info(f'Having a path')

            self._plan(self.goal_msg_)
        
    def goal_reached_cb(self, msg: Bool):
        if msg.data:
            self.goal_msg_ = None
            self.get_logger().info(f'Clearing last path')

    def pose_cb(self, msg: PoseWithCovarianceStamped):
        self.robot_x_ = msg.pose.pose.position.x
        self.robot_y_ = msg.pose.pose.position.y

    def goal_cb(self, msg: PoseStamped):
        self.goal_msg_ = msg
        if self.map_ is None:
            self.get_logger().warn('No map received yet — ignoring goal')
            return
        self._update_robot_pose_from_tf()
        self._plan(msg)

    # ── TF pose lookup ────────────────────────────────────────────────────────

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
            pass

    # ── Planning ──────────────────────────────────────────────────────────────

    def _plan(self, goal_msg: PoseStamped):
        m   = self.map_
        res = m.info.resolution
        ox  = m.info.origin.position.x
        oy  = m.info.origin.position.y
        W   = m.info.width
        H   = m.info.height

        # ── Build binary occupancy grid ───────────────────────────────────────
        # 1 = obstacle/unknown, 0 = free
        raw    = np.array(m.data, dtype=np.int8).reshape(H, W)
        binary = np.zeros((H, W), dtype=np.int8)
        binary[raw > self.occ_thresh] = 1
        if self.unknown_as_occupied:
            binary[raw < 0] = 1   # -1 = unknown → blocked

        # ── Morphological closing — fill wall gaps before inflating ───────────
        # The SLAM map often has holes in internal structures (partial walls,
        # noisy edges).  Closing with a small radius seals those gaps so the
        # planner cannot route "through" a wall that has a missing pixel.
        # This runs on the raw binary grid (before inflation) so it only
        # fills genuine small holes, not large free-space areas.
        if self.wall_closing_radius > 0:
            cr = self.wall_closing_radius
            struct_close = np.ones((2 * cr + 1, 2 * cr + 1), dtype=bool)
            binary = binary_closing(binary.astype(bool),
                                    structure=struct_close).astype(np.int8)

        # ── Inflate obstacles ─────────────────────────────────────────────────
        radius_cells = max(1, int(math.ceil(self.inflation_radius_m / res)))
        inflated = _inflate_grid(binary, radius_cells)

        # ── Build distance costmap ────────────────────────────────────────────
        if self.use_cost_map:
            cost_map = _build_cost_map(inflated, res, self.cost_weight,
                                       self.max_obstacle_cost)
        else:
            cost_map = np.zeros((H, W), dtype=np.float32)

        # ── Distance map (metres) for goal/path validation ────────────────────
        # dist_map[r,c] = distance in metres from cell (r,c) to nearest obstacle
        dist_map = distance_transform_edt(~binary.astype(bool)) * res

        # ── Coordinate helpers ────────────────────────────────────────────────
        def world_to_cell(wx, wy):
            c = int((wx - ox) / res)
            r = int((wy - oy) / res)
            return (r, c)

        def clamp_cell(cell):
            return (max(0, min(H - 1, cell[0])),
                    max(0, min(W - 1, cell[1])))

        def cell_to_world(r, c):
            return ox + (c + 0.5) * res, oy + (r + 0.5) * res

        # ── Start cell ────────────────────────────────────────────────────────
        start = clamp_cell(world_to_cell(self.robot_x_, self.robot_y_))
        if inflated[start]:
            # Intentar reducir inflación primero
            radius_cells = max(0, radius_cells // 2)
            inflated = _inflate_grid(binary, radius_cells)
            cost_map = (_build_cost_map(inflated, res, self.cost_weight,
                                        self.max_obstacle_cost)
                        if self.use_cost_map else cost_map)
            if inflated[start]:
                # Con inflación reducida sigue bloqueado — buscar celda libre más cercana
                # BFS en espiral desde la posición del robot
                found = None
                for search_r in range(1, 12):
                    for dr in range(-search_r, search_r + 1):
                        for dc in range(-search_r, search_r + 1):
                            if abs(dr) != search_r and abs(dc) != search_r:
                                continue
                            nr = start[0] + dr
                            nc = start[1] + dc
                            if 0 <= nr < H and 0 <= nc < W and not inflated[nr, nc]:
                                found = (nr, nc)
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    self.get_logger().warn(
                        f'Start {start} bloqueado — replanificando desde celda libre más cercana {found}')
                    start = found
                else:
                    self.get_logger().warn('No se encontró celda libre cercana al robot — plan abortado')
                    return
            else:
                self.get_logger().warn(
                    f'Start cell {start} is inside inflated obstacle — '
                    'shrinking inflation by 50% for this plan.')

        # ── Goal validation and reprojection ──────────────────────────────────
        goal_wx = goal_msg.pose.position.x
        goal_wy = goal_msg.pose.position.y
        goal_cell = clamp_cell(world_to_cell(goal_wx, goal_wy))

        goal_cell, goal_wx, goal_wy = self._validate_goal(
            goal_cell, goal_wx, goal_wy,
            inflated, dist_map, H, W, res,
            cell_to_world, world_to_cell, clamp_cell)

        if goal_cell is None:
            return   # goal rejected, already logged

        # ── A* with costmap ───────────────────────────────────────────────────
        cell_path = _astar(inflated, cost_map, start, goal_cell, self.cost_weight)

        if cell_path is None:
            self.get_logger().warn(
                f'A* found no path from {start} to {goal_cell}. '
                'Check that goal is reachable and not inside an obstacle.')
            return

        # ── Path safety check ─────────────────────────────────────────────────
        path_min_dist = float(min(dist_map[r, c] for r, c in cell_path))
        if path_min_dist < self.min_path_dist:
            self.get_logger().warn(
                f'Path safety warning: closest waypoint is {path_min_dist:.3f} m '
                f'from a wall (min allowed: {self.min_path_dist:.3f} m). '
                'Consider increasing inflation_radius or cost_weight.')

        # ── Convert cell path to nav_msgs/Path ────────────────────────────────
        now  = self.get_clock().now().to_msg()
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
            f'→ goal=({goal_wx:.2f},{goal_wy:.2f}) | '
            f'inflation={radius_cells} cells ({radius_cells * res:.2f} m) | '
            f'path_min_wall_dist={path_min_dist:.3f} m')

    # ── Goal validation and reprojection ──────────────────────────────────────

    def _validate_goal(self, goal_cell, goal_wx, goal_wy,
                       inflated, dist_map, H, W, res,
                       cell_to_world, world_to_cell, clamp_cell):
        """
        Check that the goal cell is free and far enough from walls.
        If not, try to reproject to the nearest safe free cell.
        Returns (cell, wx, wy) or (None, _, _) if goal must be rejected.
        """
        r, c = goal_cell
        in_bounds = (0 <= r < H) and (0 <= c < W)
        is_blocked  = (not in_bounds) or bool(inflated[r, c])
        is_too_close = in_bounds and (dist_map[r, c] < self.min_goal_dist)

        if not is_blocked and not is_too_close:
            return goal_cell, goal_wx, goal_wy   # goal is safe, use as-is

        reason = ('inside inflated obstacle' if is_blocked
                  else f'too close to wall ({dist_map[r, c]:.3f} m < {self.min_goal_dist:.3f} m)')

        if not self.allow_reprojection:
            self.get_logger().error(
                f'Goal ({goal_wx:.2f}, {goal_wy:.2f}) rejected — {reason}. '
                'Set allow_goal_reprojection:=true to enable auto-correction.')
            return None, goal_wx, goal_wy

        # Search for the nearest free cell within reproject_radius that is
        # also far enough from walls.
        search_cells = int(math.ceil(self.reproject_radius / res))
        best_cell  = None
        best_dist  = float('inf')   # distance to original goal (prefer closest)

        for dr in range(-search_cells, search_cells + 1):
            for dc in range(-search_cells, search_cells + 1):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if inflated[nr, nc]:
                    continue
                if dist_map[nr, nc] < self.min_goal_dist:
                    continue
                d = math.hypot(dr, dc)
                if d > search_cells:
                    continue
                if d < best_dist:
                    best_dist = d
                    best_cell = (nr, nc)

        if best_cell is None:
            self.get_logger().error(
                f'Goal ({goal_wx:.2f}, {goal_wy:.2f}) rejected — {reason}. '
                f'No safe cell found within {self.reproject_radius:.2f} m.')
            return None, goal_wx, goal_wy

        new_wx, new_wy = cell_to_world(best_cell[0], best_cell[1])
        self.get_logger().warn(
            f'Goal reproyectado: ({goal_wx:.3f},{goal_wy:.3f}) → '
            f'({new_wx:.3f},{new_wy:.3f}) '
            f'[motivo: {reason}, desplazamiento: {best_dist * res:.3f} m]')
        return best_cell, new_wx, new_wy


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
