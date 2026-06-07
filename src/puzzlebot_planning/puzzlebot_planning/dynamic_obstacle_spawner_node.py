"""
dynamic_obstacle_spawner_node.py — Spawner de obstáculos dinámicos.

Modos:
  on_path        — waypoint exacto de la ruta verde, a 30-70% del recorrido total
  near_path      — punto a ±0.3-0.6 m lateral de la ruta
  random_free    — celda libre aleatoria dentro del mapa
  fixed_sequence — posiciones fijas desde parámetros

desktop_mode=true: publica en /test_obstacle_pose (DOM lo inyecta en /augmented_map)
desktop_mode=false: usa 'ign service' para crear modelo físico en Gazebo Fortress
"""

import math
import random
import subprocess
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

try:
    from tf2_ros import Buffer, TransformListener
    _HAS_TF2 = True
except ImportError:
    _HAS_TF2 = False


class DynamicObstacleSpawner(Node):

    def __init__(self):
        super().__init__('dynamic_obstacle_spawner_node')

        self.declare_parameter('enabled',                    True)
        self.declare_parameter('spawn_interval_sec',         45.0)
        self.declare_parameter('spawn_delay_after_path_sec', 3.0)   # delay tras recibir ruta
        self.declare_parameter('obstacle_ttl_sec',           60.0)
        self.declare_parameter('max_active_obstacles',       2)
        self.declare_parameter('spawn_mode',                 'on_path')

        self.declare_parameter('obstacle_shape',             'box')
        self.declare_parameter('obstacle_length_m',          0.25)
        self.declare_parameter('obstacle_width_m',           0.25)
        self.declare_parameter('obstacle_height_m',          0.50)

        # on_path: el obstáculo cae en el rango [path_fraction_min, path_fraction_max]
        # de la longitud total de la ruta (0.0=robot, 1.0=goal).
        # 0.35-0.65 = zona media de la ruta, lejos del robot y del goal.
        self.declare_parameter('path_fraction_min',          0.35)
        self.declare_parameter('path_fraction_max',          0.65)

        self.declare_parameter('min_distance_from_robot_m',  0.60)
        self.declare_parameter('min_distance_from_goal_m',   0.40)
        self.declare_parameter('min_distance_from_walls_m',  0.20)

        self.declare_parameter('planned_path_topic',         '/planned_path')
        self.declare_parameter('map_topic',                  '/map')
        self.declare_parameter('goal_topic',                 '/goal_pose')
        self.declare_parameter('world_name',                 'real_arena')
        self.declare_parameter('robot_frame',                'base_footprint')
        self.declare_parameter('enable_rviz_markers',        True)
        self.declare_parameter('auto_remove',                True)
        self.declare_parameter('desktop_mode',               True)

        # Posiciones fijas para fixed_sequence
        self.declare_parameter('fixed_obstacles.0.x',        1.50)
        self.declare_parameter('fixed_obstacles.0.y',        2.00)
        self.declare_parameter('fixed_obstacles.1.x',        2.50)
        self.declare_parameter('fixed_obstacles.1.y',        3.00)

        self._enabled      = self.get_parameter('enabled').value
        self._interval     = self.get_parameter('spawn_interval_sec').value
        self._path_delay   = self.get_parameter('spawn_delay_after_path_sec').value
        self._ttl          = self.get_parameter('obstacle_ttl_sec').value
        self._max_obs      = self.get_parameter('max_active_obstacles').value
        self._mode         = self.get_parameter('spawn_mode').value
        self._shape        = self.get_parameter('obstacle_shape').value
        self._length       = self.get_parameter('obstacle_length_m').value
        self._width        = self.get_parameter('obstacle_width_m').value
        self._height       = self.get_parameter('obstacle_height_m').value
        self._frac_min     = self.get_parameter('path_fraction_min').value
        self._frac_max     = self.get_parameter('path_fraction_max').value
        self._min_robot    = self.get_parameter('min_distance_from_robot_m').value
        self._min_goal     = self.get_parameter('min_distance_from_goal_m').value
        self._min_wall     = self.get_parameter('min_distance_from_walls_m').value
        self._world        = self.get_parameter('world_name').value
        self._markers_en   = self.get_parameter('enable_rviz_markers').value
        self._auto_rm      = self.get_parameter('auto_remove').value
        self._desktop      = self.get_parameter('desktop_mode').value

        # Estado
        self._robot_x      = 0.0
        self._robot_y      = 0.0
        self._have_pose    = False
        self._goal_x       = None
        self._goal_y       = None
        self._current_path: Path = None
        self._base_map: OccupancyGrid = None
        self._active_obstacles: list = []  # [(name, spawn_t, x, y)]
        self._obstacle_counter  = 0
        self._fixed_seq_idx     = 0
        self._last_spawn_t      = 0.0     # 0 = nunca spawneado
        self._path_received_t   = 0.0     # cuando llegó la última ruta válida
        self._spawned_this_path = False   # evita spawnear 2 veces para la misma ruta

        # QoS latched para /map
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        if _HAS_TF2:
            self._tf_buffer   = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
        else:
            self._tf_buffer = None

        path_topic = self.get_parameter('planned_path_topic').value
        map_topic  = self.get_parameter('map_topic').value
        goal_topic = self.get_parameter('goal_topic').value

        self.create_subscription(Path,          path_topic, self._path_cb, 10)
        self.create_subscription(OccupancyGrid, map_topic,  self._map_cb,  latched)
        self.create_subscription(PoseStamped,   goal_topic, self._goal_cb, latched)

        self._pub_markers  = self.create_publisher(MarkerArray, '/spawner/markers', 10)
        self._pub_state    = self.create_publisher(String,      '/spawner/state',   10)
        self._pub_test_obs = self.create_publisher(PoseStamped, '/test_obstacle_pose', 10)

        self.create_timer(0.5, self._loop)

        if not self._enabled:
            self.get_logger().info('[SPAWNER] disabled')
        else:
            self.get_logger().info(
                f'[SPAWNER] OK  mode={self._mode}  interval={self._interval}s  '
                f'path_delay={self._path_delay}s  desktop={self._desktop}  '
                f'fraction=[{self._frac_min:.0%},{self._frac_max:.0%}]')

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _path_cb(self, msg: Path):
        if len(msg.poses) > 0:
            # Ruta nueva → resetear flag para permitir un spawn nuevo
            if self._current_path is None or len(self._current_path.poses) == 0:
                self._spawned_this_path = False
            self._current_path    = msg
            self._path_received_t = time.monotonic()
        else:
            self._current_path = None

    def _map_cb(self, msg: OccupancyGrid):
        if self._base_map is None:
            self._base_map = msg
            self.get_logger().info(f'[SPAWNER] mapa {msg.info.width}x{msg.info.height}px recibido')

    def _goal_cb(self, msg: PoseStamped):
        self._goal_x = msg.pose.position.x
        self._goal_y = msg.pose.position.y
        # Nuevo goal → permitir spawn nuevo
        self._spawned_this_path = False

    # ── Loop principal ─────────────────────────────────────────────────────────

    def _loop(self):
        if not self._enabled:
            return

        now = time.monotonic()
        self._update_robot_pose()
        self._remove_expired(now)

        # Estado en tópico
        has_path = self._current_path is not None
        next_in  = max(0.0, self._interval - (now - self._last_spawn_t)) if self._last_spawn_t else 0.0
        delay_left = max(0.0, self._path_delay - (now - self._path_received_t)) if has_path else -1.0
        s = String()
        s.data = (f'mode={self._mode}  active={len(self._active_obstacles)}'
                  f'  path={"SI" if has_path else "NO"}'
                  f'  delay_left={delay_left:.0f}s'
                  f'  next_in={next_in:.0f}s')
        self._pub_state.publish(s)

        # Condiciones para spawnear:
        # 1. Tiene que haber una ruta válida
        if not has_path:
            return
        # 2. El robot tiene que tener pose
        if not self._have_pose:
            return
        # 3. Esperar el delay inicial tras recibir la ruta
        if (now - self._path_received_t) < self._path_delay:
            return
        # 4. Intervalo entre spawns
        if self._last_spawn_t > 0 and (now - self._last_spawn_t) < self._interval:
            return
        # 5. No superar el máximo
        if len(self._active_obstacles) >= self._max_obs:
            return

        pos = self._choose_spawn_position()
        if pos is None:
            self.get_logger().warn('[SPAWNER] no se encontró posición válida')
            return

        self._obstacle_counter += 1
        name = f'dyn_obs_{self._obstacle_counter}'
        self._do_spawn(name, pos[0], pos[1], now)
        self._last_spawn_t = now

    # ── Selección de posición ──────────────────────────────────────────────────

    def _choose_spawn_position(self):
        if self._mode == 'fixed_sequence':
            return self._fixed_sequence_pos()
        elif self._mode == 'on_path':
            return self._on_path_pos()
        elif self._mode == 'near_path':
            return self._near_path_pos()
        else:
            return self._random_free_pos()

    def _on_path_pos(self):
        """
        Elige un punto exactamente SOBRE la ruta verde, en la zona media
        [path_fraction_min, path_fraction_max] de la longitud total.
        Garantiza que el obstáculo cae sobre la ruta que está siguiendo el robot.
        """
        if self._current_path is None:
            return None

        poses = self._current_path.poses
        if len(poses) < 4:
            return self._random_free_pos()

        # Calcular longitud total de la ruta
        total_len = 0.0
        seg_lens  = []
        for i in range(len(poses) - 1):
            dx = poses[i+1].pose.position.x - poses[i].pose.position.x
            dy = poses[i+1].pose.position.y - poses[i].pose.position.y
            d  = math.hypot(dx, dy)
            seg_lens.append(d)
            total_len += d

        if total_len < 0.2:
            return None

        # Elegir una fracción aleatoria en la zona media de la ruta
        target_frac = random.uniform(self._frac_min, self._frac_max)
        target_dist = target_frac * total_len

        # Recorrer segmentos hasta alcanzar target_dist
        acc = 0.0
        for i, seg_d in enumerate(seg_lens):
            if acc + seg_d >= target_dist:
                t  = (target_dist - acc) / max(seg_d, 1e-9)
                cx = poses[i].pose.position.x + t * (poses[i+1].pose.position.x - poses[i].pose.position.x)
                cy = poses[i].pose.position.y + t * (poses[i+1].pose.position.y - poses[i].pose.position.y)
                if self._position_valid(cx, cy):
                    return (cx, cy)
                # Si ese punto exacto está inválido, probar fracciones alternativas
                break
            acc += seg_d

        # Fallback: probar varias fracciones en la zona media
        for _ in range(10):
            frac = random.uniform(self._frac_min, self._frac_max)
            dist = frac * total_len
            acc  = 0.0
            for i, seg_d in enumerate(seg_lens):
                if acc + seg_d >= dist:
                    t  = (dist - acc) / max(seg_d, 1e-9)
                    cx = poses[i].pose.position.x + t * (poses[i+1].pose.position.x - poses[i].pose.position.x)
                    cy = poses[i].pose.position.y + t * (poses[i+1].pose.position.y - poses[i].pose.position.y)
                    if self._position_valid(cx, cy):
                        return (cx, cy)
                    break
                acc += seg_d

        return None

    def _near_path_pos(self):
        """Punto a ±0.3-0.6 m lateral de la zona media de la ruta."""
        if self._current_path is None:
            return self._random_free_pos()

        poses = self._current_path.poses
        if len(poses) < 4:
            return self._random_free_pos()

        # Elegir índice en la zona media de la ruta (40-60%)
        start_i = int(len(poses) * 0.35)
        end_i   = int(len(poses) * 0.65)
        if start_i >= end_i:
            return self._random_free_pos()

        for _ in range(15):
            idx = random.randint(start_i, min(end_i, len(poses) - 2))
            px  = poses[idx].pose.position.x
            py  = poses[idx].pose.position.y
            nx  = poses[idx + 1].pose.position.x
            ny  = poses[idx + 1].pose.position.y
            seg = math.hypot(nx - px, ny - py)
            if seg < 1e-6:
                continue
            # Vector perpendicular al segmento
            perp_x = -(ny - py) / seg
            perp_y =  (nx - px) / seg
            side   = random.choice([-1.0, 1.0])
            offset = random.uniform(0.30, 0.55)
            cx = (px + nx) / 2 + side * offset * perp_x
            cy = (py + ny) / 2 + side * offset * perp_y
            if self._position_valid(cx, cy):
                return (cx, cy)
        return None

    def _random_free_pos(self):
        if self._base_map is None:
            return None
        info = self._base_map.info
        res  = info.resolution
        ox   = info.origin.position.x
        oy   = info.origin.position.y
        W, H = info.width, info.height
        data = np.array(self._base_map.data, dtype=np.int8).reshape(H, W)
        for _ in range(60):
            r = random.randint(0, H - 1)
            c = random.randint(0, W - 1)
            if data[r, c] > 50 or data[r, c] < 0:
                continue
            wx = ox + (c + 0.5) * res
            wy = oy + (r + 0.5) * res
            if self._position_valid(wx, wy):
                return (wx, wy)
        return None

    def _fixed_sequence_pos(self):
        idx = self._fixed_seq_idx % 2
        try:
            x = self.get_parameter(f'fixed_obstacles.{idx}.x').value
            y = self.get_parameter(f'fixed_obstacles.{idx}.y').value
            self._fixed_seq_idx += 1
            if self._position_valid(x, y):
                return (x, y)
        except Exception:
            pass
        return self._on_path_pos()

    # ── Validación de posición ─────────────────────────────────────────────────

    def _position_valid(self, x: float, y: float) -> bool:
        # Distancia mínima al robot
        if self._have_pose:
            if math.hypot(x - self._robot_x, y - self._robot_y) < self._min_robot:
                return False
        # Distancia mínima al goal
        if self._goal_x is not None:
            if math.hypot(x - self._goal_x, y - self._goal_y) < self._min_goal:
                return False
        # Distancia mínima a paredes
        if self._base_map is not None:
            info = self._base_map.info
            res  = info.resolution
            ox   = info.origin.position.x
            oy   = info.origin.position.y
            W, H = info.width, info.height
            data = np.array(self._base_map.data, dtype=np.int8).reshape(H, W)
            check_r = max(1, int(math.ceil(self._min_wall / res)))
            cc = int((x - ox) / res)
            rc = int((y - oy) / res)
            if not (check_r <= cc < W - check_r and check_r <= rc < H - check_r):
                return False
            region = data[rc - check_r:rc + check_r + 1,
                          cc - check_r:cc + check_r + 1]
            if np.any(region > 50):
                return False
        # No encima de otro obstáculo activo
        for _, _, ax, ay in self._active_obstacles:
            if math.hypot(x - ax, y - ay) < self._length + 0.15:
                return False
        return True

    # ── Spawn / remove ─────────────────────────────────────────────────────────

    def _do_spawn(self, name: str, x: float, y: float, now: float):
        self.get_logger().warn(
            f'[SPAWNER] >>> SPAWNING {name} en ({x:.2f},{y:.2f})  mode={self._mode}')

        if self._desktop:
            # Modo desktop: notificar al DOM via /test_obstacle_pose
            msg = PoseStamped()
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = 'map'
            msg.pose.position.x = x
            msg.pose.position.y = y
            msg.pose.orientation.w = 1.0
            self._pub_test_obs.publish(msg)
            self.get_logger().info(f'[SPAWNER] desktop → /test_obstacle_pose ({x:.2f},{y:.2f})')
        else:
            sdf = (self._make_cylinder_sdf(name, x, y)
                   if self._shape == 'cylinder'
                   else self._make_box_sdf(name, x, y))
            ok  = self._ign_spawn(name, sdf)
            if not ok:
                # Fallback: publicar al DOM de todas formas
                msg = PoseStamped()
                msg.header.stamp    = self.get_clock().now().to_msg()
                msg.header.frame_id = 'map'
                msg.pose.position.x = x
                msg.pose.position.y = y
                msg.pose.orientation.w = 1.0
                self._pub_test_obs.publish(msg)

        self._active_obstacles.append((name, now, x, y))
        if self._markers_en:
            self._publish_marker(name, x, y)

    def _remove_expired(self, now: float):
        if not self._auto_rm:
            return
        remaining = []
        for entry in self._active_obstacles:
            name, spawn_t, ax, ay = entry
            if (now - spawn_t) > self._ttl:
                self.get_logger().info(f'[SPAWNER] TTL expirado → eliminando {name}')
                if not self._desktop:
                    self._ign_remove(name)
            else:
                remaining.append(entry)
        self._active_obstacles = remaining

    # ── Gazebo Fortress ────────────────────────────────────────────────────────

    def _make_box_sdf(self, name: str, x: float, y: float) -> str:
        return (f'<?xml version="1.0"?><sdf version="1.9">'
                f'<model name="{name}"><static>true</static>'
                f'<pose>{x} {y} {self._height/2} 0 0 0</pose>'
                f'<link name="link">'
                f'<collision name="col"><geometry><box>'
                f'<size>{self._length} {self._width} {self._height}</size>'
                f'</box></geometry></collision>'
                f'<visual name="vis"><geometry><box>'
                f'<size>{self._length} {self._width} {self._height}</size>'
                f'</box></geometry>'
                f'<material><ambient>0.9 0.3 0.1 1</ambient></material>'
                f'</visual></link></model></sdf>')

    def _make_cylinder_sdf(self, name: str, x: float, y: float) -> str:
        r = max(self._length, self._width) / 2.0
        return (f'<?xml version="1.0"?><sdf version="1.9">'
                f'<model name="{name}"><static>true</static>'
                f'<pose>{x} {y} {self._height/2} 0 0 0</pose>'
                f'<link name="link">'
                f'<collision name="col"><geometry><cylinder>'
                f'<radius>{r}</radius><length>{self._height}</length>'
                f'</cylinder></geometry></collision>'
                f'<visual name="vis"><geometry><cylinder>'
                f'<radius>{r}</radius><length>{self._height}</length>'
                f'</cylinder></geometry>'
                f'<material><ambient>0.9 0.3 0.1 1</ambient></material>'
                f'</visual></link></model></sdf>')

    def _ign_spawn(self, name: str, sdf: str) -> bool:
        sdf_esc = sdf.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
        cmd = ['ign', 'service', '-s', f'/world/{self._world}/create',
               '--reqtype', 'ignition.msgs.EntityFactory',
               '--reptype', 'ignition.msgs.Boolean',
               '--timeout', '3000',
               '--req', f'sdf: "{sdf_esc}", name: "{name}"']
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5.0)
            return r.returncode == 0
        except Exception:
            return False

    def _ign_remove(self, name: str):
        cmd = ['ign', 'service', '-s', f'/world/{self._world}/remove',
               '--reqtype', 'ignition.msgs.Entity',
               '--reptype', 'ignition.msgs.Boolean',
               '--timeout', '3000', '--req', f'name: "{name}"']
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=5.0)
        except Exception:
            pass

    # ── TF / pose ──────────────────────────────────────────────────────────────

    def _update_robot_pose(self):
        if self._tf_buffer is None:
            return
        robot_frame = self.get_parameter('robot_frame').value
        try:
            tf = self._tf_buffer.lookup_transform(
                'map', robot_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05))
            self._robot_x   = tf.transform.translation.x
            self._robot_y   = tf.transform.translation.y
            self._have_pose = True
        except Exception:
            pass

    # ── Marcadores RViz ────────────────────────────────────────────────────────

    def _publish_marker(self, name: str, x: float, y: float):
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()

        # Cubo rojo = obstáculo
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp    = now
        m.ns     = 'spawner'
        m.id     = self._obstacle_counter
        m.type   = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x    = x
        m.pose.position.y    = y
        m.pose.position.z    = self._height / 2.0
        m.pose.orientation.w = 1.0
        m.scale.x = self._length
        m.scale.y = self._width
        m.scale.z = self._height
        m.color.r = 1.0
        m.color.g = 0.15
        m.color.b = 0.0
        m.color.a = 0.9
        arr.markers.append(m)

        # Etiqueta
        t = Marker()
        t.header.frame_id = 'map'
        t.header.stamp    = now
        t.ns   = 'spawner_label'
        t.id   = self._obstacle_counter
        t.type = Marker.TEXT_VIEW_FACING
        t.action = Marker.ADD
        t.pose.position.x    = x
        t.pose.position.y    = y
        t.pose.position.z    = self._height + 0.12
        t.pose.orientation.w = 1.0
        t.scale.z = 0.12
        t.text    = name
        t.color.r = t.color.g = t.color.b = 1.0
        t.color.a = 1.0
        arr.markers.append(t)

        # Punto sobre la ruta (estrella amarilla)
        star = Marker()
        star.header.frame_id = 'map'
        star.header.stamp    = now
        star.ns   = 'spawner_point'
        star.id   = self._obstacle_counter
        star.type = Marker.SPHERE
        star.action = Marker.ADD
        star.pose.position.x    = x
        star.pose.position.y    = y
        star.pose.position.z    = 0.02
        star.pose.orientation.w = 1.0
        star.scale.x = star.scale.y = star.scale.z = 0.14
        star.color.r = 1.0
        star.color.g = 1.0
        star.color.b = 0.0
        star.color.a = 1.0
        arr.markers.append(star)

        self._pub_markers.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = DynamicObstacleSpawner()
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
