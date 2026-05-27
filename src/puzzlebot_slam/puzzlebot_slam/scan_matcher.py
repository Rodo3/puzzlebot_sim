"""Scan-to-map matcher con búsqueda de rotación + traslación.

Estrategia en dos fases desacopladas (más rápido que búsqueda 3D conjunta):

  Fase 1 — Rotación:
    Coarse: ±8° en pasos de 2°
    Fine:   ±1.5° en pasos de 0.5°
    Corrige el drift angular, que es la fuente dominante de error.

  Fase 2 — Traslación (nueva):
    Con el yaw ganador de la fase 1, busca la mejor traslación en
    una ventana ±TRANS_HALF_M en pasos de TRANS_STEP_M.
    Coste extra: (2*TRANS_HALF/TRANS_STEP + 1)² evaluaciones adicionales.
    Con los valores por defecto (±10 cm, paso 5 cm): 5×5 = 25 poses.

Total: ~22 (rot) + 25 (tras) ≈ 47 evaluaciones por scan — manejable en Python.

La búsqueda de traslación solo activa cuando el mapa tiene suficiente
contenido (WARMUP_SCANS) y el score de la fase 1 supera un umbral mínimo,
para evitar correcciones en zonas sin paredes cercanas.
"""

import numpy as np

from sensor_msgs.msg import LaserScan

from .occupancy_grid_map import OccupancyGridMap
from .slam_types import Pose2D

# ── Warmup ───────────────────────────────────────────────────────────────
_WARMUP_SCANS = 12

# ── Fase 1: búsqueda angular ─────────────────────────────────────────────
_COARSE_HALF_RAD = 0.140    # 8°
_COARSE_STEP_RAD = 0.0349   # 2°

_FINE_HALF_RAD   = 0.0262   # 1.5°
_FINE_STEP_RAD   = 0.00873  # 0.5°

# ── Fase 2: búsqueda de traslación ───────────────────────────────────────
_TRANS_HALF_M  = 0.05   # ±1 celda: evita que el matcher arrastre el mapa
_TRANS_STEP_M  = 0.05

# Score mínimo de la fase 1 para activar la búsqueda de traslación.
# Umbral bajo: activa antes cuando el mapa tiene poco contenido en la zona.
_MIN_SCORE_FOR_TRANS = 4.0

# ── Decimado de rayos ─────────────────────────────────────────────────────
_RAY_STRIDE = 3


class LocalScanMatcher:
    def __init__(self, enabled: bool = False):
        self._enabled    = enabled
        self._scan_count = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def match(
        self,
        scan: LaserScan,
        initial_pose: Pose2D,
        grid_map: OccupancyGridMap,
    ) -> Pose2D:
        if not self._enabled:
            return initial_pose

        self._scan_count += 1
        if self._scan_count <= _WARMUP_SCANS:
            return initial_pose

        return self._search(scan, initial_pose, grid_map)

    # ------------------------------------------------------------------ #

    def _search(
        self,
        scan: LaserScan,
        initial_pose: Pose2D,
        grid_map: OccupancyGridMap,
    ) -> Pose2D:
        ranges, rel_angles = self._valid_rays(scan, grid_map)
        if len(ranges) == 0:
            return initial_pose

        # ── Fase 1: búsqueda angular ─────────────────────────────────────
        best_pose  = initial_pose
        best_score = -1.0

        for dyaw in np.arange(-_COARSE_HALF_RAD,
                               _COARSE_HALF_RAD + 1e-9,
                               _COARSE_STEP_RAD):
            c = Pose2D(initial_pose.x, initial_pose.y, initial_pose.yaw + dyaw)
            s = self._score(ranges, rel_angles, c, grid_map)
            if s > best_score:
                best_score, best_pose = s, c

        coarse_yaw = best_pose.yaw
        for yaw in np.arange(coarse_yaw - _FINE_HALF_RAD,
                              coarse_yaw + _FINE_HALF_RAD + 1e-9,
                              _FINE_STEP_RAD):
            c = Pose2D(initial_pose.x, initial_pose.y, yaw)
            s = self._score(ranges, rel_angles, c, grid_map)
            if s > best_score:
                best_score, best_pose = s, c

        rot_pose  = best_pose    # yaw corregido, traslación = odometry
        rot_score = best_score

        # ── Fase 2: búsqueda de traslación ───────────────────────────────
        # Solo si el mapa tiene contenido suficiente en esta zona
        if rot_score < _MIN_SCORE_FOR_TRANS:
            return rot_pose

        offsets = np.arange(-_TRANS_HALF_M,
                             _TRANS_HALF_M + 1e-9,
                             _TRANS_STEP_M)

        trans_pose  = rot_pose
        trans_score = rot_score

        for dx in offsets:
            for dy in offsets:
                c = Pose2D(rot_pose.x + dx, rot_pose.y + dy, rot_pose.yaw)
                s = self._score(ranges, rel_angles, c, grid_map)
                if s > trans_score:
                    trans_score, trans_pose = s, c

        return trans_pose

    # ------------------------------------------------------------------ #

    @staticmethod
    def _valid_rays(scan: LaserScan, grid_map: OccupancyGridMap):
        ranges = np.array(scan.ranges, dtype=np.float32)
        n      = len(ranges)
        angles = (scan.angle_min
                  + np.arange(n, dtype=np.float32) * scan.angle_increment)

        rmin  = max(float(scan.range_min), grid_map.min_useful_range)
        rmax  = float(scan.range_max)
        if grid_map.max_mapping_range > 0.0:
            rmax = min(rmax, grid_map.max_mapping_range)
        rmax *= grid_map.max_range_factor

        valid = np.isfinite(ranges) & (ranges > rmin) & (ranges < rmax)
        idx   = np.where(valid)[0][::_RAY_STRIDE]

        return ranges[idx], angles[idx]

    @staticmethod
    def _score(
        ranges: np.ndarray,
        rel_angles: np.ndarray,
        pose: Pose2D,
        grid_map: OccupancyGridMap,
    ) -> float:
        c = np.cos(pose.yaw)
        s = np.sin(pose.yaw)
        sensor_x = pose.x + grid_map.lidar_x * c - grid_map.lidar_y * s
        sensor_y = pose.y + grid_map.lidar_x * s + grid_map.lidar_y * c

        world_angles = rel_angles + pose.yaw + grid_map.lidar_yaw
        wx = sensor_x + ranges * np.cos(world_angles)
        wy = sensor_y + ranges * np.sin(world_angles)

        res  = grid_map.resolution
        col  = ((wx - grid_map.origin_x) / res).astype(np.int32)
        row  = ((wy - grid_map.origin_y) / res).astype(np.int32)
        width = grid_map.width_pixels
        height = grid_map.height_pixels

        mask = (col >= 0) & (col < width) & (row >= 0) & (row < height)
        if not np.any(mask):
            return 0.0

        return float(np.sum(np.maximum(0.0, grid_map.grid[row[mask], col[mask]])))
