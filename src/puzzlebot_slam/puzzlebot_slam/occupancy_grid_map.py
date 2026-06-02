"""
occupancy_grid_map.py — Mapa de ocupación con log-odds e integración de rayos LiDAR.

FUNCIÓN:
  Mantiene un grid 2D donde cada celda almacena la probabilidad acumulada
  de que esté ocupada, representada en escala log-odds para facilitar la
  actualización bayesiana con sumas en lugar de multiplicaciones.

  Representación log-odds:
    l = log(p / (1-p))   → libre: l < 0,  ocupado: l > 0,  desconocido: l = 0

  Por cada scan integrado:
    1. Calcula la pose del LiDAR en el frame map (incluye offset lidar_x/y/yaw).
    2. Para cada rayo válido traza una línea Bresenham desde el sensor al hit.
    3. Celdas a lo largo del rayo → actualización libre (+l_free, negativo).
    4. Celda final (hit) → actualización ocupada (+l_occ, positivo).
    5. Los valores se recortan en ±l_clamp para evitar celdas "permanentemente" ocupadas.

  Al publicar (/map):
    l > 0.5  → 100 (ocupado)
    l < -0.5 → 0   (libre)
    resto    → -1  (desconocido)

USADO POR:
  slam_node.py — llama a integrate_scan() por cada keyframe y to_msg() cada segundo.
  path_planner_node.py — lee /map para construir el grid binario de A*.
"""

import math

import numpy as np
from nav_msgs.msg import MapMetaData, OccupancyGrid
from sensor_msgs.msg import LaserScan

from .slam_math import bresenham
from .slam_types import Pose2D


class OccupancyGridMap:
    def __init__(
        self,
        size_pixels: int,
        size_meters: float,
        origin_x: float,
        origin_y: float,
        p_occ: float,
        p_free: float,
        l_clamp: float,
        scan_step: int,
        max_range_factor: float,
        min_useful_range: float,
        max_mapping_range: float,
        lidar_x: float,
        lidar_y: float,
        lidar_yaw: float,
        width_pixels: int | None = None,
        height_pixels: int | None = None,
        resolution: float | None = None,
    ):
        self.size_pixels = size_pixels
        self.size_meters = size_meters
        self.resolution = (
            size_meters / size_pixels if resolution is None else resolution)
        self.width_pixels = size_pixels if width_pixels is None else width_pixels
        self.height_pixels = size_pixels if height_pixels is None else height_pixels
        self.width_meters = self.width_pixels * self.resolution
        self.height_meters = self.height_pixels * self.resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.l_occ = math.log(p_occ / (1.0 - p_occ))
        self.l_free = math.log(p_free / (1.0 - p_free))
        self.l_clamp = l_clamp
        self.scan_step = max(1, scan_step)
        self.max_range_factor = max_range_factor
        self.min_useful_range = min_useful_range
        self.max_mapping_range = max_mapping_range
        self.lidar_x = lidar_x
        self.lidar_y = lidar_y
        self.lidar_yaw = lidar_yaw
        self.grid = np.zeros((self.height_pixels, self.width_pixels), dtype=np.float32)

    def reset(self) -> None:
        """Reset the log-odds grid to uniform unknown state (all zeros)."""
        self.grid[:] = 0.0

    def world_to_cell(self, wx: float, wy: float):
        col = int(math.floor((wx - self.origin_x) / self.resolution))
        row = int(math.floor((wy - self.origin_y) / self.resolution))
        return col, row

    def in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.width_pixels and 0 <= row < self.height_pixels

    def integrate_scan(self, scan: LaserScan, pose: Pose2D) -> bool:
        c = math.cos(pose.yaw)
        s = math.sin(pose.yaw)
        sensor_x = pose.x + self.lidar_x * c - self.lidar_y * s
        sensor_y = pose.y + self.lidar_x * s + self.lidar_y * c

        r_col, r_row = self.world_to_cell(sensor_x, sensor_y)
        if not self.in_bounds(r_col, r_row):
            return False

        rmin = max(self.min_useful_range, scan.range_min)
        rmax = scan.range_max
        if self.max_mapping_range > 0.0:
            rmax = min(rmax, self.max_mapping_range)
        hit_threshold = rmax * self.max_range_factor

        for index in range(0, len(scan.ranges), self.scan_step):
            r = scan.ranges[index]
            if not math.isfinite(r) or r < rmin or r > rmax:
                continue

            is_hit = r < hit_threshold
            angle = scan.angle_min + index * scan.angle_increment + pose.yaw + self.lidar_yaw
            end_x = sensor_x + r * math.cos(angle)
            end_y = sensor_y + r * math.sin(angle)
            e_col, e_row = self.world_to_cell(end_x, end_y)

            if not self.in_bounds(e_col, e_row):
                e_col = max(0, min(self.width_pixels - 1, e_col))
                e_row = max(0, min(self.height_pixels - 1, e_row))
                is_hit = False

            self._integrate_ray(r_col, r_row, e_col, e_row, is_hit)

        return True

    def _integrate_ray(
        self,
        r_col: int,
        r_row: int,
        e_col: int,
        e_row: int,
        is_hit: bool,
    ) -> None:
        cells = list(bresenham(r_col, r_row, e_col, e_row))
        if not cells:
            return

        for col, row in cells[:-1]:
            if self.in_bounds(col, row):
                self.grid[row, col] = max(
                    -self.l_clamp,
                    self.grid[row, col] + self.l_free,
                )

        end_col, end_row = cells[-1]
        if not self.in_bounds(end_col, end_row):
            return

        if is_hit:
            self.grid[end_row, end_col] = min(
                self.l_clamp,
                self.grid[end_row, end_col] + self.l_occ,
            )
        else:
            self.grid[end_row, end_col] = max(
                -self.l_clamp,
                self.grid[end_row, end_col] + self.l_free,
            )

    def load_from_png(self, path: str) -> None:
        """Load a saved PNG into the log-odds grid (inverse of to_png).

        Expected pixel convention (same as to_png / ROS map_server):
          255 = free     → log-odds = -l_clamp
          127 = unknown  → log-odds =  0.0
            0 = occupied → log-odds = +l_clamp
        Pixels are clamped to [-l_clamp, +l_clamp] to match the SLAM saturation.
        The image is flipped vertically because PNG row-0 is top while the grid
        row-0 is bottom (OccupancyGrid convention).
        """
        from PIL import Image
        img = Image.open(path).convert('L')
        arr = np.array(img, dtype=np.float32)
        arr = np.flipud(arr)          # PNG top-left → grid bottom-left

        # Resize if the saved map has different pixel dimensions than the grid.
        if arr.shape != (self.height_pixels, self.width_pixels):
            img_resized = Image.fromarray(arr.astype(np.uint8)).resize(
                (self.width_pixels, self.height_pixels), Image.NEAREST)
            arr = np.array(img_resized, dtype=np.float32)

        # pixel 255→−l_clamp (free), 0→+l_clamp (occupied), 127→0 (unknown)
        self.grid = (127.0 - arr) / 127.0 * self.l_clamp
        self.grid = np.clip(self.grid, -self.l_clamp, self.l_clamp).astype(np.float32)

    def to_png(self, path: str) -> None:
        """Save the current occupancy grid as a grayscale PNG.

        Convention (matches ROS map_server):
          127 = unknown  (log-odds ≈ 0)
          255 = free     (log-odds < -0.5)
            0 = occupied (log-odds >  0.5)
        """
        from PIL import Image
        img_data = np.full(self.grid.shape, 127, dtype=np.uint8)
        img_data[self.grid < -0.5] = 255
        img_data[self.grid > 0.5] = 0
        # OccupancyGrid origin is bottom-left; PNG row 0 is top → flip vertically
        img = Image.fromarray(np.flipud(img_data), mode='L')
        img.save(path)

    def to_msg(self, stamp, frame_id: str) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id

        msg.info = MapMetaData()
        msg.info.resolution = self.resolution
        msg.info.width = self.width_pixels
        msg.info.height = self.height_pixels
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.orientation.w = 1.0

        flat = self.grid.flatten()
        data = np.full(flat.shape, -1, dtype=np.int8)
        data[flat > 0.5] = 100
        data[flat < -0.5] = 0
        msg.data = data.tolist()
        return msg
