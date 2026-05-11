"""Log-odds occupancy grid map and lidar ray integration."""

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
    ):
        self.size_pixels = size_pixels
        self.size_meters = size_meters
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.resolution = size_meters / size_pixels
        self.l_occ = math.log(p_occ / (1.0 - p_occ))
        self.l_free = math.log(p_free / (1.0 - p_free))
        self.l_clamp = l_clamp
        self.scan_step = max(1, scan_step)
        self.max_range_factor = max_range_factor
        self.min_useful_range = min_useful_range
        self.grid = np.zeros((size_pixels, size_pixels), dtype=np.float32)

    def world_to_cell(self, wx: float, wy: float):
        col = int(math.floor((wx - self.origin_x) / self.resolution))
        row = int(math.floor((wy - self.origin_y) / self.resolution))
        return col, row

    def in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.size_pixels and 0 <= row < self.size_pixels

    def integrate_scan(self, scan: LaserScan, pose: Pose2D) -> bool:
        r_col, r_row = self.world_to_cell(pose.x, pose.y)
        if not self.in_bounds(r_col, r_row):
            return False

        rmin = max(self.min_useful_range, scan.range_min)
        rmax = scan.range_max
        hit_threshold = rmax * self.max_range_factor

        for index in range(0, len(scan.ranges), self.scan_step):
            r = scan.ranges[index]
            if not math.isfinite(r) or r < rmin or r > rmax:
                continue

            is_hit = r < hit_threshold
            angle = scan.angle_min + index * scan.angle_increment + pose.yaw
            end_x = pose.x + r * math.cos(angle)
            end_y = pose.y + r * math.sin(angle)
            e_col, e_row = self.world_to_cell(end_x, end_y)

            if not self.in_bounds(e_col, e_row):
                e_col = max(0, min(self.size_pixels - 1, e_col))
                e_row = max(0, min(self.size_pixels - 1, e_row))
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

    def to_msg(self, stamp, frame_id: str) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id

        msg.info = MapMetaData()
        msg.info.resolution = self.resolution
        msg.info.width = self.size_pixels
        msg.info.height = self.size_pixels
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.orientation.w = 1.0

        flat = self.grid.flatten()
        data = np.full(flat.shape, -1, dtype=np.int8)
        data[flat > 0.5] = 100
        data[flat < -0.5] = 0
        msg.data = data.tolist()
        return msg
