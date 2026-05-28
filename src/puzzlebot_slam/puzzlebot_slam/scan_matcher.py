"""Scan-to-map angular matcher for the physical robot.

Strategy: coarse + fine angular search around the odometry yaw.
For each candidate yaw, project the scan endpoints into the grid and sum the
log-odds values at those cells.  The candidate with the highest score wins.

This directly corrects the rotation drift that causes the map to spiral when
the robot turns — the most significant source of error with wheel odometry.

Translation correction is intentionally omitted: translation drift is an order
of magnitude smaller than rotation drift on a differential-drive robot, and
adding a 2D search would triple the compute cost per scan.
"""

import numpy as np

from sensor_msgs.msg import LaserScan

from .occupancy_grid_map import OccupancyGridMap
from .slam_types import Pose2D

# Number of scans to integrate before matching activates.
# The first few scans build the initial map; without any map content the
# score function has nothing to compare against.
_WARMUP_SCANS = 8

# Coarse search: ±12° in 2° steps  →  13 candidates
_COARSE_HALF_RAD = 0.209   # 12°
_COARSE_STEP_RAD = 0.0349  # 2°

# Fine search: ±2° around the coarse winner in 0.5° steps  →  9 candidates
_FINE_HALF_RAD = 0.0349    # 2°
_FINE_STEP_RAD = 0.00873   # 0.5°

# Use every Nth ray for scoring (speed vs. accuracy trade-off).
# Every 3rd ray still gives ~120 points for a 360-ray scan.
_RAY_STRIDE = 3


class LocalScanMatcher:
    def __init__(self, enabled: bool = False):
        self._enabled = enabled
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
        # Pre-compute range/angle arrays once
        ranges, rel_angles = self._valid_rays(scan, grid_map)
        if len(ranges) == 0:
            return initial_pose

        # ── Coarse pass ──────────────────────────────────────────────────
        best_pose = initial_pose
        best_score = -1.0

        for dyaw in np.arange(-_COARSE_HALF_RAD, _COARSE_HALF_RAD + 1e-9, _COARSE_STEP_RAD):
            candidate = Pose2D(initial_pose.x, initial_pose.y, initial_pose.yaw + dyaw)
            score = self._score(ranges, rel_angles, candidate, grid_map)
            if score > best_score:
                best_score = score
                best_pose = candidate

        # ── Fine pass around coarse winner ───────────────────────────────
        coarse_yaw = best_pose.yaw
        for yaw in np.arange(
            coarse_yaw - _FINE_HALF_RAD,
            coarse_yaw + _FINE_HALF_RAD + 1e-9,
            _FINE_STEP_RAD,
        ):
            candidate = Pose2D(initial_pose.x, initial_pose.y, yaw)
            score = self._score(ranges, rel_angles, candidate, grid_map)
            if score > best_score:
                best_score = score
                best_pose = candidate

        return best_pose

    # ------------------------------------------------------------------ #

    @staticmethod
    def _valid_rays(scan: LaserScan, grid_map: OccupancyGridMap):
        """Return (ranges, relative_angles) arrays for valid hit rays."""
        ranges = np.array(scan.ranges, dtype=np.float32)
        n = len(ranges)
        angles = scan.angle_min + np.arange(n, dtype=np.float32) * scan.angle_increment

        rmin = max(float(scan.range_min), grid_map.min_useful_range)
        rmax = float(scan.range_max) * grid_map.max_range_factor

        valid = np.isfinite(ranges) & (ranges > rmin) & (ranges < rmax)
        idx = np.where(valid)[0][::_RAY_STRIDE]

        return ranges[idx], angles[idx]

    @staticmethod
    def _score(
        ranges: np.ndarray,
        rel_angles: np.ndarray,
        pose: Pose2D,
        grid_map: OccupancyGridMap,
    ) -> float:
        """Sum log-odds at scan endpoint cells. Higher means better alignment."""
        world_angles = rel_angles + pose.yaw
        wx = pose.x + ranges * np.cos(world_angles)
        wy = pose.y + ranges * np.sin(world_angles)

        res = grid_map.resolution
        col = ((wx - grid_map.origin_x) / res).astype(np.int32)
        row = ((wy - grid_map.origin_y) / res).astype(np.int32)

        size = grid_map.size_pixels
        mask = (col >= 0) & (col < size) & (row >= 0) & (row < size)
        if not np.any(mask):
            return 0.0

        return float(np.sum(np.maximum(0.0, grid_map.grid[row[mask], col[mask]])))
