"""Scan matching extension point for mapping SLAM.

The current implementation intentionally returns the odometry pose unchanged.
This keeps the validated Gazebo mapping behavior stable while making the future
real-robot scan matcher an isolated component.
"""

from sensor_msgs.msg import LaserScan

from .occupancy_grid_map import OccupancyGridMap
from .slam_types import Pose2D


class LocalScanMatcher:
    def __init__(self, enabled: bool = False):
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def match(
        self,
        scan: LaserScan,
        initial_pose: Pose2D,
        grid_map: OccupancyGridMap,
    ) -> Pose2D:
        # TODO: implement local search around initial_pose for the physical robot.
        return initial_pose
