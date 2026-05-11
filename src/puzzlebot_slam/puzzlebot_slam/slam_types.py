"""Small shared data types for the mapping SLAM package."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float
