"""Convert ROS 2 messages to JSON-serializable dicts."""

import base64
import math
import time
from typing import Any, Dict, List, Optional


def _stamp_to_float(header) -> float:
    """Convert ROS Header stamp to Unix float seconds."""
    try:
        return header.stamp.sec + header.stamp.nanosec * 1e-9
    except Exception:
        return time.time()


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Extract yaw (rotation around Z) from a quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _clean_ranges(ranges) -> List[Optional[float]]:
    """Replace inf/NaN LiDAR readings with null for JSON safety."""
    cleaned = []
    for r in ranges:
        if math.isfinite(r):
            cleaned.append(round(r, 4))
        else:
            cleaned.append(None)
    return cleaned


def odom_to_json(msg) -> Dict[str, Any]:
    q = msg.pose.pose.orientation
    cov = msg.pose.covariance  # flat 36-element array
    kp = {
        'x': round(msg.pose.pose.position.x, 4),
        'y': round(msg.pose.pose.position.y, 4),
        'theta': round(_quat_to_yaw(q.x, q.y, q.z, q.w), 4),
    }
    return {
        'type': 'robot_state',
        'timestamp': _stamp_to_float(msg.header),
        'pose': kp,          # overridden in bridge_node with SLAM pose when available
        'kalman_pose': kp,   # always the raw Kalman/odom estimate
        'odom_twist': {
            'linear_x': round(msg.twist.twist.linear.x, 4),
            'angular_z': round(msg.twist.twist.angular.z, 4),
        },
        # Kalman covariance diagonal: P_xx, P_yy, P_tt
        'cov_xx': round(float(cov[0]),  6),
        'cov_yy': round(float(cov[7]),  6),
        'cov_tt': round(float(cov[35]), 6),
    }


def pose_with_cov_to_json(msg, type_str: str) -> Dict[str, Any]:
    """Serialize a PoseWithCovarianceStamped message (ArUco pose, scan match pose)."""
    q   = msg.pose.pose.orientation
    cov = msg.pose.covariance
    return {
        'type':      type_str,
        'timestamp': _stamp_to_float(msg.header),
        'x':     round(msg.pose.pose.position.x, 4),
        'y':     round(msg.pose.pose.position.y, 4),
        'theta': round(_quat_to_yaw(q.x, q.y, q.z, q.w), 4),
        'cov_xx': round(float(cov[0]),  6),
        'cov_yy': round(float(cov[7]),  6),
        'cov_tt': round(float(cov[35]), 6),
    }


def scan_to_json(msg) -> Dict[str, Any]:
    ranges = _clean_ranges(msg.ranges)
    valid = [r for r in ranges if r is not None]
    min_distance = round(min(valid), 4) if valid else None
    return {
        'type': 'scan',
        'timestamp': _stamp_to_float(msg.header),
        'angle_min': round(msg.angle_min, 6),
        'angle_max': round(msg.angle_max, 6),
        'angle_increment': round(msg.angle_increment, 6),
        'range_min': round(msg.range_min, 4),
        'range_max': round(msg.range_max, 4),
        'min_distance': min_distance,
        'ranges': ranges,
    }


def map_to_json(msg) -> Dict[str, Any]:
    return {
        'type': 'map',
        'timestamp': _stamp_to_float(msg.header),
        'width': msg.info.width,
        'height': msg.info.height,
        'resolution': round(msg.info.resolution, 6),
        'origin': {
            'x': round(msg.info.origin.position.x, 4),
            'y': round(msg.info.origin.position.y, 4),
        },
        'data': list(msg.data),
    }


def twist_to_json(msg, source: str) -> Dict[str, Any]:
    return {
        'type': 'velocity_command',
        'timestamp': time.time(),
        'source': source,
        'linear_x': round(msg.linear.x, 4),
        'angular_z': round(msg.angular.z, 4),
    }


def camera_to_json(msg) -> Dict[str, Any]:
    return {
        'type': 'camera_frame',
        'timestamp': _stamp_to_float(msg.header),
        'data': base64.b64encode(bytes(msg.data)).decode('ascii'),
    }


def voice_to_json(
    command: Optional[str],
    confidence: Optional[float],
    status: Optional[str],
    inference_time_ms: Optional[float],
    ranked_predictions_raw: Optional[str],
) -> Dict[str, Any]:
    predictions = []
    if ranked_predictions_raw:
        try:
            import json
            predictions = json.loads(ranked_predictions_raw)
        except Exception:
            pass
    return {
        'type': 'voice_command',
        'timestamp': time.time(),
        'command': command,
        'confidence': round(confidence, 4) if confidence is not None else None,
        'status': status,
        'inference_time_ms': round(inference_time_ms, 4) if inference_time_ms is not None else None,
        'ranked_predictions': predictions,
    }
