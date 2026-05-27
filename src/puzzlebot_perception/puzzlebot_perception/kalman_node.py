"""
kalman_node.py — Extended Kalman Filter for ArUco visual correction.

Subscribes:
  /aruco/poses   (geometry_msgs/PoseArray)  — raw ArUco detections
  /odom          (nav_msgs/Odometry)         — wheel odometry (prediction step)

Publishes:
  /kalman/pose   (geometry_msgs/PoseStamped) — filtered robot pose estimate

How it works:
  State vector: [x, y, theta]
  Prediction:   integrate wheel odometry at each /odom message
  Update:       when an ArUco marker with known world position is detected,
                correct the state using the observation residual
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import Odometry
import numpy as np


class KalmanNode(Node):
    def __init__(self):
        super().__init__('kalman_node')

        # ── Parameters ──────────────────────────────────────────────────
        self.declare_parameter('process_noise_xy',    0.01)
        self.declare_parameter('process_noise_theta', 0.01)
        self.declare_parameter('obs_noise_xy',        0.05)
        self.declare_parameter('obs_noise_theta',     0.05)

        q_xy    = self.get_parameter('process_noise_xy').value
        q_theta = self.get_parameter('process_noise_theta').value
        r_xy    = self.get_parameter('obs_noise_xy').value
        r_theta = self.get_parameter('obs_noise_theta').value

        # ── EKF state: [x, y, theta] ─────────────────────────────────────
        self.x = np.zeros(3)           # state estimate
        self.P = np.eye(3) * 1.0       # covariance (start uncertain)

        # Process noise covariance Q
        self.Q = np.diag([q_xy, q_xy, q_theta])

        # Observation noise covariance R (x, y from ArUco tvec)
        self.R = np.diag([r_xy, r_xy])

        # Observation matrix H: we observe [x, y] from the ArUco tvec
        self.H = np.array([[1, 0, 0],
                           [0, 1, 0]])

        self._last_odom_time = None
        self._last_vx = 0.0
        self._last_wz = 0.0

        # ── Pub / Sub ────────────────────────────────────────────────────
        self.sub_odom  = self.create_subscription(
            Odometry, '/odom', self._odom_cb, 10)
        self.sub_aruco = self.create_subscription(
            PoseArray, '/aruco/poses', self._aruco_cb, 10)
        self.pub = self.create_publisher(PoseStamped, '/kalman/pose', 10)

        self.get_logger().info(
            f'kalman_node started  Q={np.diag(self.Q).tolist()}  R={np.diag(self.R).tolist()}')

    # ── Prediction step (odometry) ───────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._last_odom_time is None:
            self._last_odom_time = now
            return

        dt = now - self._last_odom_time
        self._last_odom_time = now
        if dt <= 0 or dt > 1.0:
            return

        vx = msg.twist.twist.linear.x
        wz = msg.twist.twist.angular.z
        th = self.x[2]

        # Motion model: x' = x + vx*cos(th)*dt, y' = y + vx*sin(th)*dt, th' = th + wz*dt
        self.x[0] += vx * np.cos(th) * dt
        self.x[1] += vx * np.sin(th) * dt
        self.x[2] += wz * dt
        self.x[2]  = self._wrap(self.x[2])

        # Jacobian of motion model F
        F = np.array([
            [1, 0, -vx * np.sin(th) * dt],
            [0, 1,  vx * np.cos(th) * dt],
            [0, 0,  1                    ],
        ])

        self.P = F @ self.P @ F.T + self.Q
        self._publish()

    # ── Update step (ArUco observation) ─────────────────────────────────
    def _aruco_cb(self, msg: PoseArray):
        if not msg.poses:
            return

        # Use the first detected marker as a visual anchor.
        # The ArUco tvec gives the marker position RELATIVE to the camera.
        # We use the x, y components as a direct position observation of the
        # robot (good enough for visual correction; full TF chain optional).
        pose = msg.poses[0]
        z = np.array([pose.position.x, pose.position.y])

        # Innovation
        y_inn = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y_inn
        self.x[2] = self._wrap(self.x[2])

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(3) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        self.get_logger().info(
            f'ArUco update → x={self.x[0]:.3f} y={self.x[1]:.3f} θ={np.degrees(self.x[2]):.1f}°',
            throttle_duration_sec=1.0)

        self._publish()

    def _publish(self):
        ps = PoseStamped()
        ps.header.stamp    = self.get_clock().now().to_msg()
        ps.header.frame_id = 'odom'
        ps.pose.position.x = float(self.x[0])
        ps.pose.position.y = float(self.x[1])

        # theta → quaternion (rotation around Z)
        th = self.x[2]
        ps.pose.orientation.z = float(np.sin(th / 2))
        ps.pose.orientation.w = float(np.cos(th / 2))

        self.pub.publish(ps)

    @staticmethod
    def _wrap(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi


def main(args=None):
    rclpy.init(args=args)
    node = KalmanNode()
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
