# Puzzlebot ROS2 Architecture Guide
**Robots Autónomos — ITESM 2026**
*Reference document for development of the autonomous differential-drive robot system*

---

## Table of Contents

1. [Hardware Context & Constraints](#1-hardware-context--constraints)
2. [Differential Drive vs Ackermann — Which Model?](#2-differential-drive-vs-ackermann--which-model)
3. [ML & YOLO Optimization on Jetson Orin 2GB](#3-ml--yolo-optimization-on-jetson-orin-2gb)
4. [LiDAR, Mapping, and SLAM — Full Pipeline](#4-lidar-mapping-and-slam--full-pipeline)
5. [Node Architecture & ROS2 Design](#5-node-architecture--ros2-design)
6. [Testing with Mocks & YAML Configs](#6-testing-with-mocks--yaml-configs)
7. [Simulator Evaluation & Integration](#7-simulator-evaluation--integration)
8. [Development Roadmap](#8-development-roadmap)

---

## 1. Hardware Context & Constraints

### What you have

| Component | Detail | Constraint |
|---|---|---|
| Drive system | 2-wheel differential drive | No access to the low-level MCU |
| Encoder topics | `/velocity_enc_r`, `/velocity_enc_l` — published by existing firmware | Read-only; you cannot reconfigure the driver |
| Motor command | `/cmd_vel` Twist message accepted by existing firmware | You publish, firmware executes |
| Camera | USB/CSI camera processed on Jetson Orin | Only 2 GB RAM available for all processes |
| LiDAR | 2D rotating LiDAR (e.g., RPLIDAR A1) | Publishes `sensor_msgs/LaserScan` on `/scan` |
| Compute | Jetson Orin 2 GB (ARM + integrated GPU) | No discrete GPU; must use TensorRT for any DNN |
| Gripper | FPGA + servo, listens to `/open_close` Bool | Treated as a black box |

### Critical architectural implication

Because you cannot touch the MCU firmware, the entire software stack is **ROS2 nodes communicating only through topics**. The physical hardware interface layer is already handled — your lowest-level access is the published topics. This is actually advantageous: it means you can mock any hardware sensor trivially by publishing fake data on those same topics.

---

## 2. Differential Drive vs Ackermann — Which Model?

### Your robot is a differential drive — and that is the right choice

An **Ackermann drive** is the steering geometry used in cars: the front wheels turn on arcs with different radii, and there is always a non-zero minimum turning radius. A pure differential robot, by contrast, can spin in place by rotating both wheels in opposite directions at equal speed.

**Do not try to implement an Ackermann model for the Puzzlebot.** Here is why:

| Property | Differential drive (your robot) | Ackermann drive |
|---|---|---|
| Zero-radius turn | ✅ Yes — spin both wheels in opposite directions | ❌ No — always needs forward motion to turn |
| Model complexity | Simple — 3 state variables `[x, y, θ]` | Higher — requires steering angle state |
| Kinematics | One equation: `v = (vR + vL)/2`, `ω = (vR - vL)/l` | Separate steering geometry + Ackermann correction |
| Sensor needed | Encoders on both wheels | Encoders + steering angle sensor |
| ROS2 standard | `diff_drive_controller` | `ackermann_steering_controller` |

**The only scenario where Ackermann thinking is relevant** is if your path planner generates trajectories with minimum curvature constraints (e.g., for smoother indoor navigation). In that case, you add a curvature limit to your planner output — but the kinematics node itself stays pure differential.

### Kinematic model (as seen in your class slides)

From the Semana 3 slides, the differential drive forward kinematics are:

```
ω = (vR - vL) / l           # angular velocity of robot body
v = (vR + vL) / 2           # linear velocity of robot body

ẋ = v · cos(θ)
ẏ = v · sin(θ)
θ̇ = ω
```

Where `vR` and `vL` are the linear velocities of the right and left wheels respectively (derived from encoder angular velocities by `v = r · ω_wheel`), `l` is the wheelbase, and `r` is wheel radius.

**Discrete integration (what your `odometry_node` actually computes each tick):**

```
Δd = (ΔdR + ΔdL) / 2
Δθ = (ΔdR - ΔdL) / l

x_k  = x_{k-1}  + Δd · cos(θ_{k-1})
y_k  = y_{k-1}  + Δd · sin(θ_{k-1})
θ_k  = θ_{k-1}  + Δθ
```

This is your dead-reckoning prediction step. It accumulates drift over time, which is why you need the Kalman filter correction step using Aruco landmarks and LiDAR scan-matching.

---

## 3. ML & YOLO Optimization on Jetson Orin 2 GB

The Jetson Orin is powerful — but 2 GB of unified memory shared between CPU and GPU means a naive PyTorch/YOLO installation will OOM immediately. You need to be deliberate about every MB.

### The optimization stack: TensorRT + INT8 quantization

**Step 1 — Choose a lightweight detector backbone**

Do not start with YOLOv8-large or YOLOv5-m. Your options, ranked by memory and latency:

| Model | Parameters | FP16 VRAM | Typical INT8 latency (Orin) | Recommended use |
|---|---|---|---|---|
| YOLOv8n | 3.2 M | ~150 MB | 4–8 ms/frame | ✅ Primary choice |
| YOLOv5n | 1.9 M | ~120 MB | 3–6 ms/frame | ✅ Alternative |
| MicroYOLO / YOLO-Fastest | <1 M | ~60 MB | 2–4 ms/frame | ✅ If memory is still tight |
| YOLOv8s | 11.2 M | ~350 MB | 8–15 ms/frame | ⚠️ Marginal |
| YOLOv8m+ | 25+ M | >600 MB | >20 ms/frame | ❌ Will OOM |

**Step 2 — Export to TensorRT**

NVIDIA's TensorRT compiler takes your trained model and produces a hardware-optimized `.engine` file that runs directly on the Orin's GPU without a Python DL runtime in the hot path.

```bash
# From a trained YOLOv8n .pt checkpoint:
pip install ultralytics
yolo export model=yolov8n.pt format=engine device=0 half=True
# 'half=True' = FP16 precision — halves memory vs FP32
# The output is yolov8n.engine (~80-150 MB)
```

**Step 3 — INT8 calibration for maximum compression**

FP16 is good. INT8 is better — it cuts the model to ~25% of FP32 size with minimal accuracy loss if you calibrate properly.

```bash
yolo export model=yolov8n.pt format=engine device=0 int8=True \
  data=calib_images/   # folder of ~100-500 representative images for calibration
```

The calibration images should be representative of your actual environment — the corridors, trailers, and pallets the robot will see.

**Step 4 — ROS2 integration via TensorRT Python bindings**

```python
# yolo_node.py  — simplified
import tensorrt as trt
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        # Load .engine file — no PyTorch loaded at runtime
        self.engine = self.load_trt_engine('yolov8n.engine')
        self.sub = self.create_subscription(Image, '/cam_img', self.cb, 10)
        self.pub = self.create_publisher(Detection2DArray, '/trailer/detection', 10)

    def load_trt_engine(self, path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, 'rb') as f, trt.Runtime(logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def cb(self, msg):
        # Preprocess → run TRT inference → parse detections → publish
        ...
```

**Step 5 — Memory budget management**

Run only what you need simultaneously. Use ROS2 lifecycle nodes so you can shut down the YOLO node when it is not needed (e.g., during pure navigation phases).

```yaml
# ros2 launch puzzlebot_bringup navigation.launch.py
# vs.
# ros2 launch puzzlebot_bringup docking.launch.py  (activates yolo_node)
```

**Step 6 — Camera resolution and inference rate**

Do not run at 1080p at 30fps. For detection tasks:

```python
declare_parameter('camera_width',  640)   # 640x480 is enough for YOLO
declare_parameter('camera_height', 480)
declare_parameter('detection_rate', 10.0) # 10 Hz detection, not 30 Hz
```

Subscribe to the camera at 30 fps but throttle detection to 10 fps using a timer. This alone cuts GPU time by 66%.

### Aruco detection — zero ML needed

Aruco marker detection runs on CPU with OpenCV — no GPU or TensorRT required. It is lightweight and deterministic. Keep it running at full camera frame rate.

```python
import cv2
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, params)
corners, ids, _ = detector.detectMarkers(gray_frame)
```

---

## 4. LiDAR, Mapping, and SLAM — Full Pipeline

This section ties directly to what was covered in the Semana 3 slides. The pipeline is:

```
Raw LiDAR scans  →  Feature extraction  →  Scan matching (RANSAC)  →  Pose estimation
                                         +  Monte Carlo particle filter  →  Occupancy grid map
```

### 4.1 What a LiDAR scan is

Your LiDAR publishes `sensor_msgs/LaserScan` on `/scan`. Each message is an array of distance measurements at angles from `angle_min` to `angle_max` in steps of `angle_increment`. For an RPLIDAR A1 this gives 360 measurements per revolution at 5–10 Hz.

```python
# What one scan message looks like:
scan.ranges      # list of ~360 float values (distances in meters)
scan.angle_min   # typically -π
scan.angle_max   # typically +π
scan.angle_increment  # typically 2π/360 ≈ 0.0175 rad
```

Converting to 2D Cartesian points (your point cloud):

```python
import numpy as np

def scan_to_points(scan):
    angles = np.arange(scan.angle_min, scan.angle_max, scan.angle_increment)
    ranges = np.array(scan.ranges)
    # Filter out inf/nan readings
    valid = np.isfinite(ranges) & (ranges > scan.range_min) & (ranges < scan.range_max)
    x = ranges[valid] * np.cos(angles[valid])
    y = ranges[valid] * np.sin(angles[valid])
    return np.vstack((x, y)).T  # shape (N, 2)
```

### 4.2 RANSAC — fitting lines and planes to the point cloud

**Random Sample Consensus (RANSAC)** is used to extract geometric features (lines = walls) from the noisy LiDAR point cloud. It is robust to outliers — stray reflections, people walking through, etc.

**How it works for wall detection:**

```
1. Randomly pick 2 points from the scan
2. Fit a line through those 2 points
3. Count how many other points are within distance ε of that line (inliers)
4. Repeat N times
5. Keep the line with the most inliers
6. Refit the line using all inliers (least squares)
7. Remove those inliers and repeat to find next wall
```

```python
def ransac_line(points, n_iter=100, threshold=0.05):
    best_inliers = []
    for _ in range(n_iter):
        # Pick 2 random points
        idx = np.random.choice(len(points), 2, replace=False)
        p1, p2 = points[idx[0]], points[idx[1]]
        
        # Line equation: ax + by + c = 0
        a = p2[1] - p1[1]
        b = p1[0] - p2[0]
        c = p2[0]*p1[1] - p1[0]*p2[1]
        norm = np.sqrt(a**2 + b**2)
        
        # Distance of every point to this line
        dists = np.abs(a*points[:,0] + b*points[:,1] + c) / norm
        inliers = np.where(dists < threshold)[0]
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
    
    return best_inliers
```

**Why RANSAC in the SLAM pipeline?** The extracted line features (walls) are much more stable landmarks than individual LiDAR points. Two successive scans of the same wall produce slightly different individual points (noise), but the fitted wall line is nearly identical. This makes scan-to-scan matching far more robust.

### 4.3 Hidden Markov Model (HMM) — the probabilistic skeleton of SLAM

As your Semana 3 slides explain, SLAM can be viewed as a **Hidden Markov Model**:

- **Hidden states** `X_t` — the robot's true pose `[x, y, θ]` at each timestep, which is never observed directly
- **Observations** `Z_t` — sensor readings (LiDAR scan, Aruco marker detections)
- **Transition model** `P(X_t | X_{t-1}, u_t)` — how the robot moves given control input `u_t` (i.e., the kinematic model above)
- **Observation model** `P(Z_t | X_t, M)` — what sensor readings we would expect if the robot is at pose `X_t` in map `M`

The goal of SLAM is to estimate the joint distribution `P(X_t, M | Z_{1:t}, u_{1:t})` — both the robot pose and the map simultaneously.

**The dynamic Bayesian network structure (from your slides):**

```
A_{t-2} ──→ A_{t-1} ──→ A_t        (hidden actions / poses)
    ↓           ↓           ↓
  Z_{t-1}    Z_t-1      Z_{t+1}     (sensor observations)
```

### 4.4 Monte Carlo Localization (MCL) — the particle filter

**MCL = the practical implementation of the HMM belief update using random samples.**

A particle is a hypothesis `[x_i, y_i, θ_i, w_i]` — a possible robot pose with an associated weight. The filter maintains N particles (N = 100–500 is usually enough) representing the probability distribution over robot poses.

**One MCL iteration (following your class slides exactly):**

```
Step 1 — Predict (motion update)
   For each particle i:
     Sample new pose from motion model:
     x_i += Δd·cos(θ_i) + noise_x
     y_i += Δd·sin(θ_i) + noise_y
     θ_i += Δθ              + noise_θ
   (Noise comes from your encoder uncertainty model)

Step 2 — Update (sensor update / weighting)
   For each particle i:
     Simulate what the LiDAR should read at pose [x_i, y_i, θ_i] using current map
     Compare simulated scan to actual scan
     w_i = P(Z_actual | pose_i, map)   # likelihood

Step 3 — Resample
   Draw N new particles with replacement, weighted by w_i
   (Particles with high weight multiply; low-weight particles disappear)

Step 4 — Map update
   Use the weighted particle set to update the occupancy grid
```

**Python skeleton for MCL in your `slam_node`:**

```python
import numpy as np

class ParticleFilter:
    def __init__(self, n_particles=200, map_size=(800, 800), resolution=0.05):
        self.N = n_particles
        # Initialize uniformly across the map
        self.particles = np.zeros((n_particles, 3))  # [x, y, theta]
        self.particles[:, 0] = np.random.uniform(0, map_size[0] * resolution, n_particles)
        self.particles[:, 1] = np.random.uniform(0, map_size[1] * resolution, n_particles)
        self.particles[:, 2] = np.random.uniform(-np.pi, np.pi, n_particles)
        self.weights = np.ones(n_particles) / n_particles

    def predict(self, delta_d, delta_theta, noise_std=(0.02, 0.02, 0.01)):
        """Motion model — propagate each particle with Gaussian noise."""
        self.particles[:, 0] += delta_d * np.cos(self.particles[:, 2]) \
                                + np.random.normal(0, noise_std[0], self.N)
        self.particles[:, 1] += delta_d * np.sin(self.particles[:, 2]) \
                                + np.random.normal(0, noise_std[1], self.N)
        self.particles[:, 2] += delta_theta \
                                + np.random.normal(0, noise_std[2], self.N)

    def update(self, scan, occupancy_map):
        """Sensor model — score each particle against the LiDAR scan."""
        for i, p in enumerate(self.particles):
            # Simulate LiDAR reading at particle pose p
            # Compare to actual scan using beam model or correlation
            self.weights[i] = self.score_particle(p, scan, occupancy_map)
        self.weights += 1e-300  # avoid zero
        self.weights /= self.weights.sum()

    def resample(self):
        """Systematic resampling."""
        indices = np.random.choice(self.N, self.N, replace=True, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.N) / self.N

    def score_particle(self, pose, scan, occ_map):
        """
        Project scan rays from 'pose' into the current occupancy map.
        Sum of occupancy values at ray endpoints = likelihood score.
        Higher score = particle is in a location consistent with the scan.
        """
        score = 0.0
        angles = np.linspace(-np.pi, np.pi, len(scan))
        for r, a in zip(scan, angles):
            if not np.isfinite(r):
                continue
            world_x = pose[0] + r * np.cos(pose[2] + a)
            world_y = pose[1] + r * np.sin(pose[2] + a)
            # Convert to map cell
            cx = int(world_x / occ_map.resolution)
            cy = int(world_y / occ_map.resolution)
            if 0 <= cx < occ_map.width and 0 <= cy < occ_map.height:
                score += occ_map.grid[cy, cx]  # occupied cells near ray endpoints = good
        return score

    @property
    def best_estimate(self):
        """Weighted mean pose."""
        x = np.average(self.particles[:, 0], weights=self.weights)
        y = np.average(self.particles[:, 1], weights=self.weights)
        # Circular mean for angle
        sin_sum = np.sum(self.weights * np.sin(self.particles[:, 2]))
        cos_sum = np.sum(self.weights * np.cos(self.particles[:, 2]))
        theta = np.arctan2(sin_sum, cos_sum)
        return np.array([x, y, theta])
```

### 4.5 Occupancy grid — the 2D map representation

The map is a 2D grid where each cell holds a value from 0 (definitely free) to 100 (definitely occupied), with −1 for unknown. Each LiDAR scan updates the map using the **log-odds update rule**:

```python
def update_map(grid, robot_pose, scan, resolution=0.05):
    """
    Bresenham ray-cast: cells along each ray → mark as free.
    Cell at ray endpoint → mark as occupied.
    """
    L_OCCUPIED = 0.85   # log-odds increment for occupied
    L_FREE     = -0.4   # log-odds decrement for free
    
    for r, angle in zip(scan.ranges, angles):
        if not np.isfinite(r):
            continue
        # Endpoint in world frame
        end_x = robot_pose[0] + r * np.cos(robot_pose[2] + angle)
        end_y = robot_pose[1] + r * np.sin(robot_pose[2] + angle)
        
        # All cells along the ray → free
        ray_cells = bresenham(robot_pose[:2], [end_x, end_y], resolution)
        for cell in ray_cells[:-1]:
            grid[cell] += L_FREE
        
        # Endpoint cell → occupied
        grid[ray_cells[-1]] += L_OCCUPIED
    
    # Clamp log-odds to [-5, 5] to prevent saturation
    np.clip(grid, -5, 5, out=grid)
```

### 4.6 CoreSLAM / BreezySLAM — your reference implementation

Your class explicitly suggests **BreezySLAM** as a didactic implementation of CoreSLAM. BreezySLAM implements the full Monte Carlo + occupancy grid pipeline in ~200 lines. The ROS2 integration approach is:

```python
# In your slam_node — wrap BreezySLAM
from breezyslam.algorithms import CoreSLAM
from breezyslam.sensors import RPLidarA1

class SlamNode(Node):
    def __init__(self):
        super().__init__('slam_node')
        laser = RPLidarA1()
        self.slam = CoreSLAM(laser, MAP_SIZE_PIXELS=800, MAP_SIZE_METERS=16)
        self.map_bytes = bytearray(800 * 800)
        
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.map_pub  = self.create_publisher(OccupancyGrid, '/map', 1)

    def scan_cb(self, msg):
        distances = [int(r * 1000) for r in msg.ranges]  # mm
        velocities = [self.current_v, self.current_omega]
        x_mm, y_mm, theta_deg = self.slam.update(distances, velocities)
        self.slam.getmap(self.map_bytes)
        self.publish_map()
```

### 4.7 The complete SLAM pipeline in your system

```
/velocity_enc_r, /velocity_enc_l
         ↓
   odometry_node  →  /odom_raw  (dead-reckoning pose, drifts over time)
         ↓
  kalman_filter_node  ←  /aruco/poses  (absolute correction, kills drift)
         ↓                              ← /scan (scan-match correction)
        /odom  (filtered, corrected pose)
         ↓
     slam_node  ←  /scan
         ↓
        /map  (OccupancyGrid, updated in real time)
```

**Why two localization layers?**
- Dead reckoning (`odometry_node`) is fast but drifts
- EKF (`kalman_filter_node`) fuses odometry with Aruco for low-latency corrected pose
- MCL inside `slam_node` uses the corrected pose + LiDAR to build and maintain the map

---

## 5. Node Architecture & ROS2 Design

### Complete package structure

```
puzzlebot_ws/
└── src/
    ├── puzzlebot_msgs/          # Custom message, service, and action definitions
    │   └── msg/
    │       ├── HealthIndex.msg
    │       └── ParticleArray.msg
    ├── puzzlebot_bringup/       # Launch files and YAML configs
    │   ├── launch/
    │   │   ├── full_system.launch.py
    │   │   ├── slam_only.launch.py
    │   │   └── navigation.launch.py
    │   └── config/
    │       ├── robot_params.yaml
    │       ├── slam_params.yaml
    │       ├── kalman_params.yaml
    │       └── yolo_params.yaml
    ├── puzzlebot_perception/
    │   └── puzzlebot_perception/
    │       ├── camera_node.py
    │       ├── aruco_node.py
    │       ├── yolo_node.py
    │       └── encoder_node.py   # (may be provided by hardware — subscribe only)
    ├── puzzlebot_localization/
    │   └── puzzlebot_localization/
    │       ├── odometry_node.py
    │       ├── kalman_filter_node.py
    │       └── slam_node.py
    ├── puzzlebot_planning/
    │   └── puzzlebot_planning/
    │       ├── path_planner_node.py
    │       └── obstacle_avoidance_node.py
    ├── puzzlebot_control/
    │   └── puzzlebot_control/
    │       ├── state_machine_node.py
    │       └── steering_controller_node.py
    └── puzzlebot_hmi/
        └── puzzlebot_hmi/
            ├── voice_node.py
            └── web_bridge_node.py
```

### EKF Kalman filter node — implementation guide

The Kalman filter fuses the kinematic prediction with Aruco marker corrections.

**State vector:** `X = [x, y, θ]`
**Covariance matrix:** `P` (3×3)

```python
class KalmanFilterNode(Node):
    def __init__(self):
        super().__init__('kalman_filter_node')
        
        # --- State ---
        self.x = np.zeros(3)          # [x, y, theta]
        self.P = np.eye(3) * 0.1      # Initial covariance (loaded from YAML)
        
        # --- Noise matrices (tuned via YAML config) ---
        self.Q = np.diag([0.01, 0.01, 0.005])   # Process noise
        self.R = np.diag([0.05, 0.05, 0.01])     # Measurement noise (Aruco)

    def predict(self, delta_d, delta_theta):
        """EKF prediction step — apply motion model."""
        theta = self.x[2]
        # State transition
        self.x[0] += delta_d * np.cos(theta)
        self.x[1] += delta_d * np.sin(theta)
        self.x[2] += delta_theta
        self.x[2]  = self.normalize_angle(self.x[2])
        
        # Jacobian of motion model (linearization)
        F = np.array([
            [1, 0, -delta_d * np.sin(theta)],
            [0, 1,  delta_d * np.cos(theta)],
            [0, 0,  1]
        ])
        self.P = F @ self.P @ F.T + self.Q

    def update_aruco(self, z_x, z_y, z_theta):
        """EKF update step — correct with Aruco marker measurement."""
        z = np.array([z_x, z_y, z_theta])
        H = np.eye(3)                   # Observation matrix (direct measurement)
        S = H @ self.P @ H.T + self.R  # Innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman gain
        
        innovation = z - H @ self.x
        innovation[2] = self.normalize_angle(innovation[2])
        
        self.x = self.x + K @ innovation
        self.P = (np.eye(3) - K @ H) @ self.P
```

---

## 6. Testing with Mocks & YAML Configs

### The isolation testing principle

Every ROS2 node communicates only through topics, services, and actions. This means **you can replace any hardware source with a Python script that publishes fake data on the same topic**. No hardware needed. No simulation needed.

### Pattern 1 — Topic mock publisher

Create a `mock_` node for each hardware topic you want to fake:

```python
# puzzlebot_bringup/puzzlebot_bringup/mock_encoders.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import math

class MockEncoders(Node):
    """Publishes fake encoder velocities simulating straight-line motion."""
    def __init__(self):
        super().__init__('mock_encoders')
        self.declare_parameter('linear_vel', 0.2)   # m/s
        self.declare_parameter('angular_vel', 0.0)  # rad/s
        
        v   = self.get_parameter('linear_vel').value
        w   = self.get_parameter('angular_vel').value
        l   = 0.18   # wheelbase in meters
        r   = 0.05   # wheel radius in meters
        
        vR = (v + w * l / 2) / r   # rad/s
        vL = (v - w * l / 2) / r   # rad/s
        
        self.pub_r = self.create_publisher(Float32, '/velocity_enc_r', 10)
        self.pub_l = self.create_publisher(Float32, '/velocity_enc_l', 10)
        self.create_timer(0.05, lambda: [
            self.pub_r.publish(Float32(data=vR)),
            self.pub_l.publish(Float32(data=vL))
        ])

def main():
    rclpy.init()
    rclpy.spin(MockEncoders())
```

Now you can test `odometry_node` in isolation with no robot:

```bash
ros2 run puzzlebot_bringup mock_encoders --ros-args -p linear_vel:=0.3
ros2 run puzzlebot_localization odometry_node
ros2 topic echo /odom  # Watch the pose integrate correctly
```

### Pattern 2 — Mock LiDAR from a recorded bag

Record one real LiDAR session and replay it forever for offline testing:

```bash
# On the real robot — record 30 seconds of LiDAR:
ros2 bag record /scan -o lidar_corridor_test

# On your dev laptop — replay infinitely:
ros2 bag play lidar_corridor_test --loop

# Now run your slam_node against recorded data — no robot needed
ros2 run puzzlebot_localization slam_node
```

### Pattern 3 — Mock Aruco detections

```python
# mock_aruco.py — publishes a fake marker at a known world position
class MockAruco(Node):
    def __init__(self):
        super().__init__('mock_aruco')
        self.pub = self.create_publisher(PoseArray, '/aruco/poses', 10)
        self.create_timer(0.5, self.publish_marker)
    
    def publish_marker(self):
        pose = Pose()
        pose.position.x = 2.5   # known world position
        pose.position.y = 1.0
        # ... set orientation
        msg = PoseArray()
        msg.poses = [pose]
        self.pub.publish(msg)
```

### Pattern 4 — Unit testing node logic without ROS2 spin

Extract the math from your nodes into plain Python functions and test them with `pytest` — no ROS2 context needed:

```python
# tests/test_odometry.py
import pytest
import numpy as np
from puzzlebot_localization.odometry_math import integrate_pose

def test_straight_line():
    x, y, theta = 0.0, 0.0, 0.0
    # Moving forward 1 meter
    x, y, theta = integrate_pose(x, y, theta, delta_d=1.0, delta_theta=0.0)
    assert abs(x - 1.0) < 0.001
    assert abs(y - 0.0) < 0.001

def test_turn_in_place():
    x, y, theta = 0.0, 0.0, 0.0
    x, y, theta = integrate_pose(x, y, theta, delta_d=0.0, delta_theta=np.pi/2)
    assert abs(theta - np.pi/2) < 0.001
```

```bash
pytest tests/  # Runs instantly, no robot, no simulator
```

### YAML configuration — the right way to manage parameters

Never hardcode tunable values in node source code. All physical constants, noise parameters, and thresholds live in YAML files loaded at launch time.

```yaml
# config/robot_params.yaml
/**:
  ros__parameters:
    wheel_radius: 0.05         # meters
    wheelbase: 0.18            # meters — distance between wheel centers

# config/kalman_params.yaml
kalman_filter_node:
  ros__parameters:
    process_noise_x:     0.01   # Q matrix diagonal
    process_noise_y:     0.01
    process_noise_theta: 0.005
    meas_noise_x:        0.05   # R matrix diagonal (Aruco noise)
    meas_noise_y:        0.05
    meas_noise_theta:    0.01
    initial_covariance:  0.1

# config/slam_params.yaml
slam_node:
  ros__parameters:
    map_size_pixels:   800
    map_size_meters:   16.0
    n_particles:       200
    particle_noise_x:  0.02    # Noise injected in predict step
    particle_noise_y:  0.02
    particle_noise_t:  0.01
    update_frequency:  5.0     # Hz — how often to run MCL update

# config/yolo_params.yaml
yolo_node:
  ros__parameters:
    engine_path:       "models/yolov8n.engine"
    confidence_thresh: 0.45
    nms_thresh:        0.5
    detection_rate_hz: 10.0
    camera_width:      640
    camera_height:     480
```

Loading parameters in a node:

```python
class OdometryNode(Node):
    def __init__(self):
        super().__init__('odometry_node')
        self.declare_parameter('wheel_radius', 0.05)
        self.declare_parameter('wheelbase',    0.18)
        
        self.r = self.get_parameter('wheel_radius').value
        self.l = self.get_parameter('wheelbase').value
```

Loading via launch file:

```python
# launch/slam_only.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config_dir = os.path.join(
        get_package_share_directory('puzzlebot_bringup'), 'config')
    
    return LaunchDescription([
        Node(
            package='puzzlebot_localization',
            executable='odometry_node',
            parameters=[
                os.path.join(config_dir, 'robot_params.yaml')
            ]
        ),
        Node(
            package='puzzlebot_localization',
            executable='kalman_filter_node',
            parameters=[
                os.path.join(config_dir, 'robot_params.yaml'),
                os.path.join(config_dir, 'kalman_params.yaml')
            ]
        ),
        Node(
            package='puzzlebot_localization',
            executable='slam_node',
            parameters=[
                os.path.join(config_dir, 'slam_params.yaml')
            ]
        ),
    ])
```

**Changing a parameter at runtime (no recompilation):**

```bash
ros2 param set /slam_node n_particles 500
ros2 param set /kalman_filter_node process_noise_x 0.05
```

---

## 7. Simulator Evaluation & Integration

### Should you use a simulator?

**Yes — emphatically yes.** Testing exclusively on the physical robot in an academic project with limited robot access, limited time, and no automated testing infrastructure leads to slow iteration cycles and untraceable bugs. Here is the structured argument:

| Factor | Physical robot only | With simulator |
|---|---|---|
| Iteration speed | One test = walking to robot, deploying, running, debugging | One test = `ros2 launch`, instant feedback |
| Team parallelism | One robot = one person testing at a time | Everyone tests on their own machine simultaneously |
| Reproducibility | Environment changes (lighting, obstacles, people) | Deterministic — exact same scenario every time |
| Safety | A buggy `cmd_vel` command can damage robot or property | Crash the simulator all you want |
| Edge cases | Cannot easily put robot in exact failure scenario | Inject any scenario (perfect darkness, obstacle spawns, encoder failure) |
| CI/CD | Cannot run automated tests in a pipeline | `pytest` + simulator runs headlessly on every git push |

### Simulator comparison for your use case

| Simulator | ROS2 support | Physics | LiDAR simulation | Camera sim | Memory footprint | Verdict |
|---|---|---|---|---|---|---|
| **Gazebo (Ignition/Harmonic)** | ✅ Native, first-class | Good rigid body | ✅ Ray-cast LiDAR plugin | ✅ Camera plugin | Medium (~1 GB) | ✅ **Best choice** |
| CoppeliaSim (V-REP) | ✅ via ROS2 bridge | Good | ✅ via sensors | ✅ | Medium | ✅ Good alternative |
| MuJoCo | ⚠️ Community bridge | Excellent (contact) | ❌ No native LiDAR | Limited | Low | ❌ Wrong tool for this project |
| Webots | ✅ Native ROS2 node | Good | ✅ | ✅ | Low | ✅ Lightweight option |

### Recommendation: Gazebo Harmonic

Gazebo (now called Ignition/Harmonic) is the standard ROS2 simulator. It publishes on the **exact same topics** as your real robot. If your nodes work in Gazebo, they work on the physical robot with zero code changes — only the launch file changes.

### Gazebo integration for the Puzzlebot

**Step 1 — Create a URDF/SDF model of the Puzzlebot**

```xml
<!-- puzzlebot.urdf.xacro — simplified -->
<robot name="puzzlebot">
  <link name="base_link">
    <visual>
      <geometry><box size="0.2 0.15 0.1"/></geometry>
    </visual>
    <collision>
      <geometry><box size="0.2 0.15 0.1"/></geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <!-- Differential drive plugin — publishes /cmd_vel subscriber and encoder topics -->
  <gazebo>
    <plugin filename="libgz-sim-diff-drive-system.so"
            name="gz::sim::systems::DiffDrive">
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.18</wheel_separation>
      <wheel_radius>0.05</wheel_radius>
      <odom_publish_frequency>20</odom_publish_frequency>
      <topic>/cmd_vel</topic>
    </plugin>
  </gazebo>

  <!-- LiDAR plugin — publishes /scan -->
  <gazebo reference="lidar_link">
    <sensor type="gpu_lidar" name="lidar">
      <update_rate>10</update_rate>
      <ray>
        <scan><horizontal>
          <samples>360</samples>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal></scan>
        <range><min>0.15</min><max>12.0</max></range>
      </ray>
      <plugin filename="libgz-sim-sensors-system.so"
              name="gz::sim::systems::Sensors">
        <render_engine>ogre2</render_engine>
      </plugin>
    </sensor>
  </gazebo>
</robot>
```

**Step 2 — The bridge: same topics, zero code changes**

The Gazebo `ros_gz_bridge` package maps Gazebo internal topics to ROS2 topics:

```python
# In your Gazebo launch file:
from launch_ros.actions import Node

gz_bridge = Node(
    package='ros_gz_bridge',
    executable='parameter_bridge',
    arguments=[
        '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
        '/cam_img@sensor_msgs/msg/Image[gz.msgs.Image',
    ]
)
```

After this, `/scan`, `/cmd_vel`, and `/cam_img` exist in ROS2 exactly as they would from real hardware. Your `slam_node`, `odometry_node`, and `yolo_node` subscribe to these topics **without any modification**.

**Step 3 — Switching between simulator and real robot**

Use conditional launch arguments, not code changes:

```python
# full_system.launch.py
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition

use_sim = LaunchConfiguration('use_sim')

DeclareLaunchArgument('use_sim', default_value='true')

# Simulator bringup (only when use_sim:=true)
IncludeLaunchDescription(
    PythonLaunchDescriptionSource(gazebo_launch),
    condition=IfCondition(use_sim)
)

# Hardware bringup (only when use_sim:=false)
Node(package='puzzlebot_drivers', executable='hardware_interface',
     condition=UnlessCondition(use_sim))
```

```bash
# Development on laptop:
ros2 launch puzzlebot_bringup full_system.launch.py use_sim:=true

# Deployment on real robot:
ros2 launch puzzlebot_bringup full_system.launch.py use_sim:=false
```

### CoppeliaSim as an alternative

CoppeliaSim is easier to install on Windows/macOS for team members who cannot run Linux. It integrates via the `simROS2` plugin and publishes to the same topic namespace. Your class activities for Semana 3 explicitly use CoppeliaSim. If your team is already familiar with it, keep it — the topic-level abstraction means it is interchangeable with Gazebo from your nodes' perspective.

### Recommended development workflow

```
Phase 1 — Algorithm development
  Write node, test with pytest (pure Python, no ROS)
         ↓
Phase 2 — Node integration
  Run node with mock topic publishers (no simulator)
         ↓
Phase 3 — System integration
  Run full system in Gazebo (no physical robot)
         ↓
Phase 4 — Hardware validation
  Deploy to real robot with use_sim:=false
  Only tune YAML parameters (noise values, thresholds)
```

This pipeline means you only need access to the physical robot in Phase 4, and you arrive there with high confidence that the software works.

---

## 8. Development Roadmap

> **Note:** 6 weeks remaining (April–May 2026). All development runs in Gazebo simulation; hardware transfer is the final step only if time allows.

### Sprint breakdown

| Sprint | Goal | Key deliverable |
|---|---|---|
| 1 (week 1) | Topic plumbing + sim bringup | All nodes launch in Gazebo, topics verified with `ros2 topic echo` |
| 2 (week 2) | Odometry + EKF | Robot drives in Gazebo, pose tracked with <5 cm drift over 3 m |
| 3 (week 3) | SLAM | Occupancy grid generated in Gazebo corridor environment |
| 4 (week 4) | Perception | ArUco + YOLO node detects targets at ≥10 Hz in simulation |
| 5 (week 5) | Planning + control | A* path + steering controller executes full mission in Gazebo |
| 6 (week 6) | Full integration + (optional) hardware | Complete autonomous mission in Gazebo; tune YAML params on physical robot if available |

### Key references from your class

- **Probabilistic Robotics** — Thrun, Burgard, Fox (the foundational text for everything in this guide)
- **Robotics, Vision and Control** — Corke (Python-based, practical implementations)
- **BreezySLAM** — `github.com/simondlevy/BreezySLAM` (your MCL SLAM starting point)
- **CoreSLAM paper** — `researchgate.net/publication/228374722` (the algorithm under BreezySLAM)

---

## 9. Agent Development Guide — Working with These Nodes
### Ground rules for any agent or developer working on this codebase

Never hardcode physical constants — wheel radius, wheelbase, noise values, thresholds, and frequencies all live in the YAML files under puzzlebot_bringup/config/. If you are adding a new tunable value, declare it as a parameter first.
Never modify topic names in node source code — topic names are remapped at launch time via the launch files. The node always uses its local name (e.g., "odom_raw"); the launch file decides the actual namespace.
Never assume hardware is present — every node must be testable with a mock publisher on its input topics. If you write a node that segfaults or throws when no sensor data arrives, it is broken.
One responsibility per node — if a node is doing two conceptually distinct things, split it.
Language per node — follow this table strictly; do not deviate without a documented reason:

| Node | Language | Reason |
|---|---|---|
| `odometry_node` | C++ | 20–50 Hz tight loop — latency directly affects pose accuracy |
| `kalman_filter_node` | C++ | High-rate matrix math; Eigen is faster than NumPy |
| `steering_controller_node` | C++ | PID loop — microsecond jitter causes oscillation |
| `slam_node` | Python | MCL math is compute-heavy but not latency-sensitive; NumPy/BreezySLAM ecosystem |
| `yolo_node` | Python | TensorRT Python bindings are standard; ultralytics is Python-first |
| `aruco_node` | Python | OpenCV Python is fine; not latency-critical |
| `path_planner_node` | Python | A* runs once per goal, not in a tight loop |
| `state_machine_node` | Python | Logic-heavy; benefits from Python readability |
| `voice_node` | Python | NLP ecosystem is Python-only |

*Document version: April 2026 — Robots Autónomos ITESM*
*Hardware: Puzzlebot differential drive · Jetson Orin 2 GB · RPLIDAR · USB Camera*
