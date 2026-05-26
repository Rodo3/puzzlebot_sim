# puzzlebot_perception

Owns camera feed, visual detection, and camera calibration. Does **not** make navigation decisions — it publishes data for `puzzlebot_localization` and `puzzlebot_planning` to consume.

Put here:
- Camera capture/bridge nodes.
- ArUco marker detection.
- YOLO/TensorRT object detection.
- Camera calibration pipeline.
- Perception messages derived from images.

Do not put here:
- Localization filters (only publish raw visual measurements like `/aruco/poses`).
- Navigation or control decisions.

---

## Nodes

### `image_viewer_node`
Displays the camera feed in an OpenCV window. Supports optional distortion correction using the calibration YAML — no separate `calib_apply_node` needed.

```bash
# Raw feed (no correction)
ros2 run puzzlebot_perception image_viewer_node

# With distortion correction (uses installed camera_calibration.yaml)
ros2 run puzzlebot_perception image_viewer_node --ros-args -p rectify:=true
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `topic` | `/camera/image/compressed` | Auto-detects CompressedImage vs raw Image by topic name |
| `rectify` | `false` | Apply distortion correction using `calib_yaml` |
| `calib_yaml` | installed `camera_calibration.yaml` | Calibration YAML path |
| `window_title` | `Puzzlebot Camera` | OpenCV window title |
| `show_fps` | `true` | Overlay FPS, size, and timestamp |
| `window_width/height` | `960 / 480` | Initial window size |

---

### `calib_capture_node`
Captures chessboard images for intrinsic camera calibration. Board: **9×6 internal corners, 2.6 cm squares**.

```bash
# Auto mode (default): captures automatically when board moves
ros2 run puzzlebot_perception calib_capture_node

# Manual mode: press SPACE to capture
ros2 run puzzlebot_perception calib_capture_node --ros-args -p auto_capture:=false
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `board_cols / board_rows` | `9 / 6` | Internal corner count |
| `square_length` | `0.026` | Square size in meters |
| `save_dir` | `~/calib_images` | Output directory for PNGs |
| `target_captures` | `50` | Target image count |
| `auto_capture` | `true` | Auto vs manual (SPACE) capture |
| `capture_interval` | `2.0 s` | Minimum time between captures |
| `min_corner_displacement` | `30 px` | Minimum board movement to trigger capture |

Preview is also published on `/calib/preview` for `rqt_image_view`.

---

### `calib_compute_node`
Reads saved PNGs and computes intrinsic parameters (K, D). Displays best/worst image comparison.

```bash
ros2 run puzzlebot_perception calib_compute_node
# Output: ~/calib_images/camera_calibration.yaml
```

Quality guide: RMS < 0.5 px excellent · 0.5–1.0 px acceptable · > 1.0 px retake images.

---

### `calib_apply_node`
Applies calibration to the camera stream and publishes `/cam_img_rect` + `/cam_info`. Used when other nodes need a rectified image topic (not required for `image_viewer_node` or `aruco_node` which load the YAML directly).

```bash
ros2 run puzzlebot_perception calib_apply_node
```

---

### `aruco_node`
Detects ArUco markers, estimates 6-DOF pose via `solvePnP`, and computes the robot's absolute pose from the known marker map.

Publishes `/aruco/poses` (`geometry_msgs/PoseArray`) → consumed by `kalman_filter_node`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_topic` | `/camera/image/compressed` | Camera source |
| `camera_info_file` | installed `camera_calibration.yaml` | Intrinsics |
| `extrinsics_file` | installed `camera_extrinsics.yaml` | Camera mount pose |
| `marker_map_file` | installed `aruco_map.yaml` | Known marker poses in map frame |
| `marker_length` | `0.08 m` | Physical marker side length |
| `max_detection_distance` | `2.0 m` | Ignore farther markers |

---

### `yolo_node`
Object detection with YOLOv8 — PyTorch in simulation, TensorRT INT8 on Jetson Orin.

---

## Camera Calibration Workflow

```
Step 1 — Capture
  ros2 run puzzlebot_perception calib_capture_node
  → 50 PNG images saved to ~/calib_images/

Step 2 — Compute
  ros2 run puzzlebot_perception calib_compute_node
  → ~/calib_images/camera_calibration.yaml  (RMS: 0.96 px)

Step 3 — Install
  cp ~/calib_images/camera_calibration.yaml \
     src/puzzlebot_bringup/config/camera_calibration.yaml
  colcon build --packages-select puzzlebot_bringup

Step 4 — Verify
  ros2 run puzzlebot_perception image_viewer_node --ros-args -p rectify:=true
  → Straight lines in the scene should appear straight in the window
```

## Topic Map

```
/camera/image/compressed ──→ image_viewer_node  → OpenCV window (optional rectification)
                         ├──→ aruco_node         → /aruco/poses → kalman_filter_node
                         └──→ calib_apply_node   → /cam_img_rect, /cam_info
                               yolo_node          → /detections
```
