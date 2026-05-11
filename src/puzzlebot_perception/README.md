# puzzlebot_perception

Owns camera and visual detection.

Put here:
- Camera capture/bridge nodes.
- ArUco marker detection.
- YOLO/TensorRT object detection.
- Perception messages derived from images.

Do not put here:
- Localization filters, except publishing raw visual measurements.
- Navigation or control decisions.

Expected role:
- Publish measurements such as `/aruco/poses` or object detections.
- Let localization/control packages decide how to consume those measurements.
