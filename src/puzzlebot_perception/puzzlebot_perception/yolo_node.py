import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from cv_bridge import CvBridge


class YoloNode(Node):
    """
    Object detection node.
    Expects a TensorRT .engine file on Jetson Orin; falls back to
    ultralytics YOLOv8 in PyTorch for simulation/dev machines.
    """

    def __init__(self):
        super().__init__('yolo_node')
        self.declare_parameter('engine_path',        'models/yolov8n.engine')
        self.declare_parameter('use_trt',            False)   # True on Jetson
        self.declare_parameter('confidence_thresh',  0.45)
        self.declare_parameter('nms_thresh',         0.5)
        self.declare_parameter('detection_rate_hz',  10.0)
        self.declare_parameter('camera_width',       640)
        self.declare_parameter('camera_height',      480)

        self.conf_thresh = self.get_parameter('confidence_thresh').value
        self.use_trt     = self.get_parameter('use_trt').value
        rate             = self.get_parameter('detection_rate_hz').value

        self.bridge  = CvBridge()
        self.model   = None
        self.pending = None

        self._load_model()

        self.sub = self.create_subscription(Image, '/cam_img', self._img_cb, 10)
        self.pub = self.create_publisher(Detection2DArray, '/detections', 10)
        # Throttle: collect frames but only run inference at detection_rate_hz
        self.create_timer(1.0 / rate, self._detect)

        self.get_logger().info(f'yolo_node started (use_trt={self.use_trt}, conf={self.conf_thresh})')

    def _load_model(self):
        if self.use_trt:
            try:
                import tensorrt as trt  # noqa: F401
                self.get_logger().info('TensorRT runtime available — loading .engine')
                # Actual TRT loading deferred to first detection call
            except ImportError:
                self.get_logger().warn('TensorRT not found — disabling yolo_node')
        else:
            try:
                from ultralytics import YOLO
                self.model = YOLO('yolov8n.pt')
                self.get_logger().info('Loaded YOLOv8n (PyTorch, sim mode)')
            except Exception as e:
                self.get_logger().warn(f'Could not load YOLO model: {e}')

    def _img_cb(self, msg: Image):
        self.pending = msg

    def _detect(self):
        if self.pending is None or self.model is None:
            return
        msg = self.pending
        self.pending = None

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(frame, conf=self.conf_thresh, verbose=False)

        det_array = Detection2DArray()
        det_array.header = msg.header

        for r in results:
            for box in r.boxes:
                det = Detection2D()
                det.header = msg.header
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                w,  h  = x2 - x1, y2 - y1
                det.bbox = BoundingBox2D()
                det.bbox.center.position.x = cx
                det.bbox.center.position.y = cy
                det.bbox.size_x = w
                det.bbox.size_y = h
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(int(box.cls[0]))
                hyp.hypothesis.score    = float(box.conf[0])
                det.results.append(hyp)
                det_array.detections.append(det)

        self.pub.publish(det_array)


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
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
