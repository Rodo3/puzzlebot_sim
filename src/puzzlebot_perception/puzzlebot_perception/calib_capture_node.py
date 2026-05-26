"""
Nodo 1/3 — Captura de imágenes para calibración intrínseca.

Modos de captura:
  auto_capture=true  — captura sola cuando:
    • El tablero está detectado
    • Pasaron >= capture_interval segundos desde la última captura
    • El tablero se movió >= min_corner_displacement px (diversidad)
  auto_capture=false — presiona SPACE para capturar manualmente

El preview se publica como tópico /calib/preview (Image) para verlo con:
  ros2 run rqt_image_view rqt_image_view /calib/preview
También abre ventana local si hay display disponible.

Uso (Checkerboard 9×6 esquinas internas, cuadro 2.6 cm):
  ros2 run puzzlebot_perception calib_capture_node

  # Modo manual:
  ros2 run puzzlebot_perception calib_capture_node \\
    --ros-args -p auto_capture:=false

  # ChArUco (alternativa):
  ros2 run puzzlebot_perception calib_capture_node \\
    --ros-args -p board_type:=charuco \\
               -p board_cols:=7 -p board_rows:=5 \\
               -p square_length:=0.026
"""

import os
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import cv2
import numpy as np

_LEGACY = not hasattr(cv2.aruco, 'ArucoDetector')


def _get_aruco_dict(dict_name: str):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))


def _get_detector_params():
    try:
        return cv2.aruco.DetectorParameters_create()
    except AttributeError:
        return cv2.aruco.DetectorParameters()


class CalibCaptureNode(Node):
    def __init__(self):
        super().__init__('calib_capture_node')

        self.declare_parameter('image_topic',             '/camera/image/compressed')
        self.declare_parameter('save_dir',                os.path.expanduser('~/calib_images'))
        self.declare_parameter('board_type',              'checkerboard')  # checkerboard | charuco
        self.declare_parameter('board_cols',               9)    # esquinas internas en X
        self.declare_parameter('board_rows',               6)    # esquinas internas en Y
        self.declare_parameter('square_length',            0.026)  # 2.6 cm
        self.declare_parameter('marker_length',            0.019)  # solo ChArUco
        self.declare_parameter('aruco_dict',               'DICT_4X4_50')
        self.declare_parameter('min_captures',             30)
        self.declare_parameter('target_captures',          50)
        self.declare_parameter('auto_capture',             True)
        self.declare_parameter('capture_interval',         2.0)
        self.declare_parameter('min_corner_displacement',  30.0)
        self.declare_parameter('window_width',             960)
        self.declare_parameter('window_height',            540)

        self.save_dir         = self.get_parameter('save_dir').value
        self.image_topic      = self.get_parameter('image_topic').value
        board_type            = self.get_parameter('board_type').value
        cols                  = self.get_parameter('board_cols').value
        rows                  = self.get_parameter('board_rows').value
        sq_len                = self.get_parameter('square_length').value
        mk_len                = self.get_parameter('marker_length').value
        dict_name             = self.get_parameter('aruco_dict').value
        self.min_captures     = self.get_parameter('min_captures').value
        self.target_captures  = self.get_parameter('target_captures').value
        self.auto_capture     = self.get_parameter('auto_capture').value
        self.capture_interval = self.get_parameter('capture_interval').value
        self.min_displacement = self.get_parameter('min_corner_displacement').value
        win_w                 = self.get_parameter('window_width').value
        win_h                 = self.get_parameter('window_height').value

        os.makedirs(self.save_dir, exist_ok=True)

        self.use_charuco       = (board_type == 'charuco')
        self.board_size        = (cols, rows)
        self.bridge            = CvBridge()
        self.frame             = None
        self._vis_frame        = None   # frame listo para mostrar, escrito por _tick
        self.count             = 0
        self.last_capture_t    = 0.0
        self.last_corners_mean = None
        self._manual_trigger   = False  # activado por SPACE en _poll_window
        self.has_display       = False
        self._no_frame_warned  = False

        if self.use_charuco:
            self._init_charuco(cols, rows, sq_len, mk_len, dict_name)
            self.get_logger().info(
                f'ChArUco {cols}×{rows}  cuadro={sq_len*100:.1f}cm  '
                f'marker={mk_len*100:.1f}cm  dict={dict_name}  '
                f'OpenCV {cv2.__version__} ({"legacy" if _LEGACY else "nueva"} API)'
            )
        else:
            self.get_logger().info(
                f'Checkerboard {cols}×{rows} esquinas internas  '
                f'cuadro={sq_len*100:.1f}cm  OpenCV {cv2.__version__}'
            )

        # QoS idéntico al image_viewer_node: BEST_EFFORT, depth=1
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        if 'compressed' in self.image_topic.lower():
            self.sub = self.create_subscription(
                CompressedImage, self.image_topic, self._cb_compressed, qos)
        else:
            self.sub = self.create_subscription(
                Image, self.image_topic, self._cb_raw, qos)

        self.pub_preview = self.create_publisher(Image, '/calib/preview', 10)

        # Crear ventana al inicio (como image_viewer_node)
        self._window_title = 'Calibracion  |  Q=salir  SPACE=captura manual'
        try:
            cv2.namedWindow(self._window_title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._window_title, win_w, win_h)
            self.has_display = True
        except Exception:
            self.get_logger().warn(
                'Sin display local. Usa rqt_image_view para ver el preview:\n'
                '  ros2 run rqt_image_view rqt_image_view /calib/preview'
            )

        # Timer de lógica (15 Hz) + timer de ventana separado (30 Hz, igual que image_viewer)
        self.create_timer(1.0 / 15.0, self._tick)
        self.create_timer(1.0 / 30.0, self._poll_window)

        mode = 'AUTO' if self.auto_capture else 'MANUAL (SPACE)'
        self.get_logger().info(
            f'Modo captura: {mode}  |  '
            f'intervalo={self.capture_interval:.1f}s  '
            f'desplaz_min={self.min_displacement:.0f}px\n'
            f'Preview: ros2 run rqt_image_view rqt_image_view /calib/preview\n'
            f'Guardando en: {self.save_dir}'
        )

    # ------------------------------------------------------------------
    def _init_charuco(self, cols, rows, sq_len, mk_len, dict_name):
        aruco_dict        = _get_aruco_dict(dict_name)
        self.aruco_dict   = aruco_dict
        self.aruco_params = _get_detector_params()
        if _LEGACY:
            self.board       = cv2.aruco.CharucoBoard_create(cols, rows, sq_len, mk_len, aruco_dict)
            self.charuco_det = None
        else:
            self.board       = cv2.aruco.CharucoBoard((cols, rows), sq_len, mk_len, aruco_dict)
            self.charuco_det = cv2.aruco.CharucoDetector(self.board)

    # ------------------------------------------------------------------
    def _cb_compressed(self, msg: CompressedImage):
        buf        = np.frombuffer(msg.data, np.uint8)
        self.frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def _cb_raw(self, msg: Image):
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    # ------------------------------------------------------------------
    # Timer principal: detección + decisión de captura + publicar preview
    # ------------------------------------------------------------------
    def _tick(self):
        if self.frame is None:
            if not self._no_frame_warned:
                self._no_frame_warned = True
                self.get_logger().warn(
                    f'Sin frames de {self.image_topic}. '
                    'Verifica que la cámara está publicando y que '
                    'ROS_DOMAIN_ID coincide en ambas máquinas.')
            return

        found, vis, corners_mean = self._detect(self.frame)
        now = time.time()

        do_capture = False
        reason_no  = ''

        if found:
            time_ok = (now - self.last_capture_t) >= self.capture_interval
            if not time_ok:
                remain    = self.capture_interval - (now - self.last_capture_t)
                reason_no = f'espera {remain:.1f}s'

            move_ok = True
            if self.last_corners_mean is not None and corners_mean is not None:
                disp = float(np.linalg.norm(corners_mean - self.last_corners_mean))
                if disp < self.min_displacement:
                    move_ok   = False
                    reason_no = f'mueve el tablero ({disp:.0f}/{self.min_displacement:.0f}px)'

            if self.auto_capture:
                do_capture = time_ok and move_ok
            elif self._manual_trigger:
                # Modo manual: respetar solo el intervalo mínimo
                do_capture           = time_ok
                self._manual_trigger = False

        vis = self._draw_overlay(vis, found, do_capture, reason_no)

        if do_capture:
            self._save_frame(corners_mean)

        # Publicar preview como tópico ROS
        try:
            preview_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            self.pub_preview.publish(preview_msg)
        except Exception:
            pass

        # Compartir frame con _poll_window (timer separado, como image_viewer_node)
        self._vis_frame = vis

    # ------------------------------------------------------------------
    # Timer de ventana: separado del timer de lógica (patrón image_viewer_node)
    # ------------------------------------------------------------------
    def _poll_window(self):
        if not self.has_display:
            return

        if self._vis_frame is not None:
            try:
                cv2.imshow(self._window_title, self._vis_frame)
            except cv2.error:
                self.has_display = False
                self.get_logger().warn(
                    'Sin display local. Usa rqt_image_view:\n'
                    '  ros2 run rqt_image_view rqt_image_view /calib/preview'
                )
                return

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            self.get_logger().info('Saliendo...')
            cv2.destroyAllWindows()
            rclpy.shutdown()
        elif key == ord(' ') and not self.auto_capture:
            self._manual_trigger = True

    # ------------------------------------------------------------------
    def _detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis  = frame.copy()
        if self.use_charuco:
            return self._detect_charuco(gray, vis)
        return self._detect_checkerboard(gray, vis)

    def _detect_checkerboard(self, gray, vis):
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                 cv2.CALIB_CB_NORMALIZE_IMAGE  |
                 cv2.CALIB_CB_FAST_CHECK)
        ret, corners = cv2.findChessboardCorners(gray, self.board_size, flags)
        corners_mean = None
        if ret:
            # Refinar a nivel sub-pixel: imprescindible para buena calibración
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners  = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(vis, self.board_size, corners, ret)
            corners_mean = corners.reshape(-1, 2).mean(axis=0)
        return ret, vis, corners_mean

    def _detect_charuco(self, gray, vis):
        found        = False
        corners_mean = None
        if _LEGACY:
            m_c, m_i, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params)
            if m_i is not None and len(m_i) >= 4:
                _, ch_c, ch_i = cv2.aruco.interpolateCornersCharuco(
                    m_c, m_i, gray, self.board)
                if ch_i is not None and len(ch_i) >= 4:
                    found = True
                    cv2.aruco.drawDetectedMarkers(vis, m_c, m_i)
                    cv2.aruco.drawDetectedCornersCharuco(vis, ch_c, ch_i)
                    corners_mean = ch_c.reshape(-1, 2).mean(axis=0)
        else:
            ch_c, ch_i, _, _ = self.charuco_det.detectBoard(gray)
            if ch_i is not None and len(ch_i) >= 4:
                found = True
                cv2.aruco.drawDetectedCornersCharuco(vis, ch_c, ch_i)
                corners_mean = ch_c.reshape(-1, 2).mean(axis=0)
        return found, vis, corners_mean

    # ------------------------------------------------------------------
    def _draw_overlay(self, vis, found, do_capture, reason_no):
        h, w  = vis.shape[:2]
        label = 'ChArUco' if self.use_charuco else 'Checkerboard'

        if found:
            if do_capture:
                txt, color = f'{label} — CAPTURANDO!', (0, 255, 255)
            elif reason_no:
                txt, color = f'{label} OK — {reason_no}', (0, 200, 255)
            else:
                txt, color = f'{label} DETECTADO', (0, 255, 0)
        else:
            txt, color = f'Buscando {label}...', (0, 50, 255)

        cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # Barra de progreso inferior
        pct   = min(self.count / max(self.target_captures, 1), 1.0)
        bar_w = int((w - 20) * pct)
        cv2.rectangle(vis, (10, h - 35), (w - 10, h - 20), (60, 60, 60), -1)
        cv2.rectangle(vis, (10, h - 35), (10 + bar_w, h - 20), (0, 200, 80), -1)
        cv2.putText(vis,
                    f'{self.count}/{self.target_captures} capturas  (min {self.min_captures})',
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 100) if self.count >= self.min_captures else (200, 200, 200), 1)
        cv2.putText(vis,
                    'Variar: inclinado, rotado, esquinas del frame, cerca, lejos',
                    (10, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

        if do_capture:
            overlay = vis.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)

        return vis

    # ------------------------------------------------------------------
    def _save_frame(self, corners_mean):
        path = os.path.join(self.save_dir, f'calib_{self.count:04d}.png')
        cv2.imwrite(path, self.frame)
        self.last_capture_t    = time.time()
        self.last_corners_mean = corners_mean
        self.count += 1
        self.get_logger().info(f'[{self.count}/{self.target_captures}] {path}')
        if self.count == self.min_captures:
            self.get_logger().info(
                f'Mínimo alcanzado ({self.min_captures} imágenes). '
                f'Sigue moviendo el tablero hasta {self.target_captures}.')
        if self.count >= self.target_captures:
            self.get_logger().info(
                f'¡{self.target_captures} capturas completas!  '
                'Presiona Ctrl+C o cierra el nodo.')

    # ------------------------------------------------------------------
    def destroy_node(self):
        if self.has_display:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CalibCaptureNode()
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
