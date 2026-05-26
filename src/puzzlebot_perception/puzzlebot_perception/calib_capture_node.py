"""
Nodo 1/3 — Captura de imágenes para calibración intrínseca.

Modos de captura:
  auto_capture=true  (default) — captura sola cuando:
    • El tablero está detectado
    • Pasaron >= capture_interval segundos desde la última captura
    • El tablero se movió >= min_corner_displacement px (diversidad)
  auto_capture=false — presiona SPACE para capturar manualmente

El preview se publica como tópico /calib/preview (Image) para verlo con:
  ros2 run rqt_image_view rqt_image_view /calib/preview
También intenta abrir ventana local (funciona si tienes display).

Uso:
  ros2 run puzzlebot_perception calib_capture_node \\
    --ros-args -p square_length:=0.026 -p target_captures:=50

  # Para checkerboard:
  ros2 run puzzlebot_perception calib_capture_node \\
    --ros-args -p board_type:=checkerboard \\
               -p board_cols:=7 -p board_rows:=5 \\
               -p square_length:=0.026
"""

import os
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
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
        self.declare_parameter('board_type',              'charuco')
        self.declare_parameter('board_cols',               7)
        self.declare_parameter('board_rows',               5)
        self.declare_parameter('square_length',            0.026)   # 2.6 cm
        self.declare_parameter('marker_length',            0.019)   # ~73% de sq para ChArUco
        self.declare_parameter('aruco_dict',               'DICT_4X4_50')
        self.declare_parameter('min_captures',             30)
        self.declare_parameter('target_captures',          50)
        self.declare_parameter('auto_capture',             True)
        self.declare_parameter('capture_interval',         2.0)     # segundos entre capturas
        self.declare_parameter('min_corner_displacement',  30.0)    # px mínimo de movimiento

        self.save_dir           = self.get_parameter('save_dir').value
        self.image_topic        = self.get_parameter('image_topic').value
        board_type              = self.get_parameter('board_type').value
        cols                    = self.get_parameter('board_cols').value
        rows                    = self.get_parameter('board_rows').value
        sq_len                  = self.get_parameter('square_length').value
        mk_len                  = self.get_parameter('marker_length').value
        dict_name               = self.get_parameter('aruco_dict').value
        self.min_captures       = self.get_parameter('min_captures').value
        self.target_captures    = self.get_parameter('target_captures').value
        self.auto_capture       = self.get_parameter('auto_capture').value
        self.capture_interval   = self.get_parameter('capture_interval').value
        self.min_displacement   = self.get_parameter('min_corner_displacement').value

        os.makedirs(self.save_dir, exist_ok=True)

        self.use_charuco      = (board_type == 'charuco')
        self.board_size       = (cols, rows)
        self.bridge           = CvBridge()
        self.frame            = None
        self.count            = 0
        self.last_capture_t   = 0.0
        self.last_corners_mean = None
        self.has_display      = True   # se desactiva si cv2.imshow falla

        if self.use_charuco:
            self._init_charuco(cols, rows, sq_len, mk_len, dict_name)
            self.get_logger().info(
                f'ChArUco {cols}x{rows}  cuadro={sq_len*100:.1f}cm  '
                f'marker={mk_len*100:.1f}cm  dict={dict_name}  '
                f'OpenCV {cv2.__version__} ({"legacy" if _LEGACY else "nueva"} API)'
            )
        else:
            self.get_logger().info(
                f'Checkerboard {cols}x{rows} esquinas internas  '
                f'cuadro={sq_len*100:.1f}cm'
            )

        if 'compressed' in self.image_topic.lower():
            self.sub = self.create_subscription(
                CompressedImage, self.image_topic, self._cb_compressed,
                qos_profile_sensor_data)
        else:
            self.sub = self.create_subscription(
                Image, self.image_topic, self._cb_raw,
                qos_profile_sensor_data)

        self.pub_preview   = self.create_publisher(Image, '/calib/preview', 10)
        self._no_frame_warned = False
        self.create_timer(1.0 / 15.0, self._tick)   # 15 Hz es suficiente para preview

        mode = 'AUTO' if self.auto_capture else 'MANUAL (SPACE)'
        self.get_logger().info(
            f'Modo captura: {mode}  |  '
            f'intervalo={self.capture_interval:.1f}s  '
            f'desplaz_min={self.min_displacement:.0f}px\n'
            f'Preview: ros2 run rqt_image_view rqt_image_view /calib/preview\n'
            f'dir={self.save_dir}'
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
    # Tick principal: detectar, decidir captura, publicar preview
    # ------------------------------------------------------------------
    def _tick(self):
        if self.frame is None:
            if not self._no_frame_warned:
                self._no_frame_warned = True
                self.get_logger().warn(
                    f'Sin frames de {self.image_topic}. '
                    'Verifica que la Jetson está publicando y que '
                    'ROS_DOMAIN_ID coincide en ambas máquinas.')
            return

        found, vis, corners_mean = self._detect(self.frame)
        now = time.time()

        # --- decidir si capturar ---
        do_capture = False
        reason_no  = ''

        if found:
            time_ok = (now - self.last_capture_t) >= self.capture_interval
            if not time_ok:
                remain = self.capture_interval - (now - self.last_capture_t)
                reason_no = f'espera {remain:.1f}s'

            move_ok = True
            if self.last_corners_mean is not None and corners_mean is not None:
                disp = float(np.linalg.norm(corners_mean - self.last_corners_mean))
                if disp < self.min_displacement:
                    move_ok = False
                    reason_no = f'mueve el tablero ({disp:.0f}/{self.min_displacement:.0f}px)'

            if self.auto_capture:
                do_capture = time_ok and move_ok
            # Manual: do_capture se activa por teclado en _show_window

        # --- overlay de estado ---
        vis = self._draw_overlay(vis, found, do_capture, reason_no, now)

        # --- capturar si aplica ---
        if do_capture:
            self._save_frame(corners_mean)

        # --- publicar preview como tópico ---
        try:
            msg        = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            self.pub_preview.publish(msg)
        except Exception:
            pass

        # --- ventana local (opcional, puede fallar en WSL2 sin display) ---
        self._show_window(vis, found, corners_mean)

    # ------------------------------------------------------------------
    def _detect(self, frame):
        """Devuelve (found, vis_frame, corners_mean_px)."""
        gray         = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis          = frame.copy()
        found        = False
        corners_mean = None

        if self.use_charuco:
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
        else:
            ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)
            if ret:
                found = True
                cv2.drawChessboardCorners(vis, self.board_size, corners, ret)
                corners_mean = corners.reshape(-1, 2).mean(axis=0)

        return found, vis, corners_mean

    # ------------------------------------------------------------------
    def _draw_overlay(self, vis, found, do_capture, reason_no, now):
        h, w = vis.shape[:2]

        # Barra de estado superior
        label = 'ChArUco' if self.use_charuco else 'Checkerboard'
        if found:
            if do_capture:
                txt   = f'{label} — CAPTURANDO!'
                color = (0, 255, 255)
            elif reason_no:
                txt   = f'{label} OK — {reason_no}'
                color = (0, 200, 255)
            else:
                txt   = f'{label} DETECTADO'
                color = (0, 255, 0)
        else:
            txt   = f'Buscando {label}...'
            color = (0, 50, 255)
        cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # Contador de capturas con barra de progreso
        pct  = min(self.count / max(self.target_captures, 1), 1.0)
        bar_w = int((w - 20) * pct)
        cv2.rectangle(vis, (10, h - 35), (w - 10, h - 20), (60, 60, 60), -1)
        cv2.rectangle(vis, (10, h - 35), (10 + bar_w, h - 20), (0, 200, 80), -1)
        cv2.putText(vis,
                    f'{self.count}/{self.target_captures} capturas  (min {self.min_captures})',
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 100) if self.count >= self.min_captures else (200, 200, 200), 1)

        # Consejos en la parte inferior
        tips = [
            'Variar: inclinado, rotado, esquinas del frame, cerca, lejos',
        ]
        for i, tip in enumerate(tips):
            cv2.putText(vis, tip, (10, h - 5 - 15 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

        # Flash verde al capturar
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
    def _show_window(self, vis, found, corners_mean):
        """Intenta mostrar ventana local; desactiva silenciosamente si falla."""
        if not self.has_display:
            return
        try:
            cv2.imshow('Calibracion (Q=salir)', vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                cv2.destroyAllWindows()
                self.get_logger().info('Saliendo...')
                rclpy.shutdown()
            elif key == ord(' ') and not self.auto_capture and found:
                self._save_frame(corners_mean)
        except cv2.error:
            self.has_display = False
            self.get_logger().warn(
                'Sin display local. Usa rqt_image_view para ver el preview:\n'
                '  ros2 run rqt_image_view rqt_image_view /calib/preview'
            )

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
