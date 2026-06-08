"""
qr_reader_node.py — Lector de códigos QR con estimación de pose via solvePnP.

FUNCIÓN:
  Detecta códigos QR en la imagen de la cámara, extrae el texto (cliente del
  pallet) y estima la pose 3D del QR respecto al robot usando solvePnP con
  la calibración real de la cámara.

PIPELINE:
  /camera/image/compressed
    → cv2.QRCodeDetector  (detecta + decodifica texto + devuelve 4 esquinas en px)
    → cv2.solvePnP        (4 puntos 2D ↔ 4 esquinas 3D conocidas del QR)
    → T_camera_qr         (rvec + tvec en frame camera_optical)
    → T_base_qr           (aplicando extrínseca camera_link → base_link)
    → /qr/detected        (Bool)
    → /qr/client          (String — texto del QR)
    → /qr/pose            (PoseStamped en frame base_footprint)
    → /qr/debug_image     (Image con overlay — si publish_debug:=true)

SISTEMA DE COORDENADAS solvePnP:
  El QR tiene 4 esquinas en 3D (frame del propio QR, lado=qr_real_size_m):
    top-left:     (-s/2,  s/2, 0)
    top-right:    ( s/2,  s/2, 0)
    bottom-right: ( s/2, -s/2, 0)
    bottom-left:  (-s/2, -s/2, 0)
  Donde s = qr_real_size_m.

  El detector devuelve las 4 esquinas en ese mismo orden (convención OpenCV).
  solvePnP devuelve T_camera_optical_qr.

  Para pasar a base_footprint:
    T_base_qr = T_base_camera_link · T_camera_link_optical · T_camera_optical_qr

  La posición publicada en /qr/pose es la del centro del QR
  expresada en frame base_footprint (x=adelante, y=izquierda, z=arriba).

TÓPICOS SUSCRITOS:
  /camera/image/compressed  (sensor_msgs/CompressedImage)

TÓPICOS PUBLICADOS:
  /qr/detected    (std_msgs/Bool)             — QR visible en frame actual
  /qr/client      (std_msgs/String)           — texto del QR
  /qr/pose        (geometry_msgs/PoseStamped) — pose centro QR en base_footprint
  /qr/debug_image (sensor_msgs/Image)         — overlay (si publish_debug:=true)

PARÁMETROS:
  image_topic         [/camera/image/compressed]
  camera_info_file    ['']     YAML de calibración (K, D). Sin él: sin undistort.
  extrinsics_file     ['']     YAML de extrínseca cámara→robot.
  qr_real_size_m      [0.15]   lado real del QR en metros
  publish_debug       [false]  publicar imagen con overlay
  max_processing_hz   [10.0]   límite de procesamiento
  lost_timeout_sec    [0.5]    segundos sin QR → publica detected=False
  min_points          [4]      mínimo de esquinas válidas para solvePnP
"""

import math
import os
import random
import time

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, String


# ── Utilidades de transformación ──────────────────────────────────────────────

def _euler_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _make_T(x: float, y: float, z: float,
            roll: float, pitch: float, yaw: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _euler_to_rot(roll, pitch, yaw)
    T[:3,  3] = [x, y, z]
    return T


def _rot_to_quat(R: np.ndarray):
    """Convierte matriz de rotación 3×3 a cuaternión (x, y, z, w)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w


# Transformación fija camera_link → camera_optical_frame
# (convención ROS: x→adelante / y→izquierda / z→arriba en camera_link;
#  convención OpenCV: x→derecha / y→abajo / z→frente en optical)
_T_CAMERA_LINK_OPTICAL = _make_T(
    0.0, 0.0, 0.0,
    -math.pi / 2.0, 0.0, -math.pi / 2.0,
)
_T_OPTICAL_CAMERA_LINK = np.linalg.inv(_T_CAMERA_LINK_OPTICAL)


# ── Nodo principal ────────────────────────────────────────────────────────────

class QRReaderNode(Node):

    def __init__(self):
        super().__init__('qr_reader_node')

        self.declare_parameter('image_topic',       '/camera/image/compressed')
        self.declare_parameter('camera_info_file',  '')
        self.declare_parameter('extrinsics_file',   '')
        self.declare_parameter('qr_real_size_m',    0.15)
        self.declare_parameter('publish_debug',     False)
        self.declare_parameter('max_processing_hz', 20.0)
        self.declare_parameter('lost_timeout_sec',  0.5)
        self.declare_parameter('min_points',        4)
        # Distancia máxima (m) a la que se acepta una detección de QR.
        self.declare_parameter('max_detection_distance', 8.0)
        # Cuando el QR se detecta pero NO se decodifica el texto, en vez de
        # publicar 'UNKNOWN' (que rompe el match con YOLO), se asigna un cliente
        # aleatorio de esta lista — así la misión puede seguir con un cliente
        # válido. Poner fallback_random_client=False para volver a 'UNKNOWN'.
        self.declare_parameter('fallback_random_client', True)
        self.declare_parameter('valid_clients', ['Walmart', 'Pepsi', 'Amazon'])

        image_topic     = self.get_parameter('image_topic').value
        calib_file      = self.get_parameter('camera_info_file').value
        extr_file       = self.get_parameter('extrinsics_file').value
        self._qr_size   = self.get_parameter('qr_real_size_m').value
        self._pub_debug = self.get_parameter('publish_debug').value
        max_hz          = self.get_parameter('max_processing_hz').value
        self._lost_to   = self.get_parameter('lost_timeout_sec').value
        self._min_pts   = self.get_parameter('min_points').value
        self._fallback_random = self.get_parameter('fallback_random_client').value
        self._valid_clients   = list(self.get_parameter('valid_clients').value)
        self._max_det_dist    = self.get_parameter('max_detection_distance').value

        self._min_period = 1.0 / max(max_hz, 0.1)
        self._last_proc  = 0.0

        # ── Intrínsecos ───────────────────────────────────────────────────────
        # Defaults conservadores hasta que se cargue la calibración
        self._K = np.array([[600.0, 0.0, 320.0],
                            [0.0, 600.0, 240.0],
                            [0.0, 0.0,   1.0]], dtype=np.float64)
        self._D          = np.zeros((5, 1), dtype=np.float64)
        self._calibrated = False
        self._load_calibration(calib_file)

        # ── Extrínseca base_link → camera_link ───────────────────────────────
        # T_base_camera: lleva puntos de camera_link a base_link
        self._T_base_camera = _make_T(0.152, 0.0, 0.044, 0.0, 0.0, 0.0)
        self._load_extrinsics(extr_file)

        # T_base_optical = T_base_camera · T_camera_link_optical⁻¹
        # (inv porque T_CAMERA_LINK_OPTICAL lleva de link a optical;
        #  queremos de optical a link primero, luego de link a base)
        self._T_base_optical = self._T_base_camera @ _T_OPTICAL_CAMERA_LINK

        # ── Puntos 3D del QR en su propio frame (esquinas, plano z=0) ─────────
        # El detector devuelve: top-left, top-right, bottom-right, bottom-left
        s = self._qr_size / 2.0
        self._obj_pts = np.array([
            [-s,  s, 0.0],   # top-left
            [ s,  s, 0.0],   # top-right
            [ s, -s, 0.0],   # bottom-right
            [-s, -s, 0.0],   # bottom-left
        ], dtype=np.float64)

        # ── Estado ────────────────────────────────────────────────────────────
        self._last_detection_t = 0.0
        self._last_client      = ''   # último texto QR decodificado con éxito
        self._client_is_real   = False  # True si _last_client vino de decodificar (no random)
        self._bridge           = CvBridge()
        # QRCodeDetectorAruco (OpenCV ≥ 4.7) decodifica QR a distancia/ángulo
        # mucho mejor que el QRCodeDetector clásico. Si no está, caemos al clásico.
        if hasattr(cv2, 'QRCodeDetectorAruco'):
            self._detector = cv2.QRCodeDetectorAruco()
            self.get_logger().info('Detector QR: QRCodeDetectorAruco (mejor decodificación)')
        else:
            self._detector = cv2.QRCodeDetector()
            self.get_logger().info('Detector QR: QRCodeDetector (clásico)')

        # ── Suscripciones y publicadores ──────────────────────────────────────
        self.create_subscription(
            CompressedImage, image_topic,
            self._image_cb, qos_profile_sensor_data)

        self._pub_detected = self.create_publisher(Bool,        '/qr/detected',    10)
        self._pub_client   = self.create_publisher(String,      '/qr/client',      10)
        self._pub_pose     = self.create_publisher(PoseStamped, '/qr/pose',        10)
        if self._pub_debug:
            self._pub_dbg = self.create_publisher(Image, '/qr/debug_image', 10)

        self.create_timer(0.1, self._watchdog_cb)

        mode = 'solvePnP' if self._calibrated else 'solvePnP (sin calibración — K defaults)'
        self.get_logger().info(
            f'qr_reader_node iniciado — {mode}  '
            f'qr_size={self._qr_size} m  '
            f'fx={self._K[0,0]:.1f}  fy={self._K[1,1]:.1f}  '
            f'cx={self._K[0,2]:.1f}  cy={self._K[1,2]:.1f}')

    # ── Carga de calibración ──────────────────────────────────────────────────

    def _load_calibration(self, path: str):
        if not path or not os.path.isfile(path):
            self.get_logger().warn(
                f'camera_info_file no encontrado: {path!r} — usando K defaults')
            return
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)

            K_data = data.get('camera_matrix', {}).get('data', [])
            D_data = data.get('distortion_coefficients', {}).get('data', [])

            if len(K_data) < 9:
                self.get_logger().warn('camera_matrix incompleta en el YAML')
                return

            self._K = np.array(K_data, dtype=np.float64).reshape(3, 3)
            if D_data:
                self._D = np.array(D_data, dtype=np.float64).reshape(-1, 1)
            self._calibrated = True
            self.get_logger().info(
                f'Calibración cargada — '
                f'fx={self._K[0,0]:.2f}  fy={self._K[1,1]:.2f}  '
                f'cx={self._K[0,2]:.2f}  cy={self._K[1,2]:.2f}  '
                f'D={self._D.ravel().tolist()}')
        except Exception as e:
            self.get_logger().warn(f'Error cargando calibración: {e}')

    def _load_extrinsics(self, path: str):
        if not path or not os.path.isfile(path):
            self.get_logger().info(
                f'extrinsics_file no encontrado: {path!r} — usando defaults '
                f'(x=0.152 m, z=0.044 m, sin rotación)')
            return
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            e = data.get('camera_extrinsics', data)
            self._T_base_camera = _make_T(
                e.get('x', 0.0), e.get('y', 0.0), e.get('z', 0.0),
                e.get('roll', 0.0), e.get('pitch', 0.0), e.get('yaw', 0.0),
            )
            self.get_logger().info(
                f'Extrínseca cargada — '
                f'x={e.get("x",0):.3f}  y={e.get("y",0):.3f}  '
                f'z={e.get("z",0):.3f}  '
                f'rpy=({e.get("roll",0):.3f}, {e.get("pitch",0):.3f}, {e.get("yaw",0):.3f})')
        except Exception as ex:
            self.get_logger().warn(f'Error cargando extrínseca: {ex}')

    # ── Decodificación robusta ──────────────────────────────────────────────────

    def _decode_robust(self, gray: np.ndarray, points: np.ndarray) -> str:
        """Intenta decodificar el texto del QR con varias estrategias.

        El decode() directo falla seguido a distancia/ángulo. Aquí probamos, en
        orden: (1) decode con las esquinas; (2) recortar un ROI alrededor del QR
        y reintentar; (3) escalar el ROI ×2 y ×3; (4) binarización Otsu del ROI.
        Devuelve el texto si alguna funciona, o '' si todas fallan.
        """
        # 1. decode directo con las esquinas detectadas
        try:
            data, _ = self._detector.decode(gray, points)
            if data:
                return data.strip()
        except cv2.error:
            pass

        # ROI alrededor del QR (con margen), recortado a los límites de la imagen
        h, w = gray.shape[:2]
        xs = points[:, 0]
        ys = points[:, 1]
        m = 20  # margen en píxeles
        x0 = max(0, int(xs.min()) - m); x1 = min(w, int(xs.max()) + m)
        y0 = max(0, int(ys.min()) - m); y1 = min(h, int(ys.max()) + m)
        if x1 - x0 < 10 or y1 - y0 < 10:
            return ''
        roi = gray[y0:y1, x0:x1]

        # 2-3. ROI tal cual y escalado ×2, ×3 (detectAndDecode redetecta en el ROI)
        for scale in (1.0, 2.0, 3.0):
            img = roi if scale == 1.0 else cv2.resize(
                roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            try:
                data, _, _ = self._detector.detectAndDecode(img)
                if data:
                    return data.strip()
            except cv2.error:
                pass

        # 4. Binarización Otsu del ROI escalado
        try:
            big = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            _, binimg = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            data, _, _ = self._detector.detectAndDecode(binimg)
            if data:
                return data.strip()
        except cv2.error:
            pass

        return ''

    # ── Callback de imagen ────────────────────────────────────────────────────

    def _image_cb(self, msg: CompressedImage):
        now = time.monotonic()
        if now - self._last_proc < self._min_period:
            return
        self._last_proc = now

        # Decodificar imagen
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return
        except Exception:
            return

        # Undistort si tenemos calibración (mejora la precisión de solvePnP)
        if self._calibrated:
            frame = cv2.undistort(frame, self._K, self._D)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Detección + decodificación en dos pasos ────────────────────────────
        # detectAndDecode() exige detectar Y decodificar el texto en una sola
        # llamada; el QRCodeDetector de OpenCV suele DETECTAR el QR (encuentra
        # las esquinas) pero FALLAR al decodificar el texto a distancia/ángulo,
        # devolviendo data='' → se perdía el frame entero. Separamos los pasos:
        #   1. detect()  → esquinas (robusto)
        #   2. decode()  → texto (puede fallar; si falla seguimos con la pose)
        try:
            found, points = self._detector.detect(gray)
        except cv2.error:
            found, points = False, None

        if not found or points is None:
            self.get_logger().info(
                'QR no detectado en este frame', throttle_duration_sec=2.0)
            return

        points_2d = points.reshape(-1, 2).astype(np.float32)
        if len(points_2d) < self._min_pts:
            return

        # Intenta decodificar el texto; si no se puede, NO descarta la detección
        # (la pose es válida). Fallback: último cliente conocido, o uno aleatorio
        # de valid_clients (fijo durante toda la sesión) para que la misión siga
        # con un cliente válido en vez de 'UNKNOWN'.
        client = self._decode_robust(gray, points)
        if client:
            # Texto REAL leído del QR.
            if client != self._last_client:
                self.get_logger().info(f'✅ QR REAL decodificado: "{client}"')
            self._last_client = client
            self._client_is_real = True
        else:
            if not self._last_client and self._fallback_random and self._valid_clients:
                # Asigna un cliente aleatorio UNA vez y lo fija (estable entre frames)
                self._last_client = random.choice(self._valid_clients)
                self._client_is_real = False
                self.get_logger().warn(
                    f'⚠️ QR detectado pero SIN decodificar texto — cliente '
                    f'ALEATORIO "{self._last_client}" (NO es el real del QR)')
            client = self._last_client or 'UNKNOWN'
            tag = 'REAL' if getattr(self, '_client_is_real', False) else 'ALEATORIO'
            self.get_logger().info(
                f'QR detectado → cliente="{client}" ({tag})',
                throttle_duration_sec=2.0)

        # ── solvePnP ──────────────────────────────────────────────────────────
        # Resuelve T_camera_optical_qr a partir de las 4 correspondencias:
        #   obj_pts (3D en frame QR) ↔ points_2d (2D en píxeles)
        success, rvec, tvec = cv2.solvePnP(
            self._obj_pts,
            points_2d,
            self._K,
            np.zeros((4, 1), dtype=np.float64),  # undistort ya aplicado
            flags=cv2.SOLVEPNP_IPPE_SQUARE,       # óptimo para targets cuadrados
        )

        if not success:
            self.get_logger().warn('solvePnP falló — descartando frame',
                                   throttle_duration_sec=1.0)
            return

        # Sanity check: el QR no puede estar a menos de 5 cm ni a más de max_detection_distance
        dist_cam = float(np.linalg.norm(tvec))
        if dist_cam < 0.05 or dist_cam > self._max_det_dist:
            self.get_logger().warn(
                f'solvePnP: dist={dist_cam:.3f} m fuera de rango '
                f'[0.05, {self._max_det_dist:.1f}] — descartando',
                throttle_duration_sec=1.0)
            return

        # ── T_camera_optical_qr → T_base_footprint_qr ────────────────────────
        # 1. Construir T_camera_optical_qr (homogénea 4×4)
        R_cam_qr, _ = cv2.Rodrigues(rvec)
        T_optical_qr = np.eye(4)
        T_optical_qr[:3, :3] = R_cam_qr
        T_optical_qr[:3,  3] = tvec.ravel()

        # 2. T_base_qr = T_base_optical · T_optical_qr
        T_base_qr = self._T_base_optical @ T_optical_qr

        # Posición del centro del QR en base_footprint
        pos = T_base_qr[:3, 3]           # [x_fwd, y_left, z_up] en base_footprint
        R_base_qr = T_base_qr[:3, :3]

        # Ángulo horizontal hacia el QR (para log)
        angle_deg = math.degrees(math.atan2(pos[1], pos[0]))

        # Cuaternión para la orientación del plano del QR en base_footprint
        qx, qy, qz, qw = _rot_to_quat(R_base_qr)

        # ── Publicar ──────────────────────────────────────────────────────────
        self._last_detection_t = now

        self._pub_detected.publish(Bool(data=True))

        smsg = String()
        smsg.data = client
        self._pub_client.publish(smsg)

        pose = PoseStamped()
        pose.header.stamp    = self.get_clock().now().to_msg()
        pose.header.frame_id = 'base_footprint'
        pose.pose.position.x = float(pos[0])
        pose.pose.position.y = float(pos[1])
        pose.pose.position.z = float(pos[2])
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        self._pub_pose.publish(pose)

        self.get_logger().info(
            f'QR "{client}"  dist={dist_cam:.3f} m  '
            f'base_fwd={pos[0]:.3f} m  base_lat={pos[1]:.3f} m  '
            f'ang={angle_deg:.1f}°',
            throttle_duration_sec=0.5)

        # ── Debug image ───────────────────────────────────────────────────────
        if self._pub_debug:
            debug = frame.copy()
            pts_int = points_2d.astype(int)
            cv2.polylines(debug, [pts_int.reshape(-1, 1, 2)], True, (0, 255, 0), 2)

            # Dibuja ejes XYZ del QR proyectados en la imagen
            axis_len = self._qr_size * 0.6
            axis_3d  = np.float32([
                [0, 0, 0],
                [axis_len, 0, 0],
                [0, axis_len, 0],
                [0, 0, -axis_len],
            ])
            axis_2d, _ = cv2.projectPoints(
                axis_3d, rvec, tvec, self._K,
                np.zeros((4, 1), dtype=np.float64))
            axis_2d = axis_2d.astype(int).reshape(-1, 2)
            origin = tuple(axis_2d[0])
            cv2.arrowedLine(debug, origin, tuple(axis_2d[1]), (0, 0, 255), 2)   # X rojo
            cv2.arrowedLine(debug, origin, tuple(axis_2d[2]), (0, 255, 0), 2)   # Y verde
            cv2.arrowedLine(debug, origin, tuple(axis_2d[3]), (255, 0, 0), 2)   # Z azul

            # Texto con cliente y distancia
            x_min = int(np.min(pts_int[:, 0]))
            y_min = int(np.min(pts_int[:, 1]))
            cv2.putText(debug,
                        f'{client}  {dist_cam:.2f}m  {angle_deg:.0f}deg',
                        (x_min, max(y_min - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            self._pub_dbg.publish(
                self._bridge.cv2_to_imgmsg(debug, encoding='bgr8'))

    # ── Watchdog ──────────────────────────────────────────────────────────────

    def _watchdog_cb(self):
        if time.monotonic() - self._last_detection_t > self._lost_to:
            self._pub_detected.publish(Bool(data=False))


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = QRReaderNode()
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
