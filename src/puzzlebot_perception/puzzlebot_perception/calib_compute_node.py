"""
Nodo 2/3 — Cómputo de parámetros de calibración intrínseca.

Compatible con OpenCV 4.5.x (API legada) y OpenCV 4.7+ (nueva API).

Lee imágenes PNG de save_dir, detecta el tablero y ejecuta calibración.
En modo legacy usa cv2.aruco.calibrateCameraCharuco directamente.
En nueva API usa board.matchImagePoints + cv2.calibrateCamera.

Uso:
  ros2 run puzzlebot_perception calib_compute_node
  ros2 run puzzlebot_perception calib_compute_node \\
    --ros-args -p output_yaml:=src/puzzlebot_bringup/config/camera_calibration.yaml \\
               -p show_results:=true
"""

import os
import glob
import yaml
import rclpy
from rclpy.node import Node
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


class CalibComputeNode(Node):
    def __init__(self):
        super().__init__('calib_compute_node')

        self.declare_parameter('save_dir',        os.path.expanduser('~/calib_images'))
        self.declare_parameter('output_yaml',     os.path.expanduser('~/calib_images/camera_calibration.yaml'))
        self.declare_parameter('board_type',      'charuco')
        self.declare_parameter('board_cols',       7)
        self.declare_parameter('board_rows',       5)
        self.declare_parameter('square_size',      0.030)
        self.declare_parameter('marker_length',    0.022)
        self.declare_parameter('aruco_dict',       'DICT_4X4_50')
        self.declare_parameter('show_results',     True)
        self.declare_parameter('image_width',      640)
        self.declare_parameter('image_height',     360)   # Jetson publica 640x360

        save_dir   = self.get_parameter('save_dir').value
        output     = self.get_parameter('output_yaml').value
        board_type = self.get_parameter('board_type').value
        cols       = self.get_parameter('board_cols').value
        rows       = self.get_parameter('board_rows').value
        sq_size    = self.get_parameter('square_size').value
        mk_len     = self.get_parameter('marker_length').value
        dict_name  = self.get_parameter('aruco_dict').value
        show       = self.get_parameter('show_results').value
        exp_w      = self.get_parameter('image_width').value
        exp_h      = self.get_parameter('image_height').value

        use_charuco = (board_type == 'charuco')
        self.get_logger().info(
            f'OpenCV {cv2.__version__} — {"legacy" if _LEGACY else "nueva"} API  |  '
            f'modo={"ChArUco" if use_charuco else "Checkerboard"} {cols}x{rows}'
        )

        # ------ preparar tablero ------
        aruco_dict = _get_aruco_dict(dict_name)
        params     = _get_detector_params()

        if use_charuco:
            if _LEGACY:
                board        = cv2.aruco.CharucoBoard_create(cols, rows, sq_size, mk_len, aruco_dict)
                charuco_det  = None
            else:
                board        = cv2.aruco.CharucoBoard((cols, rows), sq_size, mk_len, aruco_dict)
                charuco_det  = cv2.aruco.CharucoDetector(board)

        # ------ cargar imágenes ------
        images = sorted(glob.glob(os.path.join(save_dir, '*.png')))
        if not images:
            self.get_logger().error(f'No se encontraron PNG en: {save_dir}')
            return
        self.get_logger().info(f'Procesando {len(images)} imágenes...')

        img_shape  = None
        used       = 0
        criteria   = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Para API legada con ChArUco: colectar corners/ids para calibrateCameraCharuco
        all_ch_corners: list = []
        all_ch_ids:     list = []
        # Para API nueva o checkerboard: colectar objpts/imgpts para calibrateCamera
        objpoints: list = []
        imgpoints: list = []

        for fname in images:
            img  = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_shape = gray.shape[::-1]   # (width, height)

            actual_w, actual_h = img_shape
            if actual_w != exp_w or actual_h != exp_h:
                self.get_logger().warn(
                    f'Resolución {actual_w}x{actual_h} ≠ esperada {exp_w}x{exp_h}',
                    throttle_duration_sec=5.0)

            if use_charuco:
                if _LEGACY:
                    ok, ch_corners, ch_ids = self._detect_charuco_legacy(
                        gray, board, aruco_dict, params)
                else:
                    ok, ch_corners, ch_ids = self._detect_charuco_new(
                        gray, board, charuco_det)

                if ok:
                    if _LEGACY:
                        all_ch_corners.append(ch_corners)
                        all_ch_ids.append(ch_ids)
                    else:
                        obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
                        objpoints.append(obj_pts)
                        imgpoints.append(img_pts)
                    used += 1
                    if show:
                        vis = img.copy()
                        cv2.aruco.drawDetectedCornersCharuco(vis, ch_corners, ch_ids)
                        cv2.imshow('ChArUco detectado', vis)
                        cv2.waitKey(300)
                else:
                    self.get_logger().warn(f'No detectado: {os.path.basename(fname)}')

            else:  # checkerboard
                board_size = (cols, rows)
                ret, corners = cv2.findChessboardCorners(gray, board_size, None)
                if ret:
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    objp = np.zeros((rows * cols, 3), np.float32)
                    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq_size
                    objpoints.append(objp)
                    imgpoints.append(corners2)
                    used += 1
                    if show:
                        vis = img.copy()
                        cv2.drawChessboardCorners(vis, board_size, corners2, ret)
                        cv2.imshow('Esquinas detectadas', vis)
                        cv2.waitKey(300)
                else:
                    self.get_logger().warn(f'No detectado: {os.path.basename(fname)}')

        cv2.destroyAllWindows()
        self.get_logger().info(f'Imágenes válidas: {used}/{len(images)}')

        if used < 5:
            self.get_logger().error(
                f'Solo {used} imágenes válidas. Se necesitan al menos 5.')
            return

        # ------ calibración ------
        if use_charuco and _LEGACY:
            rms, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                all_ch_corners, all_ch_ids, board, img_shape, None, None)
            # Calcular error por imagen usando chessboardCorners del board
            per_errors = []
            cc3d = board.chessboardCorners     # (N_corners, 3)
            for i in range(len(all_ch_corners)):
                ids_i = all_ch_ids[i].flatten()
                obj_i = np.array([cc3d[j] for j in ids_i], dtype=np.float32)
                proj, _ = cv2.projectPoints(obj_i, rvecs[i], tvecs[i], K, D)
                err = cv2.norm(all_ch_corners[i], proj, cv2.NORM_L2) / max(len(proj), 1)
                per_errors.append(float(err))
        else:
            rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, img_shape, None, None)
            per_errors = []
            for i in range(len(objpoints)):
                proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
                err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / max(len(proj), 1)
                per_errors.append(float(err))

        sorted_idx = np.argsort(per_errors)
        self.get_logger().info(
            f'RMS global: {rms:.4f} px  |  '
            f'min={per_errors[sorted_idx[0]]:.3f}  '
            f'max={per_errors[sorted_idx[-1]]:.3f}  '
            f'mediana={float(np.median(per_errors)):.3f} px'
        )
        if rms > 1.0:
            worst = [os.path.basename(images[i]) for i in sorted_idx[-3:]]
            self.get_logger().warn(
                f'RMS={rms:.3f} px > 1.0 px. Considera eliminar: {worst}')
        elif rms < 0.5:
            self.get_logger().info('Calibración excelente (RMS < 0.5 px).')

        self.get_logger().info(
            f'fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  '
            f'cx={K[0,2]:.2f}  cy={K[1,2]:.2f}')
        self.get_logger().info(f'D = {D.ravel().tolist()}')

        self._save_yaml(output, K, D, img_shape, rms)

        if show:
            self._show_comparison(images, sorted_idx, K, D, per_errors)

    # ------------------------------------------------------------------
    def _detect_charuco_legacy(self, gray, board, aruco_dict, params):
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=params)
        if ids is None or len(ids) < 4:
            return False, None, None
        _, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board)
        if ch_ids is None or len(ch_ids) < 6:
            return False, None, None
        return True, ch_corners, ch_ids

    def _detect_charuco_new(self, gray, board, charuco_det):
        ch_corners, ch_ids, _, _ = charuco_det.detectBoard(gray)
        if ch_ids is None or len(ch_ids) < 6:
            return False, None, None
        return True, ch_corners, ch_ids

    def _save_yaml(self, output, K, D, img_shape, rms):
        calib = {
            'image_width':  int(img_shape[0]),
            'image_height': int(img_shape[1]),
            'camera_name':  'camera',
            'camera_matrix': {
                'rows': 3, 'cols': 3,
                'data': K.ravel().tolist(),
            },
            'distortion_model': 'plumb_bob',
            'distortion_coefficients': {
                'rows': 1, 'cols': 5,
                'data': D.ravel().tolist()[:5],
            },
            'rectification_matrix': {
                'rows': 3, 'cols': 3,
                'data': [1.0, 0.0, 0.0,
                         0.0, 1.0, 0.0,
                         0.0, 0.0, 1.0],
            },
            'projection_matrix': {
                'rows': 3, 'cols': 4,
                'data': np.hstack([K, np.zeros((3, 1))]).ravel().tolist(),
            },
            'rms_error': float(rms),
        }
        out_dir = os.path.dirname(output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output, 'w') as f:
            yaml.dump(calib, f, default_flow_style=False)
        self.get_logger().info(f'Calibración guardada en: {output}')
        self.get_logger().info(
            'Copia a puzzlebot_bringup/config/camera_calibration.yaml '
            'y haz colcon build.')

    def _show_comparison(self, images, sorted_idx, K, D, per_errors):
        for title, idx in [('Mejor', sorted_idx[0]), ('Peor', sorted_idx[-1])]:
            img    = cv2.imread(images[idx])
            undist = cv2.undistort(img, K, D)
            combo  = np.hstack([img, undist])
            cv2.putText(combo, f'{title} — Original',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(combo, f'err={per_errors[idx]:.3f}px — Rectificada',
                        (img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('Resultado calibracion', combo)
            self.get_logger().info(f'{title}: presiona cualquier tecla...')
            cv2.waitKey(0)
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = CalibComputeNode()
    try:
        rclpy.spin_once(node, timeout_sec=0)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
