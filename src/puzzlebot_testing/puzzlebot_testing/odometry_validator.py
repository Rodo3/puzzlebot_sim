"""
Nodo de validación de odometría — modo pasivo (tú manejas, él mide).

════════════════════════════════════════════════════════════════════════
 CÓMO USARLO
════════════════════════════════════════════════════════════════════════

1. Lanza el stack normal del robot real:
     ros2 launch puzzlebot_bringup real_robot.launch.py \\
       avoidance:=false viewer:=false lidar_topic:=/scan

2. En otra terminal, lanza este validador:
     ros2 run puzzlebot_testing odometry_validator

3. Coloca el robot mirando de frente a cualquier marcador ArUco.
   El nodo espera hasta recibir una detección válida y usa esa pose
   como origen absoluto de la vuelta 1.

4. Conduce el robot con teleop_twist_keyboard haciendo vueltas al
   rectángulo. Cada vez que el robot pase cerca del origen se registra
   una vuelta y se imprime el reporte.

5. Presiona Ctrl+C al terminar para ver el resumen estadístico.

════════════════════════════════════════════════════════════════════════
 LÓGICA DE INICIALIZACIÓN
════════════════════════════════════════════════════════════════════════

  Estado WAITING_ARUCO:
    - El nodo no cuenta distancia ni detecta vueltas
    - Muestra un log cada 3 s recordando que espera un ArUco
    - En cuanto llega /aruco/pose con covarianza válida → RUNNING

  Estado RUNNING:
    - El origen es la pose ArUco del primer frame válido
    - La odometría se integra a partir de ese momento
    - El cierre de vuelta usa dos fases (salida + regreso)

════════════════════════════════════════════════════════════════════════
 QUÉ MIDE POR VUELTA
════════════════════════════════════════════════════════════════════════

  • Error de cierre XY   — distancia entre la pose final y el origen
  • Error de yaw         — ángulo residual al cerrar la vuelta
  • Distancia recorrida  — odómetro integrado
  • Correcciones ArUco   — número de correcciones y magnitud media

════════════════════════════════════════════════════════════════════════
 DIMENSIONES DEL RECTÁNGULO (aruco_map.yaml)
════════════════════════════════════════════════════════════════════════
  Ancho (X): 3.76 m
  Alto  (Y): 4.86 m
  Perímetro: 2 × (3.76 + 4.86) = 17.24 m
"""

import math
import statistics

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node


TRACK_WIDTH_M     = 3.76
TRACK_HEIGHT_M    = 4.86
TRACK_PERIMETER_M = 2.0 * (TRACK_WIDTH_M + TRACK_HEIGHT_M)   # 17.24 m

_STATE_WAITING  = 'WAITING_ARUCO'
_STATE_RUNNING  = 'RUNNING'


def _dist(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def _norm_angle(a):
    while a >  math.pi: a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return a


class LapRecord:
    def __init__(self, lap_num, origin_x, origin_y, origin_yaw):
        self.num          = lap_num
        self.origin_x     = origin_x
        self.origin_y     = origin_y
        self.origin_yaw   = origin_yaw
        self.close_x      = origin_x
        self.close_y      = origin_y
        self.close_yaw    = origin_yaw
        self.distance     = 0.0
        self.aruco_count  = 0
        self.aruco_deltas = []
        self._closed      = False

    def close(self, x, y, yaw):
        self.close_x   = x
        self.close_y   = y
        self.close_yaw = yaw
        self._closed   = True

    @property
    def is_closed(self):
        return self._closed

    @property
    def error_xy(self):
        return _dist(self.origin_x, self.origin_y, self.close_x, self.close_y)

    @property
    def error_yaw_deg(self):
        return math.degrees(abs(_norm_angle(self.close_yaw - self.origin_yaw)))

    @property
    def aruco_mean_delta(self):
        return statistics.mean(self.aruco_deltas) if self.aruco_deltas else 0.0


class OdometryValidator(Node):

    def __init__(self):
        super().__init__('odometry_validator')

        # ── Parámetros ────────────────────────────────────────────────────
        self.declare_parameter('lap_close_radius',   0.40)   # m
        self.declare_parameter('min_lap_distance',   8.0)    # m — distancia mínima para poder cerrar
        self.declare_parameter('min_exit_radius',    1.50)   # m — debe alejarse antes de poder cerrar
        self.declare_parameter('expected_perimeter', TRACK_PERIMETER_M)
        self.declare_parameter('aruco_wait_log_sec', 3.0)    # s — intervalo de log mientras espera

        self._close_radius    = self.get_parameter('lap_close_radius').value
        self._min_lap_dist    = self.get_parameter('min_lap_distance').value
        self._min_exit_radius = self.get_parameter('min_exit_radius').value
        self._expected_perim  = self.get_parameter('expected_perimeter').value
        self._wait_log_sec    = self.get_parameter('aruco_wait_log_sec').value

        # ── Estado ────────────────────────────────────────────────────────
        self._state = _STATE_WAITING

        # Pose actual del Kalman (/odom)
        self._x   = 0.0
        self._y   = 0.0
        self._yaw = 0.0

        # Odómetro incremental entre frames
        self._prev_x = 0.0
        self._prev_y = 0.0
        self._odom_ready = False    # primer /odom recibido

        # Vueltas
        self._lap_num       = 0
        self._laps: list[LapRecord] = []
        self._current: LapRecord | None = None
        self._exited_origin = False   # fase 1 del detector de cierre

        # ── Suscripciones ────────────────────────────────────────────────
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, '/aruco/pose', self._aruco_cb, 10)

        # Timer para recordatorio mientras espera ArUco
        self._wait_timer = self.create_timer(
            self._wait_log_sec, self._waiting_log)

        self.get_logger().info(
            '\n'
            '╔══════════════════════════════════════════════════════╗\n'
            '║       PUZZLEBOT — Validador de Odometría             ║\n'
            '╠══════════════════════════════════════════════════════╣\n'
            f'║  Radio de cierre       : {self._close_radius:.2f} m                 ║\n'
            f'║  Radio de salida       : {self._min_exit_radius:.2f} m                 ║\n'
            f'║  Distancia mínima/vuelta: {self._min_lap_dist:.1f} m                ║\n'
            f'║  Perímetro esperado    : {self._expected_perim:.2f} m              ║\n'
            '╠══════════════════════════════════════════════════════╣\n'
            '║  Apunta el robot a un marcador ArUco.                ║\n'
            '║  El origen se tomará de la primera detección válida. ║\n'
            '╚══════════════════════════════════════════════════════╝'
        )

    # ── Timer de espera ────────────────────────────────────────────────────

    def _waiting_log(self):
        if self._state == _STATE_WAITING:
            self.get_logger().info(
                '⏳ Esperando detección ArUco para establecer el origen...'
                '  (asegúrate de que el robot esté mirando un marcador)')

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        q   = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Siempre actualizar pose y odómetro incremental
        if not self._odom_ready:
            self._odom_ready = True
            self._prev_x, self._prev_y = x, y

        step = _dist(self._prev_x, self._prev_y, x, y)
        self._prev_x, self._prev_y = x, y
        self._x, self._y, self._yaw = x, y, yaw

        # Mientras espera ArUco: no contar distancia ni detectar vueltas
        if self._state != _STATE_RUNNING:
            return

        # Integrar odómetro de la vuelta actual
        if self._current is not None:
            self._current.distance += step

        # Detector de cierre de vuelta — dos fases
        if self._current is not None:
            dist_origin = _dist(
                self._current.origin_x, self._current.origin_y, x, y)

            # Fase 1: esperar a que el robot salga de la zona de exclusión
            if not self._exited_origin:
                if dist_origin >= self._min_exit_radius:
                    self._exited_origin = True
                    self.get_logger().info(
                        f'↗  Robot alejado {dist_origin:.2f} m del origen — '
                        'buscando regreso para cerrar vuelta...')
                return   # no comprobar cierre hasta haber salido

            # Fase 2: detectar regreso al origen
            if (dist_origin <= self._close_radius
                    and self._current.distance >= self._min_lap_dist):
                self._register_lap_close(x, y, yaw)

    def _aruco_cb(self, msg: PoseWithCovarianceStamped):
        aruco_x   = msg.pose.pose.position.x
        aruco_y   = msg.pose.pose.position.y
        q         = msg.pose.pose.orientation
        aruco_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        # ── Inicialización desde ArUco ─────────────────────────────────
        if self._state == _STATE_WAITING:
            # Validar que la covarianza no sea degenerada
            cov_x = msg.pose.covariance[0]
            cov_y = msg.pose.covariance[7]
            if cov_x <= 0.0 or cov_y <= 0.0:
                self.get_logger().warn(
                    'ArUco recibido pero covarianza degenerada — ignorando')
                return

            # Cancelar el timer de espera
            self._wait_timer.cancel()
            self._state = _STATE_RUNNING

            # Usar la pose ArUco como origen de la vuelta 1.
            # También sincronizar el odómetro incremental desde aquí.
            self._prev_x, self._prev_y = self._x, self._y

            self._start_new_lap(aruco_x, aruco_y, aruco_yaw)

            self.get_logger().info(
                f'\n'
                f'✅ Origen establecido desde ArUco:\n'
                f'   x={aruco_x:.3f} m  y={aruco_y:.3f} m  '
                f'yaw={math.degrees(aruco_yaw):.1f}°\n'
                f'   pos_std=({math.sqrt(cov_x):.3f}, {math.sqrt(cov_y):.3f}) m\n'
                f'   Comienza a mover el robot — vuelta 1 en curso...'
            )
            return

        # ── Registrar corrección ArUco durante la vuelta ───────────────
        if self._state == _STATE_RUNNING and self._current is not None:
            # Delta entre lo que dice ArUco y donde cree estar el Kalman
            delta = _dist(self._x, self._y, aruco_x, aruco_y)
            if delta < 2.0:   # sanity check: ignorar teleports
                self._current.aruco_count  += 1
                self._current.aruco_deltas.append(delta)
                self.get_logger().debug(
                    f'ArUco Δ={delta:.3f} m  '
                    f'Kalman=({self._x:.2f},{self._y:.2f}) → '
                    f'ArUco=({aruco_x:.2f},{aruco_y:.2f})')

    # ── Gestión de vueltas ─────────────────────────────────────────────────

    def _start_new_lap(self, x, y, yaw):
        self._lap_num       += 1
        self._exited_origin  = False
        self._current        = LapRecord(self._lap_num, x, y, yaw)
        self._laps.append(self._current)

    def _register_lap_close(self, x, y, yaw):
        self._current.close(x, y, yaw)
        self._print_lap_report(self._current)
        # El nuevo origen de la siguiente vuelta es el punto de cierre actual
        self._start_new_lap(x, y, yaw)

    # ── Reportes ───────────────────────────────────────────────────────────

    def _print_lap_report(self, lap: LapRecord):
        dist_err_pct = (
            abs(lap.distance - self._expected_perim) / self._expected_perim * 100.0
            if self._expected_perim > 0 else 0.0)
        aruco_info = (
            f'{lap.aruco_count} correcciones, Δ_med={lap.aruco_mean_delta:.3f} m'
            if lap.aruco_count > 0 else 'ninguna')

        self.get_logger().info(
            f'\n'
            f'┌─────────────────────────────────────────────────────┐\n'
            f'│  VUELTA {lap.num:2d} completada                           │\n'
            f'├─────────────────────────────────────────────────────┤\n'
            f'│  Error cierre XY    : {lap.error_xy:6.3f} m                  │\n'
            f'│  Error yaw          : {lap.error_yaw_deg:6.1f} °                  │\n'
            f'│  Distancia recorrida: {lap.distance:6.2f} m  '
            f'(esperado {self._expected_perim:.2f}, err={dist_err_pct:.1f}%)  │\n'
            f'│  Correcciones ArUco : {aruco_info:<30}│\n'
            f'└─────────────────────────────────────────────────────┘\n'
            f'  → Vuelta {lap.num + 1} iniciada'
        )

    def _print_final_summary(self):
        closed = [l for l in self._laps if l.is_closed]
        if not closed:
            n_total = len(self._laps)
            self.get_logger().warn(
                f'No se completó ninguna vuelta '
                f'({"nunca se detectó ArUco" if n_total == 0 else "vuelta en curso no cerrada"}).')
            return

        errors_xy  = [l.error_xy       for l in closed]
        errors_yaw = [l.error_yaw_deg  for l in closed]
        distances  = [l.distance       for l in closed]
        aruco_c    = [float(l.aruco_count) for l in closed]

        def fmt(vals):
            if len(vals) == 1:
                return f'{vals[0]:.3f}'
            return f'{statistics.mean(vals):.3f} ± {statistics.stdev(vals):.3f}'

        mean_xy   = statistics.mean(errors_xy)
        mean_yaw  = statistics.mean(errors_yaw)
        mean_dist = statistics.mean(distances)

        diagnoses = []

        if mean_xy > 0.10:
            diagnoses.append(
                f'⚠  Error XY {mean_xy:.3f} m > 10 cm — revisar wheel_radius / wheel_separation')
        else:
            diagnoses.append(f'✓  Error XY {mean_xy:.3f} m — odometría XY aceptable')

        if mean_yaw > 5.0:
            ws_current = 0.172
            theta_err_rad = math.radians(mean_yaw)
            correction    = 1.0 - (theta_err_rad * ws_current / mean_dist)
            ws_suggested  = ws_current * correction
            diagnoses.append(
                f'⚠  Error yaw {mean_yaw:.1f}° — wheel_separation puede estar mal\n'
                f'   Actual: {ws_current:.4f} m  →  Sugerido: ~{ws_suggested:.4f} m')
        else:
            diagnoses.append(f'✓  Error yaw {mean_yaw:.1f}° — wheel_separation razonable')

        dist_err_pct = abs(mean_dist - self._expected_perim) / self._expected_perim * 100.0
        if dist_err_pct > 5.0:
            diagnoses.append(
                f'⚠  Distancia {mean_dist:.2f} m vs {self._expected_perim:.2f} m '
                f'({dist_err_pct:.1f}% error) — revisar wheel_radius')
        else:
            diagnoses.append(
                f'✓  Distancia {mean_dist:.2f} m ({dist_err_pct:.1f}% error) — wheel_radius OK')

        diag_str = '\n║  '.join(diagnoses)

        self.get_logger().info(
            f'\n'
            f'╔══════════════════════════════════════════════════════════╗\n'
            f'║        RESUMEN FINAL — {len(closed)} vuelta(s) completada(s)       ║\n'
            f'╠══════════════════════════════════════════════════════════╣\n'
            f'║  Error cierre XY    : {fmt(errors_xy):>14} m               ║\n'
            f'║  Error yaw          : {fmt(errors_yaw):>14} °               ║\n'
            f'║  Distancia/vuelta   : {fmt(distances):>14} m               ║\n'
            f'║  Correcciones ArUco : {fmt(aruco_c):>14} /vuelta           ║\n'
            f'╠══════════════════════════════════════════════════════════╣\n'
            f'║  DIAGNÓSTICO                                             ║\n'
            f'║  {diag_str:<56}║\n'
            f'╚══════════════════════════════════════════════════════════╝'
        )

    def destroy_node(self):
        self._print_final_summary()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OdometryValidator()
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
