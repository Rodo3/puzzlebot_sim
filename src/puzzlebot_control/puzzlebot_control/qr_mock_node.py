"""
qr_mock_node.py — Mock de la percepción del pallet (alignment) para pruebas sin cámara.

FUNCIÓN:
  Reemplaza a qr_reader_node simulando que el robot encuentra un pallet con QR y
  se acerca/alinea a él. Permite probar la cadena
      EXPLORE_PICKUP_AREA → APPROACH_PALLET → ALIGN_TO_PALLET → PRE_PICK_VALIDATION
  de la FSM en bench, sin cámara ni QR físico ni que el robot se mueva.

  Es un MOCK DESACOPLADO E INTERCAMBIABLE: habla el mismo contrato de tópicos que
  qr_reader_node. Cuando la cámara/QR reales estén listos, NO se lanza este nodo y
  qr_reader_node ocupa su lugar — la FSM no cambia.

CONTRATO (igual que qr_reader_node):
  Publica:   /qr/detected  (std_msgs/Bool)         QR visible en el frame actual
             /qr/client    (std_msgs/String)       texto/cliente leído del QR
             /qr/pose      (geometry_msgs/PoseStamped, frame base_footprint)
                                                    pose del QR relativa al robot
  Suscribe:  /mission_state (std_msgs/String)      estado actual de la FSM

COMPORTAMIENTO (pose que converge sola):
  - Mientras la FSM NO está explorando, no hay pallet visible (/qr/detected=False).
  - Al entrar en EXPLORE_PICKUP_AREA, tras `reveal_delay` s aparece el pallet:
    /qr/detected=True y /qr/pose con una distancia que decrece linealmente desde
    `start_dist` hasta ~0 en `converge_time` s, con el ángulo lateral también
    decreciendo. Así APPROACH_PALLET (dist ≤ approach_distance) y ALIGN_TO_PALLET
    (ángulo ≈ 0) completan solos, como si el robot avanzara y se alineara.
  - Una vez "tomado" el pallet (la FSM sale de los estados de pallet), deja de
    publicar detección para no interferir con SEARCH_TRAILER_LOGO.

PARÁMETROS:
  client          ['wolmar']  Texto del QR (debe existir en qr_logo_map del
                              mission_config: wolmar/popsi/emezon).
  start_dist      [0.80]      Distancia inicial simulada al QR (m).
  start_lateral   [0.15]      Offset lateral inicial (m) para ejercitar ALIGN.
  converge_time   [6.0]       Segundos en que la pose llega de start_dist a 0.
  reveal_delay    [3.0]       Segundos en EXPLORE antes de revelar el pallet
                              (simula que el robot barre un poco antes de verlo).
  publish_rate    [10.0]      Hz de publicación.

USO:
  ros2 run puzzlebot_control qr_mock_node -p client:=popsi
  # Forzar revelación inmediata sin FSM (debug):
  ros2 run puzzlebot_control qr_mock_node -p reveal_delay:=0.0
"""

import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String


# Estados de la FSM durante los cuales el pallet debe estar "visible".
_PALLET_STATES = {
    'EXPLORE_PICKUP_AREA',
    'APPROACH_PALLET',
    'ALIGN_TO_PALLET',
    'PRE_PICK_VALIDATION',
    'RECOVER_QR_VIEW',
}


class QrMockNode(Node):

    def __init__(self):
        super().__init__('qr_mock_node')

        self.declare_parameter('client',        'wolmar')
        self.declare_parameter('start_dist',    0.80)
        self.declare_parameter('start_lateral', 0.15)
        self.declare_parameter('converge_time', 6.0)
        self.declare_parameter('reveal_delay',  3.0)
        self.declare_parameter('publish_rate',  10.0)

        self._client     = str(self.get_parameter('client').value)
        self._start_dist = float(self.get_parameter('start_dist').value)
        self._start_lat  = float(self.get_parameter('start_lateral').value)
        self._converge   = max(0.1, float(self.get_parameter('converge_time').value))
        self._reveal     = float(self.get_parameter('reveal_delay').value)
        rate             = float(self.get_parameter('publish_rate').value)

        self._fsm_state: str = ''
        # Momento en que entramos a EXPLORE (arranca el cronómetro de revelado).
        self._explore_since: float | None = None
        # True una vez que el pallet ya fue tomado (no volver a mostrarlo).
        self._consumed = False

        self._pub_det    = self.create_publisher(Bool,        '/qr/detected', 10)
        self._pub_client = self.create_publisher(String,      '/qr/client',   10)
        self._pub_pose   = self.create_publisher(PoseStamped, '/qr/pose',     10)
        self.create_subscription(String, '/mission_state', self._on_state, 10)

        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f'qr_mock_node listo — client="{self._client}", '
            f'start_dist={self._start_dist}m, converge={self._converge}s, '
            f'reveal_delay={self._reveal}s')

    def _on_state(self, msg: String):
        new = msg.data.strip()
        if new == self._fsm_state:
            return
        self._fsm_state = new
        if new == 'EXPLORE_PICKUP_AREA' and self._explore_since is None and not self._consumed:
            self._explore_since = time.monotonic()
            self.get_logger().info('FSM explorando — revelando pallet pronto…')
        # Si la FSM avanzó más allá de los estados de pallet, marcar consumido.
        if new not in _PALLET_STATES and self._explore_since is not None:
            self._consumed = True
            self.get_logger().info('Pallet consumido — dejo de publicar QR')

    def _tick(self):
        visible = (
            self._fsm_state in _PALLET_STATES
            and self._explore_since is not None
            and not self._consumed
            and (time.monotonic() - self._explore_since) >= self._reveal
        )

        if not visible:
            self._pub_det.publish(Bool(data=False))
            return

        # Progreso de convergencia [0,1] desde que se reveló el pallet.
        t = time.monotonic() - self._explore_since - self._reveal
        frac = max(0.0, min(1.0, t / self._converge))
        dist = self._start_dist * (1.0 - frac)
        lat  = self._start_lat * (1.0 - frac)

        self._pub_det.publish(Bool(data=True))
        self._pub_client.publish(String(data=self._client))

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'base_footprint'
        pose.pose.position.x = dist     # adelante del robot
        pose.pose.position.y = lat      # lateral
        # Orientación: yaw apuntando hacia el QR (informativo).
        yaw = math.atan2(lat, max(dist, 1e-3))
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        self._pub_pose.publish(pose)


def main(args=None):
    rclpy.init(args=args)
    node = QrMockNode()
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
