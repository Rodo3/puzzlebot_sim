"""
yolo_mock_node.py — Mock del detector de logos (YOLO) para pruebas en bench.

FUNCIÓN:
  Reemplaza al detector ONNX de logos simulando que el robot encuentra el tráiler
  del cliente correcto y se centra sobre él. Permite probar la segunda mitad de
  la misión en bench, sin cámara:
      SEARCH_TRAILER_LOGO → MATCH_TRAILER_CLIENT → ALIGN_TO_DOCK →
      ENTER_TRAILER → DROP_PALLET → EXIT_TRAILER → MISSION_COMPLETE

  Es un MOCK DESACOPLADO E INTERCAMBIABLE: publica el mismo contrato que el YOLO
  real (/detections con vision_msgs/Detection2DArray). Cuando uses el YOLO de la
  Jetson, NO lances este mock — el detector real ocupa su lugar y la FSM no cambia.

CONTRATO (igual que el YOLO real):
  Publica:   /detections    (vision_msgs/Detection2DArray)
                             det.results[0].hypothesis.class_id (string, cliente)
                             det.results[0].hypothesis.score    (float)
                             det.bbox.center.position.x         (cx normalizado [0,1])
  Suscribe:  /mission_state  (std_msgs/String) — estado de la FSM
             /mission_client (std_msgs/String) — cliente objetivo leído del QR

COMPORTAMIENTO (logo que aparece y se centra solo):
  - Mientras la FSM NO busca tráiler, no publica detecciones (lista vacía).
  - Al entrar en SEARCH_TRAILER_LOGO, tras `reveal_delay` s aparece el logo del
    cliente objetivo (el que publica /mission_client; si está vacío, usa
    `client` por defecto). Esto dispara SEARCH → MATCH.
  - El centro horizontal `cx` empieza descentrado (`start_cx`) y converge a 0.5
    en `converge_time` s una vez en ALIGN_TO_DOCK, para que el centrado por
    visión (|cx-0.5| < dock_center_thresh) se cumpla y pase a ENTER_TRAILER.
  - Tras depositar (la FSM sale de los estados de tráiler), deja de publicar.

PARÁMETROS:
  client          ['']    Cliente a detectar si /mission_client está vacío.
                          Vacío → usa el de /mission_client (recomendado).
  score           [0.90]  Confianza publicada (debe superar logo_confidence_thresh).
  start_cx        [0.30]  Centro horizontal inicial del logo [0,1] (0.5=centrado).
  converge_time   [3.0]   Segundos en que cx llega de start_cx a 0.5 en ALIGN_TO_DOCK.
  reveal_delay    [2.0]   Segundos buscando antes de revelar el logo.
  publish_rate    [10.0]  Hz de publicación (mantener > 2 Hz: ALIGN exige <0.5s).

USO:
  ros2 run puzzlebot_control yolo_mock_node
  # Forzar un cliente fijo (debug, ignora /mission_client):
  ros2 run puzzlebot_control yolo_mock_node -p client:=Wolmar
"""

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


# Estados de la FSM durante los cuales el logo del tráiler debe estar "visible".
_TRAILER_STATES = {
    'SEARCH_TRAILER_LOGO',
    'MATCH_TRAILER_CLIENT',
    'ALIGN_TO_DOCK',
    'RECOVER_LOGO_VIEW',
}


class YoloMockNode(Node):

    def __init__(self):
        super().__init__('yolo_mock_node')

        self.declare_parameter('client',        '')
        self.declare_parameter('score',         0.90)
        self.declare_parameter('start_cx',      0.30)
        self.declare_parameter('converge_time', 3.0)
        self.declare_parameter('reveal_delay',  2.0)
        self.declare_parameter('publish_rate',  10.0)

        self._client_param = str(self.get_parameter('client').value)
        self._score        = float(self.get_parameter('score').value)
        self._start_cx     = float(self.get_parameter('start_cx').value)
        self._converge     = max(0.1, float(self.get_parameter('converge_time').value))
        self._reveal       = float(self.get_parameter('reveal_delay').value)
        rate               = float(self.get_parameter('publish_rate').value)

        self._fsm_state: str = ''
        self._mission_client: str = ''
        # Momento en que entramos a SEARCH (arranca el cronómetro de revelado).
        self._search_since: float | None = None
        # Momento en que entramos a ALIGN_TO_DOCK (arranca la convergencia de cx).
        self._align_since: float | None = None
        # True una vez depositado el pallet (no volver a mostrar el logo).
        self._consumed = False

        self._pub_det = self.create_publisher(Detection2DArray, '/detections', 10)
        self.create_subscription(String, '/mission_state',  self._on_state,  10)
        self.create_subscription(String, '/mission_client', self._on_client, 10)

        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f'yolo_mock_node listo — score={self._score}, start_cx={self._start_cx}, '
            f'converge={self._converge}s, reveal_delay={self._reveal}s')

    def _on_client(self, msg: String):
        if msg.data:
            self._mission_client = msg.data.strip()

    def _on_state(self, msg: String):
        new = msg.data.strip()
        if new == self._fsm_state:
            return
        self._fsm_state = new
        if new == 'SEARCH_TRAILER_LOGO' and self._search_since is None and not self._consumed:
            self._search_since = time.monotonic()
            self.get_logger().info('FSM buscando tráiler — revelando logo pronto…')
        if new == 'ALIGN_TO_DOCK' and self._align_since is None:
            self._align_since = time.monotonic()
            self.get_logger().info('FSM alineando al dock — cx convergerá a 0.5')
        # Si la FSM salió de los estados de tráiler tras haber buscado → consumido.
        if new not in _TRAILER_STATES and self._search_since is not None:
            self._consumed = True
            self.get_logger().info('Tráiler consumido — dejo de publicar logo')

    def _target_client(self) -> str:
        # Prioriza el cliente real de la misión (del QR); si no, el parámetro.
        return self._mission_client or self._client_param

    def _tick(self):
        visible = (
            self._fsm_state in _TRAILER_STATES
            and self._search_since is not None
            and not self._consumed
            and (time.monotonic() - self._search_since) >= self._reveal
        )

        arr = Detection2DArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        arr.header.frame_id = 'camera_link'

        if not visible:
            self._pub_det.publish(arr)   # lista vacía
            return

        target = self._target_client()
        if not target:
            # Aún no sabemos el cliente (no llegó /mission_client): no inventamos.
            self._pub_det.publish(arr)
            return

        # cx converge a 0.5 solo una vez en ALIGN_TO_DOCK; antes, fijo en start_cx.
        if self._align_since is not None:
            frac = max(0.0, min(1.0, (time.monotonic() - self._align_since) / self._converge))
            cx = self._start_cx + (0.5 - self._start_cx) * frac
        else:
            cx = self._start_cx

        det = Detection2D()
        det.header = arr.header
        det.bbox.center.position.x = float(cx)        # normalizado [0,1]
        det.bbox.center.position.y = 0.5
        det.bbox.size_x = 0.2
        det.bbox.size_y = 0.3

        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = target              # = cliente del QR
        hyp.hypothesis.score    = self._score
        det.results.append(hyp)
        arr.detections.append(det)
        self._pub_det.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = YoloMockNode()
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
