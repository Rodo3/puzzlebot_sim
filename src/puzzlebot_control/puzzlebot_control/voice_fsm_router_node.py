"""
voice_fsm_router_node.py — Puente entre el reconocedor de voz y la FSM de misión.

FUNCIÓN:
  Traduce los comandos de voz crudos (/voice/command + /voice/confidence) en
  eventos de control de la FSM (/mission_state_in), aplicando las políticas que
  el reconocedor NO implementa: umbral de confianza, cooldown global, debounce
  por palabra y PRIORIDAD de STOP.

  La FSM (mission_manager_node) queda agnóstica del modelo de voz: solo recibe
  eventos limpios por /mission_state_in, el mismo tópico que ya usa el dashboard
  o una terminal. Si mañana se cambia el modelo de voz, solo se toca este router.

CONTRATO:
  Suscribe:  /voice/command     (std_msgs/String)   palabra reconocida
             /voice/confidence  (std_msgs/Float32)  margen/score del reconocedor
  Publica:   /mission_state_in  (std_msgs/String)   evento para la FSM

MAPEO DE COMANDOS (etapa actual — solo control de misión):
  'alto'    → 'PAUSE'   (máxima prioridad; se acepta SIEMPRE, sin cooldown)
  'inicio'  → 'START'   (la FSM decide si es arranque o reanudación según su
                         estado: START_M* en WAIT_FOR_VOICE_START, CONTINUE en
                         GLOBAL_VOICE_STOP)

  El resto de palabras del vocabulario (avanzar, retroceder, izquierda, derecha,
  subir, bajar, tomar, soltar) son TELEOP y las maneja el web_bridge, no la FSM.
  Este router las ignora a propósito.

POLÍTICAS:
  confidence_threshold  — descarta comandos por debajo de este score.
  cooldown_sec          — tras aceptar un comando, ignora cualquier comando
                          (excepto 'alto') durante este tiempo. Evita que una
                          misma palabra dispare varias transiciones seguidas.
  STOP siempre pasa: 'alto' nunca se filtra por cooldown ni por debounce, solo
  por umbral de confianza (configurable aparte con stop_confidence_threshold).

PARÁMETROS:
  confidence_threshold       [0.0]   Umbral mínimo para comandos normales.
  stop_confidence_threshold  [0.0]   Umbral mínimo para 'alto' (suele ser menor:
                                     preferimos un STOP de más que uno de menos).
  cooldown_sec               [3.0]   Segundos de bloqueo tras aceptar un comando.
  confidence_stale_sec       [1.0]   Si la confianza recibida es más vieja que
                                     esto, se trata como confianza desconocida.
  command_topic              [/voice/command]
  confidence_topic           [/voice/confidence]
  fsm_event_topic            [/mission_state_in]

USO:
  ros2 run puzzlebot_control voice_fsm_router_node
  # Prueba sin micrófono:
  ros2 topic pub --once /voice/confidence std_msgs/Float32 "data: 0.9"
  ros2 topic pub --once /voice/command    std_msgs/String  "data: 'inicio'"
  ros2 topic echo /mission_state_in       # → 'START'
"""

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String


# Palabras de voz → evento que entiende la FSM (/mission_state_in).
_COMMAND_MAP = {
    'alto':   'PAUSE',
    'inicio': 'START',
}

# Palabras tratadas como STOP de máxima prioridad (saltan cooldown/debounce).
_STOP_COMMANDS = {'alto'}


class VoiceFsmRouterNode(Node):

    def __init__(self):
        super().__init__('voice_fsm_router_node')

        self.declare_parameter('confidence_threshold',      0.0)
        self.declare_parameter('stop_confidence_threshold', 0.0)
        self.declare_parameter('cooldown_sec',              3.0)
        self.declare_parameter('confidence_stale_sec',      1.0)
        self.declare_parameter('command_topic',    '/voice/command')
        self.declare_parameter('confidence_topic', '/voice/confidence')
        self.declare_parameter('fsm_event_topic',  '/mission_state_in')

        self._conf_thresh      = float(self.get_parameter('confidence_threshold').value)
        self._stop_conf_thresh = float(self.get_parameter('stop_confidence_threshold').value)
        self._cooldown         = float(self.get_parameter('cooldown_sec').value)
        self._conf_stale       = float(self.get_parameter('confidence_stale_sec').value)
        cmd_topic              = self.get_parameter('command_topic').value
        conf_topic             = self.get_parameter('confidence_topic').value
        event_topic            = self.get_parameter('fsm_event_topic').value

        # Última confianza recibida (se empareja con el siguiente /voice/command).
        self._last_conf:      float = 0.0
        self._last_conf_time: float = 0.0
        # Momento en que termina el cooldown del último comando aceptado.
        self._cooldown_until: float = 0.0

        self._pub_event = self.create_publisher(String, event_topic, 10)
        self.create_subscription(Float32, conf_topic, self._on_confidence, 10)
        self.create_subscription(String,  cmd_topic,  self._on_command,    10)

        self.get_logger().info(
            f'voice_fsm_router_node listo — umbral={self._conf_thresh}, '
            f'umbral_stop={self._stop_conf_thresh}, cooldown={self._cooldown}s | '
            f'{cmd_topic} + {conf_topic} → {event_topic}')

    def _on_confidence(self, msg: Float32):
        self._last_conf = float(msg.data)
        self._last_conf_time = time.monotonic()

    def _current_confidence(self) -> float:
        # Si la confianza es demasiado vieja, no la consideramos válida.
        if time.monotonic() - self._last_conf_time > self._conf_stale:
            return 0.0
        return self._last_conf

    def _on_command(self, msg: String):
        word = msg.data.strip().lower()
        event = _COMMAND_MAP.get(word)
        if event is None:
            # Palabra de teleop u desconocida: no es asunto de la FSM.
            return

        conf = self._current_confidence()
        is_stop = word in _STOP_COMMANDS

        # ── Umbral de confianza (STOP usa su propio umbral, normalmente menor) ──
        thresh = self._stop_conf_thresh if is_stop else self._conf_thresh
        if conf < thresh:
            self.get_logger().warn(
                f'voz "{word}" descartada — confianza {conf:.3f} < {thresh}')
            return

        now = time.monotonic()

        # ── STOP: máxima prioridad, ignora cooldown ───────────────────────────
        if is_stop:
            self._emit(event, word, conf)
            # Un STOP no impone cooldown a sí mismo: repetir 'alto' debe poder
            # reforzar la pausa. Pero sí reiniciamos el cooldown de otros comandos
            # para que un 'inicio' no se cuele inmediatamente tras un 'alto'.
            self._cooldown_until = now + self._cooldown
            return

        # ── Comandos normales: respetar cooldown ──────────────────────────────
        if now < self._cooldown_until:
            self.get_logger().info(
                f'voz "{word}" en cooldown ({self._cooldown_until - now:.1f}s) — ignorada')
            return

        self._emit(event, word, conf)
        self._cooldown_until = now + self._cooldown

    def _emit(self, event: str, word: str, conf: float):
        self._pub_event.publish(String(data=event))
        self.get_logger().info(
            f'voz "{word}" (conf={conf:.3f}) → /mission_state_in: {event}')


def main(args=None):
    rclpy.init(args=args)
    node = VoiceFsmRouterNode()
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
