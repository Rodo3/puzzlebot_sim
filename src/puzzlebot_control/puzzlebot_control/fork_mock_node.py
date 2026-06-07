"""
fork_mock_node.py — Mock del actuador de horquillas (lifter) para pruebas sin hardware.

FUNCIÓN:
  Reemplaza al nodo receptor real de la Jetson que mueve el actuador y publica el
  estado del limit switch. Permite probar la FSM completa (PICK_MANEUVER →
  VERIFY_PICK → … → DROP_PALLET) en el robot real mientras el actuador físico de
  las horquillas todavía no existe o no está conectado.

  Es un MOCK DESACOPLADO E INTERCAMBIABLE: habla exactamente el mismo contrato de
  tópicos que tendrá el driver real. Cuando el hardware esté listo, simplemente
  NO se lanza este nodo y el driver real ocupa su lugar — la FSM no cambia.

CONTRATO (igual que el lifter real esperado):
  Suscribe:  /fork/command  (std_msgs/Bool)  True=subir, False=bajar
  Publica:   /fork/status   (std_msgs/Bool)  estado del limit switch
                                              True=horquillas arriba/cargadas

COMPORTAMIENTO:
  Al recibir /fork/command, simula que la maniobra física tarda `lift_time`
  segundos. Durante ese lapso mantiene el estado anterior; al cumplirse el tiempo
  publica /fork/status con el nuevo valor (como si el limit switch se activara).
  Republica /fork/status periódicamente (latcheado por timer) para que la FSM lo
  vea aunque se suscriba tarde.

PARÁMETROS:
  lift_time      [2.0]   Segundos que tarda la maniobra simulada (subir o bajar).
  publish_rate   [5.0]   Hz de republicación del estado actual de /fork/status.
  start_up       [false] Estado inicial del limit switch al arrancar.

USO:
  ros2 run puzzlebot_control fork_mock_node
  # En otra terminal:
  ros2 topic pub --once /fork/command std_msgs/Bool "data: true"
  ros2 topic echo /fork/status     # → true tras lift_time segundos
"""

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


class ForkMockNode(Node):

    def __init__(self):
        super().__init__('fork_mock_node')

        self.declare_parameter('lift_time',    2.0)
        self.declare_parameter('publish_rate', 5.0)
        self.declare_parameter('start_up',     False)

        self._lift_time = float(self.get_parameter('lift_time').value)
        rate            = float(self.get_parameter('publish_rate').value)

        # Estado actual del "limit switch" simulado.
        self._fork_up = bool(self.get_parameter('start_up').value)
        # Maniobra en curso: (valor_objetivo, t_de_finalizacion) o None.
        self._pending: tuple | None = None

        self._pub = self.create_publisher(Bool, '/fork/status', 10)
        self.create_subscription(Bool, '/fork/command', self._on_command, 10)

        # Republica el estado actual y resuelve la maniobra pendiente.
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f'fork_mock_node listo — lift_time={self._lift_time}s, '
            f'estado inicial={"ARRIBA" if self._fork_up else "ABAJO"}')

    def _on_command(self, msg: Bool):
        target = bool(msg.data)
        if target == self._fork_up and self._pending is None:
            # Ya está en ese estado y no hay maniobra en curso: nada que hacer.
            return
        self._pending = (target, time.monotonic() + self._lift_time)
        self.get_logger().info(
            f'/fork/command={target} → simulando maniobra '
            f'({"subir" if target else "bajar"}) durante {self._lift_time}s…')

    def _tick(self):
        # Resolver maniobra pendiente cuando se cumpla el tiempo simulado.
        if self._pending is not None:
            target, t_done = self._pending
            if time.monotonic() >= t_done:
                self._fork_up = target
                self._pending = None
                self.get_logger().info(
                    f'Maniobra completa — limit switch: '
                    f'{"ARRIBA ↑" if self._fork_up else "ABAJO ↓"}')

        self._pub.publish(Bool(data=self._fork_up))


def main(args=None):
    rclpy.init(args=args)
    node = ForkMockNode()
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
