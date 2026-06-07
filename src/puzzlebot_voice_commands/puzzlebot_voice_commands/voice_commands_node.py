"""
voice_commands_node.py — Nodo ROS 2 para reconocimiento de voz con micrófono local.

NOTA: Este nodo asume que el micrófono está conectado a la misma máquina donde
corre ROS (ej: Jetson). Si el micrófono está en la computadora del dashboard,
usa el endpoint POST /audio del puzzlebot_web_bridge en su lugar.

Flujo:
  1. Recibe trigger en /voice/trigger (std_msgs/String, data='record')
  2. Graba `duration` segundos con sounddevice en hilo separado (no bloquea spin)
  3. Ejecuta inferencia con VoiceInferenceEngine (KMeans + HMM)
  4. Publica resultados en /voice/*

Parámetros ROS:
  artifact_dir        (string)  — ruta a la carpeta con los modelos entrenados
  duration            (float)   — segundos de grabación (default: 1.5)
  confidence_threshold (float)  — margen mínimo KMeans para publicar comando (default: 0.0)
"""
import json
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String

from .voice_inference import VoiceInferenceEngine


class VoiceCommandsNode(Node):
    def __init__(self):
        super().__init__('voice_commands_node')

        self.declare_parameter('artifact_dir', 'artifacts_final')
        self.declare_parameter('duration', 1.5)
        self.declare_parameter('confidence_threshold', 0.0)

        artifact_dir = self.get_parameter('artifact_dir').get_parameter_value().string_value
        self._duration = self.get_parameter('duration').get_parameter_value().double_value
        self._threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value

        self.get_logger().info(f'Cargando modelos desde: {artifact_dir}')
        try:
            self._engine = VoiceInferenceEngine.load(artifact_dir)
            self.get_logger().info('Modelos KMeans + HMM cargados.')
        except FileNotFoundError as e:
            self.get_logger().error(str(e))
            raise

        # Publishers
        self._pub_command = self.create_publisher(String, '/voice/command', 10)
        self._pub_confidence = self.create_publisher(Float32, '/voice/confidence', 10)
        self._pub_status = self.create_publisher(String, '/voice/status', 10)
        self._pub_ranked = self.create_publisher(String, '/voice/ranked_predictions', 10)
        self._pub_time = self.create_publisher(Float32, '/voice/inference_time_ms', 10)

        # Subscriber
        self.create_subscription(String, '/voice/trigger', self._on_trigger, 10)

        self._busy = False
        self._publish_status('idle')
        self.get_logger().info('voice_commands_node listo. Esperando trigger en /voice/trigger')

    def _on_trigger(self, msg: String):
        if self._busy:
            self.get_logger().warn('Grabación en curso, trigger ignorado.')
            return
        self._busy = True
        threading.Thread(target=self._record_and_infer, daemon=True).start()

    def _record_and_infer(self):
        import sounddevice as sd

        try:
            self._publish_status('listening')
            self.get_logger().info(f'Grabando {self._duration}s...')

            frames = int(self._duration * 16000)
            audio = sd.rec(frames, samplerate=16000, channels=1, dtype='float32')
            sd.wait()

            self._publish_status('processing')
            result = self._engine.infer(audio.flatten())

            if result.confidence < self._threshold:
                self.get_logger().warn(
                    f'Confianza baja ({result.confidence:.4f} < {self._threshold}), descartado.'
                )
            else:
                self._publish_result(result)

        except Exception as e:
            self.get_logger().error(f'Error en inferencia: {e}')
        finally:
            self._publish_status('idle')
            self._busy = False

    def _publish_result(self, result):
        ranked_json = json.dumps({
            'kmeans': result.ranked_kmeans[:3],
            'hmm': result.ranked_hmm[:3],
        })

        self._pub_command.publish(String(data=result.command))
        self._pub_confidence.publish(Float32(data=result.confidence))
        self._pub_ranked.publish(String(data=ranked_json))
        self._pub_time.publish(Float32(data=result.inference_time_ms))

        self.get_logger().info(
            f'Comando: {result.command.upper()}  '
            f'confianza={result.confidence:.4f}  '
            f'tiempo={result.inference_time_ms:.1f}ms'
        )

    def _publish_status(self, status: str):
        self._pub_status.publish(String(data=status))


def main(args=None):
    rclpy.init(args=args)
    node = VoiceCommandsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
