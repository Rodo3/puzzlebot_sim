"""
Graba 3 videos de entrenamiento desde la cámara del Puzzlebot.
Un video por logo: Pepsi, Amazon, Walmart.

Uso:
  ros2 run puzzlebot_logo_detector record_logos
  ros2 run puzzlebot_logo_detector record_logos --ros-args -p duration:=30 -p output_dir:=/home/jorge/videos

Controles durante grabación:
  q  — terminar este video antes de tiempo y pasar al siguiente
  Ctrl+C — salir
"""

import os
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

LOGOS = ["Pepsi", "Amazon", "Walmart"]

_WINDOW = "Grabación — Puzzlebot"


class RecorderNode(Node):

    def __init__(self):
        super().__init__("logo_recorder")

        self.declare_parameter("duration", 30)
        self.declare_parameter("output_dir", os.path.expanduser("~/logo_recordings"))
        self.declare_parameter("camera_topic", "/camera/image/compressed")

        self.duration = self.get_parameter("duration").value
        self.output_dir = self.get_parameter("output_dir").value
        self.camera_topic = self.get_parameter("camera_topic").value

        os.makedirs(self.output_dir, exist_ok=True)

        self.latest_frame = None
        self.frame_lock = threading.Lock()

        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.sub = self.create_subscription(
            CompressedImage, self.camera_topic, self._image_callback, camera_qos
        )

        self.get_logger().info(f"Escuchando cámara en: {self.camera_topic}")
        self.get_logger().info(f"Guardando videos en:  {self.output_dir}")
        self.get_logger().info(f"Duración por video:   {self.duration} segundos")

    def _image_callback(self, msg: CompressedImage):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is not None:
            with self.frame_lock:
                self.latest_frame = frame

    def get_frame(self):
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def wait_for_camera(self, timeout: float = 10.0) -> bool:
        """Espera hasta recibir el primer frame."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.get_frame() is not None:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def record(self, logo_name: str) -> str:
        """Graba un video para el logo dado. Retorna la ruta del archivo guardado."""
        output_path = os.path.join(self.output_dir, f"{logo_name.lower()}.avi")

        # Obtener resolución del primer frame
        sample = self.get_frame()
        if sample is None:
            self.get_logger().error("No hay frames disponibles.")
            return ""
        h, w = sample.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path, fourcc, 15.0, (w, h))

        cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_WINDOW, w * 3, h * 3)

        self.get_logger().info(f"Grabando {logo_name}... Presiona 'q' para terminar antes.")

        start = time.time()
        frames_written = 0
        stopped_early = False

        while True:
            elapsed = time.time() - start
            remaining = self.duration - elapsed

            if remaining <= 0:
                break

            rclpy.spin_once(self, timeout_sec=0.05)
            frame = self.get_frame()
            if frame is None:
                continue

            writer.write(frame)
            frames_written += 1

            # Overlay de estado
            display = frame.copy()
            bar_w = int((elapsed / self.duration) * w)
            cv2.rectangle(display, (0, h - 8), (bar_w, h), (0, 200, 0), -1)
            cv2.putText(
                display,
                f"REC  {logo_name}  {int(remaining)}s restantes  ({frames_written} frames)",
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
            )
            cv2.putText(
                display, "q = terminar antes",
                (6, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
            )

            cv2.imshow(_WINDOW, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                stopped_early = True
                break

        writer.release()
        cv2.destroyWindow(_WINDOW)

        duration_real = time.time() - start
        self.get_logger().info(
            f"{'Detenido' if stopped_early else 'Completado'}: {output_path} "
            f"({frames_written} frames, {duration_real:.1f}s)"
        )
        return output_path


def _wait_enter(prompt: str):
    print(prompt, end="", flush=True)
    sys.stdin.readline()


def main(args=None):
    rclpy.init(args=args)
    node = RecorderNode()

    print("\n" + "=" * 60)
    print("  GRABADOR DE LOGOS — Puzzlebot")
    print("=" * 60)
    print(f"  Logos a grabar: {', '.join(LOGOS)}")
    print(f"  Duración:       {node.duration}s por logo")
    print(f"  Destino:        {node.output_dir}")
    print("=" * 60)

    print("\nEsperando frames de la cámara...", end="", flush=True)
    if not node.wait_for_camera():
        print(" TIMEOUT — verifica que la cámara esté activa.")
        node.destroy_node()
        rclpy.shutdown()
        return
    print(" OK")

    saved = []
    try:
        for logo in LOGOS:
            print(f"\n{'─'*60}")
            print(f"  Siguiente: {logo}")
            print(f"  Pon el logo de {logo} frente a la cámara.")
            _wait_enter(f"  Presiona ENTER para empezar a grabar {logo}... ")
            path = node.record(logo)
            if path:
                saved.append(path)

    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")

    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

    print(f"\n{'='*60}")
    print("  Videos guardados:")
    for p in saved:
        size_mb = os.path.getsize(p) / 1e6
        print(f"    {p}  ({size_mb:.1f} MB)")
    print("=" * 60)
    print("\nPara extraer frames en otra computadora:")
    print("  python3 -c \"")
    print("  import cv2, os")
    print("  for video in ['pepsi.avi','amazon.avi','walmart.avi']:")
    print("      cap = cv2.VideoCapture(video)")
    print("      name = video.split('.')[0]")
    print("      os.makedirs(name, exist_ok=True)")
    print("      i = 0")
    print("      while True:")
    print("          ok, f = cap.read()")
    print("          if not ok: break")
    print("          cv2.imwrite(f'{name}/{i:05d}.jpg', f)")
    print("          i += 1")
    print("      print(f'{name}: {i} frames')")
    print("  \"")
