"""
grabar.py — Script de grabación para práctica de reconocimiento de voz
Graba 15 repeticiones de cada palabra a 16 kHz y las guarda en data/<palabra>/
"""

import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf

# ── Configuración ──────────────────────────────────────────────────────────────
PALABRAS = ['avanzar', 'retroceder', 'izquierda', 'derecha', 'alto', 'inicio']
FS = 16000          # frecuencia de muestreo requerida
DURACION = 1.5      # segundos por grabación (ajusta si tus palabras son más largas)
REPETICIONES = 20
DATA_DIR = 'data'
# ──────────────────────────────────────────────────────────────────────────────

def grabar(duracion, fs):
    """Graba `duracion` segundos de audio mono a `fs` Hz."""
    frames = int(duracion * fs)
    audio = sd.rec(frames, samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def countdown(segundos=3):
    """Cuenta regresiva visible en terminal."""
    for i in range(segundos, 0, -1):
        print(f'  {i}...', end='', flush=True)
        time.sleep(1)
    print()

def verificar_dispositivo():
    """Muestra el dispositivo de entrada activo."""
    info = sd.query_devices(kind='input')
    print(f"\n🎙️  Micrófono detectado: {info['name']}")
    print(f"    Canales disponibles: {info['max_input_channels']}")
    print(f"    Sample rate por defecto: {int(info['default_samplerate'])} Hz\n")

def main():
    print("=" * 55)
    print("  GRABADOR DE PALABRAS — Reconocimiento de Voz LPC/VQ")
    print("=" * 55)
    verificar_dispositivo()

    print(f"Palabras a grabar : {', '.join(PALABRAS)}")
    print(f"Repeticiones      : {REPETICIONES} por palabra")
    print(f"Duración          : {DURACION}s por grabación")
    print(f"Sample rate       : {FS} Hz")
    print(f"Directorio salida : {DATA_DIR}/\n")
    print("Instrucciones:")
    print("  - Graba en una habitación SIN ruido")
    print("  - Habla tú mismo TODAS las grabaciones")
    print("  - Di la palabra claramente cuando veas ▶ GRABANDO")
    print("  - Presiona Enter para pasar a la siguiente")
    print("  - Escribe 'q' + Enter para salir\n")

    input("Presiona Enter cuando estés listo para empezar...")

    # Crear estructura de carpetas
    for palabra in PALABRAS:
        os.makedirs(os.path.join(DATA_DIR, palabra), exist_ok=True)

    total_grabadas = 0

    for palabra in PALABRAS:
        print(f"\n{'━'*55}")
        print(f"  PALABRA: « {palabra.upper()} »")
        print(f"{'━'*55}")

        # Ver cuántas ya existen (por si se reanuda)
        existentes = len([
            f for f in os.listdir(os.path.join(DATA_DIR, palabra))
            if f.endswith('.wav')
        ])
        if existentes >= REPETICIONES:
            print(f"  ✅ Ya tiene {existentes} grabaciones. Saltando...")
            continue
        if existentes > 0:
            print(f"  ⚠️  Ya existen {existentes} grabaciones. Continuando desde la {existentes+1}...")

        for rep in range(existentes + 1, REPETICIONES + 1):
            print(f"\n  [{rep:02d}/{REPETICIONES:02d}] Di la palabra «{palabra}»")
            print("  Cuenta regresiva:", end=' ')
            countdown(3)

            print("  ▶  GRABANDO...", end='', flush=True)
            audio = grabar(DURACION, FS)
            print(" ✓")

            # Guardar
            nombre = f"{palabra}_{rep:02d}.wav"
            ruta = os.path.join(DATA_DIR, palabra, nombre)
            sf.write(ruta, audio, FS)
            total_grabadas += 1

            # Reproducir para verificar
            print("  🔊 Reproduciendo...", end='', flush=True)
            sd.play(audio, FS)
            sd.wait()
            print(" listo")

            # Opción de repetir
            respuesta = input("  ¿Repetir esta grabación? (Enter = continuar / r = repetir / q = salir): ").strip().lower()
            if respuesta == 'q':
                print(f"\n⛔ Grabación pausada. Se guardaron {total_grabadas} archivos.")
                return
            elif respuesta == 'r':
                print("  Repitiendo...")
                rep -= 1  # no incrementa en el loop, pero borramos el archivo actual
                os.remove(ruta)
                total_grabadas -= 1
                # Volver a grabar la misma repetición
                print(f"\n  [{rep:02d}/{REPETICIONES:02d}] Di la palabra «{palabra}» (repetición)")
                print("  Cuenta regresiva:", end=' ')
                countdown(3)
                print("  ▶  GRABANDO...", end='', flush=True)
                audio = grabar(DURACION, FS)
                print(" ✓")
                sf.write(ruta, audio, FS)
                total_grabadas += 1
                sd.play(audio, FS)
                sd.wait()
                print("  🔊 Guardado.")

        print(f"\n  ✅ «{palabra}» completada — {REPETICIONES} grabaciones guardadas.")

    print(f"\n{'='*55}")
    print(f"  🎉 ¡Grabación completada!")
    print(f"  Total de archivos guardados: {total_grabadas}")
    print(f"  Carpeta: {os.path.abspath(DATA_DIR)}/")
    print(f"{'='*55}")
    print("\nSiguiente paso:")
    print("  git add data/")
    print("  git commit -m 'feat: grabaciones reales de voz'")
    print("  git push origin grabaciones-reales\n")

if __name__ == '__main__':
    main()
