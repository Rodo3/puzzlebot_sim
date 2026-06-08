# Handoff — Misión logística Puzzlebot (branch `Jesus-fsm`)

**Fecha:** 2026-06-07  
**Branch:** `Jesus-fsm` (local, trackea `origin/Jesus-fsm`)  
**Robot:** físico — mapa conocido + EKF/ArUco + navegación A\*  
**Arquitectura de lanzamiento:** dos launches en capas:
- `real_robot.launch.py` — base (localización, navegación, sensores)
- `mission.launch.py` — capa de misión (FSM, QR, fork mock/real, voz)

---

## Lo que funcionó (verificado en log)

- Barrido de conveyors vía `/goal_pose` — el robot recorre la franja moviéndose de verdad.
- Detección del QR durante el barrido → transición `EXPLORE_PICKUP_AREA → APPROACH_PALLET`.
- Mock de alineación + pick (`mock_alignment=True`) → `PICK_MANEUVER → VERIFY_PICK` con `fork_mock` publicando `/fork/status`.
- Navegación directa al camión por waypoint según cliente (sin YOLO) → `NAVIGATE_TO_TRAILER` llegó al goal correcto `(3.14, 0.40)` para "Pepsi".
- Misión completa de punta a punta: `START → barrido → QR → pick → expedición → camión`.
- `QRCodeDetectorAruco` (OpenCV 4.11) reemplazó al detector clásico — detecta el QR (encuentra esquinas) de forma fiable.

---

## Lo que NO funcionó / queda pendiente

| Problema | Detalle |
|---|---|
| **Decodificación del texto del QR** | Detecta el QR pero casi nunca lee el contenido. El cliente "Pepsi" del log fue un fallback aleatorio, no el QR real. Se mejoró con `_decode_robust()` (ROI + upscale + Otsu) pero sin verificar lectura real. |
| **Alineación fina real** | Nunca convergió — el QR se ve muy lateral (ang ≈ −70°). Por eso está en modo mock. |
| **`max_linear_vel: 0.09` rompió el movimiento** | El robot dejó de moverse con los waypoints. Causa probable: por debajo del umbral de arranque del motor. Subido a `0.12` pero **no verificado**. |
| **Conflicto de coordenadas** | `mission_config.yaml` dice origen "esquina inferior derecha"; `aruco_map.yaml` usa "esquina inferior izquierda". Waypoints de camiones definidos con origen izquierdo (pared sur, Y ≈ 0). Verificar en pista. |

---

## Estado actual del código (sin commitear)

| Archivo | Cambios |
|---|---|
| `puzzlebot_control/mission_manager_node.py` | +284 / −110 líneas: barrido por `goal_pose`, `mock_alignment`, `NAVIGATE_TO_TRAILER`, `RECOVER_QR_VIEW` reescrito (giro lento mirando al norte). |
| `puzzlebot_perception/qr_reader_node.py` | Detector Aruco, `_decode_robust()`, cliente random, `max_detection_distance=8.0`, log `REAL` vs `ALEATORIO`. |
| `puzzlebot_controller/config/controller_params.yaml` | `max_linear_vel` 0.14 → 0.12. |
| `puzzlebot_bringup/launch/real_robot.launch.py` | Revertido a baseline (sin cambios). |

> **Nota:** El último cambio (`RECOVER_QR_VIEW` giro lento + velocidad 0.09) coincide con que "el robot ya no se mueve". No está confirmado cuál de los dos lo causó.

---

## Próximos pasos recomendados

### 1. Diagnosticar "el robot no se mueve" (hacer esto primero)

```bash
ros2 topic echo /cmd_vel
ros2 topic echo /goal_pose
ros2 topic echo /avoidance/status
ros2 topic echo /localization/status
```

Distinguir entre:
- Velocidad muy baja → motor no arranca (umbral de arranque del Puzzlebot ≈ 0.11 m/s)
- Steering sin path → el planificador no genera path
- Avoidance frenando → obstacle avoidance bloquea el movimiento

**Regla:** no bajar `max_linear_vel` por debajo de ~0.11. Para ir lento solo durante el barrido, cambiar el parámetro dinámicamente por estado (vía `set_parameters()` en runtime) en lugar de modificarlo globalmente.

### 2. Verificar `_decode_robust` con QR real

Buscar en log: `✅ QR REAL decodificado`. Si sigue fallando, considerar `pyzbar` — decodifica mucho mejor que OpenCV para imágenes de baja calidad o ángulos complicados.

### 3. Resolver origen de coordenadas

Alinear `mission_config.yaml` con `aruco_map.yaml` de una vez. Decidir un solo origen (recomendado: esquina inferior izquierda, consistente con `aruco_map.yaml`) y actualizar todos los waypoints.

### 4. Verificar `RECOVER_QR_VIEW`

Confirmar que el giro lento mirando al norte detecta el QR. Ajustar `recover_scan_range` si el QR queda fuera del arco de barrido.

### 5. Limpieza antes de commitear

- Eliminar helpers muertos en el FSM: `_yaw_from_odom`, `_dist`.
- Revisar el mapeo `_CRITICAL_RESUME` — apunta a `ALIGN_TO_DOCK` / `ALIGN_TO_PALLET` que en modo mock/directo ya no se usan igual.

---

## Comandos de referencia

```bash
# Build
colcon build --packages-select puzzlebot_control puzzlebot_perception puzzlebot_bringup
source install/setup.bash

# T1 — base (mapa conocido + nav)
ros2 launch puzzlebot_bringup real_robot.launch.py \
  slam:=false mcl:=false use_map:=true kalman:=true aruco:=true \
  navigation:=true lidar_topic:=/scan rviz:=true

# T2 — misión (QR real + fork mock)
ros2 launch puzzlebot_control mission.launch.py \
  mission_number:=1 mock_qr:=false mock_fork:=true qr_publish_debug:=true

# T3 — arrancar misión
ros2 topic pub --once /mission_state_in std_msgs/String "data: 'START'"
```
