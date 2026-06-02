# Claude Handoff 8 — Bug Navigation: Evasión Reactiva de Obstáculos Dinámicos

**Date:** 2026-05-30
**Branch:** `mi_rama_de_pruebas`
**Platform:** Puzzlebot Differential Drive — Jetson Orin (sensors) + PC (all compute)
**Previous handoff:** `handoffClaude7.md`

---

## Objective

Implementar navegación reactiva Bug0/Bug2 para evadir obstáculos dinámicos no incluidos en el mapa conocido, integrada con el stack existente de A* + pure pursuit + EKF + ArUco.

---

## Result

✅ **`bug_navigation_node.py` creado** — detecta obstáculos, inyecta en `/map`, fuerza replan del A*
✅ **`obstacle_avoidance_node.py` mejorado** — retroceso automático al llevar N segundos bloqueado
✅ **`steering_controller_node.cpp` mejorado** — frenado proporcional al ángulo + closest waypoint al recibir ruta nueva
✅ **`path_planner_node.py` mejorado** — BFS para encontrar celda libre cuando start está bloqueado por obstáculo dinámico
⚠️ **Problema pendiente crítico:** el robot NO se detiene al detectar el obstáculo antes de recibir la nueva ruta
⚠️ **Problema pendiente crítico:** la ruta de evasión no es suficientemente agresiva — el A* genera rutas que pasan demasiado cerca del obstáculo o por el mismo corredor bloqueado

---

## Architecture

```
/scan_stamped ──→ bug_navigation_node
                  │  detecta obstáculo frontal (< front_stop_distance)
                  │  publica Path vacío → /planned_path (detener steering)
                  │  inyecta blob en /map con radio safe_radius
                  │
                  ▼
              path_planner_node (A*)
                  │  recibe /map con obstáculo pintado
                  │  BFS para celda libre si start está bloqueado
                  │  replantea ruta desde posición actual
                  ▼
              /planned_path (ruta nueva)
                  │
              steering_controller_node (pure pursuit)
                  │  empieza desde waypoint más cercano al robot
                  │  frena proporcionalmente al ángulo de giro
                  ▼
              /cmd_vel_steering
                  │
              obstacle_avoidance_node
                  │  stop_distance: 0.22 m (emergencia real)
                  │  retroceso automático tras stuck_timeout_sec bloqueado
                  ▼
              /cmd_vel → motores
```

**Topic flow completo:**
```
/scan_stamped  →  bug_navigation_node  →  /map (aumentado)
                                       →  /planned_path (vacío para detener)
/map           →  path_planner_node    →  /planned_path (ruta nueva)
/planned_path  →  steering_controller →  /cmd_vel_steering
/cmd_vel_steering → bug_navigation_node → /cmd_vel_in (pass-through)
/cmd_vel_in    →  obstacle_avoidance   →  /cmd_vel
```

---

## Files Changed This Session

| Archivo | Cambio |
|---|---|
| `src/puzzlebot_planning/puzzlebot_planning/bug_navigation_node.py` | **NUEVO** — nodo completo de evasión reactiva |
| `src/puzzlebot_planning/puzzlebot_planning/obstacle_avoidance_node.py` | Retroceso automático (`stuck_timeout_sec`) |
| `src/puzzlebot_planning/puzzlebot_planning/path_planner_node.py` | BFS para celda libre cuando start bloqueado |
| `src/puzzlebot_controller/src/steering_controller_node.cpp` | Frenado por ángulo + closest waypoint en path_cb |
| `src/puzzlebot_bringup/launch/navigation.launch.py` | Agrega `bug_navigation_node`, remapping `/cmd_vel_steering` |
| `src/puzzlebot_bringup/config/controller_params.yaml` | Parámetros de bug_nav + obstacle_avoidance actualizados |
| `src/puzzlebot_planning/setup.py` | Registra `bug_navigation_node` como entry point |

---

## Key Code Details

### bug_navigation_node.py

**Lógica principal (10 Hz):**
1. Expirar obstáculos viejos del mapa (decay configurable)
2. Si `min_front < front_stop_distance` durante `obstacle_confirm_cycles` ciclos consecutivos:
   - Calcular posición del obstáculo: `obs = robot_pos + min_front * robot_direction`
   - Calcular `safe_radius = min(inj_radius, dist_robot_to_obs - 0.20)` — nunca cubre al robot
   - Publicar `Path()` vacío en `/planned_path` → steering controller para
   - Publicar `/map` con blob de `safe_radius` en `obs` → A* replantea
3. Cooldown: no re-inyectar el mismo punto por 15 s (si está a < 0.50 m del anterior)

**Parámetros clave:**
```yaml
front_stop_distance:        0.50    # [m] detecta antes que obstacle_avoidance (0.22)
obstacle_confirm_cycles:    3       # 0.3 s a 10 Hz para confirmar
obstacle_inject_radius_m:   0.45   # radio del blob en el mapa
obstacle_inject_decay_sec:  25.0   # segundos hasta borrar
blocked_inject_radius_m:    1.10   # radio en recovery (bloquea corredor)
blocked_inject_decay_sec:   60.0   # decay en recovery
```

### obstacle_avoidance_node.py — Retroceso automático

```python
# Si lleva stuck_timeout_sec bloqueado:
# Fase 1 (0–1.5 s): linear.x = -reverse_speed, angular.z = 0.25
# Fase 2 (1.5–rec_wait s): solo girar
# Fase 3: evaluar frontal → si libre → GO_TO_WAYPOINT
```

Parámetros:
```yaml
stop_distance:        0.22   # m — bajado de 0.30 para dar margen al bug_nav
slow_distance:        0.45   # m
stuck_timeout_sec:    3.0    # s — tiempo bloqueado antes de retroceder
reverse_speed:        0.07   # m/s
reverse_duration_sec: 2.0    # s
```

### steering_controller_node.cpp — Frenado por ángulo

```cpp
// Frenado proporcional: 0°→100% vel, 20°→100%, 90°→20%
double speed_scale = 1.0 - 0.8 * ((abs_err - 0.35) / (M_PI/2 - 0.35));
speed_scale = clamp(speed_scale, 0.20, 1.0);
v *= speed_scale;

// Freno cerca del goal
if (dist_to_goal < 0.40) v = min(v, max_v_ * (dist_to_goal / 0.40));
```

Al recibir ruta nueva, busca el waypoint más cercano al robot (no siempre idx=0).

### path_planner_node.py — BFS para start bloqueado

```python
# Si start cell está bloqueada incluso con inflación reducida:
# BFS en espiral hasta radio 12 celdas (0.60 m) para celda libre
for search_r in range(1, 12):
    for dr, dc in spiral(search_r):
        if not inflated[start[0]+dr, start[1]+dc]:
            start = (start[0]+dr, start[1]+dc)
            break
```

---

## Current Parameters (controller_params.yaml)

```yaml
bug_navigation_node:
  bug_algorithm:              "bug2"
  front_stop_distance:        0.50    # [m]
  front_angle_deg:            30.0
  obstacle_confirm_cycles:    3
  obstacle_inject_radius_m:   0.45
  obstacle_inject_decay_sec:  25.0
  blocked_inject_radius_m:    1.10
  blocked_inject_decay_sec:   60.0
  enable_rviz_markers:        true

steering_controller_node:
  lookahead_distance:  0.30   # m — reducido para curvas cerradas
  max_linear_vel:      0.14   # m/s
  max_angular_vel:     0.80   # rad/s
  goal_tolerance:      0.12   # m

obstacle_avoidance_node:
  stop_distance:        0.22   # m
  slow_distance:        0.45   # m
  stuck_timeout_sec:    3.0
  reverse_speed:        0.07
  reverse_duration_sec: 2.0
  cov_slow_threshold:   0.15
  cov_stop_threshold:   0.8
  cov_timeout_sec:      2.0
```

---

## Pending Problems (Critical)

### 1. [Crítico] Robot no se detiene al detectar obstáculo

**Síntoma:** el `bug_navigation_node` publica `Path()` vacío en `/planned_path` al detectar obstáculo, pero el steering controller NO para el robot. El robot sigue avanzando y choca.

**Causa probable:** el `steering_controller` suscribe `/planned_path` pero puede estar ignorando el path vacío porque en `path_cb` solo actualiza `path_` sin verificar si es vacío explícitamente antes de que el control_loop lo procese, o hay un race condition entre el path vacío y la ruta nueva que llega inmediatamente después del replan.

**Lo que hay que verificar:**
- Confirmar con `ros2 topic echo /planned_path` que el path vacío llega al steering controller
- Confirmar que `path_.empty()` en `control_loop` devuelve Twist(0)
- Posible solución alternativa: el `bug_navigation_node` también publique directamente Twist(0) en `/cmd_vel_in` durante N ciclos mientras el A* replantea, en lugar de depender del path vacío

### 2. [Crítico] Ruta de evasión no es suficientemente agresiva

**Síntoma:** el A* genera rutas que pasan demasiado cerca del obstáculo dinámico o por el mismo corredor bloqueado, el robot intenta seguirlas y vuelve a chocar.

**Causa:** el radio de inyección (`safe_radius`) se calcula como `min(0.45, dist_obs - 0.20)`. Cuando el robot está a 0.22 m del obstáculo, `safe_radius = 0.02 m` — demasiado pequeño para que el A* genere una ruta realmente diferente.

**Solución propuesta:**
- Aumentar `obstacle_inject_radius_m` a 0.70–0.80 m para que el A* genere rutas que pasen mínimo 0.70 m alejadas del obstáculo
- En el cálculo de `safe_radius`, usar el máximo posible sin cubrir al robot en lugar del mínimo: `safe_radius = dist_obs - 0.25` clampado a `[0.20, inj_radius]`
- Alternativamente, inyectar SIEMPRE en la posición calculada del obstáculo (no del robot) para que el robot nunca quede dentro del blob
- Considerar inyectar obstáculos adicionales formando una "pared virtual" más larga perpendicular al movimiento del robot

### 3. [Media] Spam de re-inyección del mismo obstáculo

**Síntoma:** en los logs aparecen 4–5 inyecciones del mismo obstáculo en pocas segundos, acumulando hasta 5 blobs en el mapa aunque el cooldown sea de 15 s.

**Causa:** el cooldown compara la posición del obstáculo calculado (que varía levemente con el LiDAR) contra `_last_obs_x/_last_obs_y`. Con variación de 0.05 m en el cálculo y umbral de 0.50 m, funciona — pero si el robot hace retroceso y el obstáculo queda en una posición diferente, recalifica como "nuevo".

**Solución:** aumentar el cooldown a 20 s y el umbral de posición a 0.70 m.

---

## What Is Working

- ✅ Detección de obstáculos con LiDAR (`min_front < front_stop_distance`)
- ✅ Inyección de obstáculo en `/map` con radio seguro (no cubre al robot)
- ✅ El A* replantea automáticamente al recibir el mapa nuevo (`replan_on_new_map: true`)
- ✅ BFS en path_planner encuentra celda libre cuando start está dentro del blob
- ✅ Steering controller usa waypoint más cercano al recibir ruta nueva (no idx=0)
- ✅ Frenado proporcional al ángulo en el pure pursuit
- ✅ Retroceso automático en obstacle_avoidance tras N segundos bloqueado
- ✅ Marcadores RViz para visualizar blobs de obstáculos dinámicos

---

## What Is NOT Working

- ❌ Robot no se detiene al publicar path vacío — sigue moviéndose hacia el obstáculo
- ❌ Ruta alternativa generada pasa demasiado cerca del obstáculo
- ❌ Re-inyección en bucle cuando el cooldown falla

---

## Recommended Next Steps

### Paso 1 — Arreglar la detención inmediata (prioridad máxima)

En vez de depender de que el steering controller procese el path vacío, hacer que el `bug_navigation_node` publique directamente Twist(0) en `/cmd_vel_in` durante 2 segundos mientras el A* replantea:

```python
# En _inject_and_publish():
# Activar flag de "deteniendo para replan"
self._stopping_for_replan = True
self._replan_stop_until = time.monotonic() + 2.0  # 2 s de parada

# En _steering_cb():
def _steering_cb(self, msg):
    now = time.monotonic()
    if self._stopping_for_replan and now < self._replan_stop_until:
        self._pub_cmd.publish(Twist())  # publicar cero en vez de pasar el cmd
        return
    self._stopping_for_replan = False
    self._pub_cmd.publish(msg)
```

### Paso 2 — Ruta de evasión más agresiva

Opciones en orden de menor a mayor cambio:

**Opción A (YAML, sin código):**
```yaml
obstacle_inject_radius_m: 0.80   # blob más grande → A* se aleja más
obstacle_inject_decay_sec: 30.0
```

**Opción B (código, moderado):**
En `bug_navigation_node._loop()`, al detectar obstáculo, inyectar dos blobs:
- Blob 1: en la posición del obstáculo con radio `inj_radius`
- Blob 2: offset lateral (derecha o izquierda según `dist_left` vs `dist_right`) con radio `inj_radius * 0.7`
Esto forma una "L" virtual que obliga al A* a rodear más ampliamente.

**Opción C (código, agresivo — recomendada):**
Inyectar una "pared virtual" de 3–5 blobs formando una línea perpendicular a la dirección del robot, cubriendo todo el ancho del corredor potencialmente bloqueado. Esto garantiza que el A* genere ruta por el lado completamente opuesto.

### Paso 3 — Verificar que path vacío llega al steering controller

```bash
# En una terminal separada mientras navega:
ros2 topic echo /planned_path --field poses | head -5
# Al detectar obstáculo debe aparecer un mensaje con poses: []
```

---

## Build Status

```
puzzlebot_planning    ✓  (bug_navigation_node, obstacle_avoidance, path_planner)
puzzlebot_controller  ✓  (steering_controller con frenado + closest waypoint)
puzzlebot_bringup     ✓  (navigation.launch.py + controller_params.yaml)
```

```bash
cd ~/Documents/puzzlebot_sim
colcon build --packages-select puzzlebot_planning puzzlebot_controller puzzlebot_bringup
source install/setup.bash
```

---

## Launch Command

```bash
ros2 launch puzzlebot_bringup real_robot.launch.py \
  slam:=false mcl:=false kalman:=true aruco:=true use_map:=true \
  rviz:=true lidar_topic:=/scan navigation:=true
```

**Monitoreo:**
```bash
# Estado de detección de obstáculos
ros2 topic echo /bug_nav/state

# Verificar que path vacío se publica al detectar obstáculo
ros2 topic echo /planned_path --field header

# Ver velocidades reales enviadas a motores
ros2 topic echo /cmd_vel

# Ver velocidades del steering controller (antes de obstacle_avoidance)
ros2 topic echo /cmd_vel_steering
```

---

## TF Chain (sin cambios desde handoff 7)

```
map ──aruco_map_odom──▶ odom ──kalman_filter_node──▶ base_footprint
```

---

## Archivos modificados esta sesión (desde commit 1408a98)

Los 15 archivos en `git diff --stat HEAD` incluyen todos los cambios de las sesiones 7 y 8.
El próximo commit debería incluir al mínimo:
- `bug_navigation_node.py` (nuevo)
- `obstacle_avoidance_node.py` (retroceso)
- `steering_controller_node.cpp` (frenado + closest waypoint)
- `path_planner_node.py` (BFS start bloqueado)
- `navigation.launch.py` (bug_navigation_node integrado)
- `controller_params.yaml` (parámetros nuevos)
