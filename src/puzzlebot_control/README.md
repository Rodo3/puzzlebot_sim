# puzzlebot_control

Owns high-level mission logic for the warehouse delivery mission.

Do not put here:
- Low-level steering loops.
- SLAM or localization algorithms.
- Perception model inference.
- Path-planning internals.

---

## Nodo principal: `state_machine_node`

Máquina de estados que coordina la misión logística completa:
**escanear QR → recoger pallet (montacargas) → navegar a docks → identificar tráiler por logo → depositar pallet**.

### Estados

| Estado | Descripción |
|---|---|
| `IDLE` | Esperando comando de inicio de misión |
| `WAITING_FOR_GOAL` | Solo Misión 2 — espera click en mapa como punto de inicio |
| `GOING_TO_START` | Solo Misión 2 — navega al punto elegido por el usuario |
| `SCANNING_QR` | Patrulla waypoints buscando el QR con `/qr/detections` |
| `FORKLIFT_UP` | Activa el montacargas para recoger el pallet (stub hasta implementación) |
| `NAVIGATING_TO_DOCKS` | Navega al waypoint `dock_scan` (único hardcodeado) |
| `SCANNING_LOGOS` | Desde `dock_scan`, compara `/logo_detection/result` con el target del QR |
| `FORKLIFT_DOWN` | Baja el montacargas para depositar el pallet (stub) |
| `DONE` | Misión completada — regresa a IDLE |
| `ERROR` | Fallo irrecuperable — publica velocidad cero y espera reset |

### Diagrama de transiciones

```
/mission_start "1" ──────────────────────────────→ SCANNING_QR
                                                          │
/mission_start "2" ──→ WAITING_FOR_GOAL                  │ QR estable detectado
                              │                           │ (N frames consecutivos)
                         /goal_pose                       ▼
                         recibido                     FORKLIFT_UP
                              │                           │
                              ▼                      timeout/confirmación
                       GOING_TO_START                     │
                              │                           ▼
                         llegó (odom)           NAVIGATING_TO_DOCKS
                              │                           │
                              ▼                      llegó a dock_scan
                         SCANNING_QR                      │
                                                          ▼
                                                   SCANNING_LOGOS
                                                          │ match logo == target
                                                          │ conf ≥ umbral
                                                          ▼
                                                   FORKLIFT_DOWN
                                                          │
                                                          ▼
                                                        DONE → IDLE

/mission_stop (cualquier estado) ──────────────────→ IDLE
```

### Tópicos suscritos

| Tópico | Tipo | Cuándo se usa |
|---|---|---|
| `/mission_start` | std_msgs/String | `"1"` o `"2"` — inicia misión |
| `/qr/detections` | std_msgs/String | JSON con `[{data, corners}]` — en SCANNING_QR |
| `/logo_detection/result` | std_msgs/String | JSON con `[{class_name, confidence, bbox}]` — en SCANNING_LOGOS |
| `/odom` | nav_msgs/Odometry | detección de llegada a waypoints por distancia euclidiana |
| `/goal_pose` | geometry_msgs/PoseStamped | captura el punto de inicio en WAITING_FOR_GOAL (solo Misión 2) |

### Tópicos publicados

| Tópico | Tipo | Cuándo |
|---|---|---|
| `/mission_state` | std_msgs/String | Estado actual a 2 Hz (para dashboard y bridge) |
| `/navigate_to_waypoint` | std_msgs/String | Nombre del waypoint destino (o `"stop"` para cancelar) |
| `/forklift/command` | std_msgs/String | `"up"` o `"down"` (stub hasta implementación del montacargas) |
| `/mission/markers` | visualization_msgs/MarkerArray | Punto de color + etiqueta en RViz donde se confirmó cada QR/logo |
| `cmd_vel_topic` | geometry_msgs/Twist | **Solo** un `Twist` cero como parada de seguridad en `ERROR`/`stop` |

### Parada de seguridad (`cmd_vel_topic`)

En `ERROR` y al recibir `mission_stop`, el nodo cancela la navegación (`/navigate_to_waypoint` ← `"stop"`)
y publica **un** `Twist` cero en `cmd_vel_topic`. Ese parámetro **debe** coincidir con el tópico final
del robot, igual que `cmd_vel_out_topic` del bridge:

| Entorno | `cmd_vel_topic` |
|---|---|
| Robot real | `/cmd_vel` (default) |
| Gazebo | `/model/puzzlebot/cmd_vel` |

### Markers en RViz

Cada vez que se confirma un QR (en `SCANNING_QR`) o un logo (en `SCANNING_LOGOS`), el nodo planta una
esfera + etiqueta de texto en `/mission/markers`, frente a la pose actual del robot. Colores: QR cyan,
Walmart verde, Pepsi rojo, Amazon ámbar. Se limpian al iniciar una nueva misión.

**Para verlos:** en RViz, *Add → By topic → `/mission/markers` → MarkerArray*. El frame por default es
`odom` (RViz lo lleva a `map` por TF); ajustable con el parámetro `marker_frame`.

### Mapeo QR → logo

El QR contiene el nombre del cliente en formato interno:

| String en QR | Logo del tráiler |
|---|---|
| `wolmar` | `Walmart` |
| `popsi` | `Pepsi` |
| `emezon` | `Amazon` |

### Detección de llegada a waypoint

El nodo carga el mismo `waypoints.yaml` que usa `waypoint_navigator_node` para conocer las coordenadas `(x, y)` de cada waypoint. Compara con la pose actual de `/odom`. Umbral configurable: `goal_reached_tolerance` (default 0.20 m).

### Config: `mission_config.yaml`

Ubicado en `puzzlebot_control/config/mission_config.yaml`:

```yaml
# Mapeo string QR → nombre clase del logo_detector
qr_logo_map:
  wolmar: Walmart
  popsi: Pepsi
  emezon: Amazon

# Waypoint único de la zona de docks (desde donde se ven los 3 logos)
dock_waypoint: dock_scan

# Waypoints que el robot recorre buscando el QR (usan los de bringup/config/waypoints.yaml)
patrol_waypoints: [estacion_a, estacion_b, estacion_c, estacion_d, estacion_e]

# Segundos en cada waypoint de patrulla antes de avanzar al siguiente
scan_timeout_sec: 8.0

# Confianza mínima del logo_detector para aceptar un match
logo_confidence_threshold: 0.70

# Tolerancia de llegada a waypoint (metros)
goal_reached_tolerance: 0.20

# Frames consecutivos del mismo QR para aceptarlo como válido
qr_stable_frames: 3

# Frames consecutivos con el mismo logo para aceptar el match
logo_stable_frames: 5

# Ruta al waypoints.yaml de puzzlebot_bringup (para leer coordenadas)
waypoints_file: ""   # rellenar con path absoluto o configurar en launch file
```

### Build y ejecución

```bash
colcon build --packages-select puzzlebot_control
source install/setup.bash

ros2 run puzzlebot_control state_machine_node \
  --ros-args \
  -p mission_config_file:=src/puzzlebot_control/config/mission_config.yaml \
  -p waypoints_file:=src/puzzlebot_bringup/config/waypoints.yaml

# En Gazebo, apuntar la parada de seguridad al tópico del DiffDrive:
ros2 run puzzlebot_control state_machine_node \
  --ros-args \
  -p mission_config_file:=src/puzzlebot_control/config/mission_config.yaml \
  -p waypoints_file:=src/puzzlebot_bringup/config/waypoints.yaml \
  -p cmd_vel_topic:=/model/puzzlebot/cmd_vel
```

### Lanzar misión desde terminal (para testing)

```bash
# Misión 1 — robot ya está en zona de transportadores
ros2 topic pub --once /mission_start std_msgs/String '{data: "1"}'

# Misión 2 — el nodo espera un goal_pose del dashboard (click en mapa)
ros2 topic pub --once /mission_start std_msgs/String '{data: "2"}'

# Detener misión
ros2 topic pub --once /mission_start std_msgs/String '{data: "stop"}'

# Monitorear estado
ros2 topic echo /mission_state
```

### Notas de diseño

- **No** lanzar junto con el `state_machine_node` original (eran el mismo nodo antes de esta expansión).
- El bridge solo retransmite `/mission_start` desde el dashboard; la lógica de decisión vive aquí, no en el bridge.
- Los estados `FORKLIFT_UP` / `FORKLIFT_DOWN` hacen timeout automático mientras el montacargas no esté implementado — publican `/forklift/command` y avanzan tras `forklift_timeout_sec` segundos.
- En WAITING_FOR_GOAL, el primer `/goal_pose` recibido se usa como destino de inicio. No compite con la navegación normal porque el robot solo entra en este estado cuando el usuario inicia Misión 2 explícitamente.

---

# 🎯 DISEÑO OBJETIVO (pendiente de implementar)

> Esta sección captura el **diseño completo de la misión** según las descripciones
> oficiales (Misión 1: carga desde final de línea; Misión 2: carga desde estantes).
> La implementación actual (arriba) es una versión reducida. **Aún NO implementado.**
> Retomar desde aquí cuando haya tiempo/tokens.

## Contexto físico confirmado

- **Mapa:** `slam_map_20260529_235356.png` (4.30 × 5.40 m, 0.05 m/px, origen `(-0.25,-0.25)`).
  Pista 3.76 × 4.86 m. Estructuras (transportadores/estantes) al centro; **los 3 docks
  de tráileres en la esquina inferior-derecha (SE)**.
- **ArUcos:** `aruco_map.yaml` define 5 (IDs 0–4) en paredes **N/E/O** (el usuario menciona
  4 físicos — reconciliar). **Hay ArUcos alrededor de los logos** → ayudan a localizar en el dock.
- **Localización:** EKF con `init_from_aruco: true` → arranca sin fijar y se ancla con la
  **primera lectura de ArUco** (pose absoluta del marcador conocido). **Sin `/initialpose`.**
- **Pallet:** 1 solo, con QR **siempre de frente al pasillo**. QR mide 4.5×4.5 cm y 9×9 cm.
  El QR solo indica el **destino** (qué tráiler), no desambigua pallets.
- **Docks/logos:** posiciones **fijas** (no se mueven). Los logos probablemente fijos
  también, pero **NO hardcodear** el mapeo logo→dock: determinarlo dinámicamente por la
  **posición horizontal del bbox** del logo en la imagen (izq/centro/der).

## Máquina de estados objetivo

```
LOCALIZING            ← girar en sitio hasta fix de ArUco
   │
   ├─ Misión 1 ─→ EXPLORE_CONVEYORS ──┐
   │                                   │
   └─ Misión 2 ─→ GO_TO_SHELF_ZONE     │   (punto definido / click humano)
                       │               │
                       ▼               │
                  EXPLORE_SHELVES ──────┤   ← barre el frente de la estructura
                                        │     monitoreando /qr/detections
                          (QR visible)  ▼
                                  ALIGN_TO_PALLET   ← aproximación controlada + alinear
                                        │             (ver "Alineación", PENDIENTE)
                            (alineado y QR legible → cliente identificado)
                                        ▼
                                  PICKUP_PALLET     ← aprox final + horquillas ↑ [stub lifter]
                                        ▼
                                  NAV_TO_EXPEDITION ← navega a dock_scan
                                        ▼
                                  SCAN_LOGOS        ← ve los 3 logos, ubica el target
                            (target logo → dock izq/centro/der por posición del bbox)
                                        ▼
                                  ENTER_TRAILER     ← navega al dock correcto y entra
                                        ▼
                                  DEPOSIT_PALLET    ← horquillas ↓ en el piso [stub lifter]
                                        ▼
                                  EXIT_DOCK         ← reversa, sale del tráiler
                                        ▼
                                      DONE → IDLE
```

`EXPLORE_CONVEYORS` y `EXPLORE_SHELVES` son el mismo comportamiento (barrer un frente
buscando el QR); cambia la zona y cómo se llega (M1 ya arranca ahí; M2 navega primero al
punto humano). De `ALIGN_TO_PALLET` en adelante todo es compartido.

## Mapeo specs oficiales → estados

| Paso (Misión 1) | Estado |
|---|---|
| 1. Exploración zona transportadores | `EXPLORE_CONVEYORS` |
| 2. Identificación del pallet (alinear + leer QR) | `ALIGN_TO_PALLET` |
| 3. Toma del pallet (horquillas) | `PICKUP_PALLET` |
| 4. Navegación a zona de expedición | `NAV_TO_EXPEDITION` |
| 5. Identificación del tráiler (logo vs QR) | `SCAN_LOGOS` |
| 6. Ingreso al tráiler | `ENTER_TRAILER` |
| 7. Depósito del pallet | `DEPOSIT_PALLET` |
| 8. Salida del dock | `EXIT_DOCK` |

Misión 2 = pasos 1 (ir a zona de estantes) + 2 (explorar) y luego **idéntico a M1 pasos 2–8**.

## Alineación frente al pallet — PENDIENTE (definir)

`ALIGN_TO_PALLET` necesita control fino de movimiento. **Aún no decidido el método.**
Ideas en evaluación (todas con medidas conocidas: cámara calibrada, tamaño QR 4.5/9 cm,
dimensiones del pallet, separación de las tenazas del lifter):

- **Caso 1 — geometría del pallet:** si el robot ve el pallet completo, con sus dimensiones
  conocidas + cámara calibrada se calcula la pose alineada (teniendo en cuenta la separación
  de las tenazas). Más robusto para la inserción final, pero requiere detectar el pallet, no
  solo el QR.
- **Caso 2 — servo sobre el QR/ArUco:** centrarse lentamente en el centro del pallet usando
  el QR (`center.nx/ny` ya publicado por `qr_node`), estimar la distancia cámara→QR por el
  tamaño aparente (`area_px` + tamaño físico conocido), y a cierto **umbral de distancia**
  intentar levantar las horquillas.

**Separación de responsabilidades sugerida:** el control de velocidad de la alineación NO
debe vivir en el `state_machine` (que solo orquesta). Opción limpia: un nodo dedicado
`pallet_approach_node` que haga el visual-servoing y publique `cmd_vel`, activado/desactivado
por el state_machine. Coarse (navegación normal a un `goal_pose` ~30 cm enfrente, estimado
por la pose del robot + QR) + fine (el approach node) es la combinación recomendada.

> El `qr_node` ya expone `center` (px + normalizado `nx/ny`) y `area_px` justamente para esto.

## Zona de docks (refinada)

Ya no es un solo `dock_scan` + depósito ahí mismo:
1. `dock_scan` — vantage donde se ven los **3 logos** (a distancia, esquina SE).
2. `SCAN_LOGOS` ubica el target y, por la **posición del bbox**, decide dock izq/centro/der.
3. `ENTER_TRAILER` → navega al dock elegido y entra.
4. `DEPOSIT_PALLET` → horquillas abajo.
5. `EXIT_DOCK` → reversa para salir.

Hay ArUcos alrededor de los logos → usarlos para localizar con precisión al entrar.

## Waypoints a medir en RViz (coords TODO)

| Waypoint | Descripción |
|---|---|
| `start_m1` | Zona de transportadores, **orientado hacia la pared de docks** (lo pide la spec) |
| `conveyor_sweep_1..n` | Frente a los transportadores (barrido M1) |
| `shelf_zone_ref` | Punto de referencia de estantes (M2; o el click humano) |
| `shelf_sweep_1..n` | Frente a los estantes (barrido M2) |
| `dock_scan` | Vantage donde se ven los 3 logos (SE) |
| `dock_left` / `dock_center` / `dock_right` | Entrada a cada uno de los 3 tráileres |

## Qué cambia vs la implementación actual

| Hoy | Objetivo |
|---|---|
| `SCANNING_QR` patrulla `estacion_a..e` | `EXPLORE_*` barre el frente de la estructura |
| Lee QR al vuelo en el dwell | `ALIGN_TO_PALLET` primero, luego confirma lectura |
| `dock_scan` único + depósito ahí | `dock_scan` + 3 docks + `ENTER_TRAILER`/`DEPOSIT`/`EXIT_DOCK` |
| Sin localización inicial | `LOCALIZING` (girar hasta fix ArUco) |
| Sin alineación de pickup | `ALIGN_TO_PALLET` + `pallet_approach_node` (método pendiente) |

## Preguntas abiertas / pendientes

- Definir el **método de alineación** (Caso 1 vs Caso 2 vs híbrido).
- Confirmar **cobertura de ArUco desde `dock_scan`** (los docks están al SE; marcadores en N/E/O,
  aunque hay ArUcos alrededor de los logos → ayuda).
- Medir todas las coordenadas de waypoints en el mapa real.
- Reconciliar **4 ArUcos físicos vs 5 en `aruco_map.yaml`**.
