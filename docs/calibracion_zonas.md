# Guía de calibración de zonas — misión logística

Cómo medir y ajustar los bounding-boxes de las zonas (conveyor / rack / dock) y el
punto de expedición en
[`mission_config.yaml`](../src/puzzlebot_control/config/mission_config.yaml).

La FSM **no usa waypoints discretos** por zona: barre un rectángulo (bounding-box)
y la detección (QR del pallet o logo del tráiler) interrumpe el barrido. Por eso
calibrar bien estos 4 números por zona es lo único que necesitas para que el robot
busque en el lugar correcto.

---

## 1. Sistema de coordenadas

Igual que [`waypoints.yaml`](../src/puzzlebot_bringup/config/waypoints.yaml):

- Origen `(0,0)` = esquina inferior **derecha** de la arena (sureste).
- **X** crece hacia la **izquierda** (Oeste).
- **Y** crece hacia **arriba** (Norte).
- Frame de referencia: **`map`** (el mismo de los goals y RViz).
- yaw: `0`=mira −X (Este) · `1.57`=mira +Y (Norte) · `3.14`=mira +X · `-1.57`=mira −Y.

---

## 2. Cómo medir un punto en RViz

1. Lanza el robot con localización y RViz (Fixed Frame = `map`):
   ```bash
   ros2 launch puzzlebot_bringup real_robot.launch.py use_map:=true mcl:=true rviz:=true
   ```
2. Espera a que la localización converja (el robot debe estar bien posicionado
   en el mapa). Verifica:
   ```bash
   ros2 topic echo /localization/status   # debe decir OK
   ```
3. En la barra de herramientas de RViz usa **"Publish Point"** y haz clic sobre
   el punto del mapa que quieres medir. Las coordenadas salen en la terminal:
   ```bash
   ros2 topic echo /clicked_point
   ```
   Anota `point.x` y `point.y`.

> Truco: mide las **4 esquinas** del rectángulo que cubre cada zona. El bbox son
> los mínimos y máximos de esos clics.

---

## 3. Qué representa cada bounding-box

Cada zona se define con `*_x_min`, `*_x_max`, `*_y_min`, `*_y_max` y `*_sweep_yaw`.
El robot genera un barrido lineal **dentro** de la caja:

| Zona | Barrido | El robot se sitúa en… | Recorre |
|------|---------|------------------------|---------|
| **conveyor** (Misión 1) | en **X** | `y_min + 0.15` (Y fija) | de `x_min` a `x_max` |
| **dock** (tráiler)      | en **X** | `y_min + 0.15` (Y fija) | de `x_min` a `x_max` |
| **rack** (Misión 2)     | en **Y** | `x_min + 0.15` (X fija) | de `y_min` a `y_max` |

- El `+0.15 m` es el margen interior desde el borde de la caja donde se planta el
  robot para barrer. Tenlo en cuenta: el robot NO entra hasta el borde, se queda
  a 15 cm.
- `*_sweep_yaw` = hacia dónde mira el robot durante el barrido. **La cámara debe
  apuntar a los objetos** (conveyors/racks/tráilers) para ver el QR/logo.
- Pasos del barrido: cada `sweep_step_m` (default 0.50 m) hay un punto de parada.

```
Ejemplo conveyor (barrido en X, Y fija):

  y_max ┌─────────────────────────┐   ← pared / fondo de la zona
        │                         │
 y_min  │ ●───●───●───●───●───●    │   ← el robot barre aquí (y_min+0.15)
  +0.15 └─────────────────────────┘
        x_min                  x_max
        (el robot va de x_min → x_max mirando hacia +Y, la cámara ve los conveyors)
```

---

## 4. Procedimiento por zona

Para **cada** zona (conveyor, rack, dock):

1. Identifica el rectángulo físico donde están los objetos (la franja de
   conveyors, el bloque de racks, la fila de docks).
2. Mide con Publish Point las esquinas. Define:
   - `x_min` = X más pequeña del rectángulo · `x_max` = X más grande.
   - `y_min` = Y más pequeña · `y_max` = Y más grande.
3. Decide el `sweep_yaw` para que la **cámara apunte a los objetos** desde donde
   barre el robot (convención: `1.57`=+Y Norte, `-1.57`=−Y Sur, `0`=−X Este,
   `3.14`=+X Oeste):
   - **conveyor**: barre en `y_min+0.15`, los conveyors están arriba (+Y) →
     mira **`1.57`** (Norte).
   - **dock**: barre en `y_min+0.15`, los tráilers están arriba (+Y) →
     mira **`1.57`** (Norte).
   - **rack**: barre en `x_min+0.15` recorriendo Y. Mira hacia la cara del
     bloque del rack: si el rack se extiende hacia +X desde donde barre →
     **`3.14`** (Oeste); si hacia −X → **`0.0`** (Este). **Verifícalo en pista.**
4. Escribe los valores en `mission_config.yaml` bajo
   `mission_manager_node: ros__parameters:`.

### Punto de expedición
`expedition_x/y/yaw` es un punto **libre de obstáculos** entre la zona de pickup y
los docks, por donde pasa el robot cargando el pallet. Mídelo con Publish Point en
una zona despejada del centro.

---

## 5. Verificar la calibración sin moverte de zona

Después de editar el YAML, **reconstruye** e inspecciona el barrido generado:

```bash
colcon build --packages-select puzzlebot_control && source install/setup.bash

# Arranca solo la FSM con el config y mira el log de generación de sweep:
ros2 run puzzlebot_control mission_manager_node \
  --ros-args --params-file install/puzzlebot_control/share/puzzlebot_control/config/mission_config.yaml \
             -p waypoints_file:=install/puzzlebot_bringup/share/puzzlebot_bringup/config/waypoints.yaml

# En otra terminal, fuerza la misión hasta el barrido y observa:
ros2 topic pub --once /localization/status std_msgs/String "data: 'OK'"
ros2 topic pub --once /mission_state_in   std_msgs/String "data: 'START'"
```

En el log de `mission_manager_node` verás algo como:
```
bbox=[0.60–3.20] × [3.80–4.40]  →  N puntos de barrido
```
Confirma que el rango y el número de puntos cubren la zona real.

### Validación visual en RViz
La FSM publica cada destino en `/goal_pose` y los puntos de barrido los recorre
con `/cmd_vel_in`. Para ver a dónde apunta:
```bash
ros2 topic echo /goal_pose          # destino A* actual (NAVIGATE_TO_*)
ros2 topic echo /mission_state      # estado de la FSM
```

---

## 6. Parámetros relacionados que quizá quieras ajustar

| Parámetro | Default | Ajusta si… |
|-----------|---------|-----------|
| `sweep_step_m` | 0.50 | Quieres más/menos paradas de escaneo por zona. |
| `sweep_wp_tol` | 0.20 | El robot "no llega" (subir) o se pasa (bajar) — depende de la odometría. |
| `sweep_speed` | 0.10 | El barrido es muy lento/rápido para que la cámara detecte. |
| `approach_distance` | 0.40 | Distancia frontal a la que se detiene antes de alinear al pallet. |
| `dock_center_thresh` | 0.05 | Qué tan centrado debe quedar el logo antes de entrar al tráiler. |
| `qr_search_timeout` / `logo_search_timeout` | 60.0 | La zona es grande y el barrido tarda más en cubrirla. |

---

## 7. Errores comunes

- **El robot barre con la cámara de espaldas a los objetos** → revisa `*_sweep_yaw`.
- **LOCALIZATION_CHECK no avanza** → `/localization/status` no da `OK`; calibra
  localización antes de calibrar zonas.
- **El robot "no llega" a los puntos de barrido** → `sweep_wp_tol` muy chico para
  la precisión de tu odometría; súbelo a 0.25–0.30.
- **Coordenadas con signo invertido** → recuerda que X crece hacia la IZQUIERDA.
