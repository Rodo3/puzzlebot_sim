# SLAM Mapping del Puzzlebot

Este documento explica el nodo `slam_node`, los cambios recientes que mejoraron el mapa en Gazebo y las consideraciones para usarlo en el robot fisico.

## Resumen

El nodo actual construye un `nav_msgs/OccupancyGrid` en `/map` usando:

- `/scan`: mediciones del lidar 2D.
- `/odom`: pose del robot en el frame `odom`.
- TF `map -> odom`: publicado como identidad durante mapping.

Tecnicamente, esta implementacion es **occupancy grid mapping con pose conocida o asumida**, no SLAM completo con cierre de lazo. El mapa se actualiza scan por scan usando log-odds. En Gazebo, para obtener un mapa limpio, el launch usa por defecto una odometria ground-truth de Gazebo durante `mode:=mapping`.

## Que se cambio

### 1. Odometria ground-truth para mapping en simulacion

Antes, `mode:=mapping` usaba `dead_reckoning` desde los joint states simulados. Eso funciona para mover y estimar pose, pero al girar acumula error angular o slip. Si la pose usada para insertar el scan esta mal, las mismas paredes se dibujan en celdas distintas y parece que el mapa se sobreescribe o se deforma.

Ahora `gz_sim.launch.py` agrega el argumento:

```bash
odom_source:=ground_truth
```

Este es el default para `mode:=mapping`. El launch bridgea:

```text
/world/<world>/dynamic_pose/info
```

y lanza `ground_truth_odom`, que publica:

```text
/odom
TF odom -> base_footprint
```

Esto hace que el mapa en simulacion dependa principalmente de la geometria del lidar y no del error de encoder.

Para comparar contra odometria por ruedas:

```bash
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze mode:=mapping odom_source:=dead_reckoning
```

### 2. Sincronizacion scan-pose por timestamp

Antes el nodo integraba cada scan con "la ultima pose recibida". Durante giros rapidos, la ultima pose puede no corresponder exactamente al timestamp del scan. Ese desfase produce paredes inclinadas, duplicadas o barridas.

Ahora `slam_node` mantiene un buffer corto de odometria:

```text
pose_buffer_sec: 3.0
max_scan_pose_age: 0.20
```

Cuando llega un `LaserScan`, busca/interpola la pose de `/odom` mas cercana al `scan.header.stamp`. Si no hay pose suficientemente cercana, salta ese scan. Esto reduce mucho el error al girar.

### 3. Sensor model menos destructivo

Se ajustaron los parametros de log-odds:

```yaml
p_occ:  0.75
p_free: 0.45
l_clamp: 5.0
```

`p_occ` hace mas fuerte la evidencia de obstaculo en el endpoint del rayo. `p_free` ahora esta mas cerca de 0.5, por lo que los rayos libres limpian espacio de forma mas conservadora. Esto evita que un pequeño error de pose borre paredes que ya fueron observadas.

## Fundamento teorico

### Occupancy Grid

El mapa es una grilla 2D. Cada celda representa la probabilidad de estar ocupada:

- `100`: ocupada.
- `0`: libre.
- `-1`: desconocida.

Internamente no se guarda probabilidad directa, sino **log-odds**:

```text
l = log(p / (1 - p))
```

Esto permite acumular evidencia sumando:

```text
l_t(c) = l_{t-1}(c) + inverse_sensor_model(c) - l_0
```

Como el prior inicial es `p = 0.5`, entonces `l_0 = 0`.

### Modelo inverso del sensor

Para cada rayo del lidar:

1. Se calcula el angulo global:

```text
angle_world = scan.angle_min + i * scan.angle_increment + robot_yaw
```

2. Se calcula el endpoint:

```text
end_x = robot_x + range * cos(angle_world)
end_y = robot_y + range * sin(angle_world)
```

3. Se traza una linea desde la celda del robot hasta la celda del endpoint usando Bresenham.

4. Las celdas intermedias reciben evidencia de espacio libre:

```text
l(c) += log(p_free / (1 - p_free))
```

5. Si el rayo termino en un impacto real, la ultima celda recibe evidencia de ocupado:

```text
l(c) += log(p_occ / (1 - p_occ))
```

6. El valor se limita con `l_clamp` para evitar saturacion infinita.

### Por que antes se deformaba al girar

El algoritmo asume que la pose del robot es correcta. Si el robot gira y `/odom` reporta un yaw incorrecto o atrasado, el mismo scan se proyecta con un angulo equivocado. En una grilla de ocupacion eso tiene dos efectos:

- Las paredes se insertan en una posicion rotada o desplazada.
- Los rayos libres pasan por celdas que antes estaban marcadas como ocupadas y reducen su probabilidad.

Por eso visualmente parecia que el mapa se estaba "sobreescribiendo". La grilla estaba haciendo lo que debe, pero con una pose incorrecta o desfasada.

## Como correrlo

Build:

```bash
colcon build --packages-select puzzlebot_slam puzzlebot_bringup
source install/setup.bash
```

Mapping recomendado en Gazebo:

```bash
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze mode:=mapping
```

Teleop en otra terminal:

```bash
source install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args --remap cmd_vel:=/model/puzzlebot/cmd_vel
```

Prueba con odometria realista de ruedas:

```bash
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze mode:=mapping odom_source:=dead_reckoning
```

## Diferencia entre simulacion y robot fisico

En Gazebo, `odom_source:=ground_truth` usa la pose real del simulador. En el robot fisico eso no existe. El robot real solo tendra:

- Encoders de ruedas.
- Lidar real.
- Posiblemente IMU, ArUco u otras correcciones futuras.

Eso significa que en el robot fisico el nodo depende de la calidad de `/odom`. Si `/odom` deriva, el mapa tambien deriva.

### Lo que debes considerar en el robot real

1. **Calibrar parametros cinematicos**

   `wheel_radius` y `wheel_separation` deben medirse en el robot real. Un error pequeno en `wheel_separation` se vuelve un error grande de yaw al girar.

2. **Controlar slip**

   Giros bruscos, pisos lisos, ruedas flojas o aceleraciones altas provocan slip. El mapping por grilla lo sufre mucho porque proyecta los scans con la pose estimada.

3. **Usar velocidades moderadas**

   Para mapear, conviene moverse lento:

   - baja velocidad lineal;
   - giros suaves;
   - evitar girar en sitio durante mucho tiempo si la odometria angular no esta bien calibrada.

4. **Sincronizar timestamps**

   `/scan` y `/odom` deben usar el mismo reloj. En robot real, `use_sim_time` debe ser `False` en todos los nodos.

5. **Validar frame del lidar**

   El `LaserScan.header.frame_id` debe corresponder al frame real del lidar y debe existir el TF hacia `base_footprint`. Si hay offset fisico del lidar, debe modelarse.

6. **No esperar cierre de lazo**

   Este nodo no corrige globalmente el mapa cuando regresas al mismo lugar. Para mapas grandes o recorridos largos, necesitas agregar scan matching, MCL contra mapa previo, EKF con landmarks, o usar una libreria SLAM completa como `slam_toolbox`.

7. **Guardar mapas cortos y controlados**

   Para validar el robot fisico, empieza con un cuarto pequeno o un pasillo simple. Si el mapa sale bien ahi, escala a entornos mas grandes.

## Division interna actual

El `slam_node.py` ya no contiene todo el algoritmo en un solo archivo. Ahora es
un orquestador ROS que conecta componentes pequenos:

| Componente | Archivo | Responsabilidad |
|---|---|---|
| `SlamNode` | `slam_node.py` | Subscripciones, publishers, timers y TF |
| `Pose2D` | `slam_types.py` | Tipo comun para poses 2D |
| `slam_math` | `slam_math.py` | Bresenham, yaw, normalizacion angular, timestamps |
| `OdometryBuffer` | `odometry_buffer.py` | Buffer temporal de `/odom` e interpolacion por timestamp |
| `OccupancyGridMap` | `occupancy_grid_map.py` | Log-odds, conversion mundo-celda, integracion de rayos y publicacion de mapa |
| `KeyframeManager` | `keyframe_manager.py` | Decide si un scan debe integrarse al mapa |
| `LocalScanMatcher` | `scan_matcher.py` | Punto de extension para el scan matching real |

Por default, `KeyframeManager` y `LocalScanMatcher` no cambian el comportamiento
validado en Gazebo:

```yaml
use_keyframes: false
scan_matching_enabled: false
```

La siguiente fase para robot fisico debe implementar `LocalScanMatcher.match()`
para buscar una correccion local `(dx, dy, dtheta)` alrededor de la pose de odometria.

## Parametros importantes

Archivo: `src/puzzlebot_bringup/config/slam_params.yaml`

| Parametro | Valor actual | Efecto |
|---|---:|---|
| `map_size_pixels` | 500 | Ancho/alto en celdas |
| `map_size_meters` | 25.0 | Tamano fisico del mapa |
| `map_origin_x` | -12.5 | Origen X del OccupancyGrid |
| `map_origin_y` | -12.5 | Origen Y del OccupancyGrid |
| `p_occ` | 0.75 | Fuerza de evidencia ocupada |
| `p_free` | 0.45 | Fuerza de evidencia libre |
| `l_clamp` | 5.0 | Limite de saturacion log-odds |
| `scan_step` | 1 | Usa todos los rayos |
| `min_useful_range` | 0.20 | Ignora retornos demasiado cercanos |
| `pose_buffer_sec` | 3.0 | Historial de poses |
| `max_scan_pose_age` | 0.20 | Tolerancia scan-pose |
| `use_keyframes` | false | Si se activa, integra solo scans separados por movimiento minimo |
| `keyframe_min_translation` | 0.10 | Distancia minima entre keyframes |
| `keyframe_min_rotation` | 0.0873 | Rotacion minima entre keyframes, 5 grados |
| `scan_matching_enabled` | false | Hook para activar el scan matcher futuro |

## Criterio de exito

En Gazebo, una exploracion razonable del maze debe producir:

- paredes exteriores rectas;
- cajas internas cerradas;
- espacio libre consistente;
- pocas paredes duplicadas al girar;
- mapa estable si el robot gira sobre su eje.

Si falla con `odom_source:=dead_reckoning` pero funciona con `ground_truth`, el problema no es el mapper: es calibracion o deriva de odometria.
