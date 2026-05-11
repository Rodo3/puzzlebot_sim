# Agent Context — Puzzlebot ROS 2 + Gazebo Fortress Integration

Este documento es para un agente de Claude que va a integrar el stack de simulación
del Puzzlebot (ROS 2 Humble + Gazebo Fortress) en un repositorio nuevo.
Contiene todo el contexto necesario para reproducir exactamente lo que ya funciona,
sin necesidad de adivinar versiones, nombres de plugins, o convenciones.

---

## 1. Stack exacto que funciona

| Capa | Software | Versión |
|---|---|---|
| OS | Ubuntu 22.04 | — |
| Middleware | ROS 2 | Humble |
| Simulador | Gazebo | **Fortress (ignition-gazebo 6)** |
| Bridge | ros_gz_bridge | `ros-humble-ros-gz` (Fortress) |
| Python | Python | 3.10 |
| Dep extra | Pillow | instalado vía `apt python3-pil` |

**Regla crítica:** `ros-humble-ros-gz` instala el bridge de **Fortress**, no de Harmonic.
El binario correcto es `ign gazebo`, no `gz sim`.
Mezclar los dos genera un ABI mismatch silencioso donde el DiffDrive plugin nunca registra
su subscriber y el robot jamás se mueve.

---

## 2. Por qué NO se usa Harmonic (gz sim 8)

Cuando haces `sudo apt install ros-humble-ros-gz` en Ubuntu 22.04 obtienes el bridge de
Fortress (`ignition.msgs`). Si además tienes Harmonic instalado (`gz-harmonic`) y lanzas
`gz sim` en lugar de `ign gazebo`, los namespaces de mensajes no coinciden:

| Paquete | Namespace de mensajes | Binario |
|---|---|---|
| `ros-humble-ros-gz` (Fortress) | `ignition.msgs` | `ign gazebo` |
| `gz-harmonic` | `gz.msgs` | `gz sim` |

Para detectar el problema en runtime:
```bash
ign topic -i -t /model/<robot_name>/cmd_vel
# Debe mostrar DOS subscribers: uno del bridge y uno del DiffDrive plugin.
# Si solo aparece uno, hay mismatch de versiones.
```

---

## 3. Estructura de paquetes requerida

El workspace actual separa descripción, bringup, SLAM/localización, planeación y
percepción. El agente puede adaptarlos, pero los roles deben mantenerse separados:

```
ws/
└── src/
    ├── puzzlebot_description/   # meshes, URDF/SDF, worlds, RViz configs
    ├── puzzlebot_bringup/       # launch principal gz_sim.launch.py + configs
    ├── puzzlebot_localization/  # odometry_node, kalman_filter_node, pose debug sources
    ├── puzzlebot_slam/          # slam_node, map representation, scan matcher, mcl
    ├── puzzlebot_planning/      # A*, obstacle avoidance
    ├── puzzlebot_control/       # state machine / control de alto nivel
    └── puzzlebot_perception/    # ArUco, cámara, YOLO skeletons
```

### puzzlebot_description — CMakeLists.txt mínimo

```cmake
cmake_minimum_required(VERSION 3.8)
project(puzzlebot_description)
find_package(ament_cmake REQUIRED)
install(DIRECTORY meshes urdf rviz DESTINATION share/${PROJECT_NAME})
ament_package()
```

### puzzlebot_bringup — CMakeLists.txt/setup mínimo

En este repo `puzzlebot_bringup` contiene el launch principal y configs. Mantener
instalados `launch/` y `config/` en el paquete.

### puzzlebot_bringup — package.xml (dependencias ejecutables)

```xml
<exec_depend>puzzlebot_description</exec_depend>
<exec_depend>puzzlebot_slam</exec_depend>
<exec_depend>robot_state_publisher</exec_depend>
<exec_depend>ros_gz_sim</exec_depend>
<exec_depend>ros_gz_bridge</exec_depend>
<exec_depend>rviz2</exec_depend>
```

### puzzlebot_slam — setup.py

Paquete `ament_python`. Requiere tres archivos (`setup.py`, `setup.cfg`,
`pyproject.toml`) porque setuptools ≥ 64 rompió `--editable` sin `pyproject.toml`.

```python
# setup.py
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'puzzlebot_slam'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # incluir el mapa PNG junto al paquete instalado:
        (os.path.join('share', package_name, 'puzzlebot_slam'),
            glob('puzzlebot_slam/*.png') + glob('puzzlebot_slam/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'slam_node = puzzlebot_slam.slam_node:main',
            'mcl = puzzlebot_slam.mcl:main',
        ],
    },
)
```

```toml
# pyproject.toml  — OBLIGATORIO con setuptools >= 64
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"
```

```ini
# setup.cfg
[develop]
script_dir=$base/lib/puzzlebot_slam
[install]
install_scripts=$base/lib/puzzlebot_slam
```

### puzzlebot_slam — package.xml

```xml
<depend>rclpy</depend>
<depend>geometry_msgs</depend>
<depend>nav_msgs</depend>
<depend>sensor_msgs</depend>
<depend>std_msgs</depend>
<depend>tf2_ros</depend>
<exec_depend>python3-pil</exec_depend>   <!-- requerido por mcl.py -->
```

---

## 4. SDF del robot — reglas críticas

Archivo: `src/puzzlebot_description/sdf/puzzlebot_gz.sdf`

### Plugins de Fortress (nombres exactos — NO cambiar)

```xml
<!-- DiffDrive: filename completo con .so, class con ignition:: -->
<plugin filename='libignition-gazebo-diff-drive-system.so'
        name='ignition::gazebo::systems::DiffDrive'>
  <left_joint>wheel_l_joint</left_joint>
  <right_joint>wheel_r_joint</right_joint>
  <wheel_separation>0.19</wheel_separation>
  <wheel_radius>0.05</wheel_radius>
  <max_linear_velocity>0.5</max_linear_velocity>
  <max_angular_velocity>2.0</max_angular_velocity>
  <cmd_vel_timeout>0.5</cmd_vel_timeout>   <!-- detiene el robot si no llega cmd en 0.5s -->
</plugin>

<!-- JointStatePublisher: misma convención de nombres -->
<plugin filename='libignition-gazebo-joint-state-publisher-system.so'
        name='ignition::gazebo::systems::JointStatePublisher'>
  <joint_name>wheel_l_joint</joint_name>
  <joint_name>wheel_r_joint</joint_name>
</plugin>
```

Si el nuevo repo usa **Harmonic** (ROS 2 Jazzy + `gz-harmonic`), los nombres cambian a:
- `gz-sim-diff-drive-system` (sin `lib`, sin `.so`)
- `gz::sim::systems::DiffDrive`
- mensajes `gz.msgs.*`

### Meshes — URIs y variable de entorno

Los meshes usan `model://` URIs:
```xml
<uri>model://puzzlebot_description/meshes/Puzzlebot_Wheel.stl</uri>
```

Gazebo resuelve `model://puzzlebot_description/` buscando un directorio con ese nombre
en `IGN_GAZEBO_RESOURCE_PATH`. El path debe apuntar al **padre** del share directory,
no al share directory mismo.

```python
# En el launch file — SIEMPRE antes de lanzar Gazebo:
desc_share_parent = os.path.dirname(get_package_share_directory('puzzlebot_description'))
SetEnvironmentVariable(name='IGN_GAZEBO_RESOURCE_PATH', value=desc_share_parent)
```

### Altura de las ruedas

El z del joint debe ser igual al radio de la rueda (0.05 m). Si `z < radio`, la rueda
clipa el suelo y la fricción bloquea el robot para siempre:
```xml
<joint name='wheel_l_joint' type='revolute'>
  <pose relative_to='base_footprint'>0.052 0.095 0.05 0 0 0</pose>
  ...
```

---

## 5. Mundo SDF — plugins requeridos

Archivo: `src/puzzlebot_description/worlds/<world_name>.sdf`

```xml
<!-- Fortress: todos usan libignition- prefix -->
<plugin filename="libignition-gazebo-physics-system.so"
        name="ignition::gazebo::systems::Physics"/>
<plugin filename="libignition-gazebo-user-commands-system.so"
        name="ignition::gazebo::systems::UserCommands"/>   <!-- sin esto el spawn cuelga -->
<plugin filename="libignition-gazebo-scene-broadcaster-system.so"
        name="ignition::gazebo::systems::SceneBroadcaster"/>
<plugin filename="libignition-gazebo-sensors-system.so"
        name="ignition::gazebo::systems::Sensors">
  <render_engine>ogre2</render_engine>   <!-- sin esto el lidar no produce datos -->
</plugin>
<plugin filename="libignition-gazebo-contact-system.so"
        name="ignition::gazebo::systems::Contact"/>
```

El nombre del mundo (`<world name="maze">`) aparece en el topic de joint state:
`/world/maze/model/puzzlebot/joint_state`
Si renombras el mundo debes actualizar el bridge y el remapping del dead_reckoning.

---

## 6. Bridge de tópicos — sintaxis y tipos

### Sintaxis del argumento

```
/topic@ros_type@gz_type      # bidireccional
/topic@ros_type[gz_type      # gz → ROS 2 únicamente
/topic@ros_type]gz_type      # ROS 2 → gz únicamente
```

### Tópicos que necesita este stack (Fortress: ignition.msgs)

```python
arguments=[
    # cmd_vel: teleop/controlador → plugin DiffDrive
    '/model/puzzlebot/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist',

    # odometría: Gazebo → ROS 2
    '/model/puzzlebot/odometry@nav_msgs/msg/Odometry@ignition.msgs.Odometry',

    # clock: Gazebo → ROS 2 (use_sim_time=True lo necesita)
    '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',

    # lidar: Gazebo → ROS 2
    '/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan',

    # joint states: Gazebo → ROS 2 (para dead_reckoning y robot_state_publisher)
    '/world/<world_name>/model/puzzlebot/joint_state'
    '@sensor_msgs/msg/JointState[ignition.msgs.Model',

    # dynamic pose: Gazebo → ROS 2 (para ground_truth_odom en mode:=mapping)
    '/world/<world_name>/dynamic_pose/info'
    '@geometry_msgs/msg/PoseArray[ignition.msgs.Pose_V',
]
```

Sustituir `<world_name>` por el nombre real del mundo (`flat_plane`, `maze`, etc.).

### QoS override recomendado

```python
parameters=[{
    'qos_overrides./model/puzzlebot.subscriber.reliability': 'reliable',
}]
```

---

## 7. Launch file — orden y detalles

```python
def generate_launch_description():
    bringup_pkg = get_package_share_directory('puzzlebot_bringup')
    slam_pkg    = get_package_share_directory('puzzlebot_slam')
    desc_pkg   = get_package_share_directory('puzzlebot_description')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # ── 0. IGN_GAZEBO_RESOURCE_PATH — DEBE ser el primer action ──────────────
    desc_share_parent = os.path.dirname(desc_pkg)
    existing = os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')
    ign_resource_path = desc_share_parent + (':' + existing if existing else '')

    set_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=ign_resource_path,
    )

    # ── 1. Gazebo Fortress ────────────────────────────────────────────────────
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': f'-r {world_file}',
            'gz_version': '6',    # 6 = Fortress; activa el code path de ign gazebo
        }.items(),
    )

    # ── 2. robot_state_publisher ──────────────────────────────────────────────
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
    )

    # ── 3. bridge ─────────────────────────────────────────────────────────────
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_bridge',
        arguments=[...],    # ver sección 6
        parameters=[{'qos_overrides./model/puzzlebot.subscriber.reliability': 'reliable'}],
    )

    # ── 4. Spawn — 5 s de delay para que Gazebo registre el servicio ──────────
    spawn = TimerAction(
        period=5.0,
        actions=[ExecuteProcess(
            cmd=[
                'ign', 'service',                      # Fortress CLI: ign, no gz
                '-s', '/world/<world_name>/create',
                '--reqtype', 'ignition.msgs.EntityFactory',
                '--reptype', 'ignition.msgs.Boolean',
                '--timeout', '5000',
                '--req', f'sdf_filename: "{sdf_file}", name: "puzzlebot", '
                         f'pose: {{position: {{z: 0.05}}}}',
            ],
            additional_env={'IGN_GAZEBO_RESOURCE_PATH': ign_resource_path},
        )]
    )

    # ── 5. dead_reckoning ─────────────────────────────────────────────────────
    dead_reckoning = Node(
        package='puzzlebot_localization',
        executable='dead_reckoning_debug',
        parameters=[{
            'use_sim_time': True,
            'wheel_radius': 0.05,
            'wheel_separation': 0.19,
        }],
        remappings=[
            ('/joint_states', '/world/<world_name>/model/puzzlebot/joint_state'),
        ],
    )

    # ── 5b. ground_truth_odom (solo simulación mapping) ───────────────────────
    # En este repo actual, gz_sim.launch.py usa odom_source:=ground_truth por
    # default cuando mode:=mapping. No lanzar ground_truth_odom y dead_reckoning
    # publicando /odom al mismo tiempo.
    ground_truth_odom = Node(
        package='puzzlebot_localization',
        executable='ground_truth_odom',
        parameters=[{
            'use_sim_time': True,
            'pose_topic': '/world/<world_name>/dynamic_pose/info',
            'odom_frame': 'odom',
            'base_frame': 'base_footprint',
        }],
    )

    # ── 6. MCL (opcional) ─────────────────────────────────────────────────────
    mcl = Node(
        package='puzzlebot_slam',
        executable='mcl',
        parameters=[{
            'use_sim_time': True,
            'map_path':       map_file,       # ruta al PNG del mapa
            'map_resolution': 0.05,           # metros por pixel
            'map_origin_x':  -5.8350,         # coordenada world en col=0, row=height-1
            'map_origin_y':  -8.3950,
            'num_particles':  500,
            'top_k':          150,
            'noise_xy':       0.05,
            'noise_theta':    0.05,
            'score_rays':     36,
        }],
    )

    return LaunchDescription([
        set_resource_path,   # SIEMPRE primero
        arg_rviz,
        gz_sim,
        rsp,
        bridge,
        spawn,
        dead_reckoning,
        mcl,
        rviz,
    ])
```

---

## 8. Nodo odometry / dead_reckoning_debug

Archivo oficial robot real: `src/puzzlebot_localization/src/odometry_node.cpp`
Archivo debug Python: `src/puzzlebot_localization/scripts/dead_reckoning_debug`

### Qué hace

Integra velocidades de rueda con cinemática diferencial:

```
v  = r * (ω_r + ω_l) / 2      [m/s]
ω  = r * (ω_r − ω_l) / L      [rad/s]

x   += v * cos(θ) * dt
y   += v * sin(θ) * dt
θ   += ω * dt
```

`odometry_node.cpp` publica `/odom_raw` para que `kalman_filter_node.cpp` pueda
publicar `/odom`. El script `dead_reckoning_debug` publica `/odom` directamente
cuando se usa como fuente rápida de simulación/debug.

### Parámetros

| Parámetro | Default | Descripción |
|---|---|---|
| `wheel_radius` | 0.05 | metros |
| `wheel_separation` | 0.19 | metros, centro a centro |
| `odom_frame` | `"odom"` | frame ID del mensaje Odometry |
| `base_frame` | `"base_footprint"` | child frame ID |
| `input_source` | `"joint_states"` | `"joint_states"` = sim, `"encoders"` = robot real |

### Modo simulación

Suscribe `/joint_states` (remapeado al topic de Gazebo).
Extrae `wheel_l_joint` y `wheel_r_joint` por nombre, no por índice.

### Modo robot real

Suscribe `/velocity_enc_r` y `/velocity_enc_l` (`std_msgs/Float32`, rad/s).
Integra a 20 Hz con un timer. El nodo no cambia, solo el parámetro `input_source`.

---

## 9. Nodo MCL (Monte Carlo Localization)

Archivo: `src/puzzlebot_slam/puzzlebot_slam/mcl.py`

### Qué hace (pasos D–I de la tarea)

```
D. Inicializar N partículas (x, y, θ) uniformemente en celdas libres del mapa.
E. Puntuar cada partícula: ray-marching sobre el mapa PNG vs. scan real.
F. Conservar las top-K partículas por puntaje.
G. Calcular Δpose del robot desde mensajes /odom consecutivos (dead reckoning).
H. Mover todas las partículas supervivientes por Δpose + ruido gaussiano.
I. Resamplear a N duplicando supervivientes con ruido pequeño → volver a D.
```

### Entradas / Salidas

| Dirección | Tópico | Tipo |
|---|---|---|
| Entrada | `/scan` | `sensor_msgs/LaserScan` |
| Entrada | `/odom` | `nav_msgs/Odometry` |
| Salida | `/mcl/particles` | `geometry_msgs/PoseArray` |
| Salida | `/mcl/pose` | `geometry_msgs/PoseStamped` |
| Salida | `/mcl/map` | `nav_msgs/OccupancyGrid` (latched) |
| Salida | TF `map → odom` | corrección de localización |

### Parámetros principales

| Parámetro | Default | Descripción |
|---|---|---|
| `map_path` | `maze_map.png` junto al script | ruta al PNG del mapa |
| `map_resolution` | 0.05 | metros por pixel |
| `map_origin_x` | -5.8350 | x world en columna 0, fila height-1 (esquina inferior-izq) |
| `map_origin_y` | -8.3950 | y world en columna 0, fila height-1 |
| `num_particles` | 500 | N total de partículas |
| `top_k` | 150 | partículas que sobreviven el filtrado |
| `noise_xy` | 0.05 | desviación estándar del ruido de posición [m] |
| `noise_theta` | 0.05 | desviación estándar del ruido de heading [rad] |
| `score_rays` | 36 | rayos del lidar muestreados por partícula para scoring |
| `ray_step` | 0.025 | paso del ray-marching [m] |
| `hit_sigma` | 0.20 | sigma de la gaussiana de likelihood [m] |

### Convención del mapa PNG

- Blanco (valor > 127) = espacio libre
- Negro (valor ≤ 127) = obstáculo / pared
- Fila 0 de la imagen = y máxima del mundo (parte superior del mapa)
- `map_origin_x`, `map_origin_y` = coordenada world en pixel (col=0, row=height-1),
  es decir la esquina **inferior-izquierda** de la imagen

### Dependencia: Pillow

El nodo usa `PIL.Image` para cargar el PNG. Asegurarse de que esté instalado:
```bash
sudo apt install python3-pil
# o en package.xml:
# <exec_depend>python3-pil</exec_depend>
```

---

## 9b. Nodo SLAM mapping (`slam_node`)

Archivo actual: `src/puzzlebot_slam/puzzlebot_slam/slam_node.py`

Documento técnico: `docs/slam_mapping.md`

### Qué hace

Construye `/map` (`nav_msgs/OccupancyGrid`) en tiempo real desde `/scan` y `/odom`.
Es occupancy-grid mapping con pose conocida/asumida, no SLAM completo con cierre
de lazo. En Gazebo se usa `ground_truth_odom` por default para que el mapa no se
deforme por deriva de wheel odometry.

### Fundamento

- Guarda log-odds por celda: `l = log(p / (1 - p))`.
- Cada rayo del lidar se proyecta al frame global usando la pose de `/odom`.
- Bresenham marca las celdas entre robot y endpoint como libres.
- Si el endpoint es un hit real, esa celda recibe evidencia ocupada.
- Los valores se saturan con `l_clamp`.
- Al publicar `/map`, log-odds positivos se convierten a `100`, negativos a `0`
  y valores cercanos a cero a `-1`.

### Mejoras implementadas

1. Buffer temporal de poses:
   - `pose_buffer_sec: 3.0`
   - `max_scan_pose_age: 0.20`
   - El nodo interpola/busca la pose de `/odom` correspondiente a
     `scan.header.stamp`.

2. Sensor model conservador:
   - `p_occ: 0.75`
   - `p_free: 0.45`
   - Reduce el efecto de borrar paredes por pequeños errores de pose.

3. QoS latched para `/map`:
   - `DurabilityPolicy.TRANSIENT_LOCAL`
   - RViz recibe el mapa aunque se abra despues.

4. División interna para SLAM real:
   - `slam_node.py`: orquestador ROS.
   - `odometry_buffer.py`: sincroniza `/odom` con `/scan`.
   - `occupancy_grid_map.py`: log-odds, Bresenham y mensaje `OccupancyGrid`.
   - `keyframe_manager.py`: gate opcional de keyframes.
   - `scan_matcher.py`: hook para implementar scan matching local.
   - `slam_math.py` / `slam_types.py`: utilidades puras y `Pose2D`.

Por default, `use_keyframes=false` y `scan_matching_enabled=false` para preservar
el comportamiento validado en Gazebo. La implementación real del robot debe
empezar por `LocalScanMatcher.match()`.

### Launch actual

```bash
# Mapping recomendado en Gazebo: usa ground_truth_odom
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze mode:=mapping

# Mapping con deriva realista de ruedas
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze mode:=mapping odom_source:=dead_reckoning
```

### Robot real

En robot fisico no existe `ground_truth_odom`. Para usar este mapper:

- `use_sim_time: False`
- `/odom` debe venir de encoders, IMU/EKF, o una fuente equivalente.
- Calibrar `wheel_radius` y `wheel_separation`.
- Mapear lento y con giros suaves.
- No esperar correccion global ni cierre de lazo. Para mapas grandes, considerar
  scan matching o `slam_toolbox`.

---

## 10. Generador del mapa PNG

Archivo: `src/puzzlebot_slam/puzzlebot_slam/generate_maze_map.py`

Script independiente (no un nodo ROS). Parsea la geometría de `maze.sdf` y
rasteriza cada pared y caja en un PNG a 0.05 m/px.

**Cómo usarlo para un mundo nuevo:**

1. Extraer de cada `<model>` estático en el SDF:
   - La pose del modelo: `(cx, cy, yaw)`
   - El tamaño de la geometría box: `(length, width)` → `half_len = length/2`, `half_wid = width/2`

2. Modificar las listas `WALLS` y `BOXES` en el script.

3. Ejecutar:
   ```bash
   python3 generate_maze_map.py
   # Imprime map_origin_x, map_origin_y y tamaño del canvas
   # Guarda maze_map.png junto al script
   ```

4. Copiar los valores de `map_origin_x` y `map_origin_y` al parámetro `mcl` en
   el launch file.

**Verificar la alineación:**
```python
# Script de sanity check — ejecutar después de generar el mapa:
from PIL import Image
import numpy as np

img = Image.open('maze_map.png')
arr = np.array(img)
h, w = arr.shape
MAP_RES = 0.05
ORIG_X  = <map_origin_x>
ORIG_Y  = <map_origin_y>

def world_to_px(wx, wy):
    col = int((wx - ORIG_X) / MAP_RES)
    row = h - 1 - int((wy - ORIG_Y) / MAP_RES)
    return col, row

# Probar puntos conocidos del SDF:
# Un centro de pared debe dar pixel=0 (negro)
# El centro del mundo libre debe dar pixel=255 (blanco)
col, row = world_to_px(0.0, 0.0)
print(arr[row, col])   # esperado: 255 (libre)
```

---

## 11. TF estático del lidar

El sensor lidar en el SDF publica con `frame_id = "lidar_link"`, pero el link en
Gazebo se llama `puzzlebot/base_footprint/lidar` (Fortress scoping). Se necesita
un `static_transform_publisher` para que RViz y MCL tengan el frame correcto:

```python
lidar_tf = Node(
    package='tf2_ros',
    executable='static_transform_publisher',
    name='lidar_frame_fix',
    arguments=['0', '0', '0', '0', '0', '0',
               'lidar_link', 'puzzlebot/base_footprint/lidar'],
)
```

---

## 12. Build y ejecución

```bash
cd ~/ws
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash

# Simulación simple:
ros2 launch puzzlebot_bringup gz_sim.launch.py

# Simulación + MCL:
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze

# Simulación + SLAM mapping:
ros2 launch puzzlebot_bringup gz_sim.launch.py world:=maze mode:=mapping

# Teleop (terminal separado con TTY real):
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args --remap cmd_vel:=/model/puzzlebot/cmd_vel
```

### Rebuild parcial (más rápido)

```bash
colcon build --packages-select puzzlebot_slam
colcon build --packages-select puzzlebot_bringup
source install/setup.bash   # SIEMPRE después de build
```

---

## 13. Checklist de verificación en runtime

```bash
# 1. Topics activos después del spawn
ros2 topic list
# Esperados: /clock, /model/puzzlebot/cmd_vel, /model/puzzlebot/odometry,
#            /odom, /scan, /world/<world>/model/puzzlebot/joint_state, /tf, /tf_static

# 2. Clock fluyendo (sim time)
ros2 topic hz /clock   # ~1000 Hz

# 3. DiffDrive subscriber registrado (2 subscribers = OK)
ign topic -i -t /model/puzzlebot/cmd_vel

# 4. Mover el robot directo por ign transport (test sin bridge)
ign topic -t /model/puzzlebot/cmd_vel -m ignition.msgs.Twist \
  -p "linear: {x: 0.3}, angular: {z: 0.0}"

# 5. SLAM mapping topics
ros2 topic hz /map
ros2 topic echo /odom --once

# 6. MCL topics
ros2 topic hz /mcl/particles   # activo cuando llega /scan
ros2 topic echo /mcl/pose --once
```

---

## 14. Errores frecuentes y sus causas

| Síntoma | Causa | Fix |
|---|---|---|
| Robot no se mueve | DiffDrive subscriber faltante (mismatch de versión) | Verificar `ign topic -i`, usar `libignition-` prefix en SDF |
| Robot flota o se traba | `z` del wheel joint ≠ `wheel_radius` | Poner `z = 0.05` |
| Meshes invisibles | `IGN_GAZEBO_RESOURCE_PATH` no visible para Gazebo | `SetEnvironmentVariable` como primer action, no `additional_env` |
| Spawn cuelga | `UserCommands` plugin faltante en el mundo | Añadirlo al SDF del mundo |
| `ros2 topic hz /clock` = 0 | Bridge no conectado a Gazebo (todavía arrancando) | Esperar 10 s, luego reiniciar si persiste |
| `--editable not recognized` en colcon | Falta `pyproject.toml` | Añadir el archivo con `setuptools.build_meta` |
| MCL particles no convergen | Mapa PNG no coincide con el mundo simulado | Regenerar con `generate_maze_map.py` y actualizar `map_origin_x/y` |
| MCL scores siempre ~0 | `map_origin_x/y` incorrectos, o frame del lidar mal | Verificar sanity check de alineación, revisar `lidar_tf` |
| `/map` se duplica al girar | `/odom` deriva o scan usa pose desfasada | En Gazebo usar `odom_source:=ground_truth`; en real calibrar odometría y mapear lento |
| `slam_node` salta scans | No hay pose de `/odom` cercana al timestamp del scan | Verificar clocks, `use_sim_time`, frecuencia de `/odom`, `max_scan_pose_age` |
| RViz "jump back in time" | RViz arrancó antes de que el clock de Gazebo se estabilizara | Delay de 15 s antes de lanzar RViz (`TimerAction`) |

---

## 15. Transición simulación → robot real

Para robot real, preferir `odometry_node.cpp` como fuente oficial:

```python
# Robot real:
odometry_node: /velocity_enc_r + /velocity_enc_l -> /odom_raw
kalman_filter_node: /odom_raw (+ ArUco/IMU futuras) -> /odom

# Simulación/debug:
dead_reckoning_debug puede publicar /odom desde joint states si se requiere
comparar contra ground_truth_odom.
```

El nodo `mcl.py` no necesita cambios de algoritmo para el robot real: suscribe
`/scan` y `/odom`, ambos disponibles en el robot físico con los mismos tipos de
mensaje. El `slam_node.py` tampoco requiere Gazebo, pero en robot real su calidad
depende completamente de la calidad de `/odom`, porque no habrá `ground_truth_odom`.

---

## 16. Notas sobre el código recibido

Si el agente recibe los archivos fuente del repositorio original:

- `odometry_node.cpp` — fuente oficial de odometría real.
- `dead_reckoning_debug` — script Python de debug, no pertenece a SLAM.
- `ground_truth_odom` — usar solo en Gazebo para mapping; publica `/odom` desde
  `/world/<world>/dynamic_pose/info`.
- `slam_node.py` — occupancy-grid mapping log-odds desde `/scan` + `/odom`; usa
  buffer temporal de poses. Ver `docs/slam_mapping.md`.
- `mcl.py` — listo para usar. Depende de `Pillow` (`from PIL import Image`).
- `generate_maze_map.py` — script de utilidad, no un nodo ROS. Editar `WALLS` y
  `BOXES` con la geometría del nuevo mundo y ejecutar para generar el PNG.
- `maze_map.png` — válido únicamente para el mundo `maze.sdf` incluido en el repo.
  Para cualquier otro mundo, regenerar con `generate_maze_map.py`.
- `puzzlebot_gz.sdf` — SDF del robot Puzzlebot. Reutilizable sin cambios si el
  robot es el mismo. Los nombres de joints (`wheel_l_joint`, `wheel_r_joint`) y
  parámetros físicos (`wheel_radius=0.05`, `wheel_separation=0.19`) deben coincidir
  con los del launch de dead_reckoning.
- `maze.sdf` — mundo de ejemplo. Adaptarlo o reemplazarlo con el mundo del nuevo repo.
  Mantener los cinco plugins Fortress en la sección `<world>`.
