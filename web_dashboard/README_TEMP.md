# Web Dashboard — Contexto para próxima sesión

## Objetivo del siguiente paso
Tomar la **rama nueva del robot** (localización robusta) como base y aplicar encima
todos los cambios de dashboard y bridge que están en `rama_pruebas_dashboard`,
sin romper el funcionamiento existente del robot.

**Estrategia:** NO es un git merge automático. Es revisar diff entre ramas
y trasladar los cambios manualmente archivo por archivo.

---

## Archivos modificados en esta rama que deben trasladarse a la rama nueva

### ROS — bridge
| Archivo | Cambio |
|---|---|
| `src/puzzlebot_web_bridge/puzzlebot_web_bridge/bridge_node.py` | Publisher `/forklift/command`; voz y dashboard conectados a ese topic |
| `src/puzzlebot_bringup/launch/real_robot.launch.py` | `scan_topic: '/scan_stamped'` en el nodo bridge; docstring con flujo de 3 terminales |

### ROS — nuevos archivos
| Archivo | Qué es |
|---|---|
| `src/puzzlebot_bringup/launch/jetson_sensors.launch.py` | Launch para LiDAR + cámara + micro-ROS en la Jetson |

### Dashboard — todo el frontend
El dashboard de la rama nueva está desactualizado. El que se usa es el de esta rama.
Trasladar completo: `web_dashboard/src/` tal como está.

---

## Pendientes tras el merge

### 1. Botones SSH del SensorPanel
- Los 3 botones (LiDAR, Cámara, micro-ROS) ya están en el UI pero son placeholder
- Implementación: bridge recibe WebSocket command y ejecuta SSH al Puzzlebot
- SSH: `puzzlebot@10.42.0.1`, credenciales via parámetro (NO hardcoded)
- **Falta:** comandos exactos de cada sensor (el usuario los pasará)

### 2. Lifter / montacargas
- Bridge ya tiene `/forklift/command` publisher listo
- Llega en rama posterior, no en este merge

---

## Qué tiene de nuevo la rama base
- Localización más robusta (principal razón para usarla como base)
- Probablemente toca `real_robot.launch.py` — revisar diff antes de trasladar

## Correr hoy (sin merge)
```bash
# Bridge (PC operador):
ros2 run puzzlebot_web_bridge bridge_node

# Frontend:
npm run dev -- --host 0.0.0.0
```
