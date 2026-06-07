import React from 'react';

const SENSORS = [
  {
    id:    'lidar',
    label: 'LiDAR',
    icon:  '◎',
    description: 'sllidar_ros2',
  },
  {
    id:    'camera',
    label: 'Cámara',
    icon:  '⬡',
    description: 'v4l2_camera',
  },
  {
    id:    'microros',
    label: 'micro-ROS',
    icon:  '⬡',
    description: 'Encoders + motores',
  },
];

const BTN_LABEL = {
  idle:     'Iniciar',
  starting: '…',
  online:   'Detener',
  error:    'Reintentar',
};

/**
 * Panel de inicialización de sensores de la Jetson.
 *
 * Controlado por el bridge: cada botón envía un comando toggle
 * { type: 'launch_sensor', sensor: id } por WebSocket. El bridge abre/cierra
 * un túnel SSH a la Jetson y reporta el estado de vuelta con mensajes
 * { type: 'sensor_status', sensor, status }. El estado lo provee el padre
 * vía la prop `status`; este componente no mantiene estado propio.
 *
 * Props:
 *   onCommand(obj)  → envía un comando por WebSocket (sendCommand de App).
 *   status          → { lidar, camera, microros } con 'idle'|'starting'|'online'|'error'.
 */
export default function SensorPanel({ onCommand, status = {} }) {
  function handleLaunch(id) {
    onCommand?.({ type: 'launch_sensor', sensor: id });
  }

  return (
    <div className="sensor-panel panel">
      <h3 className="sensor-title">Sensores Jetson</h3>
      <div className="sensor-list">
        {SENSORS.map(({ id, label, icon, description }) => {
          const s = status[id] ?? 'idle';
          return (
            <div key={id} className={`sensor-row sensor-row-${s}`}>
              <div className="sensor-info">
                <div className={`sensor-dot sensor-dot-${s}`} />
                <div className="sensor-text">
                  <span className="sensor-label">{label}</span>
                  <span className="sensor-desc">{description}</span>
                </div>
              </div>
              <button
                className={`sensor-btn sensor-btn-${s}`}
                onClick={() => handleLaunch(id)}
                disabled={s === 'starting'}
                title={`${s === 'online' ? 'Detener' : 'Inicializar'} ${label}`}
              >
                {BTN_LABEL[s] ?? 'Iniciar'}
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
