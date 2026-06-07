import React, { useState } from 'react';

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

export default function SensorPanel() {
  // Estado local por sensor: 'idle' | 'starting' | 'online' | 'error'
  const [status, setStatus] = useState({
    lidar:    'idle',
    camera:   'idle',
    microros: 'idle',
  });

  // TODO: conectar a la lógica real de inicialización cuando se haga el merge.
  // Por ahora cada botón solo cicla el estado visual para validar el diseño.
  function handleLaunch(id) {
    // PENDIENTE: implementar inicialización real del sensor via SSH/WebSocket command
    // Placeholder: ciclar estado visual
    setStatus(prev => {
      const next = prev[id] === 'idle' ? 'starting'
                 : prev[id] === 'starting' ? 'online'
                 : 'idle';
      return { ...prev, [id]: next };
    });
  }

  return (
    <div className="sensor-panel panel">
      <h3 className="sensor-title">Sensores Jetson</h3>
      <div className="sensor-list">
        {SENSORS.map(({ id, label, icon, description }) => {
          const s = status[id];
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
                title={`Inicializar ${label}`}
              >
                {s === 'idle'     && 'Iniciar'}
                {s === 'starting' && '…'}
                {s === 'online'   && 'Online'}
                {s === 'error'    && 'Retry'}
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
