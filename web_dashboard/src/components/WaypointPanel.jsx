import React, { useState } from 'react';

// Waypoints from src/puzzlebot_bringup/config/waypoints.yaml
const WAYPOINTS = [
  { name: 'home',       label: 'Home       (0.50, 0.50)' },
  { name: 'centro',     label: 'Centro     (1.88, 2.43)' },
  { name: 'esquina_sw', label: 'Esq SW     (0.30, 0.30)' },
  { name: 'esquina_se', label: 'Esq SE     (3.46, 0.30)' },
  { name: 'esquina_ne', label: 'Esq NE     (3.46, 4.56)' },
  { name: 'esquina_nw', label: 'Esq NW     (0.30, 4.56)' },
  { name: 'estacion_a', label: 'Estación A (0.60, 1.20)' },
  { name: 'estacion_b', label: 'Estación B (1.88, 0.50)' },
  { name: 'estacion_c', label: 'Estación C (3.10, 2.43)' },
  { name: 'estacion_d', label: 'Estación D (1.88, 4.20)' },
  { name: 'estacion_e', label: 'Estación E (0.60, 3.50)' },
];

export default function WaypointPanel({ connected, mode, onCommand, addLog }) {
  const [selected, setSelected] = useState('');
  const disabled = !connected || mode !== 'navigation';

  const sendWaypoint = () => {
    if (!selected || disabled) return;
    onCommand({ type: 'navigate_to_waypoint', name: selected });
    addLog(`Waypoint → ${selected}`);
  };

  const sendStop = () => {
    onCommand({ type: 'navigate_to_waypoint', name: 'stop' });
    addLog('Navigation stop requested');
  };

  return (
    <div className="panel waypoint-panel">
      <h3>Waypoints</h3>
      {mode !== 'navigation' && (
        <div className="muted small" style={{ marginBottom: 6 }}>
          Activa el modo Navegación para usar waypoints
        </div>
      )}
      <select
        className="waypoint-select"
        value={selected}
        onChange={e => setSelected(e.target.value)}
        disabled={disabled}
      >
        <option value="">-- Seleccionar waypoint --</option>
        {WAYPOINTS.map(wp => (
          <option key={wp.name} value={wp.name}>{wp.label}</option>
        ))}
      </select>
      <div className="waypoint-actions">
        <button
          className="btn-waypoint"
          onClick={sendWaypoint}
          disabled={disabled || !selected}
        >
          Ir al Waypoint
        </button>
        <button
          className="btn-waypoint btn-stop-nav"
          onClick={sendStop}
          disabled={disabled}
        >
          Detener
        </button>
      </div>
    </div>
  );
}
