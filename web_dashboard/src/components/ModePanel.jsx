import React, { useState, useEffect, useRef } from 'react';

export default function ModePanel({
  mode, connected, onModeChange, onSlamReset,
  availableMaps, mapSource, onCommand,
}) {
  const [selectedMap, setSelectedMap] = useState('');
  const [loading,     setLoading]     = useState(false);
  const loadTimerRef = useRef(null);

  // Auto-request map list when connection is established
  useEffect(() => {
    if (connected) onCommand({ type: 'list_maps' });
  }, [connected]); // eslint-disable-line

  const handleMapping = () => {
    onModeChange('mapping');
    onSlamReset();
  };

  return (
    <div className="panel">
      <h3>Modo</h3>
      <div className="mode-buttons">
        <button
          className={`btn-mode ${mode === 'mapping' ? 'btn-mode-active' : ''}`}
          disabled={true}
          title="Bloqueado — solo visualización"
        >Iniciar Mapeo</button>
        <button
          className={`btn-mode ${mode === 'navigation' ? 'btn-mode-active' : ''}`}
          disabled={true}
          title="Bloqueado — solo visualización"
        >Navegar</button>
      </div>

      <div className="muted small">
        {mode === 'mapping'
          ? 'SLAM activo — recorre el entorno para construir el mapa'
          : 'Modo navegación — clic en el mapa para enviar goal'}
      </div>

      <hr />

      <div className="map-source-section">
        <div className="map-source-header">
          <span className="muted small">Mapa guardado</span>
          {mapSource === 'static'
            ? <span className="badge badge-warning">Mapa estático</span>
            : <span className="badge badge-ok">SLAM en vivo</span>
          }
        </div>

        <div className="map-source-row">
          <select
            className="map-select"
            value={selectedMap}
            onChange={e => setSelectedMap(e.target.value)}
            disabled={true}
          >
            <option value="">-- seleccionar mapa --</option>
            {(availableMaps ?? []).map(m => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
          <button
            className="btn-sm"
            disabled={true}
            title="Bloqueado — solo visualización"
          >Refrescar</button>
        </div>

        <div className="map-actions">
          <button
            className="btn-sm btn-sm-accent"
            style={{ flex: 1 }}
            disabled={true}
            title="Bloqueado — solo visualización"
          >Cargar mapa</button>
        </div>
      </div>
    </div>
  );
}
