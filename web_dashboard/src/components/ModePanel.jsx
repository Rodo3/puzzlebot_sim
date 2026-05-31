import React from 'react';

export default function ModePanel({ mode, connected, onModeChange, onSlamReset }) {
  const handleMapping = () => {
    onModeChange('mapping');
    onSlamReset();
  };

  const handleNavigation = () => {
    onModeChange('navigation');
  };

  return (
    <div className="panel mode-panel">
      <h3>Mode</h3>
      <div className="mode-buttons">
        <button
          className={`btn-mode ${mode === 'mapping' ? 'btn-mode-active' : ''}`}
          onClick={handleMapping}
          disabled={!connected}
          title="Reset SLAM map and start mapping"
        >
          Iniciar Mapeo
        </button>
        <button
          className={`btn-mode ${mode === 'navigation' ? 'btn-mode-active' : ''}`}
          onClick={handleNavigation}
          disabled={!connected}
          title="Enable click-to-goal on the map"
        >
          Navegar
        </button>
      </div>
      <div className="mode-hint muted small">
        {mode === 'mapping'
          ? 'SLAM activo — recorre el entorno para construir el mapa'
          : 'Modo navegación — clic en el mapa para enviar goal'}
      </div>
    </div>
  );
}
