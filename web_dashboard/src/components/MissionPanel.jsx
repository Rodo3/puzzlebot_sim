import React from 'react';

// Etiquetas legibles para cada estado de la máquina de misión.
const STATE_LABELS = {
  IDLE:                'Inactivo',
  WAITING_FOR_GOAL:    'Haz click en el mapa…',
  GOING_TO_START:      'Yendo al punto de inicio',
  SCANNING_QR:         'Escaneando QR',
  FORKLIFT_UP:         'Subiendo pallet',
  NAVIGATING_TO_DOCKS: 'Navegando a docks',
  SCANNING_LOGOS:      'Buscando logo del tráiler',
  FORKLIFT_DOWN:       'Depositando pallet',
  DONE:                'Misión completada',
  ERROR:               'Error',
  MAPPING:             'Mapeando',
  RUNNING_M1:          'Misión 1 en curso',
  RUNNING_M2:          'Misión 2 en curso',
};

function getLabel(state) {
  if (STATE_LABELS[state]) return STATE_LABELS[state];
  // NAVIGATING_TO_WP_N → "Navegando al punto N"
  const wpMatch = state?.match(/^NAVIGATING_TO_WP_(\d+)$/);
  if (wpMatch) return `Navegando al punto ${wpMatch[1]}`;
  return state ?? 'Inactivo';
}

function badgeClass(state) {
  if (!state || state === 'IDLE')        return 'mission-badge-idle';
  if (state === 'DONE')                  return 'mission-badge-done';
  if (state === 'ERROR')                 return 'mission-badge-error';
  if (state === 'WAITING_FOR_GOAL')      return 'mission-badge-wait';
  return 'mission-badge-active';
}

export default function MissionPanel({ missionState, connected, onCommand, onSwitchToNav }) {
  const state  = missionState || 'IDLE';
  const isIdle = state === 'IDLE';
  const isDone = state === 'DONE';
  const label  = getLabel(state);

  const startMission1 = () => {
    onCommand({ type: 'mission_start', mission: '1' });
  };

  const startMission2 = () => {
    // Cambiar a modo navegación automáticamente para que el click en el mapa funcione
    if (onSwitchToNav) onSwitchToNav();
    onCommand({ type: 'mission_start', mission: '2' });
  };

  const stopMission = () => onCommand({ type: 'mission_stop' });

  return (
    <div className="panel mission-panel">
      <div className="mission-row">
        <span className="mission-title">MISIÓN</span>

        {(isIdle || isDone) ? (
          <div className="mission-btns">
            <button
              className="btn-mission btn-mission-1"
              disabled={!connected}
              onClick={startMission1}
            >Misión 1</button>
            <button
              className="btn-mission btn-mission-2"
              disabled={!connected}
              onClick={startMission2}
            >Misión 2</button>
          </div>
        ) : (
          <button
            className="btn-mission btn-mission-stop"
            disabled={!connected}
            onClick={stopMission}
          >Detener</button>
        )}

        <span className={`mission-badge ${badgeClass(state)}`}>{label}</span>
      </div>
    </div>
  );
}
