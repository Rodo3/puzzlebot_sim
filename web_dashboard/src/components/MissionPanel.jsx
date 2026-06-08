import React from 'react';

// Todos los estados del mission_manager_node (Jesus-fsm)
const STATE_LABELS = {
  // Idle / init
  IDLE:                                      'Inactivo',
  INIT_SYSTEM:                               'Inicializando sistema…',
  WAIT_FOR_VOICE_START:                      'Esperando inicio por voz…',
  GLOBAL_VOICE_STOP:                         'Pausado por voz',

  // Localización
  LOCALIZATION_CHECK:                        'Verificando localización',
  LOCALIZATION_RECOVERY:                     'Recuperando localización',
  FAIL_LOCALIZATION:                         'Error: localización perdida',

  // Selección de misión
  SELECT_MISSION:                            'Seleccionando misión',

  // Misión 1 — flujo pose de observación
  NAVIGATE_TO_CONVEYOR_OBSERVATION_POSE:     'Navegando a pose de observación',
  ORIENT_TO_NORTH_ARUCO_1:                   'Orientando al norte',
  STATIC_CONVEYOR_SCAN:                      'Escaneando conveyors (estático)',
  EXTRACT_TARGET_WAREHOUSE:                  'Identificando almacén destino',
  TRIGGER_MOCKS:                             'Activando sensores mock',
  NAVIGATE_TO_TARGET_WAREHOUSE:              'Navegando al almacén',
  FAIL_MARKER_NOT_FOUND:                     'Error: marcador no encontrado',

  // Navegación / barrido
  NAVIGATE_TO_ZONE:                          'Navegando a zona de búsqueda',
  EXPLORE_PICKUP_AREA:                       'Barriendo área de pickup',
  APPROACH_PALLET:                           'Aproximándose al pallet',
  ALIGN_TO_PALLET:                           'Alineando con pallet',
  PRE_PICK_VALIDATION:                       'Validando posición de pick',
  PICK_MANEUVER:                             'Subiendo pallet',
  VERIFY_PICK:                               'Verificando pick',

  // Expedición / tráiler
  NAVIGATE_TO_EXPEDITION:                    'Navegando a expedición',
  NAVIGATE_TO_TRAILER:                       'Navegando al tráiler',
  SEARCH_TRAILER_LOGO:                       'Buscando logo del tráiler',
  MATCH_TRAILER_CLIENT:                      'Verificando cliente del tráiler',
  ALIGN_TO_DOCK:                             'Alineando con dock',
  ENTER_TRAILER:                             'Entrando al tráiler',
  DROP_PALLET:                               'Depositando pallet',
  EXIT_TRAILER:                              'Saliendo del tráiler',
  MISSION_COMPLETE:                          'Misión completada',

  // Recovery
  RECOVER_QR_VIEW:                           'Recuperando vista de QR',
  RECOVER_LOGO_VIEW:                         'Recuperando vista de logo',
  FAIL_SAFE_STOP:                            'Parada de seguridad',
  FAIL_QR_NOT_FOUND:                         'Error: QR no encontrado',
  FAIL_TRAILER_NOT_FOUND:                    'Error: tráiler no encontrado',
  FAIL_PICK:                                 'Error: pick fallido',

  // Alias legacy (por compatibilidad)
  DONE:                'Misión completada',
  ERROR:               'Error',
  WAITING_FOR_GOAL:    'Esperando objetivo en mapa',
  GOING_TO_START:      'Yendo al punto de inicio',
  SCANNING_QR:         'Escaneando QR',
  FORKLIFT_UP:         'Subiendo pallet',
  NAVIGATING_TO_DOCKS: 'Navegando a docks',
  SCANNING_LOGOS:      'Buscando logo del tráiler',
  FORKLIFT_DOWN:       'Depositando pallet',
  RUNNING_M1:          'Misión 1 en curso',
  RUNNING_M2:          'Misión 2 en curso',
};

function getLabel(state) {
  if (STATE_LABELS[state]) return STATE_LABELS[state];
  const wpMatch = state?.match(/^NAVIGATING_TO_WP_(\d+)$/);
  if (wpMatch) return `Navegando al punto ${wpMatch[1]}`;
  return state ?? 'Inactivo';
}

function badgeClass(state) {
  if (!state || state === 'IDLE')                                     return 'mission-badge-idle';
  if (state === 'DONE' || state === 'MISSION_COMPLETE')               return 'mission-badge-done';
  if (state.startsWith('FAIL_') || state === 'ERROR')                 return 'mission-badge-error';
  if (state === 'GLOBAL_VOICE_STOP')                                  return 'mission-badge-wait';
  if (state === 'WAITING_FOR_GOAL' || state === 'WAIT_FOR_VOICE_START') return 'mission-badge-wait';
  return 'mission-badge-active';
}

function locBadgeClass(status) {
  if (status === 'OK')           return 'loc-badge-ok';
  if (status === 'LOST')         return 'loc-badge-lost';
  if (status === 'INITIALIZING') return 'loc-badge-init';
  return 'loc-badge-init';
}

export default function MissionPanel({
  missionState,
  connected,
  onCommand,
  onSwitchToNav,
  localizationStatus,
  missionClient,
  forkStatus,
}) {
  const state  = missionState || 'IDLE';
  const label  = getLabel(state);

  return (
    <div className="panel mission-panel">
      <div className="mission-row">
        <span className="mission-title">MISIÓN</span>

        {/* Botones bloqueados — solo visualización */}
        <div className="mission-btns" title="Control bloqueado — solo visualización">
          <button className="btn-mission btn-mission-1" disabled>Misión 1</button>
          <button className="btn-mission btn-mission-2" disabled>Misión 2</button>
        </div>

        <span className={`mission-badge ${badgeClass(state)}`}>{label}</span>
      </div>

      {/* Info de monitoreo de Jesus-fsm */}
      <div className="mission-info-row">
        {localizationStatus && (
          <span className={`loc-badge ${locBadgeClass(localizationStatus)}`}
                title="Estado de localización">
            Loc: {localizationStatus}
          </span>
        )}
        {missionClient && (
          <span className="mission-client-badge" title="Cliente activo del QR">
            Cliente: {missionClient}
          </span>
        )}
        {forkStatus !== null && forkStatus !== undefined && (
          <span className={`fork-badge ${forkStatus ? 'fork-up' : 'fork-down'}`}
                title="Estado de horquillas">
            {forkStatus ? '↑ Fork' : '↓ Fork'}
          </span>
        )}
        <span className="locked-badge" title="Botones deshabilitados — solo visualización">
          🔒 Solo visualización
        </span>
      </div>
    </div>
  );
}
