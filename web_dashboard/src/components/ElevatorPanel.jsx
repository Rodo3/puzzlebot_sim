import React from 'react';

export default function ElevatorPanel({ connected, onCommand }) {
  const send = (action) => onCommand({ type: 'elevator', action });

  return (
    <div className="panel">
      <h3>Elevador</h3>
      <div className="elevator-btns">
        <button
          className="btn-elevator btn-elev-up"
          style={{ borderColor: 'var(--green)', color: 'var(--green)' }}
          disabled={!connected}
          onClick={() => send('up')}
        >↑ Subir</button>
        <button
          className="btn-elevator btn-elev-stop"
          style={{ borderColor: 'var(--err)', color: 'var(--err)' }}
          disabled={!connected}
          onClick={() => send('stop')}
        >■ Parar</button>
        <button
          className="btn-elevator btn-elev-down"
          style={{ borderColor: 'var(--blue)', color: 'var(--blue)' }}
          disabled={!connected}
          onClick={() => send('down')}
        >↓ Bajar</button>
      </div>
      <span className="badge badge-warning">Sin backend — pendiente de implementación</span>
    </div>
  );
}
