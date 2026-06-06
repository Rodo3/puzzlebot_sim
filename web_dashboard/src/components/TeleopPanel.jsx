import React, { useState, useCallback, useEffect, useRef } from 'react';

const DEFAULT_LINEAR  = 0.20;
const DEFAULT_ANGULAR = 0.80;
const INTERVAL_MS     = 100; // 10 Hz

export default function TeleopPanel({ connected, onCommand }) {
  const [teleMode,     setTeleMode]     = useState('robot'); // 'robot' | 'elevator'
  const [linearSpeed,  setLinearSpeed]  = useState(DEFAULT_LINEAR);
  const [angularSpeed, setAngularSpeed] = useState(DEFAULT_ANGULAR);

  // speedRef lets the interval closure always read the latest slider values
  const speedRef    = useRef({ linear: DEFAULT_LINEAR, angular: DEFAULT_ANGULAR });
  const intervalRef = useRef(null);
  const dirRef      = useRef(null);

  useEffect(() => { speedRef.current.linear  = linearSpeed;  }, [linearSpeed]);
  useEffect(() => { speedRef.current.angular = angularSpeed; }, [angularSpeed]);

  const velForDir = (dir) => {
    const { linear, angular } = speedRef.current;
    switch (dir) {
      case 'forward':  return { linear_x:  linear,  angular_z: 0 };
      case 'backward': return { linear_x: -linear,  angular_z: 0 };
      case 'left':     return { linear_x: 0,         angular_z:  angular };
      case 'right':    return { linear_x: 0,         angular_z: -angular };
      default:         return { linear_x: 0,         angular_z: 0 };
    }
  };

  const clearInterval_ = useCallback(() => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
    dirRef.current = null;
  }, []);

  const sendStop = useCallback(() => {
    onCommand({ type: 'cmd_vel', linear_x: 0, angular_z: 0 });
  }, [onCommand]);

  const emergencyStop = useCallback(() => {
    onCommand({ type: 'cmd_vel', linear_x: 0, angular_z: 0 });
    onCommand({ type: 'mission_stop' });
  }, [onCommand]);

  const hardEmergencyStop = useCallback(() => {
    clearInterval_();
    onCommand({ type: 'emergency_stop' });
  }, [clearInterval_, onCommand]);

  const release = useCallback(() => {
    if (dirRef.current !== null) { clearInterval_(); sendStop(); }
  }, [clearInterval_, sendStop]);

  // Global pointerup/pointercancel so releasing outside button still stops robot
  useEffect(() => {
    window.addEventListener('pointerup',     release);
    window.addEventListener('pointercancel', release);
    return () => {
      window.removeEventListener('pointerup',     release);
      window.removeEventListener('pointercancel', release);
    };
  }, [release]);

  // Safety stop on disconnect
  useEffect(() => { if (!connected) release(); }, [connected, release]);

  // Stop any held robot motion when switching to elevator mode
  useEffect(() => { if (teleMode !== 'robot') release(); }, [teleMode, release]);

  const press = useCallback((dir) => {
    if (!connected) return;
    clearInterval_(); // clear any previous interval
    dirRef.current = dir;
    onCommand({ type: 'cancel_navigation' });
    onCommand({ type: 'cmd_vel', ...velForDir(dir) }); // immediate send
    intervalRef.current = setInterval(() => {
      if (dirRef.current) onCommand({ type: 'cmd_vel', ...velForDir(dirRef.current) });
    }, INTERVAL_MS);
  }, [connected, onCommand, clearInterval_]); // eslint-disable-line

  // ── Robot-mode d-pad button props (held = continuous cmd_vel) ──────────────
  const robotBtn = (dir) => ({
    className: 'dpad-btn',
    disabled: !connected,
    onPointerDown: (e) => { e.preventDefault(); press(dir); },
  });

  // ── Elevator-mode button props (single click → forklift command) ───────────
  const sendElevator = (action) => onCommand({ type: 'elevator', action });

  const isElevator = teleMode === 'elevator';

  return (
    <div className="panel teleop-panel">
      <div className="teleop-header">
        <h3 style={{ margin: 0 }}>Teleop</h3>
        <div className="teleop-toggle">
          <button
            className={teleMode === 'robot' ? 'toggle-active' : ''}
            onClick={() => setTeleMode('robot')}
          >🤖 Robot</button>
          <button
            className={isElevator ? 'toggle-active' : ''}
            onClick={() => setTeleMode('elevator')}
          >↕ Elevador</button>
        </div>
      </div>

      <div className="dpad">
        <div className="dpad-row">
          {isElevator ? (
            <button
              className="dpad-btn"
              disabled={!connected}
              onClick={() => sendElevator('up')}
            >↑</button>
          ) : (
            <button {...robotBtn('forward')}>↑</button>
          )}
        </div>
        <div className="dpad-row">
          {isElevator ? (
            <button className="dpad-btn" disabled>←</button>
          ) : (
            <button {...robotBtn('left')}>←</button>
          )}
          <button
            className={`dpad-btn dpad-stop${!isElevator ? ' dpad-emergency' : ''}`}
            disabled={!connected}
            title={isElevator ? 'Parar elevador' : 'Parada de emergencia — cancela misión y movimiento'}
            onClick={isElevator ? () => sendElevator('stop') : undefined}
            onPointerDown={isElevator ? undefined : (e) => { e.preventDefault(); clearInterval_(); emergencyStop(); }}
          >■</button>
          {isElevator ? (
            <button className="dpad-btn" disabled>→</button>
          ) : (
            <button {...robotBtn('right')}>→</button>
          )}
        </div>
        <div className="dpad-row">
          {isElevator ? (
            <button
              className="dpad-btn"
              disabled={!connected}
              onClick={() => sendElevator('down')}
            >↓</button>
          ) : (
            <button {...robotBtn('backward')}>↓</button>
          )}
        </div>
      </div>

      {isElevator ? (
        <div className="elevator-hint">↑ subir · ■ parar · ↓ bajar — montacargas (stub)</div>
      ) : (
        <div className="speed-controls">
          <button
            className="dpad-btn dpad-emergency"
            disabled={!connected}
            onClick={hardEmergencyStop}
          >Emergencia</button>
          <label className="speed-label">
            <span>Linear: <strong>{linearSpeed.toFixed(2)}</strong> m/s</span>
            <input type="range" min="0.05" max="0.5" step="0.05"
              value={linearSpeed}
              onChange={e => setLinearSpeed(Number(e.target.value))} />
          </label>
          <label className="speed-label">
            <span>Angular: <strong>{angularSpeed.toFixed(2)}</strong> rad/s</span>
            <input type="range" min="0.1" max="2.0" step="0.1"
              value={angularSpeed}
              onChange={e => setAngularSpeed(Number(e.target.value))} />
          </label>
        </div>
      )}
    </div>
  );
}
