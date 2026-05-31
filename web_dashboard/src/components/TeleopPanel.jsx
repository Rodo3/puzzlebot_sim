import React, { useState, useCallback, useEffect, useRef } from 'react';

const DEFAULT_LINEAR  = 0.20;
const DEFAULT_ANGULAR = 0.80;

export default function TeleopPanel({ connected, onCommand }) {
  const [linearSpeed,  setLinearSpeed]  = useState(DEFAULT_LINEAR);
  const [angularSpeed, setAngularSpeed] = useState(DEFAULT_ANGULAR);
  const activeRef = useRef(null); // tracks which direction is currently held

  const sendVel = useCallback((lx, az) => {
    onCommand({ type: 'cmd_vel', linear_x: lx, angular_z: az });
  }, [onCommand]);

  const stop = useCallback(() => {
    activeRef.current = null;
    onCommand({ type: 'cmd_vel', linear_x: 0, angular_z: 0 });
  }, [onCommand]);

  // Safety: send stop if connection drops while holding a button
  useEffect(() => {
    if (!connected && activeRef.current) {
      activeRef.current = null;
    }
  }, [connected]);

  // Global mouseup/touchend so release outside the button still stops the robot
  useEffect(() => {
    const onUp = () => {
      if (activeRef.current !== null) stop();
    };
    window.addEventListener('mouseup',    onUp);
    window.addEventListener('touchend',   onUp);
    window.addEventListener('touchcancel', onUp);
    return () => {
      window.removeEventListener('mouseup',    onUp);
      window.removeEventListener('touchend',   onUp);
      window.removeEventListener('touchcancel', onUp);
    };
  }, [stop]);

  const press = useCallback((direction) => {
    if (!connected) return;
    activeRef.current = direction;
    switch (direction) {
      case 'forward':  sendVel( linearSpeed,  0);           break;
      case 'backward': sendVel(-linearSpeed,  0);           break;
      case 'left':     sendVel(0,  angularSpeed);           break;
      case 'right':    sendVel(0, -angularSpeed);           break;
      default: break;
    }
  }, [connected, linearSpeed, angularSpeed, sendVel]);

  const btnProps = (direction) => ({
    className: 'dpad-btn',
    disabled: !connected,
    onMouseDown: (e) => { e.preventDefault(); press(direction); },
    onTouchStart: (e) => { e.preventDefault(); press(direction); },
  });

  return (
    <div className="panel teleop-panel">
      <h3>Teleop</h3>
      <div className="dpad">
        <div className="dpad-row">
          <button {...btnProps('forward')}>↑</button>
        </div>
        <div className="dpad-row">
          <button {...btnProps('left')}>←</button>
          <button
            className="dpad-btn dpad-stop"
            disabled={!connected}
            onMouseDown={(e) => { e.preventDefault(); stop(); }}
            onTouchStart={(e) => { e.preventDefault(); stop(); }}
          >■</button>
          <button {...btnProps('right')}>→</button>
        </div>
        <div className="dpad-row">
          <button {...btnProps('backward')}>↓</button>
        </div>
      </div>

      <div className="speed-controls">
        <label className="speed-label">
          <span>Linear: <strong>{linearSpeed.toFixed(2)}</strong> m/s</span>
          <input
            type="range" min="0.05" max="0.5" step="0.05"
            value={linearSpeed}
            onChange={e => setLinearSpeed(Number(e.target.value))}
          />
        </label>
        <label className="speed-label">
          <span>Angular: <strong>{angularSpeed.toFixed(2)}</strong> rad/s</span>
          <input
            type="range" min="0.1" max="2.0" step="0.1"
            value={angularSpeed}
            onChange={e => setAngularSpeed(Number(e.target.value))}
          />
        </label>
      </div>
    </div>
  );
}
