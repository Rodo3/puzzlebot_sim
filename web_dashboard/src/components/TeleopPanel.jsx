import React, { useState, useCallback, useEffect, useRef } from 'react';

const DEFAULT_LINEAR  = 0.20;
const DEFAULT_ANGULAR = 0.80;
const INTERVAL_MS     = 100; // 10 Hz

export default function TeleopPanel({ connected, onCommand }) {
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

  const press = useCallback((dir) => {
    if (!connected) return;
    clearInterval_(); // clear any previous interval
    dirRef.current = dir;
    onCommand({ type: 'cmd_vel', ...velForDir(dir) }); // immediate send
    intervalRef.current = setInterval(() => {
      if (dirRef.current) onCommand({ type: 'cmd_vel', ...velForDir(dirRef.current) });
    }, INTERVAL_MS);
  }, [connected, onCommand, clearInterval_]); // eslint-disable-line

  const btnProps = (dir) => ({
    className: 'dpad-btn',
    disabled: !connected,
    onPointerDown: (e) => { e.preventDefault(); press(dir); },
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
            onPointerDown={(e) => { e.preventDefault(); clearInterval_(); sendStop(); }}
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
    </div>
  );
}
