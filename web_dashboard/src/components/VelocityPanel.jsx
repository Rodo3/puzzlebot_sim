import React from 'react';

const STOP_THRESHOLD = 0.02;

const DOM_STATE_META = {
  NORMAL:            { cls: 'badge-ok',      label: 'NORMAL' },
  FOLLOW_NEW_PATH:   { cls: 'badge-info',    label: 'FOLLOWING' },
  BRAKE_FOR_REPLAN:  { cls: 'badge-warning', label: 'BRAKING' },
  REPLAN:            { cls: 'badge-warning', label: 'REPLANNING' },
  RECOVERY_REVERSE:  { cls: 'badge-err',     label: 'RECOVERY ↩' },
  RECOVERY_TURN:     { cls: 'badge-err',     label: 'RECOVERY ↻' },
  SAFE_STOP:         { cls: 'badge-err',     label: 'SAFE STOP' },
};

function VelocityRow({ label, data, highlight }) {
  if (!data) return (
    <div className="velocity-row muted">{label}: waiting…</div>
  );
  return (
    <div className={`velocity-row ${highlight ? 'velocity-row-highlight' : ''}`}>
      <span className="vel-label">{label}</span>
      <span>lin: <b>{data.linear_x.toFixed(3)}</b></span>
      <span>ang: <b>{data.angular_z.toFixed(3)}</b></span>
    </div>
  );
}

export default function VelocityPanel({ cmdVel, cmdVelIn, cmdVelSteering, navState, mode }) {
  // Detect obstacle stop: cmd_vel_in requested movement but cmd_vel is zero
  let avoidNote = null;
  if (cmdVelIn && cmdVel) {
    const linDiff = Math.abs(cmdVelIn.linear_x - cmdVel.linear_x);
    const angDiff = Math.abs(cmdVelIn.angular_z - cmdVel.angular_z);
    if (linDiff > STOP_THRESHOLD || angDiff > STOP_THRESHOLD) {
      const obstacleStop =
        Math.abs(cmdVelIn.linear_x) > STOP_THRESHOLD &&
        Math.abs(cmdVel.linear_x)   < STOP_THRESHOLD &&
        Math.abs(cmdVel.angular_z)  < STOP_THRESHOLD;
      avoidNote = obstacleStop
        ? { text: 'OBSTACLE STOP', cls: 'badge-err' }
        : { text: 'CMD MODIFIED', cls: 'badge-warning' };
    }
  }

  const domMeta = DOM_STATE_META[navState] ?? { cls: 'badge-info', label: navState ?? '—' };

  return (
    <div className="panel">
      <h3>Velocity</h3>
      <div className="velocity-rows">
        {/* Show steering output only in navigation mode (DOM is active) */}
        {mode === 'navigation' && (
          <VelocityRow label="Steering" data={cmdVelSteering} />
        )}
        <VelocityRow label="Pre-avoidance" data={cmdVelIn} />
        <VelocityRow label="Final /cmd_vel" data={cmdVel} highlight />
      </div>
      <div className="velocity-badges">
        {avoidNote && (
          <span className={`badge ${avoidNote.cls}`}>{avoidNote.text}</span>
        )}
        {mode === 'navigation' && (
          <span className={`badge ${domMeta.cls}`}>{domMeta.label}</span>
        )}
      </div>
    </div>
  );
}
