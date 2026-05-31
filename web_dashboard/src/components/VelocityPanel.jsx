import React from 'react';

const STOP_THRESHOLD = 0.02;

function isNearZero(v) {
  return Math.abs(v) < STOP_THRESHOLD;
}

function VelocityRow({ label, data }) {
  if (!data) return (
    <div className="velocity-row muted">{label}: waiting for data</div>
  );
  return (
    <div className="velocity-row">
      <span className="vel-label">{label}</span>
      <span>linear: <b>{data.linear_x.toFixed(3)} m/s</b></span>
      <span>angular: <b>{data.angular_z.toFixed(3)} rad/s</b></span>
    </div>
  );
}

export default function VelocityPanel({ cmdVel, cmdVelIn }) {
  let statusNote = null;

  if (cmdVelIn && cmdVel) {
    const linDiff  = Math.abs(cmdVelIn.linear_x  - cmdVel.linear_x);
    const angDiff  = Math.abs(cmdVelIn.angular_z  - cmdVel.angular_z);
    const modified = linDiff > STOP_THRESHOLD || angDiff > STOP_THRESHOLD;

    if (modified) {
      const obstacleStop =
        !isNearZero(cmdVelIn.linear_x) &&
        isNearZero(cmdVel.linear_x) &&
        isNearZero(cmdVel.angular_z);

      statusNote = obstacleStop
        ? { text: 'Possible obstacle stop', cls: 'badge-warning' }
        : { text: 'Command modified by avoidance', cls: 'badge-info' };
    }
  }

  return (
    <div className="panel">
      <h3>Velocity</h3>
      <VelocityRow label="Desired (/cmd_vel_in)" data={cmdVelIn} />
      <VelocityRow label="Final  (/cmd_vel)"     data={cmdVel} />
      <div className={`badge ${statusNote ? statusNote.cls : ''}`}
           style={{ visibility: statusNote ? 'visible' : 'hidden' }}>
        {statusNote ? statusNote.text : 'placeholder'}
      </div>
    </div>
  );
}
