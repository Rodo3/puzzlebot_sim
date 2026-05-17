import React from 'react';

const DOT = ({ active }) => (
  <span className={`dot ${active ? 'dot-active' : 'dot-inactive'}`} />
);

export default function StatusPanel({ connected, lastUpdate, topicStatus }) {
  const lastUpdateStr = lastUpdate
    ? new Date(lastUpdate * 1000).toLocaleTimeString()
    : '—';

  const topics = [
    { key: 'odom',    label: '/odom' },
    { key: 'scan',    label: '/scan' },
    { key: 'map',     label: '/map' },
    { key: 'cmdVel',  label: '/cmd_vel' },
    { key: 'voice',   label: '/voice/*' },
  ];

  return (
    <div className="panel">
      <h3>Status</h3>
      <div className="status-row">
        <DOT active={connected} />
        <span>{connected ? 'WebSocket connected' : 'WebSocket disconnected'}</span>
      </div>
      <div className="status-row muted">
        Last update: {lastUpdateStr}
      </div>
      <hr />
      <div className="topic-list">
        {topics.map(({ key, label }) => (
          <div key={key} className="status-row">
            <DOT active={!!topicStatus[key]} />
            <span>{label}</span>
            <span className="muted">{topicStatus[key] ? 'active' : 'waiting'}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
