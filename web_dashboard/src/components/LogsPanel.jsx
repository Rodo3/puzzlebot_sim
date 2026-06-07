import React, { useRef, useEffect } from 'react';

const MAX_LOGS = 50;

export default function LogsPanel({ logs }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  return (
    <div className="panel logs-panel">
      <h3>System Logs</h3>
      <div className="logs-body">
        {logs.slice(-MAX_LOGS).map((entry, i) => (
          <div key={i} className="log-entry">
            <span className="log-time">{entry.time}</span>
            <span className="log-msg">{entry.msg}</span>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
