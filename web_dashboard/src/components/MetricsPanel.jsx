import React, { useMemo } from 'react';

// ── DOM FSM display maps ──────────────────────────────────────────────────────
const DOM_COLORS = {
  NORMAL:            '#2d2d2d',
  FOLLOW_NEW_PATH:   '#22d3ee',
  BRAKE_FOR_REPLAN:  '#facc15',
  REPLAN:            '#facc15',
  RECOVERY_REVERSE:  '#f87171',
  RECOVERY_TURN:     '#f87171',
  SAFE_STOP:         '#ef4444',
};

const DOM_LABELS = {
  NORMAL:            'NORMAL',
  FOLLOW_NEW_PATH:   'FOLLOWING',
  BRAKE_FOR_REPLAN:  'BRAKING',
  REPLAN:            'REPLANNING',
  RECOVERY_REVERSE:  'RECOVERY ↩',
  RECOVERY_TURN:     'RECOVERY ↻',
  SAFE_STOP:         'SAFE STOP',
};

// ── SVG Line Chart ────────────────────────────────────────────────────────────
// data: [{time: number, [key]: number, ...}]
// series: [{key, label, color}]
// dangerBelow: if set, shades values below this threshold red
function LineChart({ data, series, height = 90, zeroline = false, dangerBelow = null }) {
  const W = 520, H = height;
  const P = { top: 8, right: 8, bottom: 22, left: 44 };
  const IW = W - P.left - P.right;
  const IH = H - P.top - P.bottom;

  if (!data || data.length < 2) {
    return (
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height }}>
        <text x={W / 2} y={H / 2} textAnchor="middle" fill="#555" fontSize="11">
          Sin datos aún…
        </text>
      </svg>
    );
  }

  const times = data.map(d => d.time);
  const tMin = times[0];
  const tMax = times[times.length - 1];
  const tRange = Math.max(tMax - tMin, 0.001);

  const allVals = series.flatMap(s =>
    data.map(d => d[s.key]).filter(v => v != null && isFinite(v))
  );
  if (!allVals.length) return null;

  const rawMin = Math.min(...allVals);
  const rawMax = Math.max(...allVals);
  const vPad = Math.max((rawMax - rawMin) * 0.12, 0.02);
  const vMin = rawMin - vPad;
  const vMax = rawMax + vPad;

  const xS = t => P.left + ((t - tMin) / tRange) * IW;
  const yS = v => P.top + IH * (1 - (v - vMin) / (vMax - vMin));

  const yTicks = [0, 0.25, 0.5, 0.75, 1].map(f => vMin + f * (vMax - vMin));
  const xTicks = [0, 0.25, 0.5, 0.75, 1].map(f => tMin + f * tRange);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height }}>
      {/* Danger zone — values below threshold */}
      {dangerBelow != null && dangerBelow > vMin && (
        <rect
          x={P.left} y={P.top} width={IW}
          height={Math.max(0, Math.min(IH, yS(dangerBelow) - P.top))}
          fill="rgba(248,113,113,0.12)"
        />
      )}
      {dangerBelow != null && dangerBelow > vMin && dangerBelow < vMax && (
        <line
          x1={P.left} x2={P.left + IW}
          y1={yS(dangerBelow)} y2={yS(dangerBelow)}
          stroke="#f87171" strokeWidth="0.8" strokeDasharray="4 3" opacity="0.6"
        />
      )}
      {/* Y gridlines + labels */}
      {yTicks.map((v, i) => (
        <g key={i}>
          <line
            x1={P.left} x2={P.left + IW} y1={yS(v)} y2={yS(v)}
            stroke="#1c1c1c" strokeWidth="0.8"
          />
          <text x={P.left - 3} y={yS(v) + 3} textAnchor="end" fill="#555" fontSize="8">
            {v.toFixed(2)}
          </text>
        </g>
      ))}
      {/* Zero line */}
      {zeroline && vMin < 0 && vMax > 0 && (
        <line
          x1={P.left} x2={P.left + IW} y1={yS(0)} y2={yS(0)}
          stroke="#333" strokeWidth="1" strokeDasharray="4 3"
        />
      )}
      {/* X axis + time labels */}
      <line
        x1={P.left} x2={P.left + IW}
        y1={P.top + IH} y2={P.top + IH}
        stroke="#333" strokeWidth="0.5"
      />
      {xTicks.map((t, i) => (
        <text
          key={i} x={xS(t)} y={P.top + IH + 13}
          textAnchor="middle" fill="#444" fontSize="8"
        >
          {new Date(t * 1000).toLocaleTimeString([], { minute: '2-digit', second: '2-digit' })}
        </text>
      ))}
      {/* Series paths */}
      {series.map(s => {
        const pts = data
          .map(d => ({ t: d.time, v: d[s.key] }))
          .filter(p => p.v != null && isFinite(p.v));
        if (pts.length < 2) return null;
        const path = pts
          .map((p, i) => `${i === 0 ? 'M' : 'L'}${xS(p.t).toFixed(1)},${yS(p.v).toFixed(1)}`)
          .join(' ');
        return (
          <path key={s.key} d={path} stroke={s.color} strokeWidth="1.5" fill="none" />
        );
      })}
      {/* Chart border */}
      <rect
        x={P.left} y={P.top} width={IW} height={IH}
        stroke="#222" strokeWidth="0.5" fill="none"
      />
      {/* Inline legend */}
      {series.map((s, i) => (
        <g key={s.key} transform={`translate(${P.left + 4 + i * 145}, ${H - 3})`}>
          <line x1="0" x2="12" y1="-2" y2="-2" stroke={s.color} strokeWidth="1.5" />
          <text x="15" y="0" fill={s.color} fontSize="9">{s.label}</text>
        </g>
      ))}
    </svg>
  );
}

// ── DOM FSM State Timeline ────────────────────────────────────────────────────
// events: [{time: number, state: string}]
function StateTimeline({ events }) {
  const W = 520, H = 38;
  const P = { left: 44, right: 8, top: 6, bottom: 12 };
  const IW = W - P.left - P.right;
  const IH = H - P.top - P.bottom;

  if (!events || events.length === 0) {
    return (
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H }}>
        <text x={W / 2} y={H / 2} textAnchor="middle" fill="#555" fontSize="11">
          Sin transiciones aún…
        </text>
      </svg>
    );
  }

  const tMin = events[0].time;
  const tMax = events[events.length - 1].time;
  const tRange = Math.max(tMax - tMin, 0.001);
  const xS = t => P.left + ((t - tMin) / tRange) * IW;

  const segments = events.map((e, i) => ({
    x1: xS(e.time),
    x2: i < events.length - 1 ? xS(events[i + 1].time) : P.left + IW,
    state: e.state,
  }));

  const fmtTime = t =>
    new Date(t * 1000).toLocaleTimeString([], { minute: '2-digit', second: '2-digit' });

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H }}>
      {segments.map((seg, i) => (
        <rect
          key={i}
          x={seg.x1} y={P.top}
          width={Math.max(1, seg.x2 - seg.x1)} height={IH}
          fill={DOM_COLORS[seg.state] ?? '#333'}
          opacity="0.9"
        />
      ))}
      <rect
        x={P.left} y={P.top} width={IW} height={IH}
        stroke="#333" strokeWidth="0.5" fill="none"
      />
      <text x={P.left - 3} y={P.top + IH / 2 + 3} textAnchor="end" fill="#555" fontSize="8">
        FSM
      </text>
      <text x={P.left + 2} y={H - 1} fill="#444" fontSize="8">{fmtTime(tMin)}</text>
      <text x={P.left + IW - 2} y={H - 1} textAnchor="end" fill="#444" fontSize="8">
        {fmtTime(tMax)}
      </text>
    </svg>
  );
}

// ── Metric card ───────────────────────────────────────────────────────────────
function MetricCard({ label, value, unit, color = 'var(--accent)' }) {
  return (
    <div className="metric-card">
      <span className="metric-value" style={{ color }}>{value}</span>
      {unit && <span className="metric-unit">{unit}</span>}
      <span className="metric-label">{label}</span>
    </div>
  );
}

// ── Chart section wrapper ─────────────────────────────────────────────────────
function ChartSection({ title, children }) {
  return (
    <div className="chart-section">
      <div className="chart-title">{title}</div>
      <div className="chart-body">{children}</div>
    </div>
  );
}

// ── CSV export ────────────────────────────────────────────────────────────────
function exportCSV(velHistory, lidarHist, domStateLog, stats) {
  const lines = [
    '# Puzzlebot Metrics Export',
    `# Generated: ${new Date().toLocaleString()}`,
    '',
    '# Session Summary',
    `start_time,${new Date(stats.startTime * 1000).toLocaleString()}`,
    `distance_m,${stats.distanceTraveled.toFixed(3)}`,
    `max_linear_vel_ms,${stats.maxLinearVel.toFixed(4)}`,
    `replan_count,${stats.replanCount}`,
    `obstacle_stops,${stats.obstacleStops}`,
    `goals_sent,${stats.goalsSent}`,
    '',
    '# Velocity History',
    'time_s,linear_x_ms,angular_z_rads',
    ...velHistory.map(d => `${d.time.toFixed(3)},${d.linear.toFixed(4)},${d.angular.toFixed(4)}`),
    '',
    '# LiDAR Min Distance History',
    'time_s,min_dist_m',
    ...lidarHist.map(d => `${d.time.toFixed(3)},${d.min.toFixed(4)}`),
    '',
    '# DOM FSM State Log',
    'time_s,state',
    ...domStateLog.map(d => `${d.time.toFixed(3)},${d.state}`),
  ];

  const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `puzzlebot_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ── PDF / Print report ────────────────────────────────────────────────────────
function openPrintReport(stats, domStateLog) {
  const dur  = Math.max(0, Math.floor(Date.now() / 1000 - stats.startTime));
  const mins = Math.floor(dur / 60);
  const secs = dur % 60;

  const stateRows = domStateLog.map(e =>
    `<tr><td>${new Date(e.time * 1000).toLocaleTimeString()}</td><td><b>${e.state}</b></td></tr>`
  ).join('');

  const cards = [
    ['Duración',       `${mins}m ${secs}s`, ''],
    ['Distancia',      stats.distanceTraveled.toFixed(2), 'm'],
    ['Vel. máxima',    stats.maxLinearVel.toFixed(3), 'm/s'],
    ['Replans',        stats.replanCount, ''],
    ['Obstacle stops', stats.obstacleStops, ''],
    ['Goals enviados', stats.goalsSent, ''],
  ].map(([lbl, val, unit]) =>
    `<div class="card"><div class="cv">${val}${unit ? ' <span class="cu">' + unit + '</span>' : ''}</div><div class="cl">${lbl}</div></div>`
  ).join('');

  const win = window.open('', '_blank', 'width=820,height=700');
  win.document.write(`<!DOCTYPE html><html><head>
<title>Puzzlebot Report — ${new Date().toLocaleString()}</title>
<style>
  body{font-family:monospace;margin:36px;color:#111;background:#fff}
  h1{font-size:16px;margin-bottom:2px}
  .sub{color:#666;font-size:11px;margin-bottom:20px}
  h2{font-size:12px;text-transform:uppercase;letter-spacing:1px;color:#555;
     border-bottom:1px solid #ddd;padding-bottom:4px;margin:20px 0 10px}
  .cards{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:4px}
  .card{border:1px solid #ccc;border-radius:4px;padding:10px 14px;min-width:110px}
  .cv{font-size:18px;font-weight:bold}.cu{font-size:12px;color:#555}
  .cl{font-size:11px;color:#888;margin-top:2px}
  table{border-collapse:collapse;width:100%;font-size:11px;margin-top:4px}
  th,td{border:1px solid #ddd;padding:4px 8px;text-align:left}
  th{background:#f5f5f5;font-weight:bold}
  .btn{padding:6px 14px;border:1px solid #aaa;border-radius:4px;
       background:#f5f5f5;cursor:pointer;font-size:12px;margin-right:8px}
  @media print{.no-print{display:none}}
</style></head><body>
<h1>Puzzlebot Metrics Report</h1>
<div class="sub">Generado: ${new Date().toLocaleString()} &nbsp;|&nbsp;
Sesión iniciada: ${new Date(stats.startTime * 1000).toLocaleString()}</div>
<h2>Resumen de sesión</h2>
<div class="cards">${cards}</div>
<h2>Transiciones FSM — dynamic_obstacle_manager</h2>
<table>
  <tr><th>Hora</th><th>Estado</th></tr>
  ${stateRows || '<tr><td colspan="2">Sin transiciones registradas</td></tr>'}
</table>
<div class="no-print" style="margin-top:24px">
  <button class="btn" onclick="window.print()">Imprimir / Guardar PDF</button>
  <button class="btn" onclick="window.close()">Cerrar</button>
</div>
</body></html>`);
  win.document.close();
  win.focus();
}

// ── Main component ────────────────────────────────────────────────────────────
export default function MetricsPanel({ velHistory, lidarHist, domStateLog, sessionStats, onReset }) {
  const dur  = Math.max(0, Math.floor(Date.now() / 1000 - sessionStats.startTime));
  const mins = Math.floor(dur / 60);
  const secs = dur % 60;

  // Only show state names that actually appeared in the log (for legend)
  const seenStates = useMemo(
    () => [...new Set(domStateLog.map(e => e.state))].filter(s => s !== 'NORMAL'),
    [domStateLog]
  );

  return (
    <div className="metrics-panel">
      {/* ── Action buttons ── */}
      <div className="metrics-actions">
        <button className="btn-sm" onClick={onReset} title="Reiniciar contadores">↺ Reset</button>
        <button
          className="btn-sm btn-sm-accent"
          onClick={() => exportCSV(velHistory, lidarHist, domStateLog, sessionStats)}
        >↓ CSV</button>
        <button
          className="btn-sm btn-sm-green"
          onClick={() => openPrintReport(sessionStats, domStateLog)}
        >⎙ PDF</button>
      </div>

      {/* ── Counter cards ── */}
      <div className="metrics-cards">
        <MetricCard label="Duración"       value={`${mins}m ${secs}s`} color="var(--muted)" />
        <MetricCard label="Distancia"      value={sessionStats.distanceTraveled.toFixed(2)} unit="m"   color="var(--accent)" />
        <MetricCard label="Vel. máx"       value={sessionStats.maxLinearVel.toFixed(3)}     unit="m/s" color="var(--blue)" />
        <MetricCard label="Replans"        value={sessionStats.replanCount}                            color="var(--warn)" />
        <MetricCard label="Obst. stops"    value={sessionStats.obstacleStops}                          color="var(--err)" />
        <MetricCard label="Goals"          value={sessionStats.goalsSent}                              color="var(--green)" />
      </div>

      {/* ── Velocity chart ── */}
      <ChartSection title="Velocidad — /cmd_vel (m/s · rad/s)">
        <LineChart
          data={velHistory}
          series={[
            { key: 'linear',  label: 'linear (m/s)',    color: '#60a5fa' },
            { key: 'angular', label: 'angular (rad/s)', color: '#22d3ee' },
          ]}
          height={95}
          zeroline
        />
      </ChartSection>

      {/* ── LiDAR min distance chart ── */}
      <ChartSection title="Distancia mínima LiDAR (m)  — zona roja: < 0.30 m">
        <LineChart
          data={lidarHist}
          series={[{ key: 'min', label: 'dist. mín (m)', color: '#f87171' }]}
          height={75}
          dangerBelow={0.30}
        />
      </ChartSection>

      {/* ── DOM FSM state timeline ── */}
      <ChartSection title="FSM — dynamic_obstacle_manager">
        <StateTimeline events={domStateLog} />
        {seenStates.length > 0 && (
          <div className="dom-legend">
            {seenStates.map(s => (
              <span key={s} className="dom-legend-item">
                <span className="dom-legend-dot" style={{ background: DOM_COLORS[s] ?? '#666' }} />
                {DOM_LABELS[s] ?? s}
              </span>
            ))}
          </div>
        )}
      </ChartSection>
    </div>
  );
}
