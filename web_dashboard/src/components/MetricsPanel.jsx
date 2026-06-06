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

// ── SVG helpers ───────────────────────────────────────────────────────────────
const W = 500;
const P = { top: 8, right: 8, bottom: 22, left: 48 };
const IW = W - P.left - P.right;
const fmtTime = t =>
  new Date(t * 1000).toLocaleTimeString([], { minute: '2-digit', second: '2-digit' });

function NoData({ msg = 'Sin datos aún…', H = 60 }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%' }}>
      <text x={W / 2} y={H / 2} textAnchor="middle" fill="#555" fontSize="11">{msg}</text>
    </svg>
  );
}

// ── Generic line chart ────────────────────────────────────────────────────────
// thresholds: [{value, color, label}]
function LineChart({ data, series, height = 110, zeroline = false, thresholds = [] }) {
  const H = height;
  const IH = H - P.top - P.bottom;

  if (!data || data.length < 2) return <NoData H={H} />;

  const times = data.map(d => d.time);
  const tMin = times[0], tMax = times[times.length - 1];
  const tRange = Math.max(tMax - tMin, 0.001);

  const allVals = [
    ...series.flatMap(s => data.map(d => d[s.key]).filter(v => v != null && isFinite(v))),
    ...thresholds.map(th => th.value),
  ];
  if (!allVals.length) return <NoData H={H} />;

  const vPad = Math.max((Math.max(...allVals) - Math.min(...allVals)) * 0.12, 0.001);
  const vMin = Math.min(...allVals) - vPad;
  const vMax = Math.max(...allVals) + vPad;

  const xS = t => P.left + ((t - tMin) / tRange) * IW;
  const yS = v => P.top + IH * (1 - (v - vMin) / (vMax - vMin));
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map(f => vMin + f * (vMax - vMin));
  const xTicks = [0, 0.5, 1].map(f => tMin + f * tRange);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%' }}>
      {yTicks.map((v, i) => (
        <g key={i}>
          <line x1={P.left} x2={P.left + IW} y1={yS(v)} y2={yS(v)} stroke="#1c1c1c" strokeWidth="0.8" />
          <text x={P.left - 3} y={yS(v) + 3} textAnchor="end" fill="#555" fontSize="8">
            {Math.abs(v) < 1e-4 ? '0' : v.toFixed(3)}
          </text>
        </g>
      ))}
      {zeroline && vMin < 0 && vMax > 0 && (
        <line x1={P.left} x2={P.left + IW} y1={yS(0)} y2={yS(0)}
          stroke="#333" strokeWidth="1" strokeDasharray="4 3" />
      )}
      {thresholds.map((th, i) => th.value >= vMin && th.value <= vMax && (
        <g key={i}>
          <line x1={P.left} x2={P.left + IW} y1={yS(th.value)} y2={yS(th.value)}
            stroke={th.color} strokeWidth="0.9" strokeDasharray="5 3" opacity="0.8" />
          <text x={P.left + IW - 2} y={yS(th.value) - 2}
            textAnchor="end" fill={th.color} fontSize="7">{th.label}</text>
        </g>
      ))}
      <line x1={P.left} x2={P.left + IW} y1={P.top + IH} y2={P.top + IH} stroke="#333" strokeWidth="0.5" />
      {xTicks.map((t, i) => (
        <text key={i} x={xS(t)} y={P.top + IH + 13} textAnchor="middle" fill="#444" fontSize="8">
          {fmtTime(t)}
        </text>
      ))}
      {series.map(s => {
        const pts = data.map(d => ({ t: d.time, v: d[s.key] })).filter(p => p.v != null && isFinite(p.v));
        if (pts.length < 2) return null;
        const path = pts.map((p, i) =>
          `${i === 0 ? 'M' : 'L'}${xS(p.t).toFixed(1)},${yS(p.v).toFixed(1)}`
        ).join(' ');
        return <path key={s.key} d={path} stroke={s.color} strokeWidth="1.5" fill="none" />;
      })}
      <rect x={P.left} y={P.top} width={IW} height={IH} stroke="#222" strokeWidth="0.5" fill="none" />
      {series.map((s, i) => (
        <g key={s.key} transform={`translate(${P.left + 4 + i * 145}, ${H - 3})`}>
          <line x1="0" x2="10" y1="-2" y2="-2" stroke={s.color} strokeWidth="1.5" />
          <text x="13" y="0" fill={s.color} fontSize="8">{s.label}</text>
        </g>
      ))}
    </svg>
  );
}

// ── ArUco event scatter (bars + dots) ─────────────────────────────────────────
// key: which field to use as bar height ('cov_xy' or 'innovation')
function EventBars({ events, key = 'cov_xy', color = '#a78bfa', height = 75, noDataMsg }) {
  const H = height;
  const IH = H - P.top - P.bottom;
  if (!events || events.length === 0) return <NoData H={H} msg={noDataMsg} />;

  const tMin = events[0].time, tMax = events[events.length - 1].time;
  const tRange = Math.max(tMax - tMin, 0.001);
  const xS = t => P.left + ((t - tMin) / tRange) * IW;

  const vals = events.map(e => e[key]).filter(v => isFinite(v) && v >= 0);
  const vMax = vals.length > 0 ? Math.max(...vals) : 1;
  const barH = v => Math.max(3, IH * Math.min(1, v / vMax));

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%' }}>
      <line x1={P.left} x2={P.left + IW} y1={P.top + IH} y2={P.top + IH} stroke="#333" strokeWidth="0.5" />
      {events.map((e, i) => {
        const x = xS(e.time);
        const bh = barH(e[key] ?? 0);
        return (
          <g key={i}>
            <rect x={x - 1.5} y={P.top + IH - bh} width={3} height={bh} fill={color} opacity="0.6" />
            <circle cx={x} cy={P.top + IH - bh} r="2.5" fill={color} opacity="0.9" />
          </g>
        );
      })}
      <text x={P.left - 3} y={P.top + 5} textAnchor="end" fill="#555" fontSize="7">↑</text>
      <text x={P.left + 2} y={H - 2} fill="#444" fontSize="7">{fmtTime(tMin)}</text>
      <text x={P.left + IW - 2} y={H - 2} textAnchor="end" fill="#444" fontSize="7">{fmtTime(tMax)}</text>
      <text x={P.left + IW / 2} y={H - 2} textAnchor="middle" fill="#555" fontSize="7">
        {events.length} eventos
      </text>
      <rect x={P.left} y={P.top} width={IW} height={IH} stroke="#222" strokeWidth="0.5" fill="none" />
    </svg>
  );
}

// ── Trajectory map with ArUco markers ────────────────────────────────────────
function TrajectoryMap({ poseHist, arucoEvents }) {
  const H = 160;
  const MP = { top: 8, right: 8, bottom: 8, left: 8 };
  const MW = W - MP.left - MP.right;
  const MH = H - MP.top - MP.bottom;

  const hasPath  = poseHist && poseHist.length > 1;
  const hasAruco = arucoEvents && arucoEvents.length > 0;

  if (!hasPath && !hasAruco) return <NoData H={H} msg="Sin trayectoria aún…" />;

  const allX = [
    ...(hasPath  ? poseHist.map(p => p.x)    : []),
    ...(hasAruco ? arucoEvents.map(e => e.x)  : []),
  ];
  const allY = [
    ...(hasPath  ? poseHist.map(p => p.y)    : []),
    ...(hasAruco ? arucoEvents.map(e => e.y)  : []),
  ];

  const xMin = Math.min(...allX), xMax = Math.max(...allX);
  const yMin = Math.min(...allY), yMax = Math.max(...allY);
  const span = Math.max(xMax - xMin, yMax - yMin, 0.5);
  const scale = Math.min(MW, MH) / (span * 1.15);
  const cx = (xMin + xMax) / 2, cy = (yMin + yMax) / 2;

  const toSvg = (wx, wy) => ({
    sx: MP.left + MW / 2 + (wx - cx) * scale,
    sy: MP.top  + MH / 2 - (wy - cy) * scale,
  });

  let pathD = '';
  if (hasPath) {
    pathD = poseHist.map((p, i) => {
      const { sx, sy } = toSvg(p.x, p.y);
      return `${i === 0 ? 'M' : 'L'}${sx.toFixed(1)},${sy.toFixed(1)}`;
    }).join(' ');
  }

  const last = hasPath ? poseHist[poseHist.length - 1] : null;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%' }}>
      <rect x={MP.left} y={MP.top} width={MW} height={MH}
        fill="#111" stroke="#222" strokeWidth="0.5" rx="2" />
      {/* Grid lines */}
      {[-2,-1,0,1,2].map(i => {
        const gx = MP.left + MW / 2 + i * scale;
        const gy = MP.top  + MH / 2 - i * scale;
        return (
          <g key={i}>
            <line x1={gx} x2={gx} y1={MP.top} y2={MP.top + MH} stroke="#1a1a1a" strokeWidth="0.5" />
            <line x1={MP.left} x2={MP.left + MW} y1={gy} y2={gy} stroke="#1a1a1a" strokeWidth="0.5" />
          </g>
        );
      })}
      {/* Trajectory */}
      {hasPath && (
        <path d={pathD} stroke="#22d3ee" strokeWidth="1.2" fill="none" opacity="0.7" />
      )}
      {/* ArUco correction points */}
      {hasAruco && arucoEvents.map((e, i) => {
        const { sx, sy } = toSvg(e.x, e.y);
        return (
          <circle key={i} cx={sx} cy={sy} r="3.5"
            fill="#a78bfa" stroke="#7c3aed" strokeWidth="0.8" opacity="0.85" />
        );
      })}
      {/* Robot current position */}
      {last && (() => {
        const { sx, sy } = toSvg(last.x, last.y);
        return <circle cx={sx} cy={sy} r="5" fill="#60a5fa" stroke="#2563eb" strokeWidth="1" />;
      })()}
      {/* Legend */}
      <g transform={`translate(${MP.left + 4}, ${MP.top + MH - 16})`}>
        <line x1="0" x2="12" y1="4" y2="4" stroke="#22d3ee" strokeWidth="1.5" />
        <text x="15" y="7" fill="#22d3ee" fontSize="8">trayectoria</text>
        <circle cx="75" cy="4" r="3" fill="#a78bfa" />
        <text x="81" y="7" fill="#a78bfa" fontSize="8">ArUco</text>
        <circle cx="118" cy="4" r="4" fill="#60a5fa" />
        <text x="125" y="7" fill="#60a5fa" fontSize="8">robot</text>
      </g>
    </svg>
  );
}

// ── Covariance distribution bar chart ────────────────────────────────────────
// Shows last N aruco cov_xy values sorted, as a horizontal sorted-bar chart
function CovDistribution({ events, color = '#a78bfa', height = 65 }) {
  const H = height;
  const IH = H - P.top - P.bottom;
  if (!events || events.length === 0) return <NoData H={H} msg="Sin datos de covarianza…" />;

  const sorted = [...events].sort((a, b) => a.cov_xy - b.cov_xy);
  const n      = sorted.length;
  const vMax   = sorted[n - 1].cov_xy || 1;
  const barW   = IW / n;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%' }}>
      <line x1={P.left} x2={P.left + IW} y1={P.top + IH} y2={P.top + IH} stroke="#333" strokeWidth="0.5" />
      {sorted.map((e, i) => {
        const bh = Math.max(1, IH * (e.cov_xy / vMax));
        return (
          <rect key={i}
            x={P.left + i * barW} y={P.top + IH - bh}
            width={Math.max(1, barW - 0.5)} height={bh}
            fill={color} opacity="0.7"
          />
        );
      })}
      <text x={P.left - 3} y={P.top + 5} textAnchor="end" fill="#555" fontSize="7">↑ cov</text>
      <text x={P.left}         y={H - 2} fill="#444" fontSize="7">mín</text>
      <text x={P.left + IW - 2} y={H - 2} textAnchor="end" fill="#444" fontSize="7">máx</text>
      <text x={P.left + IW / 2} y={H - 2} textAnchor="middle" fill="#555" fontSize="7">
        distribución cov_xy ({n} muestras)
      </text>
      <rect x={P.left} y={P.top} width={IW} height={IH} stroke="#222" strokeWidth="0.5" fill="none" />
    </svg>
  );
}

// ── DOM FSM State Timeline ────────────────────────────────────────────────────
function StateTimeline({ events }) {
  const H = 36;
  const LP = { left: 48, right: 8, top: 4, bottom: 12 };
  const LW = W - LP.left - LP.right;
  const LH = H - LP.top - LP.bottom;

  if (!events || events.length === 0) return <NoData H={H} />;

  const tMin = events[0].time, tMax = events[events.length - 1].time;
  const tRange = Math.max(tMax - tMin, 0.001);
  const xS = t => LP.left + ((t - tMin) / tRange) * LW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%' }}>
      {events.map((e, i) => (
        <rect key={i}
          x={xS(e.time)} y={LP.top}
          width={Math.max(1, (i < events.length - 1 ? xS(events[i + 1].time) : LP.left + LW) - xS(e.time))}
          height={LH}
          fill={DOM_COLORS[e.state] ?? '#333'} opacity="0.9"
        />
      ))}
      <rect x={LP.left} y={LP.top} width={LW} height={LH} stroke="#333" strokeWidth="0.5" fill="none" />
      <text x={LP.left - 3} y={LP.top + LH / 2 + 3} textAnchor="end" fill="#555" fontSize="7">FSM</text>
      <text x={LP.left + 2} y={H - 1} fill="#444" fontSize="7">{fmtTime(tMin)}</text>
      <text x={LP.left + LW - 2} y={H - 1} textAnchor="end" fill="#444" fontSize="7">{fmtTime(tMax)}</text>
    </svg>
  );
}

// ── Semicircle gauge card ─────────────────────────────────────────────────────
function GaugeCard({ label, value, displayValue, max = 1, color = 'var(--accent)', sub = '' }) {
  const pct   = max > 0 ? Math.min(1, Math.max(0, value / max)) : 0;
  const r     = 28, cx = 42, cy = 36;
  const angle = Math.PI * (1 - pct);
  const fx    = cx + r * Math.cos(angle);
  const fy    = cy - r * Math.sin(angle);
  const large = pct > 0.5 ? 1 : 0;
  const shown = displayValue ?? `${(pct * 100).toFixed(0)}%`;

  return (
    <div className="gauge-card">
      <div className="gauge-label">{label}</div>
      <svg width="84" height="46" viewBox="0 0 84 46">
        {/* track */}
        <path d={`M ${cx-r} ${cy} A ${r} ${r} 0 1 1 ${cx+r} ${cy}`}
          fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="6" strokeLinecap="round" />
        {/* fill */}
        {pct > 0.01 && (
          <path d={`M ${cx-r} ${cy} A ${r} ${r} 0 ${large} 1 ${fx} ${fy}`}
            fill="none" stroke={color} strokeWidth="6" strokeLinecap="round" />
        )}
        {/* tick labels */}
        <text x={cx-r} y={cy+13} fontSize="7" fill="rgba(255,255,255,0.22)" textAnchor="middle">0</text>
        <text x={cx+r} y={cy+13} fontSize="7" fill="rgba(255,255,255,0.22)" textAnchor="middle">{max}</text>
      </svg>
      <div className="gauge-value" style={{ color }}>{shown}</div>
      {sub && <div className="gauge-sub">{sub}</div>}
    </div>
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

function ChartSection({ title, children }) {
  return (
    <div className="chart-section">
      <div className="chart-title">{title}</div>
      <div className="chart-body">{children}</div>
    </div>
  );
}

// ── CSV export ────────────────────────────────────────────────────────────────
function exportCSV({ velHistory, lidarHist, domStateLog, poseHist, kalmanCovHist,
                     locErrorHist, arucoEvents, scanMatchHist, stats }) {
  const lines = [
    '# Puzzlebot Localization Metrics Export',
    `# Generated: ${new Date().toLocaleString()}`,
    '',
    '# Session Summary',
    `start_time,${new Date(stats.startTime * 1000).toLocaleString()}`,
    `distance_m,${stats.distanceTraveled.toFixed(3)}`,
    `max_linear_vel_ms,${stats.maxLinearVel.toFixed(4)}`,
    `aruco_updates,${stats.arucoCount}`,
    `scan_match_updates,${stats.scanMatchCount}`,
    `replan_count,${stats.replanCount}`,
    `obstacle_stops,${stats.obstacleStops}`,
    `goals_sent,${stats.goalsSent}`,
    '',
    '# Kalman Pose History (raw odom)',
    'time_s,x_m,y_m,theta_rad',
    ...poseHist.map(d => `${d.time.toFixed(3)},${d.x.toFixed(4)},${d.y.toFixed(4)},${d.theta.toFixed(4)}`),
    '',
    '# Kalman Covariance History',
    'time_s,trace_P_xy,P_theta',
    ...kalmanCovHist.map(d => `${d.time.toFixed(3)},${d.traceP.toFixed(6)},${d.pTheta.toFixed(6)}`),
    '',
    '# Localization Error (Kalman vs SLAM)',
    'time_s,error_xy_m,error_theta_rad',
    ...locErrorHist.map(d => `${d.time.toFixed(3)},${d.error_xy.toFixed(4)},${d.error_theta.toFixed(4)}`),
    '',
    '# ArUco Correction Events',
    'time_s,x_m,y_m,theta_rad,cov_xy,innovation_m',
    ...arucoEvents.map(d =>
      `${d.time.toFixed(3)},${d.x.toFixed(4)},${d.y.toFixed(4)},${d.theta.toFixed(4)},${d.cov_xy.toFixed(6)},${(d.innovation ?? 0).toFixed(4)}`
    ),
    '',
    '# Scan Match History',
    'time_s,x_m,y_m,cov_xy',
    ...scanMatchHist.map(d => `${d.time.toFixed(3)},${d.x.toFixed(4)},${d.y.toFixed(4)},${d.cov_xy.toFixed(6)}`),
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
  a.download = `puzzlebot_loc_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function openPrintReport(stats, domStateLog, arucoEvents, kalmanCovHist, locErrorHist) {
  const dur  = Math.max(0, Math.floor(Date.now() / 1000 - stats.startTime));
  const mins = Math.floor(dur / 60);
  const secs = dur % 60;

  const avgTraceP = kalmanCovHist.length > 0
    ? (kalmanCovHist.reduce((s, d) => s + d.traceP, 0) / kalmanCovHist.length).toFixed(4)
    : 'N/A';
  const avgErr = locErrorHist.length > 0
    ? (locErrorHist.reduce((s, d) => s + d.error_xy, 0) / locErrorHist.length).toFixed(4)
    : 'N/A';
  const avgInn = arucoEvents.length > 0
    ? (arucoEvents.reduce((s, e) => s + (e.innovation ?? 0), 0) / arucoEvents.length).toFixed(4)
    : 'N/A';

  const stateRows = domStateLog.map(e =>
    `<tr><td>${new Date(e.time * 1000).toLocaleTimeString()}</td><td><b>${e.state}</b></td></tr>`
  ).join('');
  const arucoRows = arucoEvents.slice(-20).map(e =>
    `<tr><td>${new Date(e.time * 1000).toLocaleTimeString()}</td>` +
    `<td>(${e.x.toFixed(3)}, ${e.y.toFixed(3)})</td>` +
    `<td>${e.cov_xy.toFixed(5)}</td>` +
    `<td>${(e.innovation ?? 0).toFixed(4)} m</td></tr>`
  ).join('');

  const cards = [
    ['Duración',        `${mins}m ${secs}s`,   ''],
    ['Distancia',       stats.distanceTraveled.toFixed(2), 'm'],
    ['ArUco updates',   stats.arucoCount,       ''],
    ['Scan match',      stats.scanMatchCount,   ''],
    ['Avg trace(P_xy)', avgTraceP,              'm²'],
    ['Avg err Kal↔SLAM',avgErr,                 'm'],
    ['Avg innovación',  avgInn,                 'm'],
    ['Replans',         stats.replanCount,      ''],
  ].map(([lbl, val, unit]) =>
    `<div class="card"><div class="cv">${val}${unit ? ' <span class="cu">' + unit + '</span>' : ''}</div><div class="cl">${lbl}</div></div>`
  ).join('');

  const win = window.open('', '_blank', 'width=880,height=740');
  win.document.write(`<!DOCTYPE html><html><head>
<title>Puzzlebot Localization Report</title>
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
<h1>Puzzlebot Localization Report</h1>
<div class="sub">Generado: ${new Date().toLocaleString()} | Sesión: ${new Date(stats.startTime * 1000).toLocaleString()}</div>
<h2>Resumen</h2>
<div class="cards">${cards}</div>
<h2>Correcciones ArUco (últimas 20)</h2>
<table>
  <tr><th>Hora</th><th>Pose (x,y)</th><th>Cov XY</th><th>Innovación</th></tr>
  ${arucoRows || '<tr><td colspan="4">Sin correcciones</td></tr>'}
</table>
<h2>Transiciones FSM</h2>
<table>
  <tr><th>Hora</th><th>Estado</th></tr>
  ${stateRows || '<tr><td colspan="2">Sin transiciones</td></tr>'}
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
export default function MetricsPanel({
  velHistory, lidarHist, domStateLog,
  poseHist, kalmanCovHist, locErrorHist,
  arucoEvents, scanMatchHist,
  sessionStats, onReset,
}) {
  const dur  = Math.max(0, Math.floor(Date.now() / 1000 - sessionStats.startTime));
  const mins = Math.floor(dur / 60);
  const secs = dur % 60;

  const avgTraceP = useMemo(() =>
    kalmanCovHist.length > 0
      ? kalmanCovHist.reduce((s, d) => s + d.traceP, 0) / kalmanCovHist.length
      : null,
    [kalmanCovHist]
  );

  const avgLocError = useMemo(() =>
    locErrorHist.length > 0
      ? locErrorHist.reduce((s, d) => s + d.error_xy, 0) / locErrorHist.length
      : null,
    [locErrorHist]
  );

  const avgInnovation = useMemo(() =>
    arucoEvents.length > 0
      ? arucoEvents.reduce((s, e) => s + (e.innovation ?? 0), 0) / arucoEvents.length
      : null,
    [arucoEvents]
  );

  // ArUco detection frequency: events in last 10 seconds
  const arucoFreqHz = useMemo(() => {
    if (arucoEvents.length === 0) return 0;
    const now = arucoEvents[arucoEvents.length - 1].time;
    const recent = arucoEvents.filter(e => now - e.time <= 10).length;
    return (recent / 10).toFixed(2);
  }, [arucoEvents]);

  const seenStates = useMemo(
    () => [...new Set(domStateLog.map(e => e.state))].filter(s => s !== 'NORMAL'),
    [domStateLog]
  );

  return (
    <div className="metrics-panel">
      {/* ── Actions ── */}
      <div className="metrics-actions">
        <button className="btn-sm" onClick={onReset}>↺ Reset</button>
        <button className="btn-sm btn-sm-accent"
          onClick={() => exportCSV({
            velHistory, lidarHist, domStateLog, poseHist, kalmanCovHist,
            locErrorHist, arucoEvents, scanMatchHist, stats: sessionStats,
          })}>↓ CSV</button>
        <button className="btn-sm btn-sm-green"
          onClick={() => openPrintReport(sessionStats, domStateLog, arucoEvents, kalmanCovHist, locErrorHist)}>
          ⎙ PDF
        </button>
      </div>

      {/* ── Health gauges ── */}
      <div className="metrics-gauges">
        <GaugeCard
          label="Distancia"
          value={sessionStats.distanceTraveled}
          max={10}
          displayValue={`${sessionStats.distanceTraveled.toFixed(1)}m`}
          color="var(--accent)"
          sub="/ 10 m"
        />
        <GaugeCard
          label="ArUco Freq"
          value={parseFloat(arucoFreqHz)}
          max={2}
          displayValue={`${arucoFreqHz} Hz`}
          color="#a78bfa"
          sub="/ 2 Hz"
        />
        <GaugeCard
          label="Err K↔SLAM"
          value={avgLocError != null ? Math.max(0, 1 - avgLocError / 0.2) : 0}
          max={1}
          displayValue={avgLocError != null ? `${(avgLocError * 100).toFixed(1)} cm` : '—'}
          color="var(--green)"
          sub={avgLocError != null ? (avgLocError < 0.05 ? '✓ bueno' : '⚠ alto') : 'sin datos'}
        />
        <GaugeCard
          label="Kalman P"
          value={avgTraceP != null ? Math.max(0, 1 - avgTraceP / 0.5) : 0}
          max={1}
          displayValue={avgTraceP != null ? avgTraceP.toFixed(4) : '—'}
          color="var(--warn)"
          sub="trace(P_xy)"
        />
      </div>

      {/* ── Counter cards ── */}
      <div className="metrics-cards">
        <MetricCard label="Duración"      value={`${mins}m ${secs}s`}                              color="var(--muted)"  />
        <MetricCard label="ArUco upd."    value={sessionStats.arucoCount}                           color="#a78bfa"       />
        <MetricCard label="Scan match"    value={sessionStats.scanMatchCount}                        color="var(--blue)"   />
        <MetricCard label="Avg innov."    value={avgInnovation != null ? avgInnovation.toFixed(4) : '—'} unit="m" color="var(--err)" />
        <MetricCard label="Replans"       value={sessionStats.replanCount}                          color="var(--warn)"   />
        <MetricCard label="Stops"         value={sessionStats.obstacleStops}                        color="var(--err)"    />
      </div>

      {/* ══ KALMAN FILTER ══════════════════════════════════════════════════════ */}
      <div className="metrics-section-title">KALMAN FILTER</div>

      <ChartSection title="Incertidumbre — trace(P_xy) = P_xx + P_yy  (m²)">
        <LineChart
          data={kalmanCovHist}
          series={[
            { key: 'traceP',  label: 'trace(P_xy) m²',   color: '#4ade80' },
            { key: 'pTheta',  label: 'P_theta  rad²',     color: '#f59e0b' },
          ]}
          height={115}
          thresholds={[
            { value: 0.05, color: '#22d3ee', label: '0.05 ArUco activo' },
            { value: 0.30, color: '#f87171', label: '0.30 drift alto'   },
          ]}
        />
      </ChartSection>

      <ChartSection title="Posición estimada — Kalman (odom frame)  (m)">
        <LineChart
          data={poseHist}
          series={[
            { key: 'x',     label: 'x (m)',   color: '#60a5fa' },
            { key: 'y',     label: 'y (m)',   color: '#fb923c' },
            { key: 'theta', label: 'θ (rad)', color: '#a78bfa' },
          ]}
          height={110}
          zeroline
        />
      </ChartSection>

      <ChartSection title="Error de localización — Kalman vs SLAM  (m)">
        <LineChart
          data={locErrorHist}
          series={[
            { key: 'error_xy',    label: 'err XY (m)',    color: '#f87171' },
            { key: 'error_theta', label: 'err θ (rad)',   color: '#facc15' },
          ]}
          height={95}
          thresholds={[{ value: 0.05, color: '#555', label: '5 cm' }]}
        />
      </ChartSection>

      {/* ══ ARUCO ══════════════════════════════════════════════════════════════ */}
      <div className="metrics-section-title">CORRECCIONES ARUCO</div>

      <ChartSection title={`Trayectoria + puntos de corrección ArUco (${arucoEvents.length})`}>
        <TrajectoryMap poseHist={poseHist} arucoEvents={arucoEvents} />
      </ChartSection>

      <ChartSection title="Innovación ArUco — distancia entre medición y estimado Kalman  (m)">
        <EventBars events={arucoEvents} key="innovation" color="#f87171" height={80}
          noDataMsg="Sin correcciones ArUco aún…" />
        {arucoEvents.length > 0 && (
          <div className="metrics-note">
            Última corrección: ({arucoEvents[arucoEvents.length - 1].x.toFixed(3)},&nbsp;
            {arucoEvents[arucoEvents.length - 1].y.toFixed(3)}) m&nbsp;·&nbsp;
            innov = {(arucoEvents[arucoEvents.length - 1].innovation ?? 0).toFixed(4)} m
          </div>
        )}
      </ChartSection>

      <ChartSection title="Covarianza ArUco — distribución de cov_xy (incertidumbre de detección)">
        <CovDistribution events={arucoEvents} color="#a78bfa" height={65} />
      </ChartSection>

      <ChartSection title="Covarianza ArUco — cov_xy sobre tiempo  (m²)">
        <EventBars events={arucoEvents} key="cov_xy" color="#a78bfa" height={75}
          noDataMsg="Sin correcciones ArUco aún…" />
      </ChartSection>

      {/* ══ NAVEGACIÓN ═════════════════════════════════════════════════════════ */}
      <div className="metrics-section-title">NAVEGACIÓN</div>

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

      <ChartSection title="Velocidad — /cmd_vel  (auxiliar)">
        <LineChart
          data={velHistory}
          series={[
            { key: 'linear',  label: 'linear (m/s)',    color: '#60a5fa' },
            { key: 'angular', label: 'angular (rad/s)', color: '#22d3ee' },
          ]}
          height={90}
          zeroline
        />
      </ChartSection>
    </div>
  );
}
