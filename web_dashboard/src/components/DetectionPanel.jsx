import React from 'react';

const LOGO_COLOR = {
  Pepsi: '#f87171', Amazon: '#fbbf24', Walmart: '#4ade80',
  Popsi: '#f87171', Emezon: '#fbbf24', Wolmar:  '#4ade80',
};

function ConfBar({ value }) {
  const pct = Math.round((value ?? 0) * 100);
  const color = pct >= 80 ? '#4ade80' : pct >= 60 ? '#fbbf24' : '#f87171';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, flex: 1 }}>
      <div style={{
        flex: 1, height: 6, background: '#1e1e2e', borderRadius: 3, overflow: 'hidden',
      }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 3 }} />
      </div>
      <span style={{ fontSize: '0.75em', opacity: 0.8, minWidth: 30 }}>{pct}%</span>
    </div>
  );
}

export default function DetectionPanel({ logoDetections = [], qrDetections = [], arucoIds = [] }) {
  const hasLogos = logoDetections.length > 0;
  const hasQr    = qrDetections.length > 0;
  const hasAruco = arucoIds.length > 0;

  return (
    <div className="panel detection-panel">
      <h3 style={{ margin: '0 0 8px 0', fontSize: '0.85em', letterSpacing: '0.05em' }}>
        DETECCIONES
      </h3>

      {/* ── YOLO logos ─────────────────────────────────────────── */}
      <div className="det-section">
        <div className="det-section-title">
          <span className="det-dot" style={{ background: '#a78bfa' }} />
          Logos
          {hasLogos && <span className="det-count">{logoDetections.length}</span>}
        </div>
        {hasLogos ? (
          <div className="det-list">
            {logoDetections.map((d, i) => {
              const color = LOGO_COLOR[d.class_name] || '#a78bfa';
              return (
                <div key={i} className="det-item">
                  <span className="det-class-badge" style={{ background: color + '33', color, borderColor: color }}>
                    {d.class_name}
                  </span>
                  <ConfBar value={d.confidence} />
                </div>
              );
            })}
          </div>
        ) : (
          <div className="det-empty">Sin detecciones</div>
        )}
      </div>

      {/* ── QR ──────────────────────────────────────────────────── */}
      <div className="det-section">
        <div className="det-section-title">
          <span className="det-dot" style={{ background: '#22d3ee' }} />
          QR
          {hasQr && <span className="det-count">{qrDetections.length}</span>}
        </div>
        {hasQr ? (
          <div className="det-list">
            {qrDetections.map((d, i) => (
              <div key={i} className="det-item det-qr">
                <span className="det-qr-data">{d.data ?? '—'}</span>
                {d.area_px != null && (
                  <span className="det-qr-area">{Math.round(d.area_px)} px²</span>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="det-empty">Sin QR</div>
        )}
      </div>

      {/* ── ArUco IDs ───────────────────────────────────────────── */}
      <div className="det-section">
        <div className="det-section-title">
          <span className="det-dot" style={{ background: '#fb923c' }} />
          ArUco
          {hasAruco && <span className="det-count">{arucoIds.length}</span>}
        </div>
        {hasAruco ? (
          <div className="det-aruco-ids">
            {arucoIds.map(id => (
              <span key={id} className="det-aruco-badge">ID {id}</span>
            ))}
          </div>
        ) : (
          <div className="det-empty">Sin markers</div>
        )}
      </div>
    </div>
  );
}
