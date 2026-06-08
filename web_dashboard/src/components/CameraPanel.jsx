import React, { useRef, useEffect, useState, useCallback } from 'react';

// Colores por clase de logo — nombres exactos del modelo entrenado.
const LOGO_COLORS = { Pepsi: '#f87171', Amazon: '#fbbf24', Walmart: '#4ade80',
                      Popsi: '#f87171', Emezon: '#fbbf24', Wolmar: '#4ade80' };
const QR_COLOR    = '#22d3ee';

export default function CameraPanel({ cameraData, qrDetections = [], logoDetections = [] }) {
  const imgRef    = useRef(null);
  const canvasRef = useRef(null);
  const [overlay, setOverlay] = useState(true);

  const draw = useCallback(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return;

    const nw = img.naturalWidth, nh = img.naturalHeight;
    const cw = img.clientWidth,  ch = img.clientHeight;
    if (!nw || !nh || !cw || !ch) return;

    // Backing store en px CSS → dibujo 1:1 con el tamaño mostrado.
    if (canvas.width !== cw)  canvas.width  = cw;
    if (canvas.height !== ch) canvas.height = ch;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, cw, ch);
    if (!overlay) return;

    const sx = cw / nw, sy = ch / nh;
    ctx.lineWidth = 2;
    ctx.font = '12px monospace';
    ctx.textBaseline = 'bottom';

    // ── QR: polígono + etiqueta ──
    for (const d of qrDetections) {
      const c = d.corners;
      if (!Array.isArray(c) || c.length < 4) continue;
      ctx.strokeStyle = QR_COLOR;
      ctx.fillStyle   = QR_COLOR;
      ctx.beginPath();
      ctx.moveTo(c[0][0] * sx, c[0][1] * sy);
      for (let i = 1; i < c.length; i++) ctx.lineTo(c[i][0] * sx, c[i][1] * sy);
      ctx.closePath();
      ctx.stroke();
      ctx.fillText(`QR: ${d.data}`, c[0][0] * sx, Math.max(12, c[0][1] * sy - 3));
    }

    // ── Logos: caja + clase/confianza ──
    // Bridge serializa como {cx, cy, w, h} normalizados (0-1) desde Detection2DArray.
    for (const d of logoDetections) {
      const b = d.bbox;
      if (!b) continue;
      const color = LOGO_COLORS[d.class_name] || '#a78bfa';
      const bw = b.w * cw, bh = b.h * ch;
      const bx = b.cx * cw - bw / 2, by = b.cy * ch - bh / 2;
      ctx.strokeStyle = color;
      ctx.fillStyle   = color;
      ctx.strokeRect(bx, by, bw, bh);
      ctx.fillText(`${d.class_name} ${(d.confidence ?? 0).toFixed(2)}`, bx, Math.max(12, by - 3));
    }
  }, [overlay, qrDetections, logoDetections]);

  // Redibuja cuando cambian el frame, las detecciones o el toggle.
  useEffect(() => { draw(); }, [cameraData, draw]);

  // Redibuja al cambiar el tamaño de la ventana (el layout puede reescalar).
  useEffect(() => {
    window.addEventListener('resize', draw);
    return () => window.removeEventListener('resize', draw);
  }, [draw]);

  return (
    <div className="panel camera-panel">
      <div className="camera-header">
        <h3 style={{ margin: 0 }}>Camera</h3>
        <button
          className={`camera-toggle ${overlay ? 'on' : ''}`}
          onClick={() => setOverlay(o => !o)}
          title="Mostrar/ocultar detecciones"
        >{overlay ? '▣ Anotada' : '▢ Raw'}</button>
      </div>

      {cameraData ? (
        <div className="camera-stage">
          <img
            ref={imgRef}
            className="camera-img"
            src={`data:image/jpeg;base64,${cameraData.data}`}
            alt="robot camera"
            onLoad={draw}
          />
          <canvas ref={canvasRef} className="camera-overlay" />
        </div>
      ) : (
        <div className="camera-placeholder">Waiting for /camera/image/compressed…</div>
      )}
    </div>
  );
}
