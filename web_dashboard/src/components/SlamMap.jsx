import React, { useRef, useEffect, useCallback, useState } from 'react';
import { renderGridToImageData, worldToCell, cellToCanvas } from '../utils/mapUtils.js';
import { drawCircle, drawArrow } from '../utils/geometry.js';

const CANVAS_W       = 520;
const CANVAS_H       = 520;
const ROBOT_RADIUS   = 6;
const GOAL_RADIUS    = 8;
const DRAG_THRESHOLD = 4;
const ZOOM_MIN       = 0.25;
const ZOOM_MAX       = 12;

export default function SlamMap({ mapData, augMapData, robotPose, trajectory, mode, goalMarker, onGoalPose }) {
  const canvasRef    = useRef(null);
  const offscreenRef = useRef(null);
  const lastMapRef   = useRef(null);
  const dragRef      = useRef(null);
  const mapOffsetRef = useRef({ x: 0, y: 0 });

  const [zoom,     setZoom]     = useState(1);
  const [pan,      setPan]      = useState({ x: 0, y: 0 });
  const [grabbing, setGrabbing] = useState(false);
  // Toggle: show augmented map (with dynamic obstacles) vs base SLAM map
  const [showAug, setShowAug] = useState(true);

  const zoomRef = useRef(1);
  const panRef  = useRef({ x: 0, y: 0 });
  useEffect(() => { zoomRef.current = zoom; }, [zoom]);
  useEffect(() => { panRef.current  = pan;  }, [pan]);

  // Active map: augmented when available and toggle is on, else base map
  const activeMapData = (showAug && augMapData) ? augMapData : mapData;

  // Rebuild offscreen canvas when the active map changes
  useEffect(() => {
    if (!activeMapData) return;
    const { width, height, data } = activeMapData;
    const off = document.createElement('canvas');
    off.width  = width;
    off.height = height;
    const octx      = off.getContext('2d');
    const imageData = octx.createImageData(width, height);
    renderGridToImageData(imageData, data, width, height);
    octx.putImageData(imageData, 0, 0);
    offscreenRef.current = off;
    lastMapRef.current   = activeMapData;
  }, [activeMapData]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const z   = zoomRef.current;
    const { x: panX, y: panY } = panRef.current;

    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

    ctx.save();
    ctx.translate(panX + CANVAS_W / 2, panY + CANVAS_H / 2);
    ctx.scale(z, z);
    ctx.translate(-CANVAS_W / 2, -CANVAS_H / 2);

    const map = lastMapRef.current;
    if (!map || !offscreenRef.current) {
      ctx.fillStyle = '#444';
      ctx.font = '14px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for /map…', CANVAS_W / 2, CANVAS_H / 2);
      ctx.restore();
      return;
    }

    const { width, height, resolution, origin } = map;
    const cellSize = Math.min(CANVAS_W / width, CANVAS_H / height);

    const mapW    = width  * cellSize;
    const mapH    = height * cellSize;
    const offsetX = (CANVAS_W - mapW) / 2;
    const offsetY = (CANVAS_H - mapH) / 2;
    mapOffsetRef.current = { x: offsetX, y: offsetY };

    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(offscreenRef.current, offsetX, offsetY, mapW, mapH);

    const worldToPx = (wx, wy) => {
      const { col, row } = worldToCell(wx, wy, origin.x, origin.y, resolution);
      const { px, py }   = cellToCanvas(col, row, height, cellSize);
      return { px: px + offsetX, py: py + offsetY };
    };

    // Trajectory (cyan)
    if (trajectory.length > 1) {
      ctx.beginPath();
      const s = worldToPx(trajectory[0].x, trajectory[0].y);
      ctx.moveTo(s.px, s.py);
      for (let i = 1; i < trajectory.length; i++) {
        const p = worldToPx(trajectory[i].x, trajectory[i].y);
        ctx.lineTo(p.px, p.py);
      }
      ctx.strokeStyle = 'rgba(34,211,238,0.55)';
      ctx.lineWidth   = 1.5 / z;
      ctx.stroke();
    }

    // Goal marker (green circle + X)
    if (goalMarker) {
      const { px: gx, py: gy } = worldToPx(goalMarker.x, goalMarker.y);
      const r = GOAL_RADIUS / z;
      const s = 5 / z;
      ctx.beginPath();
      ctx.arc(gx, gy, r, 0, 2 * Math.PI);
      ctx.strokeStyle = '#4ade80';
      ctx.lineWidth   = 2 / z;
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(gx - s, gy - s); ctx.lineTo(gx + s, gy + s);
      ctx.moveTo(gx + s, gy - s); ctx.lineTo(gx - s, gy + s);
      ctx.strokeStyle = '#4ade80';
      ctx.lineWidth   = 2 / z;
      ctx.stroke();
    }

    // Robot (blue circle + yellow arrow)
    if (robotPose) {
      const { px: rx, py: ry } = worldToPx(robotPose.x, robotPose.y);
      drawCircle(ctx, rx, ry, ROBOT_RADIUS / z, '#60a5fa');
      drawArrow(ctx, rx, ry, robotPose.theta, (ROBOT_RADIUS * 3) / z, '#facc15');
    }

    ctx.restore();

    // Overlays (not zoomed — stay at canvas corner)
    if (mode === 'navigation') {
      ctx.fillStyle = 'rgba(74,222,128,0.05)';
      ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
      ctx.fillStyle = 'rgba(74,222,128,0.65)';
      ctx.font      = '10px monospace';
      ctx.textAlign = 'left';
      ctx.fillText('NAV — clic para enviar goal', 6, 14);
    }
  }, [robotPose, trajectory, goalMarker, mode]);

  useEffect(() => { draw(); }, [draw, activeMapData, zoom, pan]);

  // Zoom (scroll wheel)
  const handleWheel = useCallback((e) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect   = canvas.getBoundingClientRect();
    const scaleX = CANVAS_W / rect.width;
    const scaleY = CANVAS_H / rect.height;
    const cx     = (e.clientX - rect.left) * scaleX;
    const cy     = (e.clientY - rect.top)  * scaleY;

    const oldZ   = zoomRef.current;
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZ   = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, oldZ * factor));

    const { x: oldPanX, y: oldPanY } = panRef.current;
    const newPanX = oldPanX + (cx - oldPanX - CANVAS_W / 2) * (1 - newZ / oldZ);
    const newPanY = oldPanY + (cy - oldPanY - CANVAS_H / 2) * (1 - newZ / oldZ);

    setZoom(newZ);
    setPan({ x: newPanX, y: newPanY });
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.addEventListener('wheel', handleWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', handleWheel);
  }, [handleWheel]);

  // Pan (drag) + click-to-goal
  const getCanvasXY = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect   = canvas.getBoundingClientRect();
    const scaleX = CANVAS_W / rect.width;
    const scaleY = CANVAS_H / rect.height;
    return {
      cx: (e.clientX - rect.left) * scaleX,
      cy: (e.clientY - rect.top)  * scaleY,
    };
  }, []);

  const canvasToWorld = useCallback((cx, cy) => {
    const map = lastMapRef.current;
    if (!map) return null;
    const z                   = zoomRef.current;
    const { x: pX, y: pY }   = panRef.current;
    const { x: offX, y: offY } = mapOffsetRef.current;

    const ux = (cx - pX - CANVAS_W / 2) / z + CANVAS_W / 2;
    const uy = (cy - pY - CANVAS_H / 2) / z + CANVAS_H / 2;

    const { width, height, resolution, origin } = map;
    const cellSize = Math.min(CANVAS_W / width, CANVAS_H / height);
    const col = (ux - offX) / cellSize;
    const row = height - 1 - (uy - offY) / cellSize;
    return {
      x: origin.x + col * resolution,
      y: origin.y + row * resolution,
    };
  }, []);

  const handleMouseDown = useCallback((e) => {
    e.preventDefault();
    const pos = getCanvasXY(e);
    if (!pos) return;
    dragRef.current = {
      startX:    pos.cx,
      startY:    pos.cy,
      startPanX: panRef.current.x,
      startPanY: panRef.current.y,
      moved:     false,
    };
    setGrabbing(true);
  }, [getCanvasXY]);

  const handleMouseMove = useCallback((e) => {
    if (!dragRef.current) return;
    const pos = getCanvasXY(e);
    if (!pos) return;
    const dx = pos.cx - dragRef.current.startX;
    const dy = pos.cy - dragRef.current.startY;
    if (Math.hypot(dx, dy) > DRAG_THRESHOLD) dragRef.current.moved = true;
    setPan({
      x: dragRef.current.startPanX + dx,
      y: dragRef.current.startPanY + dy,
    });
  }, [getCanvasXY]);

  const handleMouseUp = useCallback((e) => {
    if (!dragRef.current) return;
    const wasDrag = dragRef.current.moved;
    dragRef.current = null;
    setGrabbing(false);

    if (!wasDrag && mode === 'navigation' && onGoalPose) {
      const pos = getCanvasXY(e);
      if (pos) {
        const world = canvasToWorld(pos.cx, pos.cy);
        if (world) onGoalPose({ x: world.x, y: world.y, theta: 0 });
      }
    }
  }, [mode, onGoalPose, getCanvasXY, canvasToWorld]);

  const handleMouseLeave = useCallback(() => {
    if (dragRef.current) { dragRef.current = null; setGrabbing(false); }
  }, []);

  const resetView = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  const cursorClass = grabbing
    ? 'slam-canvas-grab'
    : mode === 'navigation'
      ? 'slam-canvas-nav'
      : 'slam-canvas-map';

  const augAvailable = !!augMapData;

  return (
    <div className="panel slam-panel">
      <div className="slam-header">
        <div className="slam-header-left">
          <span className="slam-title">
            SLAM Map{mode === 'navigation' ? ' [NAV]' : ''}
          </span>
          {lastMapRef.current && (
            <span className="slam-dims">
              {lastMapRef.current.width}×{lastMapRef.current.height} —{' '}
              {lastMapRef.current.resolution} m/cell
            </span>
          )}
          {/* Augmented map layer toggle — only shown once /augmented_map arrives */}
          {augAvailable && (
            <button
              className={`btn-aug-toggle ${showAug ? 'btn-aug-active' : ''}`}
              onClick={() => setShowAug(v => !v)}
              title={showAug ? 'Mostrando mapa aumentado (obstáculos dinámicos). Clic para ver mapa base.' : 'Mostrando mapa base. Clic para ver obstáculos dinámicos.'}
            >
              {showAug ? 'AUG' : 'BASE'}
            </button>
          )}
        </div>
        <button className="btn-reset-view" onClick={resetView} title="Reset view">⌂</button>
      </div>
      <div className="slam-canvas-wrap">
        <canvas
          ref={canvasRef}
          width={CANVAS_W}
          height={CANVAS_H}
          className={`slam-canvas ${cursorClass}`}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
        />
      </div>
    </div>
  );
}
