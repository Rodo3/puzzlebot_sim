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

export default function SlamMap({ mapData, robotPose, trajectory, mode, goalMarker, onGoalPose }) {
  const canvasRef    = useRef(null);
  const offscreenRef = useRef(null);
  const lastMapRef   = useRef(null);
  const dragRef      = useRef(null);       // { startX, startY, startPanX, startPanY, moved }
  const mapOffsetRef = useRef({ x: 0, y: 0 }); // centering offset stored for canvasToWorld

  const [zoom,     setZoom]     = useState(1);
  const [pan,      setPan]      = useState({ x: 0, y: 0 });
  const [grabbing, setGrabbing] = useState(false);

  // Refs to access current zoom/pan inside event handlers without stale closures
  const zoomRef = useRef(1);
  const panRef  = useRef({ x: 0, y: 0 });
  useEffect(() => { zoomRef.current = zoom; }, [zoom]);
  useEffect(() => { panRef.current  = pan;  }, [pan]);

  // Build offscreen canvas when map data changes
  useEffect(() => {
    if (!mapData) return;
    const { width, height, data } = mapData;
    const off = document.createElement('canvas');
    off.width  = width;
    off.height = height;
    const octx      = off.getContext('2d');
    const imageData = octx.createImageData(width, height);
    renderGridToImageData(imageData, data, width, height);
    octx.putImageData(imageData, 0, 0);
    offscreenRef.current = off;
    lastMapRef.current   = mapData;
  }, [mapData]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const z   = zoomRef.current;
    const { x: panX, y: panY } = panRef.current;

    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

    // Apply zoom/pan transform (zoom around canvas center)
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

    // Center the map inside the canvas
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

    // Goal marker (green X)
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

    // Navigation hint overlay (not zoomed/panned — stays at corner)
    if (mode === 'navigation') {
      ctx.fillStyle = 'rgba(74,222,128,0.05)';
      ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
      ctx.fillStyle = 'rgba(74,222,128,0.65)';
      ctx.font      = '10px monospace';
      ctx.textAlign = 'left';
      ctx.fillText('NAV — clic para enviar goal', 6, 14);
    }
  }, [robotPose, trajectory, goalMarker, mode]);

  // Redraw on prop or zoom/pan changes
  useEffect(() => { draw(); }, [draw, mapData, zoom, pan]);

  // --- Zoom (scroll wheel) ---
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

    // Zoom-to-cursor: adjust pan so the point under cursor stays fixed
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

  // --- Pan (drag) + click-to-goal ---
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
    const z              = zoomRef.current;
    const { x: pX, y: pY } = panRef.current;
    const { x: offX, y: offY } = mapOffsetRef.current;

    // Undo zoom/pan transform
    const ux = (cx - pX - CANVAS_W / 2) / z + CANVAS_W / 2;
    const uy = (cy - pY - CANVAS_H / 2) / z + CANVAS_H / 2;

    // Undo map centering offset, then convert to world coords
    const { width, height, resolution, origin } = map;
    const cellSize = Math.min(CANVAS_W / width, CANVAS_H / height);
    const col = (ux - offX) / cellSize;
    const row = height - 1 - (uy - offY) / cellSize; // inverse of cellToCanvas (flip Y)
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
