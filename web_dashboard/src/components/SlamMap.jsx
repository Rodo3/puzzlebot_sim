import React, { useRef, useEffect, useCallback } from 'react';
import { renderGridToImageData, worldToCell, cellToCanvas } from '../utils/mapUtils.js';
import { drawCircle, drawArrow } from '../utils/geometry.js';

const CANVAS_W = 520;
const CANVAS_H = 520;
const ROBOT_RADIUS_PX = 6;
const GOAL_RADIUS_PX  = 8;

export default function SlamMap({ mapData, robotPose, trajectory, mode, goalMarker, onGoalPose }) {
  const canvasRef     = useRef(null);
  const offscreenRef  = useRef(null);
  const lastMapRef    = useRef(null);

  // Re-render map tiles only when map data changes (expensive).
  useEffect(() => {
    if (!mapData) return;
    const { width, height, data } = mapData;

    const offscreen = document.createElement('canvas');
    offscreen.width  = width;
    offscreen.height = height;
    const octx = offscreen.getContext('2d');
    const imageData = octx.createImageData(width, height);

    renderGridToImageData(imageData, data, width, height);
    octx.putImageData(imageData, 0, 0);
    offscreenRef.current = offscreen;
    lastMapRef.current   = mapData;
  }, [mapData]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
    ctx.fillStyle = '#1e1e1e';
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

    const map = lastMapRef.current;
    if (!map || !offscreenRef.current) {
      ctx.fillStyle = '#555';
      ctx.font = '14px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for /map…', CANVAS_W / 2, CANVAS_H / 2);
      return;
    }

    const { width, height, resolution, origin } = map;
    const cellSize = Math.min(CANVAS_W / width, CANVAS_H / height);

    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(offscreenRef.current, 0, 0, width * cellSize, height * cellSize);

    const worldToPx = (wx, wy) => {
      const { col, row } = worldToCell(wx, wy, origin.x, origin.y, resolution);
      return cellToCanvas(col, row, height, cellSize);
    };

    // Trajectory
    if (trajectory.length > 1) {
      ctx.beginPath();
      const start = worldToPx(trajectory[0].x, trajectory[0].y);
      ctx.moveTo(start.px, start.py);
      for (let i = 1; i < trajectory.length; i++) {
        const p = worldToPx(trajectory[i].x, trajectory[i].y);
        ctx.lineTo(p.px, p.py);
      }
      ctx.strokeStyle = 'rgba(0, 229, 255, 0.6)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // Goal marker
    if (goalMarker) {
      const { px, py } = worldToPx(goalMarker.x, goalMarker.y);
      ctx.beginPath();
      ctx.arc(px, py, GOAL_RADIUS_PX, 0, 2 * Math.PI);
      ctx.strokeStyle = '#69f0ae';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(px - 5, py - 5); ctx.lineTo(px + 5, py + 5);
      ctx.moveTo(px + 5, py - 5); ctx.lineTo(px - 5, py + 5);
      ctx.strokeStyle = '#69f0ae';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Robot
    if (robotPose) {
      const { px, py } = worldToPx(robotPose.x, robotPose.y);
      drawCircle(ctx, px, py, ROBOT_RADIUS_PX, '#2979ff');
      drawArrow(ctx, px, py, robotPose.theta, ROBOT_RADIUS_PX * 3, '#ffcc02');
    }

    // Navigation mode hint overlay
    if (mode === 'navigation') {
      ctx.fillStyle = 'rgba(105, 240, 174, 0.08)';
      ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
      ctx.fillStyle = 'rgba(105, 240, 174, 0.7)';
      ctx.font = '11px monospace';
      ctx.textAlign = 'left';
      ctx.fillText('NAV — clic para enviar goal', 6, 14);
    }
  }, [robotPose, trajectory, goalMarker, mode]);

  useEffect(() => {
    draw();
  }, [draw, mapData]);

  // Convert canvas click to world coordinates and emit goal_pose
  const handleClick = useCallback((e) => {
    if (mode !== 'navigation' || !lastMapRef.current || !onGoalPose) return;
    const map = lastMapRef.current;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = CANVAS_W / rect.width;
    const scaleY = CANVAS_H / rect.height;
    const canvasX = (e.clientX - rect.left) * scaleX;
    const canvasY = (e.clientY - rect.top)  * scaleY;

    const cellSize = Math.min(CANVAS_W / map.width, CANVAS_H / map.height);
    const col = canvasX / cellSize;
    // row 0 in canvas is the top (north), but map row 0 is the south (origin_y).
    // cellToCanvas uses: py = (height - row) * cellSize → invert to get row
    const row = map.height - canvasY / cellSize;

    const wx = map.origin.x + col * map.resolution;
    const wy = map.origin.y + row * map.resolution;

    onGoalPose({ x: wx, y: wy, theta: 0 });
  }, [mode, onGoalPose]);

  return (
    <div className="panel slam-panel">
      <h3>
        SLAM Map
        {mode === 'navigation' && (
          <span className="mode-badge-nav"> [NAV]</span>
        )}
      </h3>
      {lastMapRef.current && (
        <div className="muted small">
          {lastMapRef.current.width}×{lastMapRef.current.height} cells —{' '}
          {lastMapRef.current.resolution} m/cell
        </div>
      )}
      <canvas
        ref={canvasRef}
        width={CANVAS_W}
        height={CANVAS_H}
        className={`slam-canvas ${mode === 'navigation' ? 'slam-canvas-nav' : ''}`}
        onClick={handleClick}
      />
    </div>
  );
}
