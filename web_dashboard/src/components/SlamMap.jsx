import React, { useRef, useEffect, useCallback } from 'react';
import { renderGridToImageData, worldToCell, cellToCanvas } from '../utils/mapUtils.js';
import { drawCircle, drawArrow } from '../utils/geometry.js';

const CANVAS_W = 520;
const CANVAS_H = 520;
const MAX_TRAJECTORY = 500;
const ROBOT_RADIUS_PX = 6;

export default function SlamMap({ mapData, robotPose, trajectory }) {
  const canvasRef     = useRef(null);
  const offscreenRef  = useRef(null);  // offscreen canvas for map tiles
  const lastMapRef    = useRef(null);  // cache last rendered map data array

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

  // Draw everything on the visible canvas.
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

    // Draw map.
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(offscreenRef.current, 0, 0, width * cellSize, height * cellSize);

    const worldToPx = (wx, wy) => {
      const { col, row } = worldToCell(wx, wy, origin.x, origin.y, resolution);
      return cellToCanvas(col, row, height, cellSize);
    };

    // Trajectory.
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

    // Robot.
    if (robotPose) {
      const { px, py } = worldToPx(robotPose.x, robotPose.y);
      drawCircle(ctx, px, py, ROBOT_RADIUS_PX, '#2979ff');
      drawArrow(ctx, px, py, robotPose.theta, ROBOT_RADIUS_PX * 3, '#ffcc02');
    }
  }, [robotPose, trajectory]);

  useEffect(() => {
    draw();
  }, [draw, mapData]);

  return (
    <div className="panel slam-panel">
      <h3>SLAM Map</h3>
      {lastMapRef.current && (
        <div className="muted small">
          {lastMapRef.current.width}×{lastMapRef.current.height} cells —{' '}
          {lastMapRef.current.resolution} m/cell
        </div>
      )}
      <canvas ref={canvasRef} width={CANVAS_W} height={CANVAS_H} className="slam-canvas" />
    </div>
  );
}
