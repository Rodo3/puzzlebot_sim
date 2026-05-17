import React, { useRef, useEffect } from 'react';

const CANVAS_SIZE = 300;
const ROBOT_RADIUS = 8;

export default function LidarView({ scanData }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    const cx = W / 2;
    const cy = H / 2;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, W, H);

    if (!scanData) {
      ctx.fillStyle = '#555';
      ctx.font = '14px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for /scan…', cx, cy);
      return;
    }

    const { ranges, angle_min, angle_increment, range_max } = scanData;
    const scale = (Math.min(W, H) / 2 - 10) / range_max;

    // Range rings.
    ctx.strokeStyle = '#222';
    ctx.lineWidth = 1;
    for (let r = 0.5; r <= range_max; r += 0.5) {
      ctx.beginPath();
      ctx.arc(cx, cy, r * scale, 0, 2 * Math.PI);
      ctx.stroke();
    }

    // LiDAR points.
    ctx.fillStyle = '#00e5ff';
    ranges.forEach((r, i) => {
      if (r == null) return;
      const angle = angle_min + i * angle_increment;
      const px = cx + r * scale * Math.cos(angle);
      const py = cy - r * scale * Math.sin(angle);  // flip Y
      ctx.fillRect(px - 1, py - 1, 2, 2);
    });

    // Robot body.
    ctx.beginPath();
    ctx.arc(cx, cy, ROBOT_RADIUS, 0, 2 * Math.PI);
    ctx.fillStyle = '#ff6f00';
    ctx.fill();

    // Min distance label.
    if (scanData.min_distance != null) {
      ctx.fillStyle = '#ccc';
      ctx.font = '12px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`min: ${scanData.min_distance.toFixed(2)} m`, 6, 16);
    }
  }, [scanData]);

  return (
    <div className="panel lidar-panel">
      <h3>LiDAR View</h3>
      <canvas ref={canvasRef} width={CANVAS_SIZE} height={CANVAS_SIZE} className="lidar-canvas" />
    </div>
  );
}
