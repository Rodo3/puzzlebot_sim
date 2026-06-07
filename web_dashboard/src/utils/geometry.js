/**
 * Helpers for converting world coordinates (meters) to canvas pixels,
 * and for drawing robot orientation markers.
 */

/**
 * Map a world (x, y) in meters to canvas pixel (px, py).
 * @param {number} wx  World X (meters)
 * @param {number} wy  World Y (meters)
 * @param {number} originX  Map origin X (meters)
 * @param {number} originY  Map origin Y (meters)
 * @param {number} resolution  Meters per cell
 * @param {number} cellSize  Pixels per cell
 */
export function worldToCanvas(wx, wy, originX, originY, resolution, cellSize) {
  const col = (wx - originX) / resolution;
  const row = (wy - originY) / resolution;
  return {
    px: col * cellSize,
    py: row * cellSize,
  };
}

/**
 * Draw a filled circle on a 2D canvas context.
 */
export function drawCircle(ctx, cx, cy, radius, color) {
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
}

/**
 * Draw an orientation arrow from (cx, cy) in direction theta (radians).
 * Canvas Y axis is flipped vs. ROS frame, so we negate the Y component.
 */
export function drawArrow(ctx, cx, cy, theta, length, color) {
  const ex = cx + length * Math.cos(theta);
  const ey = cy - length * Math.sin(theta);  // flip Y

  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(ex, ey);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();

  // Arrowhead.
  const angle = Math.atan2(ey - cy, ex - cx);
  const headLen = length * 0.3;
  ctx.beginPath();
  ctx.moveTo(ex, ey);
  ctx.lineTo(ex - headLen * Math.cos(angle - 0.4), ey - headLen * Math.sin(angle - 0.4));
  ctx.moveTo(ex, ey);
  ctx.lineTo(ex - headLen * Math.cos(angle + 0.4), ey - headLen * Math.sin(angle + 0.4));
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();
}
