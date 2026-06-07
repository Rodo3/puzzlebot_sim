/**
 * Utilities for rendering nav_msgs/OccupancyGrid data on a Canvas.
 *
 * Cell values:
 *   -1  → unknown  (mid grey)
 *    0  → free     (white)
 *  100  → occupied (black)
 *  1-99 → partial  (scaled grey)
 */

export const CELL_COLORS = {
  UNKNOWN: '#aaaaaa',
  FREE: '#ffffff',
  OCCUPIED: '#1a1a1a',
};

/**
 * Map an OccupancyGrid cell value to an RGBA color string.
 */
export function cellToColor(value) {
  if (value === -1) return CELL_COLORS.UNKNOWN;
  if (value === 0)  return CELL_COLORS.FREE;
  if (value === 100) return CELL_COLORS.OCCUPIED;
  // Partial probability — scale from white to black.
  const grey = Math.round(255 - (value / 100) * 255);
  return `rgb(${grey},${grey},${grey})`;
}

/**
 * Render a full OccupancyGrid onto an ImageData buffer (faster than fillRect per cell).
 *
 * @param {ImageData} imageData  Pre-allocated ImageData (width × height)
 * @param {number[]}  data       Flat array of cell values (row-major, bottom-left origin in ROS)
 * @param {number}    width      Grid width in cells
 * @param {number}    height     Grid height in cells
 */
export function renderGridToImageData(imageData, data, width, height) {
  const buf = imageData.data;
  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      const cellIdx = row * width + col;
      const value = data[cellIdx];

      // ROS grid origin is bottom-left; canvas origin is top-left → flip row.
      const canvasRow = height - 1 - row;
      const pixelIdx = (canvasRow * width + col) * 4;

      let r, g, b;
      if (value === -1) {
        r = 170; g = 170; b = 170;
      } else if (value === 0) {
        r = 255; g = 255; b = 255;
      } else if (value === 100) {
        r = 26; g = 26; b = 26;
      } else {
        const grey = Math.round(255 - (value / 100) * 255);
        r = grey; g = grey; b = grey;
      }
      buf[pixelIdx]     = r;
      buf[pixelIdx + 1] = g;
      buf[pixelIdx + 2] = b;
      buf[pixelIdx + 3] = 255;
    }
  }
}

/**
 * Convert a world position (meters) to grid cell indices.
 */
export function worldToCell(wx, wy, originX, originY, resolution) {
  return {
    col: Math.floor((wx - originX) / resolution),
    row: Math.floor((wy - originY) / resolution),
  };
}

/**
 * Convert grid cell (col, row) to canvas pixel using a given cell pixel size.
 * Flips row to match canvas origin.
 */
export function cellToCanvas(col, row, gridHeight, cellSize) {
  return {
    px: col * cellSize,
    py: (gridHeight - 1 - row) * cellSize,
  };
}
