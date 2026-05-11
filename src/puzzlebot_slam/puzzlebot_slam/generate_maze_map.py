"""
Generate maze_map.png from the maze world geometry.

Run this script once (or whenever maze.sdf changes) to regenerate the PNG
that the MCL node uses for localisation.

Usage:
    python3 generate_maze_map.py
    # Prints map_origin_x/y — paste those into gz_sim.launch.py mcl parameters.
    # Saves maze_map.png next to this script.

Map convention (matches mcl.py):
  white (255) = free space
  black (0)   = obstacle / wall
  row 0 = top of image = highest y in world
  (map_origin_x, map_origin_y) = world coordinate at pixel (col=0, row=height-1)
"""

import math
import os
from PIL import Image, ImageDraw

MAP_RES  = 0.05   # metres per pixel
MARGIN   = 0.30   # extra free space around boundary [m]
WALL_PAD = 0.0    # extra half-thickness added to each wall

# ── Geometry from maze.sdf ────────────────────────────────────────────────────
# (cx, cy, half_length, half_width, yaw_rad)
WALLS = [
    (-5.46,  -6.22,   3.75/2, 0.15/2, -math.pi/2),
    (-0.41,  -8.02,  10.25/2, 0.15/2,  0.0),
    ( 4.64,  -2.595, 11.00/2, 0.15/2,  math.pi/2),
    (-5.46,  -3.495,  2.00/2, 0.15/2,  math.pi/2),
    (-5.46,  -0.145,  5.00/2, 0.15/2,  math.pi/2),
    (-0.41,   2.83,  10.25/2, 0.15/2,  math.pi),
    (-5.46,   2.555,  0.70/2, 0.15/2, -math.pi/2),
]

# (cx, cy, half_sx, half_sy) — all axis-aligned 1×1 m boxes
BOXES = [
    (-2.599,  0.229, 0.5, 0.5),
    ( 1.091, -2.438, 0.5, 0.5),
    (-1.383, -3.322, 0.5, 0.5),
    (-3.371, -6.146, 0.5, 0.5),
    ( 2.467,  0.506, 0.5, 0.5),
    ( 2.364, -5.329, 0.5, 0.5),
    (-4.104, -1.578, 0.5, 0.5),
    ( 0.208, -7.051, 0.5, 0.5),
]


def _rotated_rect_corners(cx, cy, hl, hw, yaw):
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    corners = [(hl, hw), (-hl, hw), (-hl, -hw), (hl, -hw)]
    return [(cx + lx*cy_ - ly*sy_, cy + lx*sy_ + ly*cy_) for lx, ly in corners]


def _world_to_pixel(wx, wy, orig_x, orig_y, res, img_h):
    return (wx - orig_x) / res, img_h - 1 - (wy - orig_y) / res


all_pts = []
for (cx, cy, hl, hw, yaw) in WALLS:
    all_pts.extend(_rotated_rect_corners(cx, cy, hl + WALL_PAD, hw + WALL_PAD, yaw))
for (cx, cy, hsx, hsy) in BOXES:
    all_pts += [(cx-hsx, cy-hsy), (cx+hsx, cy-hsy),
                (cx+hsx, cy+hsy), (cx-hsx, cy+hsy)]

xs = [p[0] for p in all_pts]
ys = [p[1] for p in all_pts]
world_min_x = min(xs) - MARGIN
world_min_y = min(ys) - MARGIN
world_max_x = max(xs) + MARGIN
world_max_y = max(ys) + MARGIN
img_w = math.ceil((world_max_x - world_min_x) / MAP_RES)
img_h = math.ceil((world_max_y - world_min_y) / MAP_RES)
map_origin_x = world_min_x
map_origin_y = world_min_y

print(f"Canvas: {img_w}×{img_h} px  ({img_w*MAP_RES:.2f}×{img_h*MAP_RES:.2f} m)")
print(f"map_origin_x: {map_origin_x:.4f}")
print(f"map_origin_y: {map_origin_y:.4f}")
print(f"map_resolution: {MAP_RES}")
print()
print("Paste into gz_sim.launch.py mcl parameters:")
print(f"  'map_origin_x':  {map_origin_x:.4f},")
print(f"  'map_origin_y':  {map_origin_y:.4f},")
print(f"  'map_resolution': {MAP_RES},")

img  = Image.new('L', (img_w, img_h), color=255)
draw = ImageDraw.Draw(img)


def px(wx, wy):
    return _world_to_pixel(wx, wy, map_origin_x, map_origin_y, MAP_RES, img_h)


for (cx, cy, hl, hw, yaw) in WALLS:
    corners_w = _rotated_rect_corners(cx, cy, hl + WALL_PAD, hw + WALL_PAD, yaw)
    draw.polygon([px(wx, wy) for wx, wy in corners_w], fill=0)

for (cx, cy, hsx, hsy) in BOXES:
    x0, y0 = px(cx - hsx, cy + hsy)
    x1, y1 = px(cx + hsx, cy - hsy)
    draw.rectangle([x0, y0, x1, y1], fill=0)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'maze_map.png')
img.save(out_path, optimize=False)
print(f"\nSaved: {out_path}  ({img_w}×{img_h} px)")
