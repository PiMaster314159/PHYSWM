"""Preview the 'ring' marker: white body + thin grey heading ring around the front tip.
Pure numpy + matplotlib (no torch), so it runs anywhere. Saves a montage with three rows:
full 64x64 frame, a zoom, and the TRUE vector geometry (triangle outline + ring circle +
heading axis) so you can see the actual robot shape the rasterizer is approximating.

    python preview_ring.py
"""
import sys
from pathlib import Path
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

import config as C
from sim.render import render_frame, make_triangle

GRID   = 64
thetas = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315])

tri_local = make_triangle()                      # (3,2) body-frame vertices, centroid at origin
apex_x    = 2.0 * C.L / 3.0                       # apex local x
ring_c    = np.array([C.RING_OFFSET * apex_x, 0.0])
apex_loc  = np.array([apex_x, 0.0])


def to_world(local_pts, theta, p=(0.0, 0.0)):
    """local (body-frame) -> world: p + local @ [[cos,sin],[-sin,cos]]  (matches the rasterizer)."""
    c, s = np.cos(theta), np.sin(theta)
    Rt = np.array([[c, s], [-s, c]])
    return np.atleast_2d(local_pts) @ Rt + np.asarray(p)


fig, axes = plt.subplots(3, len(thetas), figsize=(2.0 * len(thetas), 6.8))
cpix = int(0.5 * GRID)
for j, th in enumerate(thetas):
    frame = render_frame((0.5, 0.5, th), grid_size=GRID, marker="ring")
    axes[0][j].imshow(frame, cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest")
    axes[0][j].set_title(f"{np.rad2deg(th):.0f} deg", fontsize=9)
    z = frame[cpix - 9:cpix + 9, cpix - 9:cpix + 9]
    axes[1][j].imshow(z, cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest")

    # vector geometry: true triangle + ring + heading axis, robot centered at origin
    ax = axes[2][j]
    ax.set_facecolor("black")
    triw = to_world(tri_local, th)
    cw   = to_world(ring_c, th)[0]
    apw  = to_world(apex_loc, th)[0]
    ax.add_patch(Polygon(triw, closed=True, facecolor="white", edgecolor="white", lw=1.0))
    ax.add_patch(Circle(cw, C.RING_RADIUS, fill=False, edgecolor="0.6", lw=1.6))
    ax.plot([0, apw[0]], [0, apw[1]], color="#ff5555", lw=1.0)   # heading axis (centroid -> tip)
    ax.plot(0, 0, "o", color="#55aaff", ms=3)                    # centroid (x, y)
    lim = 0.11
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")

for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
axes[0][0].set_ylabel("full 64x64", fontsize=8)
axes[1][0].set_ylabel("zoom 18x18", fontsize=8)
axes[2][0].set_ylabel("vector geometry", fontsize=8)
fig.suptitle("ring marker: pixels (top, zoom) vs true geometry (bottom)  |  white body, grey heading ring", y=0.99)
fig.tight_layout()
out = ROOT / "results" / "_compare" / "ring_preview.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"wrote {out}")
print(f"frame values {np.unique(frame)}  |  L={C.L} W={C.W}  ring r={C.RING_RADIUS} offset={C.RING_OFFSET}")
