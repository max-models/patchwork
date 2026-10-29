"""The animated figure at the top of the docs home page: ``docs/src/assets/hero.svg``.

patchwork solves a Poisson problem on a hierarchy of nested patches: a sharp, narrow charge
on the right, refined by three levels of patches around it, and a broad charge of the other
sign to its left. The figure shows the patch outlines, which appear one level at a time, and
the contours of the composite potential, which run smoothly across the patch edges; a few dots
drift along them. The solve takes about a second.

Rerun with ``make docs-hero`` or ``python docs/hero.py`` (needs ``plasma_plots.hero``).
"""

from pathlib import Path

import contourpy
import numpy as np
from plasma_plots.hero import HeroFigure
from scipy.interpolate import RegularGridInterpolator

from patchwork import AMRHierarchy, solve_poisson

OUTPUT = Path(__file__).parent / "src" / "assets" / "hero.svg"
XLIM, YLIM = (0.0, 8.0), (0.0, 3.0)
SHARP, BROAD = (5.9, 1.5), (3.4, 1.4)

mesh = AMRHierarchy(base_shape=(64, 24), bounds=(XLIM, YLIM))
# boxes in cells of their own level (spacing 1/8, 1/16, 1/32, 1/64 of a unit)
mesh.add_patch(level=1, box=((76, 112), (8, 40)))
mesh.add_patch(level=2, box=((172, 204), (32, 64)))
mesh.add_patch(level=3, box=((360, 396), (80, 112)))
mesh.allocate_field("rhs", halo=1)
for patch in mesh:
    x, y = patch.local_cell_centers(mesh)
    sharp = np.exp(-((x - SHARP[0]) ** 2 + (y - SHARP[1]) ** 2) / (2 * 0.05**2))
    broad = np.exp(-((x - BROAD[0]) ** 2 + (y - BROAD[1]) ** 2) / (2 * 0.5**2))
    patch.field("rhs").local[...] = 60.0 * sharp - 1.2 * broad
solve_poisson(mesh, "rhs", potential="phi", boundary_value=0.0)

# sample the composite potential on a uniform grid, each point from the finest patch holding it
gx = np.linspace(*XLIM, 641)
gy = np.linspace(*YLIM, 241)
X, Y = np.meshgrid(gx, gy, indexing="ij")
phi = np.full(X.shape, np.nan)
for patch in sorted(mesh, key=lambda p: p.level):
    (x0, x1), (y0, y1) = patch.physical_bounds(mesh)
    hx, hy = mesh.spacing(patch.level)
    cx = np.arange(x0 + hx / 2, x1, hx)
    cy = np.arange(y0 + hy / 2, y1, hy)
    inside = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    values = mesh.gather(patch, "phi")
    interp = RegularGridInterpolator(
        (cx, cy), values, bounds_error=False, fill_value=None
    )
    phi[inside] = interp(np.column_stack([X[inside], Y[inside]]))

fig = HeroFigure(xlim=XLIM, ylim=YLIM)
# the patch outlines, appearing one level at a time over a 12 s loop
CYCLE = 12.0
for patch in mesh:
    if patch.level == 0:
        continue
    (x0, x1), (y0, y1) = patch.physical_bounds(mesh)
    box = fig.rect(x0, y0, x1, y1, tone="low", opacity=0.35, outline="high")
    box.attrs["stroke-width"] = "1.5"
    box.animate(
        "opacity",
        [0, 0] + [0] * (patch.level - 1) + [1] * (9 - patch.level) + [0],
        CYCLE,
    )

levels = np.concatenate(
    [
        -np.geomspace(0.02, 0.25, 4)[::-1],
        np.geomspace(0.01, 0.5, 6),
        [phi.max() * 0.8, phi.max() * 0.6, phi.max() * 0.45],
    ]
)
generator = contourpy.contour_generator(gx, gy, phi.T)
lines = []
for level in levels:
    tone = "accent" if level > 0 else "muted"
    for seg in generator.lines(level):
        seg = np.vstack([seg[:: max(1, len(seg) // 160)], seg[-1:]])
        if len(seg) < 8:
            continue
        lines.append(fig.line(seg[:, 0], seg[:, 1], tone=tone, width=1.3, opacity=0.7))

for i, line in enumerate(lines[::3]):
    fig.tracer(line, duration=8 + (i % 5) * 2, offset=1.3 * i, r=3, tone="high")

print(f"Saved {fig.save(OUTPUT)}")
