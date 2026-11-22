"""The Python side of the live demo: a mesh that refines around charges, and their potential.

It runs in the browser (Pyodide); ``run`` takes the values of the controls, sets Gaussian
charges on a node-centered field, regrids the hierarchy where the density changes by more
than a threshold per cell, solves the potential in a grounded box with `NodalPoisson` and
returns the figure of the potential on the patches and their outlines, with the counts and
timings.
"""

import time

import numpy as np

from patchwork import AMRHierarchy, NodalPoisson, gradient_tags

BASE = 16
# one color per level, readable on light and dark pages
LEVEL_COLORS = ["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
# (x, y, charge) of each arrangement, in the unit square
ARRANGEMENTS = {
    "one charge": [(0.5, 0.5, 1.0)],
    "dipole": [(0.38, 0.5, 1.0), (0.62, 0.5, -1.0)],
    "quadrupole": [
        (0.35, 0.35, 1.0),
        (0.65, 0.65, 1.0),
        (0.35, 0.65, -1.0),
        (0.65, 0.35, -1.0),
    ],
    "two far apart": [(0.2, 0.25, 1.0), (0.75, 0.7, 1.0)],
    "near a wall": [(0.08, 0.5, 1.0)],
}


def density(x, y, charges, width):
    """Gaussians of unit total charge (each), of variance ``width / 2``."""
    rho = np.zeros_like(x)
    for cx, cy, q in charges:
        rho += q * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / width) / (np.pi * width)
    return rho


def deposit(mesh, charges, width):
    """Set the density on every patch."""
    for patch in mesh.local_patches:
        x, y = patch.local_node_coordinates(mesh, "rho")
        patch.field("rho").local[...] = density(x, y, charges, width)


def outline(mesh, patch):
    """The outline of a patch as a closed line."""
    (x0, x1), (y0, y1) = ((a[0], a[-1]) for a in patch.node_coordinates(mesh))
    return [x0, x1, x1, x0, x0, None], [y0, y0, y1, y1, y0, None]


def run(arrangement="dipole", width=0.002, threshold=10.0, max_levels=4, max_patch=32):
    """Refine around the charges, solve the potential and draw it.

    Args:
        arrangement: Where the charges are: one charge, dipole, quadrupole, two far apart
            or near a wall.
        width: Squared width of the Gaussians.
        threshold: Change of the density across a cell that asks for finer cells.
        max_levels: The most levels of patches above the base.
        max_patch: The longest side of a patch, in cells of its level.

    Returns:
        The figure and the counts and timings.
    """
    charges = ARRANGEMENTS[arrangement]
    width, threshold = float(width), float(threshold)
    max_levels, max_patch = int(max_levels), int(max_patch)

    begin = time.perf_counter()
    mesh = AMRHierarchy(base_shape=(BASE, BASE))
    mesh.allocate_field("rho", centering="node")
    deposit(mesh, charges, width)
    while mesh.max_level < max_levels:  # one more level per regrid
        tags = gradient_tags(mesh, "rho", threshold)
        for level in list(tags):
            if level >= max_levels:
                del tags[level]
        deeper = mesh.regrid(tags, max_patch_size=max_patch)
        deposit(deeper, charges, width)
        if deeper.max_level == mesh.max_level:
            break
        mesh = deeper
    regridded = time.perf_counter() - begin

    begin = time.perf_counter()
    solver = NodalPoisson(mesh)
    built = time.perf_counter() - begin
    begin = time.perf_counter()
    solver.solve("rho", potential="phi")
    solved = time.perf_counter() - begin

    # the potential on every patch, coarse first so that the finer ones cover it
    patches = sorted(mesh.patches, key=lambda patch: patch.level)
    phi = [np.asarray(patch.field("phi").local) for patch in patches]
    limit = max(float(np.abs(p).max()) for p in phi) or 1.0
    traces = []
    for patch, values in zip(patches, phi, strict=True):
        x, y = patch.node_coordinates(mesh)
        traces.append(
            {
                "type": "heatmap",
                "x": x,
                "y": y,
                "z": values.T,  # z[row] is one y
                "zmin": -limit,
                "zmax": limit,
                "colorscale": "RdBu",
                "showscale": patch is patches[0],
                "colorbar": {"title": {"text": "φ"}, "thickness": 12},
                "hovertemplate": f"level {patch.level}<br>x %{{x:.3f}}, y %{{y:.3f}}"
                "<br>φ %{z:.3g}<extra></extra>",
            }
        )
    for level in range(1, mesh.max_level + 1):
        xs, ys = [], []
        for patch in patches:
            if patch.level == level:
                x, y = outline(mesh, patch)
                xs += x
                ys += y
        traces.append(
            {
                "type": "scatter",
                "x": xs,
                "y": ys,
                "mode": "lines",
                "line": {
                    "color": LEVEL_COLORS[min(level, len(LEVEL_COLORS) - 1)],
                    "width": 1.5,
                },
                "hoverinfo": "skip",
                "name": f"level {level} (cells 1/{BASE * 2**level})",
            }
        )
    axis = {"range": [0.0, 1.0], "showgrid": False, "zeroline": False}
    finest = BASE * 2**mesh.max_level
    counts = [
        sum(p.level == level for p in patches) for level in range(mesh.max_level + 1)
    ]
    return {
        "figure": {
            "data": traces,
            "layout": {
                "xaxis": {**axis, "title": {"text": "x"}, "constrain": "domain"},
                "yaxis": {
                    **axis,
                    "title": {"text": "y"},
                    "scaleanchor": "x",
                    "constrain": "domain",
                },
                "legend": {"orientation": "h", "y": -0.15},
                "margin": {"t": 20},
            },
        },
        "numbers": {
            "Patches per level (base first)": " + ".join(str(c) for c in counts),
            "Unknowns (uniform mesh as fine)": f"{solver.free.size} ({(finest - 1) ** 2})",
            "Regrid [ms]": f"{1e3 * regridded:.0f}",
            "Build + solve [ms]": f"{1e3 * built:.0f} + {1e3 * solved:.0f}",
        },
    }
