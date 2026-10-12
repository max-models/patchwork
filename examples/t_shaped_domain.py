r"""A T-shaped domain: a bar with a stem, refined at its two inner corners.

The bar is $3 \times 1$ and the stem below its middle $1 \times 1$. The foot
of the stem is held at 1 V and the two ends of the bar are grounded; no flux
leaves through the other walls, as for the current in a T-shaped conductor.
The gradient of the potential is unbounded at the two corners where the stem
meets the bar, so nested patches there do what a finer mesh everywhere does,
with far fewer unknowns. The errors are measured against a uniform mesh with
eight times the resolution, which is itself only good to about 2e-4 V.
"""

from pathlib import Path

import numpy as np

from patchwork import AMRHierarchy, Dirichlet, Neumann, NodalPoisson, plot_patches

#: Cells of the coarsest mesh: the bar is 48 x 16 of them, the stem 16 x 16.
BASE = (48, 32)
SPACING = 1.0 / 16


def tee(scale: int = 1, levels: int = 0) -> AMRHierarchy:
    """The T on ``scale`` times the base cells, with nested patches at its corners."""
    nx, ny = BASE[0] * scale, BASE[1] * scale
    third, half = nx // 3, ny // 2
    mesh = AMRHierarchy(
        base_shape=(nx, ny),
        bounds=((-1.5, 1.5), (-1.0, 1.0)),
        # the bar over the whole width, and the stem under its middle third
        base_boxes=[((0, nx), (half, ny)), ((third, 2 * third), (0, half))],
    )
    width = 8 * scale
    for level in range(1, levels + 1):
        fine = 2**level
        for corner, inward in ((third, 1), (2 * third, -1)):
            x, y = corner * fine, half * fine
            # in the bar, on both sides of the corner
            mesh.add_patch(level=level, box=((x - width, x + width), (y, y + width)))
            # in the stem, on the corner's side of it
            stem = (x, x + width) if inward == 1 else (x - width, x)
            mesh.add_patch(level=level, box=(stem, (y - width, y)))
    return mesh


def solve(scale: int = 1, levels: int = 0):
    """The mesh with its potential, the unknowns and the potential on the base nodes."""
    mesh = tee(scale, levels)
    solver = NodalPoisson(
        mesh,
        faces={
            "y_lower": Dirichlet(1.0),  # the foot of the stem
            "x_lower": Dirichlet(0.0),  # the ends of the bar
            "x_upper": Dirichlet(0.0),
            "y_upper": Neumann(0.0),  # the top of the bar
            "inner": Neumann(0.0),  # the walls of the stem, the bar's underside
        },
    )
    nodes = solver.nodes
    potential = solver.solve(np.zeros(nodes.num_nodes), potential="potential")
    index = nodes.coordinates / SPACING
    on_base = np.all(np.abs(index - np.round(index)) < 1e-9, axis=1)
    values = dict(
        zip(
            map(tuple, np.round(index[on_base]).astype(int).tolist()),
            potential[on_base],
            strict=True,
        )
    )
    return mesh, solver.free.size, values


_, _, reference = solve(scale=8)
for label, scale, levels in (
    ("48 x 32 cells", 1, 0),
    ("96 x 64 cells", 2, 0),
    ("192 x 128 cells", 4, 0),
    ("48 x 32 cells + 2 levels", 1, 2),
    ("48 x 32 cells + 4 levels", 1, 4),
):
    mesh, unknowns, values = solve(scale, levels)
    error = max(abs(value - reference[node]) for node, value in values.items())
    print(f"{label:26s} {unknowns:6d} unknowns, error {error:.1e} V")

result = plot_patches(mesh, field="potential", show_grid=True)
if result is not None:
    figure, axes = result
    axes.set_title("Potential in a T-shaped conductor, refined at its inner corners")
    output = Path(__file__).with_name("t_shaped_domain.png")
    figure.savefig(output, dpi=180)
    print(f"Saved {output}")
