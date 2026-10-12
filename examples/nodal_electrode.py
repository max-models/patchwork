"""Node-centered solve around a small electrode, with nested patches where it matters.

A disc of half a base cell's radius is held at 1 V in a box whose faces carry the
exact logarithmic potential. The base mesh cannot resolve the disc; two nested
patches around it do, at a fraction of the unknowns of a uniform fine mesh.
"""

from pathlib import Path

import numpy as np

from patchwork import (
    AMRHierarchy,
    Dirichlet,
    EmbeddedDirichlet,
    NodalPoisson,
    plot_patches,
)

RADIUS = 0.03
FACES = ("x_lower", "x_upper", "y_lower", "y_upper")


def exact(x, y):
    """The potential of a disc at 1 V: logarithmic outside it."""
    r = np.maximum(np.hypot(x - 0.5, y - 0.5), 1e-12)
    return np.where(r < RADIUS, 1.0, 1.0 + 0.5 * np.log(r / RADIUS))


disc = EmbeddedDirichlet(
    inside=lambda points: np.hypot(points[0] - 0.5, points[1] - 0.5) < RADIUS, value=1.0
)

for label, patches in (
    ("32 x 32 cells", []),
    ("with two nested patches", [(1, ((24, 40), (24, 40))), (2, ((56, 72), (56, 72)))]),
):
    mesh = AMRHierarchy(base_shape=(32, 32))
    for level, box in patches:
        mesh.add_patch(level=level, box=box)
    solver = NodalPoisson(
        mesh, faces={face: Dirichlet(exact) for face in FACES}, embedded=[disc]
    )
    nodes = solver.nodes
    x, y = nodes.coordinates.T
    # no charge: the source is zero on every node
    potential = solver.solve(np.zeros(nodes.num_nodes), potential="potential")
    on_base_nodes = np.all(nodes.lattice % nodes.cell_span.max() == 0, axis=1)
    error = np.abs(potential - exact(x, y))[on_base_nodes].max()
    print(f"{label:24s} {solver.free.size:5d} unknowns, error {error:.1e} V")

result = plot_patches(mesh, field="potential", show_grid=True)
if result is not None:
    figure, axes = result
    axes.set(xlim=(0.3, 0.7), ylim=(0.3, 0.7), title="Potential around the electrode")
    output = Path(__file__).with_name("nodal_electrode.png")
    figure.savefig(output, dpi=180)
    print(f"Saved {output}")
