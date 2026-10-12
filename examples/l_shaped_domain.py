r"""An L-shaped domain: the base level as two boxes, patches at the inner corner.

The domain is the square $[-1, 1]^2$ without its lower-right quadrant. Its
harmonic potential $r^{2/3} \sin(2\theta/3)$ vanishes on the two inner edges
and has an unbounded gradient at the corner, so a uniform mesh converges slowly.
Nested patches at the corner bring the error down at a fraction of the unknowns.
The inner edges are grounded here; ``faces={"inner": Neumann(0.0)}`` would
let no flux through them instead.
"""

from pathlib import Path

import numpy as np

from patchwork import AMRHierarchy, Dirichlet, NodalPoisson, plot_patches

FACES = ("x_lower", "x_upper", "y_lower", "y_upper")


def exact(x, y):
    """The corner solution of the L: zero on both inner edges."""
    theta = np.arctan2(y, x) % (2.0 * np.pi)
    return np.hypot(x, y) ** (2.0 / 3.0) * np.sin(2.0 * theta / 3.0)


def ell(base: int, levels: int) -> AMRHierarchy:
    """The L on ``base`` cells per side, with ``levels`` nested patches at the corner."""
    half = base // 2
    mesh = AMRHierarchy(
        base_shape=(base, base),
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
        # the left half, and the upper half of the right half
        base_boxes=[((0, half), (0, base)), ((half, base), (half, base))],
    )
    for level in range(1, levels + 1):
        centre = base * 2**level // 2
        width = base // 4
        # a patch is a rectangle inside the domain: two of them around the corner
        mesh.add_patch(
            level=level,
            box=((centre - width, centre), (centre - width, centre + width)),
        )
        mesh.add_patch(
            level=level, box=((centre, centre + width), (centre, centre + width))
        )
    return mesh


def solve(base: int, levels: int) -> tuple[AMRHierarchy, int, float]:
    """The mesh with its potential, the unknowns and the error on the base nodes."""
    mesh = ell(base, levels)
    solver = NodalPoisson(mesh, faces={face: Dirichlet(exact) for face in FACES})
    nodes = solver.nodes
    x, y = nodes.coordinates.T
    potential = solver.solve(np.zeros(nodes.num_nodes), potential="potential")
    spacing = 2.0 / 32  # compare all meshes on the nodes of the 32 x 32 mesh
    on_coarse = np.all(np.abs(nodes.coordinates / spacing % 1.0) < 1e-9, axis=1)
    error = np.abs(potential - exact(x, y))[on_coarse].max()
    return mesh, solver.free.size, float(error)


for label, base, levels in (
    ("32 x 32 cells", 32, 0),
    ("64 x 64 cells", 64, 0),
    ("128 x 128 cells", 128, 0),
    ("32 x 32 cells + 2 levels", 32, 2),
    ("32 x 32 cells + 4 levels", 32, 4),
):
    mesh, unknowns, error = solve(base, levels)
    print(f"{label:26s} {unknowns:6d} unknowns, error {error:.1e}")

result = plot_patches(mesh, field="potential", show_grid=True)
if result is not None:
    figure, axes = result
    axes.set_title("Potential on an L-shaped domain, refined at its corner")
    output = Path(__file__).with_name("l_shaped_domain.png")
    figure.savefig(output, dpi=180)
    print(f"Saved {output}")
