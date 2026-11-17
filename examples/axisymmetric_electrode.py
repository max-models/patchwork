"""Axisymmetric potential around a rod, on the r–z plane with the cylindrical metric.

A rod of radius 0.23 on the axis is held at 1 V inside a grounded tube of radius 1; the
ends are Neumann faces, so the potential is the coaxial line's,
``ln(1 / r) / ln(1 / 0.23)``. ``NodalPoisson(metric="cylindrical")`` weights the control
volumes and the couplings by the rings' radii, so the axis needs no condition; the error
falls at second order overall, unevenly as the rod's surface moves through the cells.
"""

from pathlib import Path

import numpy as np

from patchwork import (
    AMRHierarchy,
    Dirichlet,
    EmbeddedDirichlet,
    Neumann,
    NodalPoisson,
    plot_patches,
)

ROD, TUBE = 0.23, 1.0

rod = EmbeddedDirichlet(inside=lambda points: points[0] < ROD, value=1.0)
faces = {"x_upper": Dirichlet(0.0), "y_lower": Neumann(), "y_upper": Neumann()}

for n in (20, 40, 80, 160):
    # the radius is the first axis of an r–z hierarchy
    mesh = AMRHierarchy(
        base_shape=(n, n // 2), bounds=((0.0, TUBE), (0.0, 0.5)), geometry="cylindrical"
    )
    solver = NodalPoisson(mesh, faces=faces, embedded=[rod], metric="cylindrical")
    r = solver.nodes.coordinates[:, 0]
    potential = solver.solve(np.zeros(r.size), potential="potential")
    exact = np.log(TUBE / np.maximum(r, ROD)) / np.log(TUBE / ROD)
    print(f"{n:4d} radial cells: error {np.abs(potential - exact).max():.1e} V")

result = plot_patches(mesh, field="potential")
if result is not None:
    figure, axes = result
    axes.set(xlabel="r", ylabel="z", title="Potential of a rod in a tube")
    output = Path(__file__).with_name("axisymmetric_electrode.png")
    figure.savefig(output, dpi=180)
    print(f"Saved {output}")
