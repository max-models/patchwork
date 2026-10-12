"""Solve a manufactured Poisson problem across coarse-fine interfaces."""

from pathlib import Path

import numpy as np

from patchwork import AMRHierarchy, plot_patches, solve_poisson

mesh = AMRHierarchy(base_shape=(32, 32))
mesh.add_patch(level=1, box=((16, 48), (16, 48)))
mesh.allocate_field("rhs", halo=1)

for patch in mesh:
    x, y = patch.local_cell_centers(mesh)
    patch.field("rhs").local[...] = (
        2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    )

solve_poisson(mesh, "rhs", potential="phi", boundary_value=0.0)
result = plot_patches(mesh, field="phi")
if result is not None:
    figure, _ = result
    output = Path(__file__).with_name("composite_poisson.png")
    figure.savefig(output, dpi=180)
    print(f"Saved {output}")
