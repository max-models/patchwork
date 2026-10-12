"""Build and plot a two-level 2D mesh, with or without MPI."""

from pathlib import Path

import numpy as np

from patchwork import AMRHierarchy, plot_patches

mesh = AMRHierarchy(base_shape=(32, 24), bounds=((0.0, 2.0), (-0.5, 1.0)))
mesh.add_patch(level=1, box=((16, 48), (12, 36)))
mesh.allocate_field("density", halo=1)

for patch in mesh:
    field = patch.field("density")
    (i0, _), (j0, _) = patch.box
    (li0, li1), (lj0, lj1) = field.layout.index_bounds
    dx, dy = mesh.spacing(patch.level)
    x0, y0 = mesh.lower_bound
    x = x0 + (i0 + np.arange(li0, li1) + 0.5) * dx
    y = y0 + (j0 + np.arange(lj0, lj1) + 0.5) * dy
    xx, yy = np.meshgrid(x, y, indexing="ij")
    field.local[...] = np.exp(-60.0 * ((xx - 1.0) ** 2 + (yy - 0.2) ** 2))

result = plot_patches(mesh, field="density")
if result is not None:
    figure, _ = result
    output = Path(__file__).with_name("two_level_mesh.png")
    figure.savefig(output, dpi=180)
    print(f"Saved {output}")
