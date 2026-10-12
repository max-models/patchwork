---
title: Quickstart
description: Create, refine, and plot a 2D patch hierarchy.
---

Patchwork uses `mpiarray` for patch fields. It runs in serial without an MPI
launcher (or MPI installed), and supports distributed patch arrays when run
under `mpiexec` after installing the `mpi` extra.

## Create a hierarchy

```python
from patchwork import AMRHierarchy, solve_poisson

mesh = AMRHierarchy(base_shape=(32, 24), bounds=((0, 2), (-0.5, 1)))
fine = mesh.add_patch(level=1, box=((16, 48), (12, 36)))
mesh.allocate_field("density", halo=1)
fine.field("density").local[:] = 1.0
solve_poisson(mesh, "density", potential="phi")
```

Patch bounds are half-open cell-index intervals at that patch's level. Fine
patch bounds align to their parent grid and must be contained in a parent patch.
The root level is the full domain and is created automatically.

## Plot the patch layout and cells

Install the plotting extra, then:

```python
from patchwork import plot_patches

fig, ax = plot_patches(mesh, field="density")
fig.savefig("patches.png", dpi=180)
```

The plot overlays the active level grids and colors each patch by its field. `solve_poisson` assembles a conservative finite-volume operator across coarse-fine faces and applies a constant Dirichlet value on the domain boundary. See `examples/composite_poisson.py` for a manufactured solution.

![Example density field on a coarse grid with a fine patch](/patchwork/figures/two_level_mesh.png)

The composite Poisson example solves a smooth source across the coarse-fine interface:

![Composite Poisson solution on a refined patch](/patchwork/figures/composite_poisson.png)
For nested levels and a complete script, run `python examples/nested_patches.py`.

For node-centered fields, faces with different conditions, periodic axes and
embedded bodies, see [Node-centered Poisson](/patchwork/getting-started/node-centered/).

## MPI and serial use

No MPI setup is required for a normal Python run. To distribute a patch array,
launch the same script collectively, for example `mpiexec -n 2 python
examples/two_level_mesh.py`. All ranks must construct the same hierarchy and
allocate fields in the same order. The Poisson solves gather the patch
fields and solve on every rank; coarse-fine halo exchange for user kernels is
not part of this release.
