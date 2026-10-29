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

The [examples](/patchwork/examples/) go further, step by step: fields on patches, the
cell-centered and node-centered Poisson solves (faces with different conditions, periodic
axes, electrodes), domains that are not rectangles, halos, and regridding.

## MPI and serial use

No MPI setup is required for a normal Python run. Under MPI, launch the same
script on every rank, for example `mpiexec -n 2 python
examples/two_level_mesh.py`. All ranks must construct the same hierarchy and
allocate fields in the same order.

Each patch lives on some of the ranks and has a communicator of its own for
them, over which mpiarray splits its fields:

```python
mesh = AMRHierarchy(base_shape=(64, 64))  # the base: on every rank
small = mesh.add_patch(level=1, box=((32, 64), (32, 64)), ranks=3)  # one rank
large = mesh.add_patch(level=1, box=((64, 128), (0, 64)), ranks=(0, 1))  # two ranks
mesh.allocate_field("density")

for patch in mesh.local_patches:  # the patches this rank holds a part of
    patch.field("density").local[...] = 1.0
whole = mesh.gather(small, "density")  # the whole patch, on every rank
```

- `ranks` is one rank or several. Without it the hierarchy's `placement`
  decides: `"all"` (the default) puts a patch on every rank, `"balanced"` on
  the one rank that holds the fewest cells so far. `base_ranks` does the same
  for the base boxes.
- `patch.ranks` is the same on every rank; `patch.is_local` says whether this
  rank holds a part, and `patch.comm` is then the patch's communicator.
  Patches on the same group of ranks share one communicator.
- `mesh.gather(patch, name, root=None)` is collective over the hierarchy's
  communicator and returns the whole field on every rank, or on `root` only.
- A field split over several ranks needs at least one cell per rank along its
  first axis.

The Poisson solves and `plot_patches` gather the patch fields and work
wherever the patches are; the solve itself runs on every rank.

## Halos of patches

A field allocated with `halo=h` has `h` cells (or nodes) around every patch.
`exchange_halos` fills them from the rest of the hierarchy, so that a stencil
can be applied to each patch on its own:

```python
mesh.allocate_field("u", halo=2, centering="node")
# ... the patches' own points of "u" are set ...
mesh.restrict("u")  # coarse points under finer patches take the fine values
mesh.exchange_halos("u")  # every patch's halo, on every rank that holds it
for patch in mesh.local_patches:
    u = patch.field("u").local_with_halos  # own points and halo
```

- A halo point inside another patch of the same level gets its value, across a
  periodic face too.
- Elsewhere it is the bilinear interpolation of the level below (renormalised
  where some of the four coarse points are not on any patch).
- Between the ranks of one patch, mpiarray's `update_halos` fills the halo.
- Halo points outside the domain are left as they are: set them for your
  boundary condition.

`restrict` gives coarse cells the mean of the fine cells over them, and coarse
nodes the fine node at the same place, finest level first. Both are collective;
their schedules are built at the first call for a field and kept until a patch
is added. Each call sends its data in one `Alltoallv` over the hierarchy's
communicator (`mpiarray.CopyPlan`), whichever ranks the patches are on.

## Conservative time steps (refluxing)

A model that advances a conserved cell field (a density, an energy density) with explicit
steps computes the fluxes through the faces of each patch from its halos. Where a patch
meets a coarser level, the coarse cell next to it would take the flux of its own face and
the fine cells the fluxes of theirs: they do not agree, and the total drifts.
`FluxRegister` matches them (Berger–Colella refluxing):

```python
from patchwork import FluxRegister

register = FluxRegister(mesh)  # the coarse faces on the edges of finer levels


def fluxes(patch):
    """Flux densities through this rank's block of the patch, positive along the axis."""
    u = patch.field("n").local_with_halos  # own cells and halo
    ...
    return fx, fy  # shapes (nx + 1, ny) and (nx, ny + 1) for a block of (nx, ny) cells


for step in range(steps):
    mesh.exchange_halos("n")
    register.update("n", fluxes, dt)  # every patch: n -= dt / V * sum(A F)
    register.reflux("n")  # coarse cells next to finer patches; restrict
```

- `update` advances the cells of every patch with the face areas and cell volumes of the
  hierarchy (`face_areas`, `cell_volumes`; rings in r–z) and records the coarse flux and
  the sum of the fine fluxes through every coarse face on the edge of a finer level.
- `reflux` gives each such coarse cell the fine fluxes instead of its own, then the cells
  under finer patches the mean of their fine cells. Every face of the leaf cells carries one
  flux, and `mesh.integral("n")` changes only through the faces of the domain, to
  round-off.
- Fine levels may subcycle: advance a level with `update(..., level=level)`, then the finer
  level `ratio` times with `dt / ratio`, then `reflux(..., level=level)`; the fine fluxes add
  up over their steps. `exchange_halos(name, level=level, below="n_between")` fills the
  halos of one level from a field that holds the level below at the time in between.
- A multi-stage method records each stage with its weight, e.g. the two stages of Heun's
  method with `update(..., dt, scale=dt / 2)`.
- `update` is local to each rank; `reflux` is collective (one `allreduce` of a value per
  coarse face). Build a new register after a regrid.
