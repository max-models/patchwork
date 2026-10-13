# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- Node-centered fields: `allocate_field(name, centering="node")`, `Patch.node_shape`,
  `Patch.node_coordinates`.
- `AMRHierarchy(periodic=...)`: axes on which the last cell neighbours the first.
- `NodalPoisson`, a node-centered composite Poisson solver: the five-point Laplacian on
  uniform parts of the mesh, hanging nodes interpolated at coarse-fine interfaces, `Dirichlet`
  and `Neumann` faces, periodic axes, embedded bodies at fixed potentials
  (`EmbeddedDirichlet`, Shortley-Weller stencils), one factorization for many solves, and
  `set_conditions` to change prescribed values without factorizing again.
  `solve_nodal_poisson` is the one-call form.
- `CompositeNodes`: the nodes and leaf cells of a hierarchy, the maps between them and the
  patches, interpolation from and to a level's uniform lattice (`from_level`, `to_level`,
  `evaluation`) and `sample_patches` for patch-wise fields such as a gradient.
- `examples/nodal_electrode.py`.
- Tutorials (executed notebooks in `tutorials/`, published in the docs): a first hierarchy, the
  cell-centered and node-centered Poisson solves, non-rectangular domains, and halos. They replace
  the placeholder tutorial and the hand-written "Node-centered Poisson" page.
- Domains that are not rectangles: `AMRHierarchy(base_boxes=...)` makes the base level a union
  of boxes (an L, a T, a notch). The edges inside the bounding rectangle are the boundary
  `"inner"`, which `NodalPoisson` takes a `Dirichlet` or `Neumann` condition for;
  `solve_poisson` applies its boundary value there too. `CompositeNodes.face_edges` lists the
  cell edges of every part of the boundary. `examples/l_shaped_domain.py`,
  `examples/t_shaped_domain.py`.

- A communicator per patch: `add_patch(..., ranks=...)` and `AMRHierarchy(base_ranks=...)` put
  a patch on one rank or on several, over which mpiarray splits its fields;
  `placement="balanced"` puts each new patch on the rank with the fewest cells. `Patch.ranks`,
  `Patch.comm`, `Patch.is_local`, `AMRHierarchy.local_patches`, `cells_on_rank` and the
  collective `gather(patch, name, root=None)`. Both Poisson solvers and `plot_patches` work
  wherever the patches are. `Patch.local_node_coordinates` for filling node-centered fields.
- Halo exchange between patches: `AMRHierarchy.exchange_halos(name)` fills the halo of every
  patch from the patches of the same level (across periodic faces too), else by bilinear
  interpolation of the level below, and between the ranks of each patch; halo points outside
  the domain are left for the boundary conditions. `AMRHierarchy.restrict(name)` gives coarse
  points under finer patches the fine values (cell means, coinciding nodes), finest level first.
  Both are schedules (`patchwork.halos.HaloExchange`, `Restriction`) built once per field and
  sent with `mpiarray.CopyPlan`, which needs mpiarray's `devel` branch from 2026-10-13 on.
- Regridding: `AMRHierarchy.regrid(tags, efficiency=, buffer=, nesting=, max_patch_size=,
  placement=)` returns a hierarchy with patches over the tagged cells of each level, properly
  nested, with the fields moved over (copied from old patches of the same level, else
  interpolated from the new level below). `gradient_tags` (a field's change across a cell),
  `function_tags` (a function of the position) and `cluster` (Berger-Rigoutsos clustering)
  in `patchwork.regrid`, also exported at the top level. Tutorial "Regridding".
- Conservative fields: `allocate_field(..., conservative=True)` (cell-centered) keeps its integral
  through `regrid`, which restricts it first and fills new cells from the level below with
  limited (MC) slopes: the fine cells of a coarse cell keep its mean, and no new extrema appear.
  `AMRHierarchy.integral(name)` integrates a cell- or node-centered field over the finest cells.
- `examples/regrid_moving_charge.py`: regridding, rebuilding `NodalPoisson` and solving at every
  step of a moving charge, timed against a uniform mesh as fine as the finest patches.
- `plot_patches(grid_linewidth=, grid_alpha=)`: stronger cell lines for small figures, such as
  the frames of an animation. The tutorial publisher turns GIF outputs into animated images.
- `Patch.local_cell_centers(hierarchy, name)` takes the field whose cells are wanted; without
  one, the first cell-centered field (it used the first field of any centering).

### Changed

- Interpolation from the level below (halos, regridding) extrapolates linearly next to a wall of
  a non-periodic axis instead of dropping the coarse point beyond it, so it is exact for linear
  fields there too.
- A patch must lie within the patches of the level below, but no longer within one of them: it
  may lie over several that touch.
- A `Neumann` gradient is evaluated at the middle of every boundary cell edge instead of at the
  nodes, so that two faces meeting at a corner each keep their own value.
- `plot_patches` colours all patches on one scale and shades node-centered fields between
  their nodes.
- `mpiarray` is taken from its `devel` branch instead of a pinned commit, so that patchwork
  installs next to packages that need a newer `mpiarray`.

## [0.1.0] - 2026-10-05

### Added

- Package layout with a console entry point and pytest tests.
- Astro + Starlight documentation site with executed tutorials and a generated API reference.
- GitHub Actions for tests, ruff, pyright, tutorials, documentation and PyPI publishing.
