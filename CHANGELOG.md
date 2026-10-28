# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Solver backends (`patchwork.backends`): `NodalPoisson(backend=...)` takes a `Backend` or
  its name, and every backend solves the same assembled operator (`LinearSystem`).
  `"lu"` (`LUBackend`, the default) factorizes it with SuperLU, or with LAPACK's tridiagonal
  LU (LDL^T when symmetric) on a line; `"fft"` (`FFTBackend`) solves a uniform mesh without
  bodies with fast transforms (FFT on periodic axes, DST-I between Dirichlet faces, DCT-I
  between Neumann faces), also the singular operators (every axis periodic or Neumann: the
  solution of zero mean), and `spectral=True` inverts the exact Laplacian on a periodic box;
  `"petsc"` (`PetscBackend(rtol=, options=)`) is the distributed solve; `"auto"` takes the
  LU on a line, the FFT where it applies and the LU elsewhere. A backend refuses what it
  cannot solve with a reason (`Backend.unsupported`).

- `AMRHierarchy.interpolate(x, y, name)`: a node-centered field at markers, each read in the
  finest patch that holds it (the transpose of `deposit`), for fields computed patch by patch
  such as an electric field that jumps at a coarse-fine edge. The markers are on their
  owners (`migrate`) and the halos filled (`exchange_halos`).

- `FluxRegister`: conservative explicit steps of a cell-centered field on the patches
  (Berger–Colella refluxing). `update` advances every patch by the divergence of the face
  fluxes a model computes from its halos and records the fluxes through coarse–fine faces;
  `reflux` gives the coarse cells next to finer levels the fine fluxes instead of their own
  and restricts onto the covered cells. The `integral` is conserved to round-off, in the
  Cartesian plane, in r–z and on a line, with one time step for all levels or with fine
  levels subcycling (their fluxes summed over their steps), and with multi-stage methods
  (`scale`).
- `AMRHierarchy.face_areas`: the face areas that go with `cell_volumes` (rings in r–z, 1 on
  a line).
- `exchange_halos(name, level=, below=)`: the halos of one level only, interpolated from
  another field of the level below (the coarse values at the time of a subcycled step);
  `restrict(name, level=)`: onto one level only.
- `NodalPoisson(backend="petsc")`: a distributed solve. Every rank assembles only the rows
  of the unknowns it owns (`CompositeNodes.node_ranks`, from the ranks of the patches), from
  the leaf cells around them, and PETSc solves them together: conjugate gradients (GMRES
  with embedded bodies) and BoomerAMG, or GAMG without hypre, to `rtol`, with GMRES and
  block Jacobi as a fallback. The potential agrees with the SuperLU solve (still the
  default) to the solver tolerance on any number of ranks, on lines and planes.
  `PetscSystem`, the `petsc` extra (petsc4py), `CompositeNodes.cell_ranks` and
  `examples/distributed_scaling.py`.
- 1D hierarchies: `AMRHierarchy(base_shape=(nx,))` is a line (a uniform 1D grid is a
  hierarchy of one level), with `bounds=((x0, x1),)` and `periodic=(flag,)`; `ndim` gives
  the number of dimensions. The 2D code paths are now written for either: patches and
  fields, `exchange_halos` and `restrict` (on MPI ranks too), `integral`, tagging,
  `cluster` (on a 1D array of tags), `regrid` with conservative fields, `forbid` and
  `min_patch_size`, `locate`, `owner`, `migrate`, `deposit` and `accumulate_halos`
  (`locate(x)`, `owner(x, name)`, `migrate(x, *arrays, name=)`, `deposit(x, charge, name)`
  in 1D), `CompositeNodes`, `NodalPoisson` (the three-point stencil; no node hangs),
  `CompositeCells`, `CellOperator` with both coarse–fine treatments, bodies, floating
  bodies and `gradient`, and `plot_patches` (a line plot). Functions of the position take
  `(x)` in 1D. Tests: `test_line_patches.py` (also under `mpirun`) and
  `test_line_solvers.py`, the 1D counterparts of the 2D ones: second-order manufactured
  solutions across coarse–fine points, charge conservation of the deposit, conservation
  through restriction and regridding, and picnic's `DirichletPoisson1D` (the same
  potential to round-off) and `SheetField1D` (deposited sheets give its exact potential at
  the nodes, on refined lines too).

### Changed

- `NodalPoisson`: `backend="superlu"` is now `"lu"`; `rtol=` and `petsc_options=` moved to
  `PetscBackend(rtol=, options=)`; `solver.system` is the `LinearSystem` and `solver.solver`
  what the backend set up (a factorization, a `PetscSystem`, an `FFTSolver`).
- `AMRHierarchy.deposit` runs mpifield's compiled linear weighting
  (`mpifield.LINEAR.scatter_block`) on each patch block instead of a NumPy `bincount`; the
  result is unchanged to round-off. New dependency: `mpifield`.
- The kernels are built with cunumpy's `KernelPackage` (`patchwork.kernels.KERNEL_PACKAGE`):
  `kernels/_build.py` is gone, the build cache is still `PATCHWORK_KERNEL_CACHE`.

- `locate` (and `owner`, `migrate`, `deposit`) runs a compiled kernel (Pyccel, the new
  `pyccel` extra, also in `test`; its NumPy version runs without it): one pass over the
  points, one division per axis and a lookup per level, instead of NumPy passes per level.
  A point's cell is now its cell on the finest lattice divided down to its patch's level,
  the leaf cell `CompositeNodes` puts it in. 25,000 markers on five patches: 4.6 to 0.9 ms;
  200,000 on 338 patches: 45 to 16 ms (one thread, a loaded machine).
- `CompositeNodes.deposit` in 2D runs a compiled kernel (one pass, no matrix): 2.9 to
  1.0 ms and 38 to 20 ms in the same two cases.
- `CompositeNodes` keeps `patch_nodes` (read-only now) and which patch `from_patches`
  reads every node from, found at the first call on a mesh; on one rank `from_patches`
  reads the patch arrays in place: 1.1 to 0.18 ms and 31 to 3 ms (the first call on a
  mesh costs about one and a half calls of before).
- `accumulate_halos` on one rank skips the transport and adds the copies along paths
  found at the first call: 0.41 to 0.12 ms and 49 to 5.5 ms.
- cunumpy is a direct dependency (the kernel dispatch and the Pyccel build cache).
- `AMRHierarchy`'s `bounds` and `periodic` default to `None`: the unit interval or square,
  and no periodic axis. `IndexBox`, `spacing`, `level_shape`, `lower_bound` and the patch
  coordinates have one entry per axis. `solve_poisson` stays 2D (use `CellOperator`), as
  do the r–z geometry and `CellOperator.curl`.
- `PetscSystem` gets its PETSc communicator from `maybempi.petsc_comm` (maybempi devel)
  instead of converting it by hand.

### Fixed

- `integral` counted a replicated field (`split=None`) once per rank of a patch on several
  ranks.

## [0.2.0] - 2026-10-22

The first release on PyPI, as `patchwork-amr`.

### Changed

- Published on PyPI as `patchwork-amr` (`patchwork` there is an unrelated project); the import
  stays `import patchwork`. The release workflow checks and publishes `patchwork-amr`.

### Added

- `EmbeddedFloating`: a floating body of `CellOperator`, one unknown (its uniform value) whose
  row is the net flux into it, set to its `source` (zero for an isolated conductor); the
  operator stays symmetric. `floating_coupling` and `floating_values`.
- `cell_body`: a body of the leaf cells a mask selects, its surface on their faces (e.g. the
  cells outside a plasma held at zero); `CompositeCells.locate` finds the leaf cell of points.
- `CellOperator.gradient`, `.curl` and `.face_gradients`: derivatives of a solution from the
  operator's own faces (coarse–fine interpolation, body surfaces, the domain's conditions);
  exact for linear fields and for the uniform axial field `A_phi = B r / 2`, second order in
  the maximum norm across coarse–fine faces. The curl is the magnetic field of a vector
  potential, `(d_y A, -d_x A)` or `(-d_z A_phi, (1/r) d_r(r A_phi))`.
- `CellOperator(factorize=False)`: assembled only, factorized on the first `solve`.

### Changed

- A zero diffusion coefficient is allowed (negative ones are not); unknowns with neither
  diffusion nor reaction are reported as a singular operator.
- Coarse values along a coarse–fine face interpolate to their mirror image beyond a face of
  the domain with no gradient (`Neumann(0.0)`) and beyond the axis (minus the value for an
  azimuthal component), instead of falling back to a one-sided line: a field that does not
  vary along such a wall is now exact next to a patch (it was off by about 1e-3 with the
  symmetric variant).

- `CellOperator`, a cell-centered composite solve of `-div(a grad u) + c u = f`: variable
  diffusion (harmonic face mean, or arithmetic), real or complex reaction, `Dirichlet` and
  `Neumann` faces, periodic axes, embedded bodies (`EmbeddedDirichlet`, the surface located
  along the line of the centres), one factorization for many solves, `set_reaction` (new `c`
  without assembling) and `set_conditions`. Coarse-fine faces interpolate the coarse value
  quadratically along the face; `coarse_fine="symmetric"` (default) keeps the operator
  symmetric (complex symmetric for a complex `c`), second order in L2 and first order next to
  coarse-fine edges; `"conservative"` takes one face flux from a quadratic ghost value,
  second order everywhere and conservative face by face, not symmetric.
- `CompositeCells`: the leaf cells of a hierarchy, their faces (from a finest-lattice owner
  table) and the maps to and from the patches.
- `AMRHierarchy(geometry="cylindrical")`: the axisymmetric r–z plane, `cell_volumes`,
  `cylindrical`, `on_axis`; `integral`, `restrict` and the conservative interpolation of
  `regrid` weight cells by their ring volumes. `CellOperator(component="azimuthal")`: the
  curl-curl of an azimuthal component, `A_phi = 0` on the axis through the half cell next to
  it, a uniform axial field a null field across coarse-fine faces too.
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
- `AMRHierarchy.locate(x, y)`: the finest patch that holds each point, as an index into
  `patches` (-1 outside the domain, in the notch of an L; periodic axes wrap), from one lookup
  table per level.
- Markers on the ranks of their patch: `AMRHierarchy.owner(x, y, name)`, the rank whose block of
  field `name` holds the lower corner of each marker's cell (-1 outside the domain), and
  `AMRHierarchy.migrate(x, y, *arrays, name=..., outside="error")`, which sends the markers
  there with `mpiarray.migrate` (call it again on the hierarchy `regrid` returns to follow a
  patch that changed ranks).
- Deposit patch by patch: `AMRHierarchy.deposit(x, y, charge, name)` (bilinear weights into the
  block and its halo) and `AMRHierarchy.accumulate_halos(name)` (the scatter-add: the halos of
  the blocks, then every copy of a node onto its owner copy, the finest patch's, which
  `CompositeNodes.from_patches` reads). The result equals `CompositeNodes.deposit` to
  round-off; patches on several ranks, periodic seams and coarse-fine edges included.
  `patchwork.accumulate.Accumulation` is the schedule.
- `regrid(..., fill_halos=True)` fills the halos of every field afterwards;
  `regrid(..., forbid=...)` (a mask per level or a function of the cell centres) keeps patches
  off cells; `regrid(..., min_patch_size=...)` and `cluster(..., min_size=...)` widen boxes that
  are narrower, into free allowed cells.
- Placement: `AMRHierarchy(placement=f)` with `f(level, box)` returning the ranks, and
  `max_cells_per_rank` to split a large `"balanced"` patch over several of the least loaded
  ranks; `regrid` keeps both.
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
- Profiling regions, off by default: halo exchanges, restriction, deposit, migration, regridding,
  tagging and both Poisson solvers record `patchwork.<operation>` scope-profiler regions tagged
  `patchwork`, an opt-in tag that an application enables with `enable_tags=["patchwork"]`,
  `SCOPE_PROFILER_ENABLE_TAGS=patchwork` or `scope-profiler run --enable-tag patchwork`.
  `scope-profiler>=0.7.5` is a new dependency, which uv takes from its `devel` branch until
  0.7.5 is released.

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
