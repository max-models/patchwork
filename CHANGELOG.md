# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- `CompositeNodes.deposit(..., patches=out)` gives every point its patch (what
  `AMRHierarchy.locate` returns) in the same pass as the deposit, from the new
  `CompositeNodes.cell_patch`; and `points` may have more columns than there are charges
  (the first are deposited), so a marker store with room to spare needs no copy. The 2D
  and 3D deposit kernels split the points over the OpenMP threads, each into its own row of
  a buffer summed in thread order (reproducible for a given number of threads), and
  `locate_points` runs its loop in parallel. On picnic's 2D spoke case (570 000 markers,
  6 patches, 4 threads) locate plus deposit took 25 ms, the fused deposit takes 3 ms.
- Placement along a Hilbert curve (`placement="hilbert"`, cunumpy's `hilbert_encode`) next to
  the Morton curve: `morton_ranks` is now `curve_ranks(..., curve="morton" | "hilbert")`.
  `placement="balanced"` takes `cost` too, cut by `knapsack_ranks` (the largest cost first to
  the least loaded rank). `cost` may also map a patch `(level, box)` to a measured cost, which
  `deposit(..., times=...)` and `interpolate(..., times=...)` collect per patch;
  `rebalance(cost, placement=...)` places by any of them.
- `deposit` and `interpolate` add the markers of all patches in one pass into a buffer of the
  blocks: cloud-in-cell by one `bincount`, the quadratic and cubic B-splines in one compiled
  call (the new `scatter_blocks` and `gather_blocks` kernels, with mpifield's `spline_cell`;
  10-35% faster than mpifield's kernels patch by patch); other shapes keep their kernels per
  patch. The `accumulate_halos` schedule is built from the nodes of all local patches at once
  (68 to 11 ms on the quadtree benchmark's 8-cell blocks); a marker's owner comes from a table where its patch is on one rank; the halo
  signatures test all patches of a level at once.
- `regrid(..., fields=[...])`: the fields whose values are moved (default: all); the others
  are allocated on the new hierarchy and zero.
- `CompositeNodes` finds the cells under finer patches from one coverage mask per level (it
  compared every patch with every patch of the next level); the `accumulate_halos` schedule
  takes the rank of a node's owner from a table; halo copies between two patches are kept
  per pair across regrids; `in_domain` is a bounds check where the domain fills its box.
- `deposit`, `interpolate`, `owner` and `migrate` group the markers by patch with one sort
  instead of a scan of all markers per patch, and `marker_counts` histograms only the finest
  level and sums it into the coarser ones: the quadtree benchmark's cycle 20-27% faster
  (with mpiarray's faster `CopyPlan`).
- `locate_points` (Pyccel) is about twice as fast (35.7 to 15.9 ns per point on a two-level
  hierarchy of picnic's extraction case), with the same results: the cell on each level is the
  finest cell divided by the level's span in double precision instead of Pyccel's integer floor
  division, which takes tens of cycles per division.

### Added

- Quadtree regridding: `AMRHierarchy.regrid(method="quadtree", block_size=16)` places the
  patches as the blocks of a quadtree (an octree in 3D) instead of Berger-Rigoutsos clusters:
  every level is cut into aligned square blocks of `block_size` cells, a block holding a tag is
  a patch, the blocks of a level are the children of those below, and the tree is balanced by
  the nesting margin. Blocks at the edge of the domain or of forbidden cells are cut to their
  allowed cells. `quadtree_boxes` gives the boxes on their own. More cells are refined than
  with clusters, but all patches have one size and a regrid keeps most of them. Tutorial
  "Quadtree patches".
- `placement="morton"`: `regrid` puts the patches on the ranks along a Morton curve
  (`curve_ranks`), all levels on one curve keyed in cells of the finest level, cut into runs
  of about equal cells with the base level's cells counted.
- Morton cuts weighted by cost: `regrid(..., placement="morton", cost=...)` weighs every patch
  by its cells plus the cost of its leaf cells (`box_costs`), e.g. the markers of
  `marker_counts` times what a marker costs against a cell; `rebalance(cost)` places the
  patches of a hierarchy again by them and moves the fields.
- The halo geometry of every patch (its copies and interpolation weights) is kept on the
  hierarchy and handed on by `regrid`: a patch with the same box and ranks and the same
  patches within reach of its halo reuses it instead of finding it again.
- Benchmark `benchmarks/quadtree`: clusters against quadtree blocks in a particle-in-cell
  cycle, serially and under MPI with both placements (`RESULTS.md`).
- `marker_tags(hierarchy, x, y, per_cell=..., levels=...)`: tag the cells that hold more than
  `per_cell` markers, counted over all ranks.
- Graded axes: `AMRHierarchy(axis_nodes=...)` takes, per axis, `None` (uniform) or the
  physical positions of the base nodes. The map from the logical coordinate (the one of the
  patches, levels and kernels) to the physical one is linear in every base cell, so every leaf
  cell is a physical box and finer levels split the base cells evenly. `node_positions`,
  `cell_sizes`, `to_logical` and `to_physical` (periodic axes map by whole periods) give the
  geometry; `cell_volumes` and `face_areas` take one index array per axis and return physical
  measures (also in r–z). `locate`, `owner`, `migrate`, `deposit` and `interpolate` convert the
  points; the coordinates of patches, `CompositeNodes`, `CompositeCells`, `CompositeEdges` and
  `CompositeFaces` are physical, and so are their sizes, volumes, lengths and areas.
  `NodalPoisson` (planar and r–z, with Shortley–Weller rows from the physical arms) and
  `CellOperator` (the coarse values on coarse-fine faces interpolated through the physical
  centres) converge at second order on graded meshes with patches; `CompositeEdges.deposit_paths`
  cuts a path where it crosses a base plane of a graded axis and keeps the charge to round-off,
  and `gather` divides by the physical cell size. The multigrid's coarse levels keep every other
  base node; the FFT backend and `solve_poisson` take uniform axes only. Tutorial "Graded axes".
- `graded_nodes(lower, upper, spacing, through=...)`: the nodes of a graded axis whose cells
  follow a local size (a function of `x` or `(x, dx)` pairs, linear between them), with
  neighbouring cells that change smoothly, and nodes at the `through` positions (electrode
  faces, aperture edges), each stretch between them graded on its own.
- `PatchHierarchy.deposit` and `interpolate` take `splines=`, the B-spline degree of the shape
  (1, the default and the cloud-in-cell weights so far, 2 or 3) or an `mpifield.Splines`;
  the field needs a halo of `mpifield.splines(p).halo` nodes (2 from degree 2 on). From degree
  2 on a stencil that would reach past the edge of the marker's patch folds onto the patch's
  edge nodes (as at a bounded wall), so `accumulate_halos` drops nothing and the total is
  kept; `interpolate` stays the transpose of the deposit. `CompositeNodes` and
  `CompositeEdges` stay at degree 1 (leaf cells with hanging nodes).
- patchwork runs in the browser with Pyodide, and a live demo page in the docs refines a mesh
  there: controls for the charges, their width, the refinement threshold, the number of levels
  and the longest patch side regrid a hierarchy around Gaussian charges and solve their
  potential with `NodalPoisson`, drawn on the patches of every level. It uses
  starlight-theme's `PyodideDemo`; the docs workflow builds the wheels it installs
  (`pyodide: true`), and a Pyodide check in CI (`.github/workflows/pyodide.yml`) installs the
  wheel, imports every module and runs the demo.
- `CompositeEdges.gather(points, e)`: the field of an edge field at points, `evaluation`
  applied to `e` with mpifield's compiled edge gather (`mpifield.WHITNEY`) on the block of each
  point's patch.

### Changed

- `AMRHierarchy.cell_volumes(level, *index)` and `face_areas(level, axis, *index)` take one
  index array per axis (they broadcast), since the measures depend on every index on a graded
  axis.
- The halo and restriction schedules find a patch's neighbours through a bin index of each
  level's boxes instead of testing every pair of patches: building them is linear in the number
  of patches. The first `exchange_halos` with 1024 patches of a 2D level took 9.9 s, now 0.4 s;
  512 patches in 3D 4.7 s, now 0.8 s (`benchmarks/amrex`, B4). The schedules are unchanged.
- `CompositeEdges.deposit_paths` runs mpifield's compiled path deposit (`mpifield.WHITNEY`)
  patch by patch in 1D, 2D and 3D: each path is clipped to the leaf cells of every patch (its
  box without its children's, shifted by whole periods on a periodic axis) and integrated on
  the patch's block, whose edges map to the composite ones. The same current as the NumPy
  pieces to round-off; 1e5 paths on a 2D three-level hierarchy 490 → 31 ms (scope-profiler,
  medians of 10 calls, four threads). patchwork's own `deposit_paths_1d` kernel is gone: on a
  line the composition takes 22 ms for 1e5 paths where it took 12 ms. Needs mpifield with
  `WHITNEY`.

- `plot_patches` options for drawing a hierarchy as part of another figure: `transpose`
  (the second axis horizontal, e.g. `x` of an `(r, x)` hierarchy), `scale` (e.g. `1e3` for
  millimetres), `mask_covered=False` (every level's grid in full, not hidden under the finer
  patches), `colors` and `labels` per level, and `title`. The legend entries are the labels
  of the plotted cell lines (or outlines), so a legend drawn later on the axes keeps them.

- `CompositeMultigrid.update_reaction(weights)`: the operator without a reaction plus the
  lumped reaction `T^T diag(w) T` (`T` the hanging nodes from the unknowns), its finest and
  first coarse levels from precomputed maps of the weights instead of Galerkin products. A
  Newton step of a Boltzmann species takes its Jacobian this way on refined hierarchies too,
  where `update_diagonal` does not apply.

- The compiled multigrid V-cycle (`smoother="gauss-seidel"`) runs on OpenMP threads
  (`MultigridBackend(num_threads=...)`, by default `OMP_NUM_THREADS` or the usable cores):
  multicolour Gauss–Seidel, red-black on seven-point stencils and the eight lattice parities
  on the coarse 27-point ones (independent sets of the graph on composite levels), each level
  numbered colour by colour. The result does not depend on the threads; on 413 000 unknowns
  a cycle takes 24 ms on one thread and 8.8 ms on four.

- `NodalPoisson(metric="cylindrical")`: the axisymmetric Poisson problem on an r–z hierarchy,
  finite volumes over the rings (each conductance weighted by 2π times the mean radius of its
  half cell, each control volume by its quarter cells', the 1/r term in the Shortley–Weller
  rows), second order also across coarse–fine interfaces and with embedded bodies; the axis
  takes no condition. `NodalPoisson.volumes` are the control volumes in the metric. Lumped
  mass only, no coefficient. Example "Axisymmetric potential around a rod".
- `NodalPoisson(reaction=...)` and `set_reaction(c)`: a lumped reaction term `c phi`, e.g. the
  Jacobian of a Newton step for a Boltzmann species; only the reduction is redone.
  `stiffness` is the operator without it, `unknowns` the order of the columns and `values`
  the prescribed values. The FFT backend refuses a reaction (`"auto"` takes the LU).
- `MultigridBackend(smoother="gauss-seidel")`: symmetric Gauss–Seidel smoothing, with the whole
  V-cycle in a compiled kernel (`multigrid_vcycle`) when the coarsest level has at most 2000
  unknowns. `CompositeMultigrid.update_diagonal` and `update_operator` take a new operator
  without the system; one that differs only on its diagonal (a reaction, `set_reaction`) gets
  its first coarse level from a precomputed map instead of a Galerkin product. With the Jacobian
  of every Newton step: a 3D Boltzmann extraction (55 000 nodes) 3.7 → 2.0 ms per V-cycle,
  25 → 10 ms per update; `preconditioner` as on `AMGSolver`.


- Compiled kernels for the node deposit in 1D and 3D (`deposit_nodes_1d`,
  `deposit_nodes_3d`; only 2D had one): `CompositeNodes.deposit` of 1e5 markers on a line
  5.0 → 0.59 ms, in 3D 21 → 7.5 ms. The deposit no longer copies points and charges that are
  already C-ordered (the copies cost as much as the kernel).
- The edge space on a line: `CompositeEdges` takes 1D hierarchies (the edges are the leaf
  cells; none hangs), and `deposit_paths` walks each path in a compiled kernel there
  (`deposit_paths_1d`): 1e5 paths 50 → 3.1 ms.
- Examples "Multigrid over patches" (3D: the LU's setup grows much faster than the multigrid's,
  1.7 s against 0.12 s at 32^3, a step with a new coefficient 11x faster; iterations do not
  grow with the patches) and "Anisotropic coefficients" (a tensor turned against the grid: second
  order with its off-diagonal terms, no convergence without them; multigrid iterations
  independent of the mesh up to an anisotropy of about ten).
- A variable coefficient for `NodalPoisson`: `-div(K grad phi)` with `K` per leaf cell (a
  number, a diagonal or a symmetric tensor, whose off-diagonal entries couple the edges of
  different axes) or per edge (`edge_coefficient`, e.g. a deposited susceptibility), lumped or
  consistent, in 1D, 2D and 3D; `set_coefficient` reassembles it for a new coefficient
  (`spaces.mass.stiffness_blocks`, `edge_cross_blocks`). `CompositeNodes.cell_centers`.
- `backend="multigrid"` (`MultigridBackend`, `CompositeMultigrid`): geometric multigrid over
  the patches. The levels are the hierarchy without its finest levels, then its base level
  coarsened by two; Galerkin operators, Chebyshev/Jacobi smoothing, CG (GMRES with bodies),
  also the singular operators. 8-12 iterations independent of the refinement levels and the
  cell count with a variable tensor coefficient; `set_coefficient` keeps the interpolations,
  and one V-cycle is a preconditioner on its own. Tutorial "Variable coefficients".
- Three dimensions. Hierarchies, `locate` (the kernels take three axes), clustering and
  regridding, halos and restriction, `CompositeNodes` (a hanging node on a coarse edge or in a
  coarse face takes the multilinear interpolant of the coarse cell's corners), the node deposit
  and `NodalPoisson` (faces `z_lower` and `z_upper`; Dirichlet, Neumann and bodies) in 3D; the
  boundary of a 3D mesh is made of quads (`CompositeNodes.face_pieces`). BoomerAMG through
  `backend="petsc"` solves the 3D nodal Poisson. `CellOperator` stays 1D and 2D and says so.
  Tests in `test_3d.py`, also under `mpirun`.
- The edge space in 3D: twelve Whitney edges per cell, hanging edges on coarse edges and inside
  coarse faces, the gather, the mass and the charge-conserving path deposit (Simpson's rule
  along each piece).
- The face space `CompositeFaces` (lowest-order Raviart–Thomas, 2D and 3D): fluxes through the
  faces of the leaf cells, hanging faces, the divergence to the leaf cells, the curl from the
  edges (3D; `cell_curl` in 2D), the face interpolant and the lumped or consistent face mass
  (`spaces.mass.face_mass_blocks`). `C G = 0`, `D C = 0` and `C T1 = T2 C T1` hold exactly.
  The mass matrices take a coefficient per leaf cell (`mass(lumped, weight=...)`).
- `CurlCurl`: `curl(nu curl A) + beta A = J` on the edges of a 2D or 3D hierarchy, electric
  or magnetic faces, lumped or consistent mass, an optional weak Coulomb gauge, solved by
  SuperLU (also complex `beta`) or by conjugate gradients with hypre's AMS
  (`solver="ams"`, rows split over the ranks; also the singular magnetostatic problem). The
  PETSc CI job runs its tests. Tutorial "Three dimensions".

- Lumped or consistent mass per space, from one set of per-cell blocks
  (`patchwork.spaces.mass`: `node_mass_blocks`, `edge_mass_blocks`, in any dimension):
  `CompositeNodes.mass(lumped=...)` (lumped: the control volumes; consistent: the Q1 mass),
  `CompositeEdges.mass` now assembled from the same blocks, and
  `NodalPoisson(mass="lumped" | "consistent")`: the finite-difference five-point stencil or
  the Q1 finite elements, both `T^T G^T M1 G T` with the edge mass of that kind and the source
  integrated with the node mass of that kind. The consistent scheme is exact for bilinear
  potentials across coarse–fine interfaces; it takes no embedded bodies, and the FFT backend
  declines it (`LinearSystem.mass`; `"auto"` takes the LU). Section "Finite differences or
  finite elements" in the edge-space tutorial.

- A guide to the MPI communication in the docs (`guides/mpi-communication`, a Guides section
  in the header and sidebar): the communicators of the patches, mpiarray's copy plan behind
  every exchange, the sources of the halo cells, the interpolation buffers, restriction, the
  deposit with its migration and accumulation, regridding, refluxing and the composite
  solves, with eight figures (`docs/mpi_figures.py`, `make docs-figures`), a summary table of
  the MPI calls of every operation and a checklist for code under MPI. An interactive
  explorer (`docs/src/components/HaloExplorer.astro`) shows, for a chosen number of ranks,
  placement and periodicity, where every halo cell comes from and the message matrix of the
  halo exchange's `Alltoallv`.

- Method cards (methodcard) on the numerical methods: the hierarchy and its $(r, z)$ metric,
  the marker deposit and its accumulation, halos, restriction and the conservative fill,
  tagging, clustering and regridding, the composite nodes and the nodal Poisson operator (with
  the cut edges and Shortley-Weller rows), the cell operator (energy, coarse-fine treatments,
  walls, axis, bodies) with its gradient and curl, the edge space (gradient, Whitney
  interpolant, mass matrices, charge-conserving path deposit), refluxing, the solver backends
  and the legacy cell Poisson solve, with their assumptions and references.
  `test_method_cards.py` checks them: SymPy rules for ten closed forms, the observed order of
  `NodalPoisson`, `CellOperator`, `CellOperator.gradient` and `HaloExchange`, and the cards of
  the docs against those of the code. Adds the `methodcard` dependency.
- A Numerics section in the docs (header and sidebar): an overview of the methods
  (`docs/src/content/docs/numerics/index.mdx`, with the main cards) and one page per module
  (starlight-theme's `methodcard: { pages: true }`, theme 0.7.4).

- `CompositeEdges`: the lowest-order edge space of a 2D hierarchy, one line integral of the
  field per edge of the leaf cells. A coarse edge on a patch edge is two hanging half-edges
  (`reduction`, which commutes with the gradient: `G T = T1 G T`). `gradient` (the incidence of
  edges and nodes), `mass(lumped=True|False)` (lumped, `G^T M1 G` is `NodalPoisson`'s
  operator), `evaluation(points)` (the field at points, the gradient of the bilinear potential
  for `G phi`) and `deposit(points, vectors)`, its transpose. `CompositeNodes.point_cells`
  gives the leaf cell of points and where in it they are. Tutorial: the edge space.

- `CompositeEdges.deposit_paths(start, end, charges)`: the current of charges moving along
  straight paths, each path split at every cell face it crosses (patch edges and periodic
  seams included) and integrated exactly against the edge functions. The charge is conserved
  to round-off: `(G T)^T j` is the change of the deposited nodal charge. A NumPy reference.

- `AMRHierarchy.wrap(points)` and `wrapped(points)`: points beyond a periodic face moved back
  into the domain, in place or as a copy (the array itself when nothing needs wrapping), and
  `AMRHierarchy.node_grid(patch)`: a patch's nodes as an `mpifield.BlockGrid`, for compiled
  kernels that run on a patch's node array. The private copies in `cells` and `nodal` are gone.

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

- `stiffness_blocks` with batched products instead of one einsum: `set_coefficient` about 4x
  faster (0.3 s instead of 1.7 s for 40^3 cells).
- Three layers (`patchwork.mesh`, `patchwork.spaces`, `patchwork.solvers`), each importing
  only the ones below it when it loads (`test_layers.py`):
  - `patchwork.mesh`: `hierarchy` (was `patchwork.mesh`), `halos`, `regrid`, `accumulate`,
    `reflux`, `plotting`.
  - `patchwork.spaces`: `nodes` (`CompositeNodes`, from `patchwork.nodal`), `edges`, `cells`
    (`CompositeCells` and `cell_body`, from `patchwork.cells`) and `boundaries` (`Dirichlet`,
    `Neumann`, `EmbeddedDirichlet`, `EmbeddedFloating`, the domain's faces and the surface
    location on mesh edges).
  - `patchwork.solvers`: `nodal_poisson` (`NodalPoisson`, `solve_nodal_poisson`, from
    `patchwork.nodal`), `cell_operator` (`CellOperator`, from `patchwork.cells`), `backends`,
    `distributed`, `poisson`.
  The names exported by `patchwork` are unchanged; code that imported the old modules
  (`patchwork.nodal`, `patchwork.cells`, `patchwork.backends`, ...) imports from the new ones.

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

- The multigrid's interpolations drop the round-off weights (below 1e-12) of fine nodes on
  coarse nodes: they widened the Galerkin stencils to two cells (10% more entries on a
  halved 3D level).
- The multigrid coarsens base levels with an odd number of cells along an axis on their node
  lattice (linear interpolation, the last interval one fine cell), where it stopped and solved
  the whole base level by LU before.
- The math in the docstrings renders on the API pages: `$$` blocks and `$...$` instead of reST
  `.. math::` and `:math:` (shown as code), and code spans instead of `:class:`, `:meth:` and
  the other roles. The module docstring of `cells` gives the radial conductance of an
  azimuthal component as the code computes it, over $r_f r_m$ (not $r_f^2$).

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
