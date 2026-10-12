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

### Changed

- `plot_patches` colours all patches on one scale and shades node-centered fields between
  their nodes.
- `mpiarray` is taken from its `devel` branch instead of a pinned commit, so that patchwork
  installs next to packages that need a newer `mpiarray`.

## [0.1.0] - 2026-10-05

### Added

- Package layout with a console entry point and pytest tests.
- Astro + Starlight documentation site with executed tutorials and a generated API reference.
- GitHub Actions for tests, ruff, pyright, tutorials, documentation and PyPI publishing.
