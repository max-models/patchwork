---
title: Examples
description: Notebooks from the tutorials directory, executed and published with their outputs.
sidebar:
  order: 0
---

Step by step, from a first hierarchy to stencils on patches:

1. [A first patch hierarchy](/patchwork/examples/01-first-hierarchy/): the base level,
   patches and the rules they follow, cell- and node-centered fields.
2. [Poisson's equation on cells](/patchwork/examples/02-cell-centered-poisson/): the
   conservative cell-centered solve and its convergence.
3. [Poisson's equation on nodes](/patchwork/examples/03-node-centered-poisson/): faces with
   different conditions, periodic axes, electrodes, solving again, uniform-grid maps, and
   the solver backends (LU, FFT, PETSc).
4. [Domains that are not rectangles](/patchwork/examples/04-non-rectangular-domains/): an L
   and a T, with patches at their inner corners.
5. [Halos: stencils on patches](/patchwork/examples/05-halo-exchange/): halo exchange and
   restriction.
6. [Regridding](/patchwork/examples/06-regridding/): tagging cells, clustering them into
   patches, and patches that follow a moving feature.
7. [Edge space](/patchwork/examples/07-edge-space/): the electric-field space on the edges of the leaf cells.
8. [Three dimensions](/patchwork/examples/08-three-dimensions/): a refined box, the nodal
   Poisson, the face space and the exact sequence, and the curl–curl equation with AMS.
9. [Variable coefficients](/patchwork/examples/09-variable-coefficients/): the operator
   $-\nabla\cdot(\mathsf{K}\nabla\phi)$ of an implicit particle step, with a tensor per leaf
   cell, and the multigrid over the patches that solves it.
10. [Multigrid over patches](/patchwork/examples/10-multigrid-over-patches/): what the multigrid
    saves against the LU on a growing 3D mesh, its iterations as patches are added, and a
    coefficient that changes in every step.
11. [Anisotropic coefficients](/patchwork/examples/11-anisotropic-coefficients/): a tensor turned
    against the grid (a magnetized plasma), with and without its off-diagonal terms, and how far
    the multigrid carries the anisotropy.

Every notebook in `tutorials/` is executed when the site is built and published here with its
outputs. Download a notebook from its page to run the steps yourself.

See [Building the docs](/patchwork/development/docs/) for the cell tags that control what
is shown.
