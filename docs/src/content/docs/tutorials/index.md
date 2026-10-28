---
title: Tutorials
description: Notebooks from the tutorials directory, executed and published with their outputs.
sidebar:
  order: 0
---

Step by step, from a first hierarchy to stencils on patches:

1. [A first patch hierarchy](/patchwork/tutorials/01-first-hierarchy/): the base level,
   patches and the rules they follow, cell- and node-centered fields.
2. [Poisson's equation on cells](/patchwork/tutorials/02-cell-centered-poisson/): the
   conservative cell-centered solve and its convergence.
3. [Poisson's equation on nodes](/patchwork/tutorials/03-node-centered-poisson/): faces with
   different conditions, periodic axes, electrodes, solving again, uniform-grid maps, and
   the solver backends (LU, FFT, PETSc).
4. [Domains that are not rectangles](/patchwork/tutorials/04-non-rectangular-domains/): an L
   and a T, with patches at their inner corners.
5. [Halos: stencils on patches](/patchwork/tutorials/05-halo-exchange/): halo exchange and
   restriction.
6. [Regridding](/patchwork/tutorials/06-regridding/): tagging cells, clustering them into
   patches, and patches that follow a moving feature.

Every notebook in `tutorials/` is executed when the site is built and published here with its
outputs. Download a notebook from its page to run the steps yourself.

See [Building the docs](/patchwork/development/docs/) for the cell tags that control what
is shown.
