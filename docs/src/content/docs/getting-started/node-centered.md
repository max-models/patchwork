---
title: Node-centered Poisson
description: Solve Poisson's equation on the nodes of a patch hierarchy, with faces, periodic axes and embedded bodies.
---

`NodalPoisson` solves $-\nabla^2\phi = f$ with the unknowns on the cell corners.
It suits codes whose own grid is node-centered, such as particle-in-cell codes
that deposit charge on nodes: the base level of the hierarchy is then their grid,
and patches refine it where the field needs it.

## The mesh and its nodes

```python
import numpy as np
from patchwork import AMRHierarchy, Dirichlet, EmbeddedDirichlet, Neumann, NodalPoisson

mesh = AMRHierarchy(base_shape=(32, 32), periodic=(False, True))
mesh.add_patch(level=1, box=((24, 40), (24, 40)))
mesh.add_patch(level=2, box=((56, 72), (56, 72)))
```

A patch of `(nx, ny)` cells has `(nx + 1, ny + 1)` nodes (`Patch.node_shape`).
Patches that touch or nest share the nodes they have in common, so the solver
numbers the nodes of the whole mesh once (`CompositeNodes`). A node of a fine
cell in the middle of a coarser cell's edge is *hanging*: its value is
interpolated between the ends of that edge, which keeps the potential
continuous across patch boundaries. On a periodic axis the last node is the
first.

## Conditions

```python
solver = NodalPoisson(
    mesh,
    faces={"x_lower": Neumann(0.0), "x_upper": Dirichlet(lambda x, y: 100.0 * y)},
    embedded=[
        EmbeddedDirichlet(
            inside=lambda points: np.hypot(points[0] - 0.5, points[1] - 0.5) < 0.05,
            value=-20.0,
        )
    ],
)
```

- `Dirichlet(value)` fixes the potential on a face, `Neumann(gradient)` its
  outward normal derivative. Both take a number or a function of `x, y`. A face
  that is not listed is grounded; the faces of a periodic axis take no condition.
- `EmbeddedDirichlet(inside, value)` is a body at a fixed potential. `inside`
  tells which points it contains; its surface is found by bisection along the
  mesh edges that enter it, and the nodes next to it use the distance to the
  surface (the Shortley-Weller stencil where four equal cells meet).

The problem needs at least one Dirichlet face or body: without one the
potential is only known up to a constant, and the solver refuses it.

## Solving, and solving again

```python
x, y = solver.nodes.coordinates.T
potential = solver.solve(np.exp(-((x - 0.3) ** 2 + (y - 0.5) ** 2) / 0.01))
```

The operator is assembled and factorized when the solver is created. Every
`solve` after that costs two triangular solves, and `set_conditions` changes
face values or body potentials without factorizing again. The source and the
result are arrays over `solver.nodes`; pass field names instead to read and
write node-centered patch fields:

```python
mesh.allocate_field("rho", centering="node")
solver.solve("rho", potential="phi")
phi_on_fine_patch = mesh.patches[-1].field("phi").local
```

## Fields on a uniform grid

A code that works on uniform arrays can move data in and out with the maps of
`solver.nodes`:

- `from_level(0)` interpolates values on the base grid to all nodes, e.g. a
  charge density that was deposited there;
- `to_level(level)` samples the solution on the nodes of a level over the whole
  domain;
- `sample_patches(arrays, level)` does the same for a field that is computed
  patch by patch, such as the gradient: each node takes the value of the finest
  patch around it.

![Potential around a small electrode, resolved by two nested patches](/patchwork/figures/nodal_electrode.png)

`examples/nodal_electrode.py` solves this case: two nested patches around an
electrode of half a base cell's radius bring the error from 4e-2 to 3e-3 with a
third more unknowns.

## Accuracy

The solution converges at second order. Hanging nodes are interpolated linearly,
so across a patch boundary the error is that of the coarser side: put a patch
boundary where the solution is smooth on the coarse mesh, a few cells away from
the feature that the patch resolves.
