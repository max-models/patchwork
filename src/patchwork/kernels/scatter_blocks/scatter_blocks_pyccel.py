"""B-spline deposit of degree 0 to 3 of marker values into the blocks of many patches at once.

The Pyccel version (CPU) of the ``scatter_blocks`` kernel; ``scatter_blocks_numpy.py``
next to this file is the NumPy version, with the same arguments. Called by
`AMRHierarchy.deposit`.

The blocks of the patches lie end to end in one flat ``buffer``. Marker ``i`` deposits
``values[i]`` onto the ``degree + 1`` nodes of its stencil along each of the ``ndim``
axes (``rel.shape[0]``, 1 to 3), found by mpifield's ``spline_cell`` from ``rel[a, i]``,
its position in nodes from the first node of its block, on a bounded axis of
``count[a, i]`` nodes (a stencil node past the edge is the edge node). The node ``k``
along axis ``a`` is at ``base[i] + sum_a k_a * stride[a, i]`` in the buffer. Serial:
the sums are in the order of the markers.
"""

import numpy as np
from mpifield.kernels.spline_cell_pyccel import spline_cell


def scatter_blocks(
    rel: "float[:,:]",
    count: "int[:,:]",
    stride: "int[:,:]",
    base: "int[:]",
    values: "float[:]",
    degree: int,
    buffer: "float[:]",
) -> None:
    """Add the values of the markers to ``buffer`` (see the module)."""
    ndim = rel.shape[0]
    node = np.zeros((3, 4), dtype=np.int64)
    weight = np.zeros((3, 4), dtype=np.float64)
    size = np.ones(3, dtype=np.int64)
    for a in range(3):
        node[a, 0] = 0
        weight[a, 0] = 1.0
    for a in range(ndim):
        size[a] = degree + 1
    for i in range(rel.shape[1]):
        value = values[i]
        if value == 0.0:
            continue
        for a in range(ndim):
            spline_cell(rel[a, i], count[a, i], 0, degree, node[a, :], weight[a, :])
        for p in range(size[0]):
            at_x = base[i] + node[0, p] * stride[0, i]
            w_x = weight[0, p] * value
            for q in range(size[1]):
                at_y = at_x
                w_y = w_x
                if ndim > 1:
                    at_y = at_x + node[1, q] * stride[1, i]
                    w_y = w_x * weight[1, q]
                for r in range(size[2]):
                    at = at_y
                    w = w_y
                    if ndim > 2:
                        at = at_y + node[2, r] * stride[2, i]
                        w = w_y * weight[2, r]
                    buffer[at] += w
