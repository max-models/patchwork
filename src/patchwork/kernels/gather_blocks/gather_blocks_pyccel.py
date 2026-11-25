"""B-spline gather of degree 0 to 3 from the blocks of many patches at once.

The Pyccel version (CPU) of the ``gather_blocks`` kernel; ``gather_blocks_numpy.py``
next to this file is the NumPy version, with the same arguments. Called by
`AMRHierarchy.interpolate`; the transpose of ``scatter_blocks`` (same stencils, see
there): ``result[i]`` is the weighted sum of the stencil nodes of marker ``i`` in the
flat ``buffer`` of the blocks.
"""

import numpy as np
from mpifield.kernels.spline_cell_pyccel import spline_cell


def gather_blocks(
    rel: "float[:,:]",
    count: "int[:,:]",
    stride: "int[:,:]",
    base: "int[:]",
    degree: int,
    buffer: "float[:]",
    result: "float[:]",
) -> None:
    """Write the gathered values of the markers to ``result`` (see the module)."""
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
        for a in range(ndim):
            spline_cell(rel[a, i], count[a, i], 0, degree, node[a, :], weight[a, :])
        total = 0.0
        for p in range(size[0]):
            at_x = base[i] + node[0, p] * stride[0, i]
            w_x = weight[0, p]
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
                    total += w * buffer[at]
        result[i] = total
