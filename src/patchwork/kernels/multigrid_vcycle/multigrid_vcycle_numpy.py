"""The NumPy version of the ``multigrid_vcycle`` kernel (see ``multigrid_vcycle_pyccel.py``)."""

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve_triangular


def _matrix(ptr, start, idx, val, rows, columns):
    """Level ``rows`` x ``columns`` of a concatenated CSR matrix."""
    pointers = np.asarray(ptr[start : start + rows + 1])
    first, last = int(pointers[0]), int(pointers[-1])
    return sparse.csr_matrix(
        (val[first:last], idx[first:last], pointers - first), shape=(rows, columns)
    )


def multigrid_vcycle(
    levels,
    offsets,
    a_ptr,
    a_start,
    a_idx,
    a_val,
    p_ptr,
    p_start,
    p_idx,
    p_val,
    r_ptr,
    r_start,
    r_idx,
    r_val,
    coarse,
    sweeps,
    b,
    x,
    bs,
    xs,
    rs,
) -> None:
    """Apply one V-cycle to ``b`` into ``x``."""
    sizes = np.diff(np.asarray(offsets))
    operators = [
        _matrix(a_ptr, a_start[k], a_idx, a_val, sizes[k], sizes[k])
        for k in range(levels)
    ]
    rights = [b]
    lefts = []
    for k in range(levels - 1):
        matrix = operators[k]
        lower = sparse.tril(matrix, format="csr")
        upper = matrix - lower
        current = np.zeros(sizes[k])
        for _ in range(sweeps):
            current = spsolve_triangular(lower, rights[k] - upper @ current, lower=True)
        lefts.append(current)
        restriction = _matrix(r_ptr, r_start[k], r_idx, r_val, sizes[k + 1], sizes[k])
        rights.append(restriction @ (rights[k] - matrix @ current))
    current = np.asarray(coarse) @ rights[-1]
    for k in range(levels - 2, -1, -1):
        matrix = operators[k]
        upper = sparse.triu(matrix, format="csr")
        lower = matrix - upper
        interpolation = _matrix(p_ptr, p_start[k], p_idx, p_val, sizes[k], sizes[k + 1])
        current = lefts[k] + interpolation @ current
        for _ in range(sweeps):
            current = spsolve_triangular(
                upper, rights[k] - lower @ current, lower=False
            )
    x[:] = current
