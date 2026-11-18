"""The NumPy version of the ``multigrid_vcycle`` kernel (see ``multigrid_vcycle_pyccel.py``)."""

import numpy as np
import scipy.sparse as sparse


def _matrix(ptr, start, idx, val, rows, columns):
    """Level ``rows`` x ``columns`` of a concatenated CSR matrix."""
    pointers = np.asarray(ptr[start : start + rows + 1])
    first, last = int(pointers[0]), int(pointers[-1])
    return sparse.csr_matrix(
        (val[first:last], idx[first:last], pointers - first), shape=(rows, columns)
    )


def colour_sweeps(matrix, x, b, groups, sweeps, forward):
    """Multicolour Gauss--Seidel: each colour (rows ``groups[c]``) at once, in order."""
    diagonal = matrix.diagonal()
    off = matrix - sparse.diags(diagonal)
    order = groups if forward else groups[::-1]
    for _ in range(sweeps):
        for group in order:
            x[group] = (b[group] - off[group] @ x) / diagonal[group]
    return x


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
    color_start,
    color_level,
    sweeps,
    num_threads,
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
    groups = []
    for k in range(levels - 1):
        bounds = color_start[color_level[k] : color_level[k + 1]]
        groups.append(
            [np.arange(bounds[c], bounds[c + 1]) for c in range(bounds.size - 1)]
        )
    rights = [np.asarray(b, dtype=float)]
    lefts = []
    for k in range(levels - 1):
        matrix = operators[k]
        current = colour_sweeps(
            matrix, np.zeros(sizes[k]), rights[k], groups[k], sweeps, True
        )
        lefts.append(current)
        restriction = _matrix(r_ptr, r_start[k], r_idx, r_val, sizes[k + 1], sizes[k])
        rights.append(restriction @ (rights[k] - matrix @ current))
    current = np.asarray(coarse) @ rights[-1]
    for k in range(levels - 2, -1, -1):
        interpolation = _matrix(p_ptr, p_start[k], p_idx, p_val, sizes[k], sizes[k + 1])
        current = lefts[k] + interpolation @ current
        current = colour_sweeps(
            operators[k], current, rights[k], groups[k], sweeps, False
        )
    x[:] = current
