"""The NumPy version of the ``deposit_paths_1d`` kernel (see ``deposit_paths_1d_pyccel.py``)."""

import numpy as np


def deposit_paths_1d(
    start,
    end,
    charges,
    origin,
    spacing,
    cells,
    periodic,
    leaf_map,
    cell_span,
    cell_edge,
    current,
) -> None:
    """Add the current of every path to ``current``."""
    a = (start[0] - origin[0]) / spacing[0]
    b = (end[0] - origin[0]) / spacing[0]
    valid = np.isfinite(a) & np.isfinite(b)
    a, b, charge = a[valid], b[valid], charges[valid]
    sign = np.where(b < a, -1.0, 1.0)
    low, high = np.minimum(a, b), np.maximum(a, b)
    first = np.floor(low).astype(np.int64)
    count = np.maximum(np.ceil(high).astype(np.int64) - first, 1)
    path = np.repeat(np.arange(a.size), count)
    offset = np.arange(path.size) - np.repeat(np.cumsum(count) - count, count)
    piece_low = np.maximum(low[path], first[path] + offset)
    piece_high = np.minimum(high[path], first[path] + offset + 1.0)
    keep = piece_high > piece_low
    path, piece_low, piece_high = path[keep], piece_low[keep], piece_high[keep]
    middle = 0.5 * (piece_low + piece_high)
    cell = np.floor(middle).astype(np.int64)
    if periodic[0]:
        cell %= cells[0]
        inside = np.ones(cell.size, dtype=bool)
    else:
        inside = (middle >= 0.0) & (middle <= cells[0])
        cell = np.clip(cell, 0, cells[0] - 1)
    leaf = np.where(inside, leaf_map[cell], -1)
    inside &= leaf >= 0
    leaf = leaf[inside]
    weight = (
        (sign * charge)[path[inside]]
        * (piece_high[inside] - piece_low[inside])
        / cell_span[leaf]
    )
    current += np.bincount(cell_edge[leaf], weights=weight, minlength=current.size)
