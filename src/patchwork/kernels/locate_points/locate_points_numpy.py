"""The NumPy version of the ``locate_points`` kernel (see ``locate_points_pyccel.py``).

The same arguments and results, vectorized: one pass over the points still pending per
level, the finest level first.
"""

import numpy as np


def locate_points(
    points,
    lower,
    upper,
    periodic,
    finest_spacing,
    finest_cells,
    span,
    table_corner,
    table_shape,
    table_offset,
    tables,
    found,
    cell,
    wrapped,
) -> None:
    """Fill ``found``, ``cell`` and ``wrapped`` for every point."""
    count = points.shape[1]
    inside = np.ones(count, dtype=bool)
    fine = []
    for axis in range(2):
        values = points[axis].astype(float, copy=True)
        low, high = lower[axis], upper[axis]
        if periodic[axis]:
            out = ~((values >= low) & (values < high))
            shifted = values[out] - low
            length = high - low
            values[out] = low + (shifted - length * (shifted // length))
        wrapped[axis] = values
        inside &= (values >= low) & (values <= high)  # False for NaN
    for axis in range(2):
        index = np.zeros(count, dtype=np.int64)
        index[inside] = (
            (wrapped[axis, inside] - lower[axis]) / finest_spacing[axis]
        ).astype(np.int64)
        fine.append(np.minimum(index, finest_cells[axis] - 1))
    found[:] = -1
    cell[:] = 0
    for level in range(table_offset.shape[0] - 2, -1, -1):
        pending = np.flatnonzero(inside & (found < 0))
        if pending.size == 0:
            break
        index = [fine[axis][pending] // span[level] for axis in range(2)]
        offset = [index[axis] - table_corner[level, axis] for axis in range(2)]
        within = np.ones(pending.size, dtype=bool)
        for axis in range(2):
            within &= (offset[axis] >= 0) & (offset[axis] < table_shape[level, axis])
        owner = np.full(pending.size, -1, dtype=np.int64)
        owner[within] = tables[
            table_offset[level]
            + offset[0][within] * table_shape[level, 1]
            + offset[1][within]
        ]
        hit = owner >= 0
        found[pending[hit]] = owner[hit]
        for axis in range(2):
            cell[axis, pending[hit]] = index[axis][hit]
