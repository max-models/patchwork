"""The NumPy version of the ``deposit_nodes_1d`` kernel (see ``deposit_nodes_1d_pyccel.py``).

Written for any number of axes; ``deposit_nodes_3d_numpy.py`` uses it too.
"""

import itertools

import numpy as np


def deposit_nodes_any(
    points,
    charges,
    origin,
    spacing,
    cells,
    periodic,
    leaf_map,
    cell_corners,
    cell_lower,
    cell_span,
    charge,
) -> None:
    """Add the charges of the points to ``charge``, on a mesh of ``points.shape[0]`` axes."""
    ndim = points.shape[0]
    position = (points - origin[:, None]) / spacing[:, None]
    valid = np.isfinite(position).all(axis=0)
    cell, local = [], []
    for axis in range(ndim):
        index = np.floor(np.where(valid, position[axis], 0.0)).astype(np.int64)
        if periodic[axis]:
            index = index % cells[axis]
            wrapped = position[axis] - cells[axis] * np.floor(
                position[axis] / cells[axis]
            )
        else:
            valid &= (position[axis] >= 0.0) & (position[axis] <= cells[axis])
            index = np.clip(index, 0, cells[axis] - 1)
            wrapped = position[axis]
        cell.append(index)
        local.append(wrapped)
    leaf = leaf_map[tuple(cell)]
    on_line = [(local[a] == np.floor(local[a])) & (cell[a] > 0) for a in range(ndim)]
    steps = sorted(
        (step for step in itertools.product((0, 1), repeat=ndim) if any(step)),
        key=lambda step: (sum(step), step[::-1]),
    )
    for step in steps:
        missing = leaf < 0
        for axis in range(ndim):
            if step[axis]:
                missing &= on_line[axis]
        if missing.any():
            leaf[missing] = leaf_map[
                tuple(cell[a][missing] - step[a] for a in range(ndim))
            ]
    valid &= leaf >= 0
    leaf = leaf[valid]
    span = cell_span[leaf]
    fraction = [(local[a][valid] - cell_lower[leaf, a]) / span for a in range(ndim)]
    value = charges[valid]
    for corner in range(2**ndim):
        weight = value.copy()
        for axis in range(ndim):
            upper = (corner >> axis) & 1
            weight *= fraction[axis] if upper else 1.0 - fraction[axis]
        charge += np.bincount(
            cell_corners[leaf, corner], weights=weight, minlength=charge.size
        )


def deposit_nodes_1d(
    points,
    charges,
    origin,
    spacing,
    cells,
    periodic,
    leaf_map,
    cell_corners,
    cell_lower,
    cell_span,
    charge,
) -> None:
    """Add the charges of the points to ``charge``."""
    deposit_nodes_any(
        points,
        charges,
        origin,
        spacing,
        cells,
        periodic,
        leaf_map,
        cell_corners,
        cell_lower,
        cell_span,
        charge,
    )
