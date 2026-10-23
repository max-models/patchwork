"""The NumPy version of the ``deposit_nodes`` kernel (see ``deposit_nodes_pyccel.py``)."""

import numpy as np


def deposit_nodes(
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
    position = (points - origin[:, None]) / spacing[:, None]
    valid = ~np.isnan(position).any(axis=0)
    cell, local = [], []
    for axis in range(2):
        index = np.floor(np.where(valid, position[axis], 0.0)).astype(np.int64)
        if periodic[axis]:
            index = index % cells[axis]
            wrapped = position[axis] - cells[axis] * np.floor(position[axis] / cells[axis])
        else:
            valid &= (position[axis] >= 0.0) & (position[axis] <= cells[axis])
            index = np.clip(index, 0, cells[axis] - 1)
            wrapped = position[axis]
        cell.append(index)
        local.append(wrapped)
    leaf = leaf_map[cell[0], cell[1]]
    on_line = [(local[a] == np.floor(local[a])) & (cell[a] > 0) for a in range(2)]
    for step in ((1, 0), (0, 1), (1, 1)):
        missing = leaf < 0
        for axis in range(2):
            if step[axis]:
                missing &= on_line[axis]
        if missing.any():
            leaf[missing] = leaf_map[
                cell[0][missing] - step[0], cell[1][missing] - step[1]
            ]
    valid &= leaf >= 0
    leaf = leaf[valid]
    span = cell_span[leaf]
    fx = (local[0][valid] - cell_lower[leaf, 0]) / span
    fy = (local[1][valid] - cell_lower[leaf, 1]) / span
    value = charges[valid]
    weights = ((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy)
    for corner, weight in enumerate(weights):
        charge += np.bincount(
            cell_corners[leaf, corner], weights=weight * value, minlength=charge.size
        )
