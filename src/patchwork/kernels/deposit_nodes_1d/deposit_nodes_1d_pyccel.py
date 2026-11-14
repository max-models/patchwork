"""The charges of point particles on the nodes of a 1D composite mesh (cloud-in-cell).

The Pyccel version (CPU) of the ``deposit_nodes_1d`` kernel; ``deposit_nodes_1d_numpy.py``
next to this file is the NumPy version, with the same arguments. Loaded by
`patchwork.kernels.deposit_nodes_1d`, compiled with Pyccel on first use; called by
`CompositeNodes.deposit`.

The line version of ``deposit_nodes`` (see there): ``points`` is ``(1, n)``, ``leaf_map``
``(nx,)`` (-1 outside the domain), ``cell_corners`` ``(cells, 2)`` (lower, upper). A point
on a node whose cell is outside the domain takes the cell below it. A periodic axis wraps;
beyond a non-periodic face, or NaN, a point deposits nothing. Its charge is split between
the two ends of its leaf cell with the linear weights of its position in the cell.
"""

from math import floor


def deposit_nodes_1d(
    points: "float[:,:]",
    charges: "float[:]",
    origin: "float[:]",
    spacing: "float[:]",
    cells: "int[:]",
    periodic: "int[:]",
    leaf_map: "int[:]",
    cell_corners: "int[:,:]",
    cell_lower: "int[:,:]",
    cell_span: "int[:]",
    charge: "float[:]",
) -> None:
    """Add the charges of the points to ``charge`` (see the module)."""
    num_x = cells[0]
    for point in range(points.shape[1]):
        position_x = (points[0, point] - origin[0]) / spacing[0]
        # NaN (and infinite) positions deposit nothing (Pyccel folds x == x to True)
        if not (position_x > -1.0e300 and position_x < 1.0e300):
            continue
        cell_x = int(floor(position_x))
        local_x = position_x
        if periodic[0] == 1:
            cell_x = cell_x % num_x
            local_x = position_x - num_x * floor(position_x / num_x)
        else:
            if position_x < 0.0 or position_x > num_x:
                continue
            if cell_x > num_x - 1:
                cell_x = num_x - 1
        leaf = leaf_map[cell_x]
        if leaf < 0 and local_x == floor(local_x) and cell_x > 0:
            leaf = leaf_map[cell_x - 1]
        if leaf < 0:
            continue
        fraction_x = (local_x - cell_lower[leaf, 0]) / cell_span[leaf]
        value = charges[point]
        charge[cell_corners[leaf, 0]] += (1.0 - fraction_x) * value
        charge[cell_corners[leaf, 1]] += fraction_x * value
