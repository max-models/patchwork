"""The charges of point particles on the nodes of a 2D composite mesh (cloud-in-cell).

The Pyccel version (CPU) of the ``deposit_nodes`` kernel; ``deposit_nodes_numpy.py``
next to this file is the NumPy version, with the same arguments. Loaded by
:mod:`patchwork.kernels.deposit_nodes`, compiled with Pyccel on first use; called by
:meth:`CompositeNodes.deposit <patchwork.nodal.CompositeNodes.deposit>`.

``points`` is ``(2, n)``. Each point is found in the leaf cell over its cell of the
finest lattice (``leaf_map``, ``(nx, ny)``, -1 outside the domain), as in
``CompositeNodes.cell_weights``: a point on a lattice line whose cell is outside the
domain takes the cell on the lower side (along x, then y, then both). A periodic axis
wraps; beyond a non-periodic face a point deposits nothing. Its charge is spread over
the four corners of the leaf cell (``cell_corners``, lower-left, lower-right,
upper-left, upper-right) with the bilinear weights of its position in the cell
(``cell_lower`` and ``cell_span``, on the finest lattice), and added to ``charge``.
"""

from math import floor


def deposit_nodes(
    points: "float[:,:]",
    charges: "float[:]",
    origin: "float[:]",
    spacing: "float[:]",
    cells: "int[:]",
    periodic: "int[:]",
    leaf_map: "int[:,:]",
    cell_corners: "int[:,:]",
    cell_lower: "int[:,:]",
    cell_span: "int[:]",
    charge: "float[:]",
) -> None:
    """Add the charges of the points to ``charge`` (see the module)."""
    num_x = cells[0]
    num_y = cells[1]
    for point in range(points.shape[1]):
        position_x = (points[0, point] - origin[0]) / spacing[0]
        position_y = (points[1, point] - origin[1]) / spacing[1]
        # NaN (and infinite) positions deposit nothing (Pyccel folds x == x to True)
        if not (position_x > -1.0e300 and position_x < 1.0e300):
            continue
        if not (position_y > -1.0e300 and position_y < 1.0e300):
            continue
        cell_x = int(floor(position_x))
        cell_y = int(floor(position_y))
        local_x = position_x
        local_y = position_y
        if periodic[0] == 1:
            cell_x = cell_x % num_x
            local_x = position_x - num_x * floor(position_x / num_x)
        else:
            if position_x < 0.0 or position_x > num_x:
                continue
            if cell_x > num_x - 1:
                cell_x = num_x - 1
        if periodic[1] == 1:
            cell_y = cell_y % num_y
            local_y = position_y - num_y * floor(position_y / num_y)
        else:
            if position_y < 0.0 or position_y > num_y:
                continue
            if cell_y > num_y - 1:
                cell_y = num_y - 1
        leaf = leaf_map[cell_x, cell_y]
        on_line_x = local_x == floor(local_x) and cell_x > 0
        on_line_y = local_y == floor(local_y) and cell_y > 0
        if leaf < 0 and on_line_x:
            leaf = leaf_map[cell_x - 1, cell_y]
        if leaf < 0 and on_line_y:
            leaf = leaf_map[cell_x, cell_y - 1]
        if leaf < 0 and on_line_x and on_line_y:
            leaf = leaf_map[cell_x - 1, cell_y - 1]
        if leaf < 0:
            continue
        span = cell_span[leaf]
        fraction_x = (local_x - cell_lower[leaf, 0]) / span
        fraction_y = (local_y - cell_lower[leaf, 1]) / span
        value = charges[point]
        charge[cell_corners[leaf, 0]] += (1.0 - fraction_x) * (1.0 - fraction_y) * value
        charge[cell_corners[leaf, 1]] += fraction_x * (1.0 - fraction_y) * value
        charge[cell_corners[leaf, 2]] += (1.0 - fraction_x) * fraction_y * value
        charge[cell_corners[leaf, 3]] += fraction_x * fraction_y * value
