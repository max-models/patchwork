"""The finest patch over each point, and the point's cell on the level of that patch.

The Pyccel version (CPU) of the ``locate_points`` kernel; ``locate_points_numpy.py``
next to this file is the NumPy version, with the same arguments. Loaded by
`patchwork.kernels.locate_points`, compiled with Pyccel on first use.

Three axes: a 1D or 2D hierarchy is passed as 3D with one cell along each missing
axis. ``points`` is ``(3, n)``. A point's cell on the finest lattice of the hierarchy is
``floor((x - lower) / finest_spacing)`` per axis (the upper face of the domain in the
last cell), and its cell on level ``l`` that index divided by ``span[l]`` (the
refinement ratio to the power of the levels between): the cells of
`CompositeNodes`, whatever the level. Every level has a table
of which patch holds each of its cells over the bounding box of its patches (-1 where
none does): all tables flattened (C order, 32-bit integers) one after the other into ``tables``, level
``l`` from ``table_offset[l]``, with its lower corner ``table_corner[l]`` and shape
``table_shape[l]`` in cells of the level. A point is looked up level by level, the
finest first, until a table has a patch over it: one pass over the points, one
division per axis and a few lookups each. On a periodic axis a point is wrapped into
the domain first (a point already inside keeps its coordinate). A point outside the
domain, or NaN, gets -1.

Outputs: ``found`` (the patch index, -1 outside), ``cell`` ``(3, n)`` (the cell on the
patch's level, 0 outside) and ``wrapped`` ``(3, n)`` (the coordinates wrapped into the
domain on periodic axes).
"""


def locate_points(
    points: "float[:,:]",
    lower: "float[:]",
    upper: "float[:]",
    periodic: "int[:]",
    finest_spacing: "float[:]",
    finest_cells: "int[:]",
    span: "int[:]",
    table_corner: "int[:,:]",
    table_shape: "int[:,:]",
    table_offset: "int[:]",
    tables: "int32[:]",  # noqa: F821 -- a Pyccel type
    found: "int[:]",
    cell: "int[:,:]",
    wrapped: "float[:,:]",
) -> None:
    """Fill ``found``, ``cell`` and ``wrapped`` for every point (see the module)."""
    count = points.shape[1]
    num_levels = table_offset.shape[0] - 1
    low_x = lower[0]
    low_y = lower[1]
    low_z = lower[2]
    high_x = upper[0]
    high_y = upper[1]
    high_z = upper[2]
    length_x = high_x - low_x
    length_y = high_y - low_y
    length_z = high_z - low_z
    for point in range(count):
        x = points[0, point]
        y = points[1, point]
        z = points[2, point]
        if periodic[0] == 1 and not (x >= low_x and x < high_x):
            shifted = x - low_x
            x = low_x + (shifted - length_x * (shifted // length_x))
        if periodic[1] == 1 and not (y >= low_y and y < high_y):
            shifted = y - low_y
            y = low_y + (shifted - length_y * (shifted // length_y))
        if periodic[2] == 1 and not (z >= low_z and z < high_z):
            shifted = z - low_z
            z = low_z + (shifted - length_z * (shifted // length_z))
        wrapped[0, point] = x
        wrapped[1, point] = y
        wrapped[2, point] = z
        found[point] = -1
        cell[0, point] = 0
        cell[1, point] = 0
        cell[2, point] = 0
        # also False for NaN
        if not (x >= low_x and x <= high_x and y >= low_y and y <= high_y):
            continue
        if not (z >= low_z and z <= high_z):
            continue
        fine_x = int((x - low_x) / finest_spacing[0])
        if fine_x > finest_cells[0] - 1:
            fine_x = finest_cells[0] - 1
        fine_y = int((y - low_y) / finest_spacing[1])
        if fine_y > finest_cells[1] - 1:
            fine_y = finest_cells[1] - 1
        fine_z = int((z - low_z) / finest_spacing[2])
        if fine_z > finest_cells[2] - 1:
            fine_z = finest_cells[2] - 1
        for level in range(num_levels - 1, -1, -1):
            index_x = fine_x // span[level]
            index_y = fine_y // span[level]
            index_z = fine_z // span[level]
            offset_x = index_x - table_corner[level, 0]
            offset_y = index_y - table_corner[level, 1]
            offset_z = index_z - table_corner[level, 2]
            if offset_x < 0 or offset_x >= table_shape[level, 0]:
                continue
            if offset_y < 0 or offset_y >= table_shape[level, 1]:
                continue
            if offset_z < 0 or offset_z >= table_shape[level, 2]:
                continue
            owner = tables[
                table_offset[level]
                + (offset_x * table_shape[level, 1] + offset_y) * table_shape[level, 2]
                + offset_z
            ]
            if owner >= 0:
                found[point] = owner
                cell[0, point] = index_x
                cell[1, point] = index_y
                cell[2, point] = index_z
                break
