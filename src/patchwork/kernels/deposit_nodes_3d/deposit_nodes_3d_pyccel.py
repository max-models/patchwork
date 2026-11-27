"""The charges of point particles on the nodes of a 3D composite mesh (cloud-in-cell).

The Pyccel version (CPU) of the ``deposit_nodes_3d`` kernel; ``deposit_nodes_3d_numpy.py``
next to this file is the NumPy version, with the same arguments. Loaded by
`patchwork.kernels.deposit_nodes_3d`, compiled with Pyccel on first use; called by
`CompositeNodes.deposit`.

The box version of ``deposit_nodes`` (see there): ``points`` is ``(3, n)``, ``leaf_map``
``(nx, ny, nz)`` (-1 outside the domain), ``cell_corners`` ``(cells, 8)`` with the x
offset varying fastest, then y, then z. A point on a lattice plane whose cell is outside
the domain takes a cell on the lower side: along x, y, z, then xy, xz, yz, then xyz. A
periodic axis wraps; beyond a non-periodic face, or NaN, a point deposits nothing. Its
charge is spread over the eight corners of its leaf cell with the trilinear weights.
``found`` gets the patch of every point, and OpenMP splits the points over per-thread
buffers, as in ``deposit_nodes``.
"""

from math import floor

from numpy import zeros
from pyccel.stdlib.internal.openmp import omp_get_max_threads, omp_get_thread_num

#: Fewest points worth the per-thread buffers.
MIN_PARALLEL_POINTS = 20000


def _deposit_chunk_3d(
    points: "float[:,:]",
    charges: "float[:]",
    origin: "float[:]",
    spacing: "float[:]",
    cells: "int[:]",
    periodic: "int[:]",
    leaf_map: "int[:,:,:]",
    cell_corners: "int[:,:]",
    cell_lower: "int[:,:]",
    cell_span: "int[:]",
    cell_patch: "int[:]",
    found: "int[:]",
    buffer: "float[:,:]",
    row: int,
    first: int,
    last: int,
) -> None:
    """Deposit the points ``first`` to ``last`` into row ``row`` of ``buffer``, find their patches."""
    cell = zeros(3, dtype=int)
    local = zeros(3)
    line = zeros(3, dtype=int)
    for point in range(first, last):
        found[point] = -1
        skip = False
        for axis in range(3):
            position = (points[axis, point] - origin[axis]) / spacing[axis]
            # NaN (and infinite) positions deposit nothing
            if not (position > -1.0e300 and position < 1.0e300):
                skip = True
                break
            number = cells[axis]
            index = int(floor(position))
            wrapped = position
            if periodic[axis] == 1:
                index = index % number
                wrapped = position - number * floor(position / number)
            else:
                if position < 0.0 or position > number:
                    skip = True
                    break
                if index > number - 1:
                    index = number - 1
            cell[axis] = index
            local[axis] = wrapped
            line[axis] = 0
            if wrapped == floor(wrapped) and index > 0:
                line[axis] = 1
        if skip:
            continue
        cx = cell[0]
        cy = cell[1]
        cz = cell[2]
        leaf = leaf_map[cx, cy, cz]
        if leaf >= 0:
            found[point] = cell_patch[leaf]
        # the lower neighbours across lattice planes, in the order of point_cells
        if leaf < 0 and line[0] == 1:
            leaf = leaf_map[cx - 1, cy, cz]
        if leaf < 0 and line[1] == 1:
            leaf = leaf_map[cx, cy - 1, cz]
        if leaf < 0 and line[2] == 1:
            leaf = leaf_map[cx, cy, cz - 1]
        if leaf < 0 and line[0] == 1 and line[1] == 1:
            leaf = leaf_map[cx - 1, cy - 1, cz]
        if leaf < 0 and line[0] == 1 and line[2] == 1:
            leaf = leaf_map[cx - 1, cy, cz - 1]
        if leaf < 0 and line[1] == 1 and line[2] == 1:
            leaf = leaf_map[cx, cy - 1, cz - 1]
        if leaf < 0 and line[0] == 1 and line[1] == 1 and line[2] == 1:
            leaf = leaf_map[cx - 1, cy - 1, cz - 1]
        if leaf < 0:
            continue
        span = cell_span[leaf]
        fx = (local[0] - cell_lower[leaf, 0]) / span
        fy = (local[1] - cell_lower[leaf, 1]) / span
        fz = (local[2] - cell_lower[leaf, 2]) / span
        gx = 1.0 - fx
        gy = 1.0 - fy
        gz = 1.0 - fz
        value = charges[point]
        buffer[row, cell_corners[leaf, 0]] += gx * gy * gz * value
        buffer[row, cell_corners[leaf, 1]] += fx * gy * gz * value
        buffer[row, cell_corners[leaf, 2]] += gx * fy * gz * value
        buffer[row, cell_corners[leaf, 3]] += fx * fy * gz * value
        buffer[row, cell_corners[leaf, 4]] += gx * gy * fz * value
        buffer[row, cell_corners[leaf, 5]] += fx * gy * fz * value
        buffer[row, cell_corners[leaf, 6]] += gx * fy * fz * value
        buffer[row, cell_corners[leaf, 7]] += fx * fy * fz * value


def deposit_nodes_3d(
    points: "float[:,:]",
    charges: "float[:]",
    origin: "float[:]",
    spacing: "float[:]",
    cells: "int[:]",
    periodic: "int[:]",
    leaf_map: "int[:,:,:]",
    cell_corners: "int[:,:]",
    cell_lower: "int[:,:]",
    cell_span: "int[:]",
    cell_patch: "int[:]",
    found: "int[:]",
    charge: "float[:]",
) -> None:
    """Add the charges of the points to ``charge`` and find their patches (see the module)."""
    # the first charges.shape[0] columns: points may have room to spare
    num_points = charges.shape[0]
    num_nodes = charge.shape[0]
    num_threads = int(omp_get_max_threads())
    if num_points < MIN_PARALLEL_POINTS or num_points < num_nodes:
        num_threads = 1
    chunk = (num_points + num_threads - 1) // num_threads
    buffer = zeros((num_threads, num_nodes))
    # fmt: off
    #$ omp parallel num_threads(num_threads) private(thread, first, last)
    # fmt: on
    thread = int(omp_get_thread_num())
    first = min(thread * chunk, num_points)
    last = min(first + chunk, num_points)
    _deposit_chunk_3d(
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
        cell_patch,
        found,
        buffer,
        thread,
        first,
        last,
    )
    # fmt: off
    #$ omp end parallel
    # fmt: on
    # fmt: off
    #$ omp parallel for private(thread, total)
    # fmt: on
    for node in range(num_nodes):
        total = 0.0
        for thread in range(num_threads):
            total += buffer[thread, node]
        charge[node] += total
