"""The charges of point particles on the nodes of a 2D composite mesh (cloud-in-cell).

The Pyccel version (CPU) of the ``deposit_nodes`` kernel; ``deposit_nodes_numpy.py``
next to this file is the NumPy version, with the same arguments. Loaded by
`patchwork.kernels.deposit_nodes`, compiled with Pyccel on first use; called by
`CompositeNodes.deposit`.

``points`` is ``(2, m)``, of which the first ``n`` columns are deposited, ``n`` the
length of ``charges`` (a store with room for more points needs no copy). Each point is found in the leaf cell over its cell of the
finest lattice (``leaf_map``, ``(nx, ny)``, -1 outside the domain), as in
``CompositeNodes.cell_weights``: a point on a lattice line whose cell is outside the
domain takes the cell on the lower side (along x, then y, then both). A periodic axis
wraps; beyond a non-periodic face a point deposits nothing. Its charge is spread over
the four corners of the leaf cell (``cell_corners``, lower-left, lower-right,
upper-left, upper-right) with the bilinear weights of its position in the cell
(``cell_lower`` and ``cell_span``, on the finest lattice), and added to ``charge``.

``found`` gets the patch of every point, as `AMRHierarchy.locate` finds it: the patch
of the leaf cell over its lattice cell (``cell_patch``), before the lower-side rule,
and -1 for a point outside the domain.

With OpenMP the points are split over the threads in contiguous chunks, each
depositing into its own row of a buffer, and the rows are summed into ``charge`` in
thread order: no atomic updates, and the result is reproducible for a given number of
threads. Few points (fewer than the nodes, or than ``MIN_PARALLEL_POINTS``) are
deposited serially.
"""

from math import floor

from numpy import zeros
from pyccel.stdlib.internal.openmp import omp_get_max_threads, omp_get_thread_num

#: Fewest points worth the per-thread buffers.
MIN_PARALLEL_POINTS = 20000


def _deposit_chunk(
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
    cell_patch: "int[:]",
    found: "int[:]",
    buffer: "float[:,:]",
    row: int,
    first: int,
    last: int,
) -> None:
    """Deposit the points ``first`` to ``last`` into row ``row`` of ``buffer``, find their patches."""
    num_x = cells[0]
    num_y = cells[1]
    inverse_x = 1.0 / spacing[0]
    inverse_y = 1.0 / spacing[1]
    for point in range(first, last):
        found[point] = -1
        position_x = (points[0, point] - origin[0]) * inverse_x
        position_y = (points[1, point] - origin[1]) * inverse_y
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
            if not (position_x >= 0.0 and position_x < num_x):
                local_x = position_x - num_x * floor(position_x / num_x)
                cell_x = int(floor(local_x))
                if cell_x > num_x - 1:
                    cell_x = num_x - 1
        else:
            if position_x < 0.0 or position_x > num_x:
                continue
            if cell_x > num_x - 1:
                cell_x = num_x - 1
        if periodic[1] == 1:
            if not (position_y >= 0.0 and position_y < num_y):
                local_y = position_y - num_y * floor(position_y / num_y)
                cell_y = int(floor(local_y))
                if cell_y > num_y - 1:
                    cell_y = num_y - 1
        else:
            if position_y < 0.0 or position_y > num_y:
                continue
            if cell_y > num_y - 1:
                cell_y = num_y - 1
        leaf = leaf_map[cell_x, cell_y]
        if leaf >= 0:
            found[point] = cell_patch[leaf]
        else:
            on_line_x = local_x == floor(local_x) and cell_x > 0
            on_line_y = local_y == floor(local_y) and cell_y > 0
            if on_line_x:
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
        lower_y = (1.0 - fraction_y) * value
        upper_y = fraction_y * value
        buffer[row, cell_corners[leaf, 0]] += (1.0 - fraction_x) * lower_y
        buffer[row, cell_corners[leaf, 1]] += fraction_x * lower_y
        buffer[row, cell_corners[leaf, 2]] += (1.0 - fraction_x) * upper_y
        buffer[row, cell_corners[leaf, 3]] += fraction_x * upper_y


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
    _deposit_chunk(
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
