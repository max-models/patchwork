"""The current of charges moving along straight paths on a 1D composite mesh.

The Pyccel version (CPU) of the ``deposit_paths_1d`` kernel; ``deposit_paths_1d_numpy.py``
next to this file is the NumPy version, with the same arguments. Loaded by
`patchwork.kernels.deposit_paths_1d`, compiled with Pyccel on first use; called by
`CompositeEdges.deposit_paths` on a line.

Every charge goes from ``start`` to ``end`` (``(1, n)``, not wrapped into the domain on
a periodic axis). Its path is cut at every node of the finest lattice; each piece lies
in one leaf cell, found from the piece's midpoint (``leaf_map``, ``(nx,)``, -1 outside
the domain), whose single edge ``cell_edge[leaf]`` gets the charge times the piece's
length in units of the cell (``cell_span``, in cells of the finest lattice). A piece
beyond a non-periodic face deposits nothing; a NaN path deposits nothing.
"""

from math import floor


def deposit_paths_1d(
    start: "float[:,:]",
    end: "float[:,:]",
    charges: "float[:]",
    origin: "float[:]",
    spacing: "float[:]",
    cells: "int[:]",
    periodic: "int[:]",
    leaf_map: "int[:]",
    cell_span: "int[:]",
    cell_edge: "int[:]",
    current: "float[:]",
) -> None:
    """Add the current of every path to ``current`` (see the module)."""
    num_x = cells[0]
    for path in range(start.shape[1]):
        a = (start[0, path] - origin[0]) / spacing[0]
        b = (end[0, path] - origin[0]) / spacing[0]
        if not (a > -1.0e300 and a < 1.0e300 and b > -1.0e300 and b < 1.0e300):
            continue
        sign = 1.0
        low = a
        high = b
        if b < a:
            sign = -1.0
            low = b
            high = a
        value = sign * charges[path]
        piece_low = low
        while piece_low < high:
            piece_high = floor(piece_low) + 1.0
            if piece_high > high:
                piece_high = high
            middle = 0.5 * (piece_low + piece_high)
            cell = int(floor(middle))
            inside = True
            if periodic[0] == 1:
                if cell < 0 or cell >= num_x:
                    cell = cell % num_x
            elif middle < 0.0 or middle > num_x:
                inside = False
            if inside:
                leaf = leaf_map[cell]
                if leaf >= 0:
                    current[cell_edge[leaf]] += (
                        value * (piece_high - piece_low) / cell_span[leaf]
                    )
            piece_low = piece_high
