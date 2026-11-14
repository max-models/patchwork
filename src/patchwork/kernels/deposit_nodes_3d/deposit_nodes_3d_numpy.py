"""The NumPy version of the ``deposit_nodes_3d`` kernel (see ``deposit_nodes_3d_pyccel.py``)."""

from patchwork.kernels.deposit_nodes_1d.deposit_nodes_1d_numpy import deposit_nodes_any


def deposit_nodes_3d(
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
