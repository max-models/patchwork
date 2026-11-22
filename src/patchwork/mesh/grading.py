"""Graded axes: node positions whose spacing follows a given local cell size.

`graded_nodes` gives the base nodes of a graded axis of `AMRHierarchy` (``axis_nodes``):
cells that follow a prescribed size ``dx(x)``, for instance fine at a wall or at the
meniscus of an extraction system and coarse elsewhere, and that change in size smoothly
from one to the next.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from methodcard import method

#: The samples per cell of the target size with which the cell count is integrated.
SAMPLES_PER_CELL = 32


def _size_function(
    spacing: Callable[[np.ndarray], Any] | Sequence[Sequence[float]],
) -> Callable[[np.ndarray], np.ndarray]:
    """The cell size as a vectorised function of ``x``."""
    if callable(spacing):
        return lambda x: np.broadcast_to(np.asarray(spacing(x), dtype=float), x.shape)
    points = np.asarray(spacing, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 1:
        raise ValueError(
            "spacing must be a function of x or a sequence of (x, dx) pairs, "
            f"got an array of shape {points.shape}"
        )
    order = np.argsort(points[:, 0], kind="stable")
    where, size = points[order, 0], points[order, 1]
    if np.any(np.diff(where) <= 0.0):
        raise ValueError("the positions of the (x, dx) pairs must differ")
    return lambda x: np.interp(x, where, size)


@method(
    equations={
        "count": r"s(x) = \int_{x_0}^{x} \frac{d\xi}{\Delta x(\xi)}, \qquad N = \max\big(1, \operatorname{round} s(x_1)\big)",
        "nodes": r"s(x_i) = \frac{i}{N}\, s(x_1), \qquad i = 0, \dots, N",
    },
    assumptions=(
        r"$\Delta x > 0$; between the given $(x, \Delta x)$ pairs linear, beyond them constant",
        r"$s$ integrated by the trapezoidal rule on a fine sampling and inverted linearly; the cells follow $\Delta x$ to a factor $s(x_1) / N$, and two neighbours differ by about $\Delta x'$",
        r"with nodes required at given positions, every stretch between them on its own",
    ),
)
def graded_nodes(
    lower: float,
    upper: float,
    spacing: Callable[[np.ndarray], Any] | Sequence[Sequence[float]],
    *,
    cells: int | None = None,
    through: Sequence[float] = (),
) -> np.ndarray:
    """The nodes of ``[lower, upper]`` whose cells follow the size ``spacing``.

    ``spacing`` is the local cell size [m]: a vectorised function of ``x``, or a
    sequence of ``(x, dx)`` pairs, linear between them and constant beyond. The number
    of cells is the integral of ``1 / dx`` over the interval, rounded (at least one), or
    ``cells`` if given (the sizes are then scaled to fit). The nodes are those where that
    integral reaches whole cells, so two neighbouring cells differ by about the change
    of ``dx`` over a cell. The first and last node are ``lower`` and ``upper``; the
    result is the ``axis_nodes`` entry of an `AMRHierarchy` with that many base cells.

    ``through`` are positions that must be nodes, such as the faces of electrodes or the
    edges of apertures, so that the surfaces fall on the mesh instead of cutting cells:
    every stretch between two of them (and the ends) is graded on its own, with at least
    one cell. Not with ``cells``.
    """
    lower, upper = float(lower), float(upper)
    if not upper > lower:
        raise ValueError(f"upper ({upper}) must exceed lower ({lower})")
    pins = sorted({float(x) for x in through if lower < float(x) < upper})
    if pins:
        if cells is not None:
            raise ValueError("cells and through cannot be combined")
        ends = [lower, *pins, upper]
        parts = [
            graded_nodes(a, b, spacing)[:-1]
            for a, b in zip(ends[:-1], ends[1:], strict=True)
        ]
        return np.concatenate([*parts, [upper]])
    size = _size_function(spacing)
    # a sampling fine against the smallest size the function takes on a first one
    probe = size(np.linspace(lower, upper, 4097))
    if not np.all(probe > 0.0):
        raise ValueError("the cell size must be positive")
    samples = int(
        min(
            max(4097, SAMPLES_PER_CELL * (upper - lower) / float(probe.min()) + 1),
            4_000_001,
        )
    )
    x = np.linspace(lower, upper, samples)
    density = 1.0 / size(x)
    count = np.zeros(samples)
    count[1:] = np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(x))
    total = float(count[-1])
    if cells is None:
        cells = max(1, round(total))
    elif int(cells) != cells or cells < 1:
        raise ValueError(f"cells must be a positive integer, got {cells!r}")
    nodes = np.interp(np.linspace(0.0, total, int(cells) + 1), count, x)
    nodes[0], nodes[-1] = lower, upper
    return nodes
