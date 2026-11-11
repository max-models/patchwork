"""The conditions on a hierarchy's boundary and on bodies inside it.

The faces of the domain's bounding box (`faces_of`), the conditions on them
(`Dirichlet`, `Neumann`), and the bodies inside the domain (`EmbeddedDirichlet`,
`EmbeddedFloating`) with the location of their surfaces on mesh edges
(`_surface_crossing`). The spaces (`patchwork.spaces.nodes`,
`patchwork.spaces.cells`) number their unknowns against them; the solvers
(`patchwork.solvers`) build their equations from them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from methodcard import method

#: The faces of the domain's bounding rectangle, as ``(axis, side)`` with side 0
#: the lower one. A 1D domain has the two ``x`` faces (see `faces_of`).
FACES = {"x_lower": (0, 0), "x_upper": (0, 1), "y_lower": (1, 0), "y_upper": (1, 1)}


def faces_of(ndim: int) -> dict[str, tuple[int, int]]:
    """The faces of the bounding box of a domain of ``ndim`` dimensions."""
    return {face: where for face, where in FACES.items() if where[0] < ndim}


#: The rest of the boundary of a domain that does not fill its rectangle
#: (``AMRHierarchy(base_boxes=...)``): the edges inside the rectangle.
INNER = "inner"

Value = float | Callable[..., Any]


@dataclass(frozen=True)
class Dirichlet:
    """A prescribed potential on a domain face: a number, ``value(x)`` in 1D or ``value(x, y)``."""

    value: Value = 0.0


@dataclass(frozen=True)
class Neumann:
    """A prescribed outward normal derivative on a domain face.

    ``gradient`` is a number or ``gradient(x, y)`` (``gradient(x)`` in 1D),
    evaluated at the middle of every cell edge on the face (at the face's node
    in 1D); zero is the natural condition.
    """

    gradient: Value = 0.0


@dataclass(frozen=True)
class EmbeddedDirichlet:
    """A body inside the domain held at one potential.

    ``inside(points)`` takes coordinates ``(ndim, n)`` inside the domain and returns
    which of them the body contains. Nodes inside are fixed at ``value``. On a
    mesh edge from a node outside to a node inside, the surface is located by
    bisection of ``inside`` (or by ``fraction(outside, inside)``, the fraction
    of the edge from the outside node at which the surface crosses it; across a
    periodic face ``inside`` is then the node's image beyond the face), and the
    equation of the outside node uses the distance to the surface instead of
    the mesh spacing: the Shortley-Weller stencil at nodes with four equal
    cells around them, a two-point flux to the surface elsewhere.
    """

    inside: Callable[[np.ndarray], Any]
    value: float = 0.0
    fraction: Callable[[np.ndarray, np.ndarray], Any] | None = None


def _evaluate(value: Value, *coordinates: np.ndarray) -> np.ndarray:
    """A face value at points (one array per axis): a number, or the result of a callable."""
    result = value(*coordinates) if callable(value) else value
    return np.broadcast_to(np.asarray(result, dtype=float), coordinates[0].shape).copy()


#: The halvings of an edge that locate an embedded surface on it: the fraction
#: is known to 2**-48 of the edge.
_SURFACE_BISECTIONS = 48

#: The cost of a call of a body's ``inside`` in points: a call with this many
#: points costs about twice an empty one. `_surface_crossing` evaluates
#: several bisection steps in one call while the points are cheaper than the call.
_POINTS_PER_CALL = 10_000


def _section_bits(count: int) -> int:
    """The bisection steps per call of ``inside`` for ``count`` edges: a divisor of 48."""
    return min(
        (1, 2, 3, 4, 6, 8),
        key=lambda bits: (
            (_SURFACE_BISECTIONS // bits) * (_POINTS_PER_CALL + count * (2**bits - 1))
        ),
    )


def _surface_crossing(
    inside: Callable[[np.ndarray], Any],
    start: np.ndarray,
    step: np.ndarray,
    wrap: Callable[[np.ndarray], np.ndarray] = lambda points: points,
) -> np.ndarray:
    """Fraction of each segment ``start + t * step`` at which it enters a body.

    ``start`` and ``step`` are ``(ndim, k)``; ``start`` is outside the body,
    ``start + step`` inside. The result is that of `_SURFACE_BISECTIONS`
    bisections of ``inside`` (the upper end of the last interval), but each
    call of ``inside`` evaluates the points of several bisection steps at once
    (all ``2**b - 1`` points that ``b`` steps might visit), which are then
    replayed: the same points and the same answer in a few calls instead of 48.
    ``wrap`` moves the points into the domain (along its periodic axes).
    """
    count = start.shape[1]
    bits = _section_bits(count)
    pieces = 2**bits
    low = np.zeros(count)
    high = np.ones(count)
    # the dyadic points of one round, exact in floating point: low + j / 2**r
    grid = np.arange(1, pieces) / pieces
    rows = np.arange(count)
    for _ in range(_SURFACE_BISECTIONS // bits):
        width = high - low
        middle = low[:, None] + width[:, None] * grid  # (k, pieces - 1)
        points = start[:, :, None] + middle[None] * step[:, :, None]
        hit = np.asarray(inside(wrap(points.reshape(start.shape[0], -1))), dtype=bool)
        # the ends of the interval: outside at 0, inside at ``pieces``
        hit = np.concatenate(
            (
                np.zeros((count, 1), dtype=bool),
                hit.reshape(count, pieces - 1),
                np.ones((count, 1), dtype=bool),
            ),
            axis=1,
        )
        lower = np.zeros(count, dtype=np.int64)
        upper = np.full(count, pieces, dtype=np.int64)
        for _ in range(bits):
            half = (lower + upper) // 2
            found = hit[rows, half]
            upper = np.where(found, half, upper)
            lower = np.where(found, lower, half)
        low, high = low + lower * (width / pieces), low + upper * (width / pieces)
    return high


@method(
    equations={
        "energy": r"E \mathrel{+}= s\,(w\,u - w_s U)^2 \quad \text{per surface coupling } s",
        "net_flux": r"\oint_{\partial B} a\,\partial_n u\, dA = S, \qquad U \text{ uniform over } B",
    },
    assumptions=(
        r"$S$ = `source`, the integral of $f$ over the body; zero for an isolated conductor",
        "the operator stays symmetric; a floating body does not anchor the solution",
    ),
)
@dataclass(frozen=True)
class EmbeddedFloating:
    r"""A floating body inside the domain: one uniform value, an unknown.

    ``inside`` and ``fraction`` are those of
    `EmbeddedDirichlet`. ``source`` is the integral of
    ``f`` over the body: the net flux the body's surface carries,
    $\oint a\,\partial_n u$ outward from the domain into it; zero for an
    isolated conductor (no net charge, or no net current for a magnetic vector
    potential).
    """

    inside: Callable[[np.ndarray], Any]
    source: complex = 0.0
    fraction: Callable[[np.ndarray, np.ndarray], Any] | None = None


Body = EmbeddedDirichlet | EmbeddedFloating
