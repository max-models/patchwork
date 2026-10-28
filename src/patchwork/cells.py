r"""Cell-centered composite linear solves on a static patch hierarchy.

:class:`CellOperator` solves :math:`-\nabla\cdot(a\nabla u) + c\,u = f` for one
value per *leaf cell* (a cell of a patch that no finer patch covers), with a
variable diffusion coefficient ``a`` (real, positive) and reaction ``c`` (real or
complex), in the Cartesian plane or in the axisymmetric r–z plane of the
hierarchy (``AMRHierarchy(geometry=...)``). In r–z it also takes the
curl-curl of an azimuthal vector component, :math:`\nabla\times(a\nabla\times
A_\phi\hat\phi) = -a(\nabla^2 - 1/r^2)A_\phi` for constant ``a``, the operator of a
magnetoquasistatic field driven by azimuthal currents (``component="azimuthal"``).

The operator is assembled from an energy, so it is symmetric (complex
symmetric, not Hermitian, for a complex ``c``):

.. math::

    E(u) = \sum_\text{faces} a_f\,c_f\,(v_f^\top u)^2 + \sum_\text{cells} c_i V_i u_i^2,

where across a face between two cells of one size :math:`v_f^\top u = w_+u_+ -
w_-u_-` and :math:`c_f` is the face area over the distance of the centres.
Across a coarse–fine face the coarse value is interpolated along the face to
the middle of the fine face, quadratically from the coarse cell and its two
neighbours along the face (linearly or not at all where they are missing), so
the face difference is consistent; the coarse cells then share that face's
coupling by the interpolation weights. This keeps the operator symmetric, which
makes an energy balance (power absorbed equals power delivered) hold to
round-off, and is second order in the mean; next to a coarse–fine edge the
error is first order, since the difference of the fine centre and the coarse
value is centred a quarter cell off the face. In 1D a face is a point, and the
coarse value is that of the coarse cell itself. ``coarse_fine="conservative"``
instead builds a fine ghost value beyond the face, quadratic through the two
fine cells and the interpolated coarse value, and adds that face flux to the
fine cell and takes it from the coarse one: second order everywhere and
conservative face by face, but not symmetric. The weights are :math:`w = 1` for a
scalar, and for an azimuthal component across a radial face :math:`w = r` (the
axial field is :math:`(1/r)\,\partial_r(rA)`) with :math:`c_f` divided by
:math:`r_f^2`. On the axis an azimuthal component vanishes: the half cell
between the axis and the first centre, where it grows linearly, adds
:math:`4\pi\,\Delta z\,a\,u^2` (without it the curl-free :math:`1/r` field is
nearly free next to the axis).

Faces of the domain take :class:`~patchwork.nodal.Dirichlet` values (at the
middle of each boundary face, half a cell from the centre) or
:class:`~patchwork.nodal.Neumann` gradients, as for
:class:`~patchwork.nodal.NodalPoisson`; periodic axes take none, nor does the
axis of a cylindrical domain. Bodies inside the domain
(:class:`~patchwork.nodal.EmbeddedDirichlet`) hold the cells whose centres they
contain at their value; across a face from a free cell to a body cell the
surface is located along the line of the centres and the free cell couples to
the surface value at its distance. A floating body (:class:`EmbeddedFloating`)
is one more unknown, its uniform value ``U``: its surface couplings enter the
energy as :math:`s\,(w\,u - w_s U)^2`, so its row is the net flux into it, set to
the body's ``source`` (the integral of ``f`` over it; zero for an isolated
conductor), and the operator stays symmetric. :func:`cell_body` makes a body of
leaf cells given by a mask, with its surface on their faces.

:meth:`CellOperator.gradient` and :meth:`CellOperator.curl` differentiate a
solution with the operator's own faces: every face gives the difference of its two
values (the weighted ones, ``w u``) over their distance, located half way between
them, with the same coarse–fine interpolation as the operator, the surfaces of the
bodies and the conditions of the domain's faces; a cell takes, along each axis,
the linear interpolation to its centre of the (face-length weighted) estimates on
its two sides. That is exact for linear fields and for the uniform axial field
``A_phi = B r / 2``, and second order on smooth ones.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
import scope_profiler as sp

from patchwork._profiling import PROFILE_TAGS
from patchwork.nodal import (
    FACES,
    INNER,
    Dirichlet,
    EmbeddedDirichlet,
    Neumann,
    _surface_crossing,
    faces_of,
)

if TYPE_CHECKING:
    from patchwork.mesh import AMRHierarchy

#: A coefficient or source on the leaf cells: a number, an array over the cells,
#: or a function of the cell-centre coordinates: ``(x,)`` in 1D, ``(x, y)`` in 2D
#: (``(r, z)`` in r–z).
CellValue = float | complex | np.ndarray | Callable[..., Any]

COMPONENTS = ("scalar", "azimuthal")


@dataclass(frozen=True)
class EmbeddedFloating:
    r"""A floating body inside the domain: one uniform value, an unknown.

    ``inside`` and ``fraction`` are those of
    :class:`~patchwork.nodal.EmbeddedDirichlet`. ``source`` is the integral of
    ``f`` over the body: the net flux the body's surface carries,
    :math:`\oint a\,\partial_n u` outward from the domain into it; zero for an
    isolated conductor (no net charge, or no net current for a magnetic vector
    potential).
    """

    inside: Callable[[np.ndarray], Any]
    source: complex = 0.0
    fraction: Callable[[np.ndarray, np.ndarray], Any] | None = None


Body = EmbeddedDirichlet | EmbeddedFloating


class CompositeCells:
    """The leaf cells of a hierarchy, their faces and the maps to its patches.

    Cells are addressed on the *finest lattice*: the cells of the finest level
    extended over the whole domain. A leaf cell of level ``L`` spans
    ``ratio**(finest - L)`` lattice cells along each axis. The arrays below
    have one column per axis (``ndim``, 1 or 2).

    Attributes:
        hierarchy: The patch hierarchy.
        patch: Index into ``hierarchy.patches`` of every leaf cell, ``(n,)``.
        index: Level indices of every leaf cell, ``(n, ndim)``.
        lattice: Finest-lattice lower corner of every leaf cell, ``(n, ndim)``.
        span: Edge length of every leaf cell on the finest lattice, ``(n,)``.
        centers: Physical cell centres, ``(n, ndim)``.
        sizes: Physical edge lengths, ``(n, ndim)``.
        volumes: Cell volumes, ``(n,)``: lengths in 1D (per unit area), areas
            in the Cartesian plane (per unit length out of it), ring volumes
            ``2 pi r dr dz`` in r–z.
        owner: The leaf cell over every finest-lattice cell, -1 outside the
            domain (``AMRHierarchy(base_boxes=...)``).
    """

    def __init__(self, hierarchy: AMRHierarchy) -> None:
        """Number the leaf cells and paint the finest lattice with them."""
        self.hierarchy = hierarchy
        ratio = hierarchy.refinement_ratio
        finest = hierarchy.max_level
        patches = hierarchy.patches
        patch_numbers, indices, spans = [], [], []
        owner = np.full(hierarchy.level_shape(finest), -1, dtype=np.int64)
        count = 0
        for number, patch in enumerate(patches):
            leaf = np.ones(patch.shape, dtype=bool)
            for finer in hierarchy.patches_at_level(patch.level + 1):
                shared = [
                    (max(f0 // ratio, i0) - i0, min(f1 // ratio, i1) - i0)
                    for (f0, f1), (i0, i1) in zip(finer.box, patch.box, strict=True)
                ]
                if all(a < b for a, b in shared):
                    leaf[tuple(slice(a, b) for a, b in shared)] = False
            where = np.nonzero(leaf)
            ids = np.full(patch.shape, -2, dtype=np.int64)
            ids[where] = count + np.arange(where[0].size)
            count += where[0].size
            scale = ratio ** (finest - patch.level)
            # coarsest first: finer patches paint over the cells they cover
            for axis in range(ids.ndim):
                ids = np.repeat(ids, scale, axis=axis)
            owner[tuple(slice(lo * scale, hi * scale) for lo, hi in patch.box)] = ids
            patch_numbers.append(np.full(where[0].size, number))
            indices.append(
                np.stack(
                    [
                        values + lo
                        for values, (lo, _) in zip(where, patch.box, strict=True)
                    ],
                    axis=1,
                )
            )
            spans.append(np.full(where[0].size, scale))
        self.owner = owner
        self.patch = np.concatenate(patch_numbers)
        self.index = np.concatenate(indices)
        self.span = np.concatenate(spans).astype(np.int64)
        self.lattice = self.index * self.span[:, None]
        self._spacing = np.asarray(hierarchy.spacing(finest), dtype=float)
        lower = np.asarray(hierarchy.lower_bound, dtype=float)
        self.sizes = self.span[:, None] * self._spacing
        self.centers = lower + (self.lattice + 0.5 * self.span[:, None]) * self._spacing
        if hierarchy.cylindrical:
            self.volumes = (
                2.0 * np.pi * self.centers[:, 0] * np.prod(self.sizes, axis=1)
            )
        else:
            self.volumes = np.prod(self.sizes, axis=1)

    @property
    def num_cells(self) -> int:
        """Number of leaf cells."""
        return int(self.span.size)

    def locate(self, points: np.ndarray) -> np.ndarray:
        """The leaf cell of every point ``(ndim, k)``; -1 outside the domain.

        Periodic axes wrap; a point on a face belongs to the cell above it, the
        upper faces of the domain to the last cell.
        """
        points = self.hierarchy.wrapped(np.asarray(points, dtype=float))
        lower = np.asarray(self.hierarchy.lower_bound, dtype=float)
        shape = np.asarray(self.owner.shape)
        lattice = np.floor((points.T - lower) / self._spacing).astype(np.int64)
        inside = np.all((lattice >= 0) & (lattice <= shape), axis=1)
        lattice = np.minimum(lattice, shape - 1)
        result = np.full(points.shape[1], -1, dtype=np.int64)
        result[inside] = self.owner[tuple(lattice[inside].T)]
        return result

    def evaluate(self, value: CellValue, dtype: Any = float) -> np.ndarray:
        """A :data:`CellValue` on the leaf cells, ``(n,)``."""
        if callable(value):
            value = value(*self.centers.T)
        result = np.asarray(value, dtype=dtype)
        return np.broadcast_to(result, (self.num_cells,)).copy()

    def _segments(self, axis: int):
        """Runs of lattice faces with one pair of cells: low, high, plane, start, count."""
        # the cells along ``axis`` first, the other axis (if any) flattened after it
        n = self.owner.shape[axis]
        own = np.moveaxis(self.owner, axis, 0).reshape(n, -1)
        m = own.shape[1]
        outside = np.full((1, m), -1, dtype=np.int64)
        # the planes between lattice cells k - 1 and k: inside, then the ends
        lows, highs, planes = [own[:-1]], [own[1:]], [np.arange(1, n)]
        if self.hierarchy.periodic[axis]:
            lows.append(own[-1:])
            highs.append(own[:1])
            planes.append(np.array([n]))
        else:
            lows += [outside, own[-1:]]
            highs += [own[:1], outside]
            planes += [np.array([0]), np.array([n])]
        low = np.concatenate([a.reshape(-1) for a in lows])
        high = np.concatenate([a.reshape(-1) for a in highs])
        plane = np.concatenate([np.repeat(p.astype(np.int64), m) for p in planes])
        tick = np.tile(np.arange(m, dtype=np.int64), plane.size // m)
        keep = (low != high) & ((low >= 0) | (high >= 0))
        low, high, plane, tick = low[keep], high[keep], plane[keep], tick[keep]
        order = np.lexsort((tick, high, low, plane))
        low, high, plane, tick = low[order], high[order], plane[order], tick[order]
        new = np.ones(low.size, dtype=bool)
        new[1:] = (
            (plane[1:] != plane[:-1])
            | (low[1:] != low[:-1])
            | (high[1:] != high[:-1])
            | (tick[1:] != tick[:-1] + 1)
        )
        first = np.nonzero(new)[0]
        count = np.diff(np.append(first, low.size))
        return low[first], high[first], plane[first], tick[first], count

    # Faces ---------------------------------------------------------------------#
    def faces(self, axis: int) -> dict[str, np.ndarray]:
        """The faces normal to ``axis`` between two leaf cells.

        Arrays over the faces: ``low`` and ``high``, the cells below and above
        it; ``plane``, its finest-lattice coordinate along ``axis`` (the
        lattice size for the wrap of a periodic axis); ``start`` and ``count``,
        its extent along the other axis in lattice cells (0 and 1 in 1D, where a
        face is a point).
        """
        low, high, plane, start, count = self._segments(axis)
        inside = (low >= 0) & (high >= 0)
        return {
            "low": low[inside],
            "high": high[inside],
            "plane": plane[inside],
            "start": start[inside],
            "count": count[inside],
        }

    def boundary(self, axis: int) -> dict[str, np.ndarray]:
        """The faces normal to ``axis`` on the boundary of the domain.

        Arrays over the faces: ``cell``, the leaf cell inside; ``side``, 0 if
        the boundary is below the cell and 1 above; ``plane``, ``start``,
        ``count`` as in :meth:`faces`; ``rectangle``, whether the face is on a
        face of the domain's rectangle (else on an edge inside it).
        """
        low, high, plane, start, count = self._segments(axis)
        outside = (low < 0) != (high < 0)
        size = self.owner.shape[axis]
        # 0: the boundary is below the cell (the cell is on the high side)
        side = (high < 0).astype(np.int64)[outside]
        plane = plane[outside]
        return {
            "cell": np.where(low < 0, high, low)[outside],
            "side": side,
            "plane": plane,
            "start": start[outside],
            "count": count[outside],
            "rectangle": (plane == 0) | (plane == size),
        }

    def face_metric(
        self, axis: int, plane: np.ndarray, start: np.ndarray, count: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Area, normal coordinate and middle of faces along the other axis.

        The area is the face length in the Cartesian plane; in r–z, ``2 pi r dz``
        for a radial face at ``r`` and ``pi (r1^2 - r0^2)`` for an axial one.
        In 1D a face is a point of area 1 (per unit area) and the middle is 0.
        """
        hierarchy = self.hierarchy
        lower = np.asarray(hierarchy.lower_bound, dtype=float)
        h = self._spacing
        position = lower[axis] + plane * h[axis]
        if hierarchy.ndim == 1:
            return np.ones(position.shape), position, np.zeros(position.shape)
        other = 1 - axis
        t0 = lower[other] + start * h[other]
        t1 = t0 + count * h[other]
        if not hierarchy.cylindrical:
            area = t1 - t0
        elif axis == 0:
            area = 2.0 * np.pi * position * (t1 - t0)
        else:
            area = np.pi * (t1**2 - t0**2)
        return area, position, 0.5 * (t0 + t1)

    # Fields on the patches -----------------------------------------------------#
    def from_patches(self, name: str) -> np.ndarray:
        """The values of the cell-centered field ``name`` on the leaf cells (collective)."""
        hierarchy = self.hierarchy
        patches = hierarchy.patches
        result = None
        for number, patch in enumerate(patches):
            values = np.asarray(hierarchy.gather(patch, name))
            mine = self.patch == number
            origin = np.array([lo for lo, _ in patch.box])
            if result is None:
                result = np.zeros(self.num_cells, dtype=values.dtype)
            result[mine] = values[tuple((self.index[mine] - origin).T)]
        assert result is not None
        return result

    def to_patches(self, values: np.ndarray, name: str) -> None:
        """Write leaf-cell values to the cell-centered field ``name`` (collective).

        The cells under finer patches get the (volume-weighted) mean of their
        fine cells (:meth:`~patchwork.mesh.AMRHierarchy.restrict`).
        """
        hierarchy = self.hierarchy
        for number, patch in enumerate(hierarchy.patches):
            if not patch.is_local:
                continue
            array = patch.field(name)
            mine = self.patch == number
            bounds = array.layout.index_bounds
            index = (self.index[mine] - np.array([lo for lo, _ in patch.box])).T
            block = np.ones(index.shape[1], dtype=bool)
            for values_, (b0, b1) in zip(index, bounds, strict=True):
                block &= (values_ >= b0) & (values_ < b1)
            local = np.asarray(array.local)
            local[
                tuple(
                    values_[block] - b0
                    for values_, (b0, _) in zip(index, bounds, strict=True)
                )
            ] = values[mine][block]
            array.local[...] = local
        hierarchy.restrict(name)


def _face_points(
    ndim: int, axis: int, position: np.ndarray, middle: np.ndarray
) -> tuple[np.ndarray, ...]:
    """The middle of faces normal to ``axis``, one array per axis: ``(x,)`` or ``(x, y)``."""
    if ndim == 1:
        return (position,)
    return (position, middle) if axis == 0 else (middle, position)


def _evaluate_complex(value: Any, *coordinates: np.ndarray) -> np.ndarray:
    """A face value at points (one array per axis), complex allowed."""
    result: Any = value(*coordinates) if callable(value) else value
    dtype = complex if np.iscomplexobj(result) else float
    return np.broadcast_to(np.asarray(result, dtype=dtype), coordinates[0].shape).copy()


def _solve(
    factorization: Any, right_hand_side: np.ndarray, complex_matrix: bool
) -> np.ndarray:
    """Solve with a real or complex LU for a real or complex right-hand side."""
    if not complex_matrix and np.iscomplexobj(right_hand_side):
        real = factorization.solve(np.ascontiguousarray(np.real(right_hand_side)))
        imag = factorization.solve(np.ascontiguousarray(np.imag(right_hand_side)))
        return real + 1j * imag
    return factorization.solve(right_hand_side)


class CellOperator:
    r"""Solve :math:`-\nabla\cdot(a\nabla u) + c\,u = f` on the leaf cells of a hierarchy.

    The operator is assembled and factorized once; :meth:`solve` then costs two
    triangular solves, and :meth:`set_reaction` refactorizes for a new ``c``
    without assembling the faces again. ``diffusion`` (``a``, real and positive)
    and ``reaction`` (``c``, real or complex) are :data:`CellValue`\ s; the face
    coefficient is the harmonic mean of its two cells (``mean="arithmetic"``
    for the plain mean). ``coarse_fine`` is ``"symmetric"`` or
    ``"conservative"``, the two treatments of coarse–fine faces of the module
    docstring. ``component`` is ``"scalar"``, or in r–z
    ``"azimuthal"`` for the curl-curl of an azimuthal vector component (see the
    module docstring).

    ``faces`` maps ``"x_lower"``, ``"x_upper"``, ``"y_lower"``, ``"y_upper"``
    (only the first two in 1D; and ``"inner"``, the edges of a domain that
    does not fill its rectangle) to a
    :class:`~patchwork.nodal.Dirichlet` or :class:`~patchwork.nodal.Neumann`
    condition; a face that is not listed is a grounded Dirichlet face. Periodic
    axes take none, and nor does the axis of a cylindrical domain
    (``"x_lower"`` when the radius starts at 0): no flux for a scalar, zero for
    an azimuthal component. A Neumann gradient is the outward normal derivative
    of ``u``; an azimuthal component takes only the natural (zero) condition on
    a radial face. ``embedded`` are bodies at fixed values
    (:class:`~patchwork.nodal.EmbeddedDirichlet`) or floating
    (:class:`EmbeddedFloating`), the last one listed wins where they overlap;
    ``min_fraction`` bounds how close to a free cell's centre their surface may
    be, as a fraction of the distance of the centres. ``diffusion`` may be zero
    on some cells (a face between two such cells then carries nothing). With
    ``factorize=False`` the LU is computed on the first :meth:`solve`, not
    before: an operator only assembled (for :attr:`stiffness`, :meth:`apply`,
    :meth:`gradient` or :meth:`curl`) never needs it, nor a condition that
    anchors it.

    Attributes:
        cells: The :class:`CompositeCells` of the hierarchy.
        solid: Whether a leaf cell is inside a body (fixed or floating).
        free: Leaf-cell numbers of the free cells; the unknowns are these, then
            one per floating body.
        stiffness: The face couplings on all leaf cells (CSR, ``n x n``), with
            the walls' and surfaces' diagonal terms; rows and columns of body
            cells are zero.
        floating_coupling: The couplings of the free cells to the floating
            bodies (CSR, free cells by floating bodies; ``-s w w_s``).
        matrix: The operator on the unknowns (CSC), scaled by the cell volumes:
            ``matrix @ u`` is ``V (-div(a grad u) + c u)`` on the free cells, and
            the net flux into each floating body.
    """

    # Conditions ----------------------------------------------------------------#
    @staticmethod
    def _face_conditions(
        hierarchy: AMRHierarchy, faces: Mapping[str, Dirichlet | Neumann] | None
    ) -> dict[str, Dirichlet | Neumann]:
        known = faces_of(hierarchy.ndim)
        unknown = set(faces or ()) - set(known) - {INNER}
        if unknown:
            raise ValueError(
                f"unknown faces {sorted(unknown)}; expected {[*known, INNER]}"
            )
        result: dict[str, Dirichlet | Neumann] = {}
        for face, (axis, side) in known.items():
            no_condition = hierarchy.periodic[axis] or (
                hierarchy.on_axis and axis == 0 and side == 0
            )
            if no_condition:
                if faces and face in faces:
                    what = "the axis" if not hierarchy.periodic[axis] else "periodic"
                    raise ValueError(f"{face} is {what}: it takes no condition")
                continue
            result[face] = (faces or {}).get(face, Dirichlet(0.0))
        result[INNER] = (faces or {}).get(INNER, Dirichlet(0.0))
        return result

    def _classify_bodies(self) -> None:
        cells = self.cells
        self._owner = np.full(cells.num_cells, -1, dtype=np.int64)
        for number, body in enumerate(self._embedded):
            inside = np.asarray(body.inside(cells.centers.T), dtype=bool)
            self._owner[inside] = number
        self.solid = self._owner >= 0
        # the column of each floating body among the extra unknowns, -1 if fixed
        self._float_column = np.full(len(self._embedded), -1, dtype=np.int64)
        floating = [
            number
            for number, body in enumerate(self._embedded)
            if isinstance(body, EmbeddedFloating)
        ]
        self._float_column[floating] = np.arange(len(floating))
        self._floating = tuple(floating)

    def _face_mean(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self._mean == "harmonic":
            total = a + b
            safe = np.where(total > 0.0, total, 1.0)
            return np.where(total > 0.0, 2.0 * a * b / safe, 0.0)
        return 0.5 * (a + b)

    def _weight(self, axis: int, cells: np.ndarray) -> np.ndarray:
        """``w`` of cells across faces normal to ``axis``.

        ``r`` for a radial face of an azimuthal component, else 1.
        """
        if self._azimuthal and axis == 0:
            return self.cells.centers[cells, 0]
        return np.ones(cells.shape)

    def _conductance(
        self,
        axis: int,
        area: np.ndarray,
        position: np.ndarray,
        distance: np.ndarray,
        middle: np.ndarray | None = None,
    ) -> np.ndarray:
        """``c`` of faces: the area over the distance.

        For a radial face of an azimuthal component, also over ``r_f r_m``: the
        face's radius and ``middle``, the mean radius of the two values it
        couples (cell centres, a wall, a surface). Then ``c (r_+ A_+ - r_- A_-)^2``
        is the energy of a uniform axial field over the dual cell exactly, and
        ``A = B r / 2`` stays a null field across coarse–fine faces too.
        """
        if self._azimuthal and axis == 0:
            radius = position if middle is None else middle
            return area / (position * radius * distance)
        return area / distance

    def _mirror(self, axis: int, side: int) -> float:
        """The sign of the mirror image across the domain's face ``side`` of ``axis``.

        ``+1`` across a face with no gradient (an azimuthal component's radial
        faces excepted, whose natural condition is no axial field) and across
        the axis for a scalar, ``-1`` across the axis for an azimuthal
        component; 0 where there is no image (a value or a gradient on the
        face).
        """
        hierarchy = self.cells.hierarchy
        if axis == 0 and side == 0 and hierarchy.on_axis:
            return -1.0 if self._azimuthal else 1.0
        name = next(face for face, (a, s) in FACES.items() if a == axis and s == side)
        condition = self._faces.get(name)
        if not isinstance(condition, Neumann) or callable(condition.gradient):
            return 0.0
        if condition.gradient != 0.0 or (self._azimuthal and axis == 0):
            return 0.0
        return 1.0

    def _interpolation(
        self, axis: int, coarse: np.ndarray, start: np.ndarray, count: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cells and weights interpolating coarse values to the middle of fine faces.

        Along the face (the other axis): quadratic through the coarse cell and
        its two neighbours of the same size, linear where one is missing (a
        finer patch, a body, a wall with a value or a gradient), the coarse
        value where both are. Across a face of the domain with no gradient
        (``Neumann(0.0)``), and across the axis of an r–z domain, the missing
        neighbour is the coarse cell's mirror image (its own value; minus it
        for an azimuthal component on the axis, which vanishes there), so a
        field symmetric about the wall, such as one that does not vary along
        it, is interpolated exactly up to the wall. In 1D the face is a point:
        the coarse cell's own value, with weight 1 (the two others are 0).
        """
        cells = self.cells
        if cells.hierarchy.ndim == 1:
            ones, zeros = np.ones(coarse.size), np.zeros(coarse.size)
            return (
                np.stack([coarse, coarse, coarse], axis=1),
                np.stack([ones, zeros, zeros], axis=1),
            )
        other = 1 - axis
        span = cells.span[coarse]
        lattice = cells.lattice[coarse]
        middle = start + 0.5 * count
        centre = lattice[:, other] + 0.5 * span
        delta = (middle - centre) / span
        neighbours = []
        for side, sign in enumerate((-1, 1)):
            spot = lattice.copy()
            spot[:, other] += sign * span
            size = cells.owner.shape[other]
            mirror = 0.0
            if self.cells.hierarchy.periodic[other]:
                spot[:, other] %= size
                inside = np.ones(coarse.size, dtype=bool)
            else:
                inside = (spot[:, other] >= 0) & (spot[:, other] < size)
                mirror = self._mirror(other, side)
            found = np.full(coarse.size, -1, dtype=np.int64)
            found[inside] = cells.owner[tuple(spot[inside].T)]
            valid = found >= 0
            valid[valid] &= (cells.span[found[valid]] == span[valid]) & np.all(
                cells.lattice[found[valid]] == spot[valid], axis=1
            )
            valid[valid] &= ~self.solid[found[valid]]
            # beyond a symmetric face of the domain: the mirror image
            factor = np.ones(coarse.size)
            if mirror:
                beyond = ~inside
                valid |= beyond
                factor[beyond] = mirror
            neighbours.append((np.where(valid & inside, found, coarse), valid, factor))
        (below, has_below, f_below), (above, has_above, f_above) = neighbours
        both = has_below & has_above
        only_below = has_below & ~has_above
        only_above = has_above & ~has_below
        w_centre = np.ones(coarse.size)
        w_below = np.zeros(coarse.size)
        w_above = np.zeros(coarse.size)
        w_below[both] = 0.5 * delta[both] * (delta[both] - 1.0)
        w_centre[both] = 1.0 - delta[both] ** 2
        w_above[both] = 0.5 * delta[both] * (delta[both] + 1.0)
        w_centre[only_below] = 1.0 + delta[only_below]
        w_below[only_below] = -delta[only_below]
        w_centre[only_above] = 1.0 - delta[only_above]
        w_above[only_above] = delta[only_above]
        return (
            np.stack([coarse, below, above], axis=1),
            np.stack([w_centre, f_below * w_below, f_above * w_above], axis=1),
        )

    def _ghost(
        self,
        axis: int,
        fine: np.ndarray,
        coarse: np.ndarray,
        fine_below: np.ndarray,
        start: np.ndarray,
        count: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Columns and coefficients of ``(w u)_ghost - (w u)_fine`` at coarse–fine faces.

        The ghost is half a fine cell beyond the face, quadratic through the next
        fine cell, the fine cell and the coarse value interpolated along the
        face (linear without the next fine cell).
        """
        cells = self.cells
        h_fine = cells.sizes[fine, axis]
        h_coarse = cells.sizes[coarse, axis]
        # normal coordinates from the face, toward the coarse side positive
        n_fine = -0.5 * h_fine
        n_inner = -1.5 * h_fine
        n_coarse = 0.5 * h_coarse
        n_ghost = 0.5 * h_fine
        coarse_cells, coarse_weights = self._interpolation(axis, coarse, start, count)

        # the next fine cell away from the face
        spot = cells.lattice[fine].copy()
        step = cells.span[fine]
        spot[:, axis] += np.where(fine_below, -step, step)
        size = cells.owner.shape[axis]
        if cells.hierarchy.periodic[axis]:
            spot[:, axis] %= size
            inside = np.ones(fine.size, dtype=bool)
        else:
            inside = (spot[:, axis] >= 0) & (spot[:, axis] < size)
        inner = np.full(fine.size, -1, dtype=np.int64)
        inner[inside] = cells.owner[tuple(spot[inside].T)]
        has_inner = inner >= 0
        has_inner[has_inner] &= (cells.span[inner[has_inner]] == step[has_inner]) & (
            np.all(cells.lattice[inner[has_inner]] == spot[has_inner], axis=1)
        )
        has_inner[has_inner] &= ~self.solid[inner[has_inner]]
        inner = np.where(has_inner, inner, fine)

        def lagrange(x, nodes, k):
            result = np.ones_like(x)
            for m, node in enumerate(nodes):
                if m != k:
                    result = result * (x - node) / (nodes[k] - node)
            return result

        quad = (n_inner, n_fine, n_coarse)
        w_inner = np.where(has_inner, lagrange(n_ghost, quad, 0), 0.0)
        w_fine = np.where(
            has_inner,
            lagrange(n_ghost, quad, 1),
            (n_coarse - n_ghost) / (n_coarse - n_fine),
        )
        w_coarse = np.where(
            has_inner,
            lagrange(n_ghost, quad, 2),
            (n_ghost - n_fine) / (n_coarse - n_fine),
        )
        # D = (w u)_ghost - (w u)_fine, over the columns inner, fine and coarse cells
        columns = np.concatenate([inner[:, None], fine[:, None], coarse_cells], axis=1)
        w = self._weight(axis, columns.reshape(-1)).reshape(columns.shape)
        coefficients = (
            np.concatenate(
                [
                    w_inner[:, None],
                    (w_fine - 1.0)[:, None],
                    w_coarse[:, None] * coarse_weights,
                ],
                axis=1,
            )
            * w
        )
        return columns, coefficients

    def _conservative_fluxes(
        self,
        axis: int,
        fine: np.ndarray,
        coarse: np.ndarray,
        fine_below: np.ndarray,
        start: np.ndarray,
        count: np.ndarray,
        strength_times_distance: np.ndarray,
        face_position: np.ndarray,
    ) -> None:
        """Coarse–fine faces as one flux, into the fine cell and out of the coarse one.

        A ghost value half a fine cell beyond the face, quadratic through the
        next fine cell, the fine cell and the coarse value interpolated along
        the face (linear without the next fine cell); the flux is its
        difference with the fine value over a fine cell.
        """
        cells = self.cells
        h_fine = cells.sizes[fine, axis]
        columns, coefficients = self._ghost(
            axis, fine, coarse, fine_below, start, count
        )
        # the face's strength with the distance of a fine cell, between the fine
        # centre and the ghost, whose mean radius is the face's own
        strength = strength_times_distance / h_fine
        if self._azimuthal and axis == 0:
            centres = 0.5 * (cells.centers[fine, 0] + cells.centers[coarse, 0])
            strength = strength * centres / face_position
        rows, cols, vals = [], [], []
        for row, sign in ((fine, -1.0), (coarse, 1.0)):
            factor = sign * strength * self._weight(axis, row)
            for k in range(columns.shape[1]):
                rows.append(row)
                cols.append(columns[:, k])
                vals.append(factor * coefficients[:, k])
        self._unsymmetric.append(
            (np.concatenate(rows), np.concatenate(cols), np.concatenate(vals))
        )

    def _fractions(self, outside: np.ndarray, inside: np.ndarray) -> np.ndarray:
        """Fraction of each centre line from an outside cell at which a body starts."""
        cells = self.cells
        hierarchy = cells.hierarchy
        result = np.ones(outside.size)
        for number, body in enumerate(self._embedded):
            mine = self._owner[inside] == number
            if not mine.any():
                continue
            start = cells.centers[outside[mine]].T
            step = cells.centers[inside[mine]].T - start
            for axis, (lower, upper) in enumerate(hierarchy.bounds):
                if hierarchy.periodic[axis]:
                    period = upper - lower
                    step[axis] -= period * np.round(step[axis] / period)
            if body.fraction is not None:
                fraction = np.asarray(body.fraction(start, start + step), dtype=float)
            else:
                fraction = _surface_crossing(
                    body.inside, start, step, hierarchy.wrapped
                )
            result[mine] = np.maximum(fraction, self._min_fraction)
        return result

    def _surfaces(
        self,
        axis: int,
        faces: dict[str, np.ndarray],
        cut: np.ndarray,
        area: np.ndarray,
        position: np.ndarray,
        distance: np.ndarray,
        diagonal: np.ndarray,
    ) -> None:
        """Faces from a free cell to a body cell: the free cell sees the surface.

        Its coupling is the face's over the distance to the surface instead of
        the next centre; for a radial face of an azimuthal component the
        surface's radius ``r_s`` is the weight of the surface value and the
        coupling is ``area / (r_f r_s d)``, as for a radial wall.
        """
        cells = self.cells
        low, high = faces["low"][cut], faces["high"][cut]
        free_low = ~self.solid[low]
        outside = np.where(free_low, low, high)
        inside = np.where(free_low, high, low)
        fraction = self._fractions(outside, inside)
        normal = fraction * distance[cut]
        kappa = self.diffusion[outside]
        if self._azimuthal and axis == 0:
            surface = cells.centers[outside, 0] + fraction * (
                cells.centers[inside, 0] - cells.centers[outside, 0]
            )
            middle = 0.5 * (cells.centers[outside, 0] + surface)
            strength = kappa * area[cut] / (position[cut] * middle * normal)
            wall_weight = surface
        else:
            strength = kappa * area[cut] / normal
            wall_weight = np.ones(outside.size)
        weight = self._weight(axis, outside)
        np.add.at(diagonal, outside, strength * weight**2)
        bodies = self._owner[inside]
        column = self._float_column[bodies]
        fixed = column < 0
        self._wall_cells.append(outside[fixed])
        self._wall_strength.append((strength * weight)[fixed])
        self._wall_weight.append(wall_weight[fixed])
        self._wall_value.append(
            lambda bodies=bodies[fixed]: np.array(
                [getattr(body, "value", 0.0) for body in self._embedded], dtype=complex
            )[bodies]
        )
        floating = ~fixed
        if floating.any():
            self._float_cells.append(outside[floating])
            self._float_cross.append((strength * weight * wall_weight)[floating])
            self._float_self.append((strength * wall_weight**2)[floating])
            self._float_body.append(column[floating])

    def _interior(
        self,
        axis: int,
        stencil_cells: list[np.ndarray],
        stencil_weights: list[np.ndarray],
        strengths: list[np.ndarray],
        diagonal: np.ndarray,
    ) -> None:
        """The faces normal to ``axis`` between two cells: stencils, or surfaces."""
        cells = self.cells
        faces = cells.faces(axis)
        low, high = faces["low"], faces["high"]
        area, position, _ = cells.face_metric(
            axis, faces["plane"], faces["start"], faces["count"]
        )
        distance = 0.5 * (cells.sizes[low, axis] + cells.sizes[high, axis])
        kappa = self._face_mean(self.diffusion[low], self.diffusion[high])
        middle = 0.5 * (cells.centers[low, axis] + cells.centers[high, axis])
        strength = kappa * self._conductance(axis, area, position, distance, middle)

        solid_low, solid_high = self.solid[low], self.solid[high]
        regular = ~solid_low & ~solid_high
        cut = solid_low ^ solid_high
        if cut.any():
            self._surfaces(axis, faces, cut, area, position, distance, diagonal)

        strength_all = strength
        low, high = low[regular], high[regular]
        strength = strength_all[regular]
        start, count = faces["start"][regular], faces["count"][regular]
        index = np.zeros((low.size, 4), dtype=np.int64)
        weight = np.zeros((low.size, 4))
        span_low, span_high = cells.span[low], cells.span[high]
        index[:, 0] = high
        weight[:, 0] = self._weight(axis, high)
        same = span_low == span_high
        index[same, 1] = low[same]
        weight[same, 1] = -self._weight(axis, low[same])
        fine_coarse = ~same
        if fine_coarse.any() and self.coarse_fine == "conservative":
            low_fine = span_low < span_high
            self._conservative_fluxes(
                axis,
                np.where(low_fine, low, high)[fine_coarse],
                np.where(low_fine, high, low)[fine_coarse],
                low_fine[fine_coarse],
                start[fine_coarse],
                count[fine_coarse],
                (strength * distance[regular])[fine_coarse],
                position[regular][fine_coarse],
            )
            keep = same
            index, weight, strength = index[keep], weight[keep], strength[keep]
        elif fine_coarse.any():
            # the finer cell's value against the coarse one interpolated to the
            # middle of the face, along it
            low_fine = span_low < span_high
            fine = np.where(low_fine, low, high)[fine_coarse]
            coarse = np.where(low_fine, high, low)[fine_coarse]
            coarse_cells, coarse_weights = self._interpolation(
                axis, coarse, start[fine_coarse], count[fine_coarse]
            )
            index[fine_coarse, 0] = fine
            weight[fine_coarse, 0] = self._weight(axis, fine)
            index[fine_coarse, 1:] = coarse_cells
            weight[fine_coarse, 1:] = -coarse_weights * self._weight(
                axis, coarse_cells.reshape(-1)
            ).reshape(coarse_cells.shape)
        stencil_cells.append(index)
        stencil_weights.append(weight)
        strengths.append(strength)

    @staticmethod
    def _face_names(axis: int, boundary: dict[str, np.ndarray]) -> np.ndarray:
        """The name of the part of the boundary every boundary face is on."""
        names = np.full(boundary["cell"].size, INNER, dtype=object)
        for face, (face_axis, side) in FACES.items():
            if face_axis == axis:
                names[boundary["rectangle"] & (boundary["side"] == side)] = face
        return names

    def _domain_boundary(self, axis: int, diagonal: np.ndarray) -> None:
        """Dirichlet and Neumann faces of the domain, and the axis."""
        cells = self.cells
        hierarchy = cells.hierarchy
        boundary = cells.boundary(axis)
        cell = boundary["cell"]
        keep = ~self.solid[cell]
        boundary = {key: value[keep] for key, value in boundary.items()}
        cell = boundary["cell"]
        if not cell.size:
            return
        area, position, middle = cells.face_metric(
            axis, boundary["plane"], boundary["start"], boundary["count"]
        )
        names = self._face_names(axis, boundary)
        on_axis = np.zeros(cell.size, dtype=bool)
        if hierarchy.on_axis and axis == 0:
            on_axis = names == "x_lower"
            if self._azimuthal:
                # the half cell between the axis and the first centre
                length = boundary["count"] * cells._spacing[1]
                np.add.at(
                    diagonal,
                    cell[on_axis],
                    self.diffusion[cell[on_axis]] * 4.0 * np.pi * length[on_axis],
                )
        half = 0.5 * cells.sizes[cell, axis]
        kappa = self.diffusion[cell]
        weight = self._weight(axis, cell)
        coordinates = _face_points(hierarchy.ndim, axis, position, middle)
        for face, condition in self._faces.items():
            mine = (names == face) & ~on_axis
            if not mine.any():
                continue
            points = tuple(values[mine] for values in coordinates)
            if isinstance(condition, Dirichlet):
                strength = kappa[mine] * self._conductance(
                    axis,
                    area[mine],
                    position[mine],
                    half[mine],
                    0.5 * (cells.centers[cell[mine], axis] + position[mine]),
                )
                np.add.at(diagonal, cell[mine], strength * weight[mine] ** 2)
                self._wall_cells.append(cell[mine])
                self._wall_strength.append(strength * weight[mine])
                self._wall_weight.append(
                    position[mine]
                    if (self._azimuthal and axis == 0)
                    else np.ones(int(mine.sum()))
                )
                self._wall_value.append(
                    lambda face=face, points=points: _evaluate_complex(
                        self._faces[face].value,  # type: ignore[union-attr]
                        *points,
                    )
                )
            else:
                if self._azimuthal and axis == 0:
                    gradient = _evaluate_complex(condition.gradient, *points)
                    if np.any(gradient != 0.0):
                        raise ValueError(
                            "an azimuthal component takes only the natural "
                            "(zero) condition on a radial face"
                        )
                    continue
                self._flux_cells.append(cell[mine])
                self._flux.append(
                    lambda face=face, points=points, k=kappa[mine], a=area[mine]: (
                        k
                        * a
                        * _evaluate_complex(
                            self._faces[face].gradient,  # type: ignore[union-attr]
                            *points,
                        )
                    )
                )

    # Assembly ------------------------------------------------------------------#
    def _assemble(self) -> None:
        """The face energy (real, symmetric) and the boundary couplings."""
        cells = self.cells
        n = cells.num_cells
        stencil_cells: list[np.ndarray] = []
        stencil_weights: list[np.ndarray] = []
        strengths: list[np.ndarray] = []
        diagonal = np.zeros(n)
        # boundary couplings: diagonal terms and the right-hand side they lift,
        # per prescribed value (filled by _update_values)
        self._wall_cells: list[np.ndarray] = []
        self._wall_strength: list[np.ndarray] = []
        self._wall_weight: list[np.ndarray] = []
        self._wall_value: list[Callable[[], np.ndarray]] = []
        self._flux_cells: list[np.ndarray] = []
        self._flux: list[Callable[[], np.ndarray]] = []
        # floating surfaces: the free cells, s w w_s, s w_s^2 and the body's column
        self._float_cells: list[np.ndarray] = []
        self._float_cross: list[np.ndarray] = []
        self._float_self: list[np.ndarray] = []
        self._float_body: list[np.ndarray] = []
        # conservative coarse-fine fluxes: rows, columns and values
        self._unsymmetric: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for axis in range(cells.hierarchy.ndim):
            self._interior(axis, stencil_cells, stencil_weights, strengths, diagonal)
            self._domain_boundary(axis, diagonal)
        if stencil_cells:
            index = np.concatenate(stencil_cells)
            weight = np.concatenate(stencil_weights)
            strength = np.concatenate(strengths)
            rows = np.concatenate([index[:, p] for p in range(4) for _ in range(4)])
            cols = np.concatenate([index[:, q] for _ in range(4) for q in range(4)])
            vals = np.concatenate(
                [
                    strength * weight[:, p] * weight[:, q]
                    for p in range(4)
                    for q in range(4)
                ]
            )
            faces = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n))
        else:
            faces = sparse.coo_matrix((n, n))
        for rows, cols, vals in self._unsymmetric:
            faces = faces + sparse.coo_matrix((vals, (rows, cols)), shape=(n, n))
        self.stiffness = sparse.csr_matrix(faces + sparse.diags(diagonal))
        self.free = np.nonzero(~self.solid)[0]
        self._stiffness_free = self.stiffness[self.free][:, self.free].tocsc()
        self._volumes_free = cells.volumes[self.free]
        self._position = np.full(n, -1, dtype=np.int64)
        self._position[self.free] = np.arange(self.free.size)
        count = len(self._floating)
        if self._float_cells:
            rows = self._position[np.concatenate(self._float_cells)]
            cols = np.concatenate(self._float_body)
            cross = -np.concatenate(self._float_cross)
            own = np.bincount(
                cols, weights=np.concatenate(self._float_self), minlength=count
            )
        else:
            rows = cols = np.zeros(0, dtype=np.int64)
            cross, own = np.zeros(0), np.zeros(count)
        if count and np.any(own <= 0.0):
            lonely = [self._floating[k] for k in np.flatnonzero(own <= 0.0)]
            raise ValueError(f"floating bodies {lonely} touch no free cell")
        self.floating_coupling = sparse.csr_matrix(
            (cross, (rows, cols)), shape=(self.free.size, count)
        )
        self._floating_self = sparse.diags(own) if count else sparse.csr_matrix((0, 0))

    @sp.profile("patchwork.cell_operator_setup", tags=PROFILE_TAGS)
    def __init__(
        self,
        hierarchy: AMRHierarchy,
        *,
        diffusion: CellValue = 1.0,
        reaction: CellValue = 0.0,
        component: str = "scalar",
        faces: Mapping[str, Dirichlet | Neumann] | None = None,
        embedded: Sequence[Body] = (),
        mean: str = "harmonic",
        coarse_fine: str = "symmetric",
        min_fraction: float = 1.0e-3,
        cells: CompositeCells | None = None,
        factorize: bool = True,
    ) -> None:
        """Assemble and factorize the operator."""
        if coarse_fine not in ("symmetric", "conservative"):
            raise ValueError(
                f"coarse_fine must be 'symmetric' or 'conservative', got {coarse_fine!r}"
            )
        self.coarse_fine = coarse_fine
        if component not in COMPONENTS:
            raise ValueError(
                f"component must be one of {COMPONENTS}, got {component!r}"
            )
        if component == "azimuthal" and not hierarchy.cylindrical:
            raise ValueError(
                "an azimuthal component needs a cylindrical hierarchy; in the "
                "Cartesian plane the curl-curl of a normal component is the scalar"
            )
        if mean not in ("harmonic", "arithmetic"):
            raise ValueError(f"mean must be 'harmonic' or 'arithmetic', got {mean!r}")
        self.cells = CompositeCells(hierarchy) if cells is None else cells
        self.component = component
        self._azimuthal = component == "azimuthal"
        self._mean = mean
        self._min_fraction = float(min_fraction)
        self._factorize = bool(factorize)
        self._embedded: tuple[Body, ...] = tuple(embedded)
        self._faces = self._face_conditions(hierarchy, faces)
        self.diffusion = self.cells.evaluate(diffusion)
        if np.any(self.diffusion < 0.0):
            raise ValueError("the diffusion coefficient must not be negative")
        self._classify_bodies()
        self._assemble()
        self.set_reaction(reaction)

    # Values ---------------------------------------------------------------------#
    def _update_values(self) -> None:
        """The share of the right-hand side the prescribed values give."""
        n = self.cells.num_cells
        constant = np.zeros(n, dtype=complex)
        for cell, strength, weight, value in zip(
            self._wall_cells,
            self._wall_strength,
            self._wall_weight,
            self._wall_value,
            strict=True,
        ):
            np.add.at(constant, cell, strength * weight * value())
        for cell, flux in zip(self._flux_cells, self._flux, strict=True):
            np.add.at(constant, cell, flux())
        self._constant = constant[self.free]
        values = np.zeros(n, dtype=complex)
        if self._embedded:
            body_values = np.array(
                [getattr(body, "value", 0.0) for body in self._embedded], dtype=complex
            )
            values[self.solid] = body_values[self._owner[self.solid]]
        self._values = values
        self._float_source = np.array(
            [self._embedded[k].source for k in self._floating],  # type: ignore[union-attr]
            dtype=complex,
        )

    def set_conditions(
        self,
        faces: Mapping[str, Dirichlet | Neumann] | None = None,
        body_values: Sequence[complex] | None = None,
    ) -> None:
        """Change prescribed values without assembling or factorizing again.

        ``faces`` replaces the values of the listed faces; each must keep its
        kind. ``body_values`` are the new values of the embedded bodies: the
        value of a fixed one, the ``source`` of a floating one.
        """
        for face, condition in (faces or {}).items():
            if face not in self._faces:
                raise ValueError(f"{face} takes no condition")
            if type(condition) is not type(self._faces[face]):
                raise ValueError(f"{face} cannot change its kind of condition")
            self._faces[face] = condition
        if body_values is not None:
            if len(body_values) != len(self._embedded):
                raise ValueError("one value per embedded body is needed")
            self._embedded = tuple(
                (
                    EmbeddedFloating(body.inside, value, body.fraction)
                    if isinstance(body, EmbeddedFloating)
                    else EmbeddedDirichlet(body.inside, value, body.fraction)  # type: ignore[arg-type]
                )
                for body, value in zip(self._embedded, body_values, strict=True)
            )
        self._update_values()

    @property
    def _num_unknowns(self) -> int:
        return int(self.free.size + len(self._floating))

    def _check_anchored(self, c: np.ndarray) -> None:
        anchored = (
            bool(self._wall_cells and any(cells.size for cells in self._wall_cells))
            or bool(np.any(c[self.free] != 0.0))
            or (self._azimuthal and self.cells.hierarchy.on_axis)
        )
        if not anchored:
            raise ValueError(
                "the operator is singular: without a Dirichlet face, a body or a "
                "reaction the solution is only known up to a constant"
            )
        if self._num_unknowns:
            empty = np.asarray(abs(self.matrix).sum(axis=1)).ravel() == 0.0
            if empty.any():
                raise ValueError(
                    f"the operator is singular: {int(empty.sum())} unknowns have "
                    "neither diffusion nor reaction"
                )

    def _factorize_matrix(self) -> None:
        self._check_anchored(self.reaction)
        if self._num_unknowns:
            self._factorization = scipy.sparse.linalg.splu(
                self.matrix, permc_spec="MMD_AT_PLUS_A"
            )

    @sp.profile("patchwork.cell_operator_factorize", tags=PROFILE_TAGS)
    def set_reaction(self, reaction: CellValue) -> None:
        """Use a new reaction coefficient ``c``: refactorize, without assembling.

        A complex ``c`` makes the operator complex symmetric.
        """
        c = self.cells.evaluate(reaction, dtype=complex)
        if np.all(np.imag(c) == 0.0):
            c = np.real(c)
        self.reaction = c
        free_block = self._stiffness_free + sparse.diags(
            c[self.free] * self._volumes_free
        )
        if self._floating:
            self.matrix = sparse.bmat(
                [
                    [free_block, self.floating_coupling],
                    [self.floating_coupling.T, self._floating_self],
                ]
            ).tocsc()
        else:
            self.matrix = free_block.tocsc()
        self._update_values()
        self._factorization = None
        if self._factorize:
            self._factorize_matrix()

    # Solving --------------------------------------------------------------------#
    @property
    def constant(self) -> np.ndarray:
        """The share of the right-hand side of the unknowns that prescribed values give."""
        return self._constant

    @sp.profile("patchwork.cell_operator_solve", tags=PROFILE_TAGS)
    def solve(
        self,
        source: CellValue | str | None = None,
        potential: str | None = None,
        *,
        integrated: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve for a source ``f``; the solution on all leaf cells, ``(n,)``.

        ``source`` is a :data:`CellValue` or the name of a cell-centered field;
        or ``integrated`` is the source already integrated over each cell,
        ``V f``. Body cells hold their values. With ``potential`` the result is
        also written to the cell-centered field of that name (allocated, complex
        if needed, on first use).
        """
        cells = self.cells
        if (source is None) == (integrated is None):
            raise ValueError("give either a source or an integrated source")
        if integrated is not None:
            load = np.asarray(integrated)
            if load.shape != (cells.num_cells,):
                raise ValueError(
                    f"expected values on the {cells.num_cells} cells, got {load.shape}"
                )
        else:
            if isinstance(source, str):
                values = cells.from_patches(source)
            else:
                values = cells.evaluate(source, dtype=complex)  # type: ignore[arg-type]
            load = values * cells.volumes
        right_hand_side = load[self.free] + self._constant
        if self._floating:
            right_hand_side = np.concatenate([right_hand_side, self._float_source])
        result = self._values.copy()
        if self._factorization is None and self._num_unknowns:
            self._factorize_matrix()
        if self._factorization is not None:
            unknowns = _solve(
                self._factorization, right_hand_side, np.iscomplexobj(self.matrix)
            )
            result[self.free] = unknowns[: self.free.size]
            for k, number in enumerate(self._floating):
                result[self._owner == number] = unknowns[self.free.size + k]
        if np.all(np.imag(result) == 0.0):
            result = np.real(result)
        if potential is not None:
            hierarchy = cells.hierarchy
            if potential not in hierarchy.field_names:
                hierarchy.allocate_field(potential, dtype=result.dtype)
            cells.to_patches(result, potential)
        return result

    def apply(self, u: np.ndarray) -> np.ndarray:
        """``V (-div(a grad u) + c u)`` on the free cells for ``u`` on all cells.

        With the prescribed values in ``u``'s body cells; the residual of a
        solution is ``apply(u) - V f - constant``.
        """
        u = np.asarray(u)
        free = self.free
        coupled = self.stiffness[free] @ u
        if self._floating:
            coupled = coupled + self.floating_coupling @ self.floating_values(u)
        return coupled + self.reaction[free] * self._volumes_free * u[free]

    def floating_values(self, u: np.ndarray) -> np.ndarray:
        """The values of the floating bodies in a solution ``u`` on all cells."""
        u = np.asarray(u)
        return np.array(
            [u[np.flatnonzero(self._owner == number)[0]] for number in self._floating],
            dtype=u.dtype,
        )

    def _boundary_estimates(self, u, axis, add, radius) -> None:
        """The estimates of the domain's faces: walls, gradients, the axis."""
        cells = self.cells
        hierarchy = cells.hierarchy
        boundary = cells.boundary(axis)
        keep = ~self.solid[boundary["cell"]]
        boundary = {key: value[keep] for key, value in boundary.items()}
        cell = boundary["cell"]
        if not cell.size:
            return
        area, position, middle = cells.face_metric(
            axis, boundary["plane"], boundary["start"], boundary["count"]
        )
        length = boundary["count"].astype(float)
        names = self._face_names(axis, boundary)
        # +1: the boundary is above the cell
        sign = np.where(boundary["side"] == 1, 1.0, -1.0)
        half = 0.5 * cells.sizes[cell, axis]
        centre = cells.centers[cell, axis]
        weighted = self._weight(axis, cell) * u[cell]
        on_axis = np.zeros(cell.size, dtype=bool)
        if hierarchy.on_axis and axis == 0:
            on_axis = names == "x_lower"
            if on_axis.any():
                c = centre[on_axis]
                if self._azimuthal:
                    # u = 0 on the axis, linear to the first centre: (1/r) d(r u)/dr
                    add(
                        cell[on_axis],
                        -0.5 * c,
                        2.0 * u[cell[on_axis]] / c,
                        length[on_axis],
                    )
                else:
                    add(cell[on_axis], -c, np.zeros(c.shape), length[on_axis])
        coordinates = _face_points(hierarchy.ndim, axis, position, middle)
        for face, condition in self._faces.items():
            mine = (names == face) & ~on_axis
            if not mine.any():
                continue
            points = tuple(values[mine] for values in coordinates)
            s, h = sign[mine], half[mine]
            if isinstance(condition, Dirichlet):
                wall = position[mine] if (self._azimuthal and axis == 0) else 1.0
                value = (
                    s
                    * (
                        wall * _evaluate_complex(condition.value, *points)
                        - weighted[mine]
                    )
                    / (h * radius(0.5 * (centre[mine] + position[mine])))
                )
                add(cell[mine], 0.5 * s * h, value, length[mine])
            elif self._azimuthal and axis == 0:
                # the natural condition: no axial field on the face
                add(cell[mine], s * h, np.zeros(h.shape), length[mine])
            else:
                gradient = _evaluate_complex(condition.gradient, *points)
                add(cell[mine], s * h, s * gradient, length[mine])

    # Derivatives -----------------------------------------------------------------#
    def face_gradients(self, u: np.ndarray, axis: int) -> dict[str, np.ndarray]:
        """Derivative estimates of ``u`` along ``axis`` from the faces normal to it.

        Every face, wall and body surface gives the difference of its two values
        over their distance: ``du/dx_axis`` for a scalar; for an azimuthal
        component across radial faces ``(1/r) d(r u)/dr`` (the axial field of
        ``u``), across axial faces ``du/dz``. Arrays over the estimates, one per
        cell each touches: ``cell``, that cell; ``offset``, the signed distance
        from its centre along ``axis`` to where the estimate is located (half
        way between the two values, on the face at a coarse–fine face of the
        conservative variant); ``value``; ``length``, the face's extent along
        the other axis (in finest-lattice cells), for averaging. Body cells get
        none. ``u`` is a solution on all leaf cells (with the bodies' values).
        """
        u = np.asarray(u)
        cells = self.cells
        parts: list[tuple[np.ndarray, ...]] = []
        dtype = complex if np.iscomplexobj(u) else float

        def add(cell, offset, value, length):
            parts.append(
                (
                    np.asarray(cell, dtype=np.int64),
                    np.asarray(offset, dtype=float),
                    np.asarray(value, dtype=dtype),
                    np.asarray(length, dtype=float),
                )
            )

        def wu(index: np.ndarray) -> np.ndarray:
            return self._weight(axis, index) * u[index]

        def radius(points: np.ndarray) -> np.ndarray:
            """The radius dividing a radial difference of an azimuthal component."""
            if self._azimuthal and axis == 0:
                return points
            return np.ones(points.shape)

        faces = cells.faces(axis)
        low, high = faces["low"], faces["high"]
        start, count = faces["start"], faces["count"]
        distance = 0.5 * (cells.sizes[low, axis] + cells.sizes[high, axis])
        regular = ~self.solid[low] & ~self.solid[high]
        span_low, span_high = cells.span[low], cells.span[high]
        same = regular & (span_low == span_high)
        mixed = regular & (span_low != span_high)
        # faces between two cells of one size
        if same.any():
            lo, hi, d = low[same], high[same], distance[same]
            middle = 0.5 * (cells.centers[lo, axis] + cells.centers[hi, axis])
            value = (wu(hi) - wu(lo)) / (d * radius(middle))
            add(lo, 0.5 * d, value, count[same])
            add(hi, -0.5 * d, value, count[same])
        # coarse–fine faces: the operator's interpolation of the coarse values
        if mixed.any():
            lo, hi, d = low[mixed], high[mixed], distance[mixed]
            low_fine = span_low[mixed] < span_high[mixed]
            fine = np.where(low_fine, lo, hi)
            coarse = np.where(low_fine, hi, lo)
            sign = np.where(low_fine, 1.0, -1.0)  # +1: the coarse cell is above
            if self.coarse_fine == "conservative":
                columns, coefficients = self._ghost(
                    axis, fine, coarse, low_fine, start[mixed], count[mixed]
                )
                h_fine = cells.sizes[fine, axis]
                face = cells.centers[fine, axis] + sign * 0.5 * h_fine
                value = (
                    sign
                    * np.sum(coefficients * u[columns], axis=1)
                    / (h_fine * radius(face))
                )
                add(fine, sign * 0.5 * h_fine, value, count[mixed])
                add(
                    coarse, -sign * 0.5 * cells.sizes[coarse, axis], value, count[mixed]
                )
            else:
                coarse_cells, coarse_weights = self._interpolation(
                    axis, coarse, start[mixed], count[mixed]
                )
                w = self._weight(axis, coarse_cells.reshape(-1)).reshape(
                    coarse_cells.shape
                )
                interpolated = np.sum(coarse_weights * w * u[coarse_cells], axis=1)
                middle = 0.5 * (cells.centers[fine, axis] + cells.centers[coarse, axis])
                value = sign * (interpolated - wu(fine)) / (d * radius(middle))
                add(fine, sign * 0.5 * d, value, count[mixed])
                add(coarse, -sign * 0.5 * d, value, count[mixed])
        # body surfaces
        cut = self.solid[low] ^ self.solid[high]
        if cut.any():
            lo, hi, d = low[cut], high[cut], distance[cut]
            free_low = ~self.solid[lo]
            outside = np.where(free_low, lo, hi)
            inside = np.where(free_low, hi, lo)
            normal = self._fractions(outside, inside) * d
            sign = np.where(free_low, 1.0, -1.0)  # +1: the surface is above
            centre = cells.centers[outside, axis]
            surface = centre + sign * normal
            wall = surface if (self._azimuthal and axis == 0) else np.ones(d.shape)
            value = (
                sign
                * (wall * u[inside] - wu(outside))
                / (normal * radius(0.5 * (centre + surface)))
            )
            add(outside, sign * 0.5 * normal, value, count[cut])
        self._boundary_estimates(u, axis, add, radius)
        if not parts:
            empty = np.zeros(0)
            return {
                "cell": empty.astype(np.int64),
                "offset": empty,
                "value": empty.astype(dtype),
                "length": empty,
            }
        return {
            key: np.concatenate([part[k] for part in parts])
            for k, key in enumerate(("cell", "offset", "value", "length"))
        }

    def _derivatives(self, u: np.ndarray) -> np.ndarray:
        """``(n, ndim)``: the estimates of :meth:`face_gradients` interpolated to the centres."""
        u = np.asarray(u)
        n = self.cells.num_cells
        ndim = self.cells.hierarchy.ndim
        dtype = complex if np.iscomplexobj(u) else float
        result = np.zeros((n, ndim), dtype=dtype)
        for axis in range(ndim):
            est = self.face_gradients(u, axis)
            sides = []
            for side in (est["offset"] < 0.0, est["offset"] > 0.0):
                cell, w = est["cell"][side], est["length"][side]
                weight = np.bincount(cell, weights=w, minlength=n)
                offset = np.bincount(cell, weights=w * est["offset"][side], minlength=n)
                value = np.bincount(
                    cell, weights=w * np.real(est["value"][side]), minlength=n
                ).astype(dtype)
                if dtype is complex:
                    value = value + 1j * np.bincount(
                        cell, weights=w * np.imag(est["value"][side]), minlength=n
                    )
                with np.errstate(invalid="ignore", divide="ignore"):
                    sides.append((weight > 0, offset / weight, value / weight))
            (has_lo, o_lo, g_lo), (has_hi, o_hi, g_hi) = sides
            both = has_lo & has_hi
            column = np.zeros(n, dtype=dtype)
            column[both] = (g_lo[both] * o_hi[both] - g_hi[both] * o_lo[both]) / (
                o_hi[both] - o_lo[both]
            )
            column[has_lo & ~has_hi] = g_lo[has_lo & ~has_hi]
            column[has_hi & ~has_lo] = g_hi[has_hi & ~has_lo]
            result[:, axis] = column
        result[self.solid] = 0.0
        return result

    def gradient(self, u: np.ndarray) -> np.ndarray:
        """The gradient of a scalar ``u`` on the leaf cells, ``(n, ndim)``; zero in bodies."""
        if self._azimuthal:
            raise ValueError("an azimuthal component has a curl, not a gradient")
        return self._derivatives(u)

    def curl(self, u: np.ndarray) -> np.ndarray:
        """The in-plane curl of the field ``u`` normal to the plane, ``(n, 2)``.

        ``curl(u z) = (d_y u, -d_x u)`` for a scalar in the Cartesian plane;
        ``curl(u phi) = (-d_z u, (1/r) d_r(r u))`` for an azimuthal component in
        r–z (the magnetic field of a vector potential). Zero in bodies. 2D only.
        """
        if self.cells.hierarchy.ndim != 2:
            raise ValueError("the in-plane curl needs a 2D hierarchy")
        derivatives = self._derivatives(u)
        if self._azimuthal:
            return np.stack([-derivatives[:, 1], derivatives[:, 0]], axis=1)
        if self.cells.hierarchy.cylindrical:
            raise ValueError("the curl of a scalar needs the Cartesian plane")
        return np.stack([derivatives[:, 1], -derivatives[:, 0]], axis=1)


def cell_body(
    cells: CompositeCells,
    mask: np.ndarray,
    value: complex = 0.0,
    *,
    floating: bool = False,
) -> EmbeddedDirichlet | EmbeddedFloating:
    """A body made of the leaf cells ``mask`` selects, with its surface on their faces.

    For a region given by cells rather than by a shape (the cells outside a
    plasma, say): a free cell next to it couples to the body's value on the
    face they share. ``value`` is the body's value, or with ``floating`` the
    ``source`` of an :class:`EmbeddedFloating` body.
    """
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != (cells.num_cells,):
        raise ValueError(f"expected a mask over the {cells.num_cells} leaf cells")

    def inside(points: np.ndarray) -> np.ndarray:
        found = cells.locate(points)
        return (found >= 0) & selected[np.maximum(found, 0)]

    def fraction(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        outer, inner = cells.locate(start), cells.locate(end)
        axis = np.argmax(np.abs(np.asarray(end) - np.asarray(start)), axis=0)
        h_out = cells.sizes[outer, axis]
        h_in = cells.sizes[inner, axis]
        return h_out / (h_out + h_in)

    if floating:
        return EmbeddedFloating(inside, value, fraction)
    return EmbeddedDirichlet(inside, value, fraction)  # type: ignore[arg-type]
