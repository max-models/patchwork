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
value is centred a quarter cell off the face. ``coarse_fine="conservative"``
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
the surface value at its distance.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
import scope_profiler as sp

from patchwork._profiling import PROFILE_TAGS
from patchwork.nodal import FACES, INNER, Dirichlet, EmbeddedDirichlet, Neumann

if TYPE_CHECKING:
    from patchwork.mesh import AMRHierarchy

#: A coefficient or source on the leaf cells: a number, an array over the cells,
#: or a function of the cell-centre coordinates ``(x, y)`` (``(r, z)`` in r–z).
CellValue = float | complex | np.ndarray | Callable[[np.ndarray, np.ndarray], Any]

COMPONENTS = ("scalar", "azimuthal")


class CompositeCells:
    """The leaf cells of a hierarchy, their faces and the maps to its patches.

    Cells are addressed on the *finest lattice*: the cells of the finest level
    extended over the whole domain. A leaf cell of level ``L`` spans
    ``ratio**(finest - L)`` lattice cells along each axis.

    Attributes:
        hierarchy: The patch hierarchy.
        patch: Index into ``hierarchy.patches`` of every leaf cell, ``(n,)``.
        index: Level indices of every leaf cell, ``(n, 2)``.
        lattice: Finest-lattice lower corner of every leaf cell, ``(n, 2)``.
        span: Edge length of every leaf cell on the finest lattice, ``(n,)``.
        centers: Physical cell centres, ``(n, 2)``.
        sizes: Physical edge lengths, ``(n, 2)``.
        volumes: Cell volumes, ``(n,)``: areas in the Cartesian plane (per unit
            length out of it), ring volumes ``2 pi r dr dz`` in r–z.
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
            (i0, i1), (j0, j1) = patch.box
            leaf = np.ones(patch.shape, dtype=bool)
            for finer in hierarchy.patches_at_level(patch.level + 1):
                (f0, f1), (g0, g1) = finer.box
                a0, a1 = max(f0 // ratio, i0), min(f1 // ratio, i1)
                b0, b1 = max(g0 // ratio, j0), min(g1 // ratio, j1)
                if a0 < a1 and b0 < b1:
                    leaf[a0 - i0 : a1 - i0, b0 - j0 : b1 - j0] = False
            ii, jj = np.nonzero(leaf)
            ids = np.full(patch.shape, -2, dtype=np.int64)
            ids[ii, jj] = count + np.arange(ii.size)
            count += ii.size
            scale = ratio ** (finest - patch.level)
            # coarsest first: finer patches paint over the cells they cover
            owner[i0 * scale : i1 * scale, j0 * scale : j1 * scale] = np.repeat(
                np.repeat(ids, scale, axis=0), scale, axis=1
            )
            patch_numbers.append(np.full(ii.size, number))
            indices.append(np.stack([ii + i0, jj + j0], axis=1))
            spans.append(np.full(ii.size, scale))
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

    def evaluate(self, value: CellValue, dtype: Any = float) -> np.ndarray:
        """A :data:`CellValue` on the leaf cells, ``(n,)``."""
        if callable(value):
            value = value(self.centers[:, 0], self.centers[:, 1])
        result = np.asarray(value, dtype=dtype)
        return np.broadcast_to(result, (self.num_cells,)).copy()

    # Faces ---------------------------------------------------------------------#
    def faces(self, axis: int) -> dict[str, np.ndarray]:
        """The faces normal to ``axis`` between two leaf cells.

        Arrays over the faces: ``low`` and ``high``, the cells below and above
        it; ``plane``, its finest-lattice coordinate along ``axis`` (the
        lattice size for the wrap of a periodic axis); ``start`` and ``count``,
        its extent along the other axis in lattice cells.
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

    def _segments(self, axis: int):
        """Runs of lattice faces with one pair of cells: low, high, plane, start, count."""
        own = self.owner if axis == 0 else self.owner.T
        n, m = own.shape
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

    def face_metric(
        self, axis: int, plane: np.ndarray, start: np.ndarray, count: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Area, normal coordinate and middle of faces along the other axis.

        The area is the face length in the Cartesian plane; in r–z, ``2 pi r dz``
        for a radial face at ``r`` and ``pi (r1^2 - r0^2)`` for an axial one.
        """
        hierarchy = self.hierarchy
        other = 1 - axis
        lower = np.asarray(hierarchy.lower_bound, dtype=float)
        h = self._spacing
        position = lower[axis] + plane * h[axis]
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
            (i0, _), (j0, _) = patch.box
            if result is None:
                result = np.zeros(self.num_cells, dtype=values.dtype)
            result[mine] = values[self.index[mine, 0] - i0, self.index[mine, 1] - j0]
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
            (b0, b1), (c0, c1) = array.layout.index_bounds
            (i0, _), (j0, _) = patch.box
            mine = self.patch == number
            i = self.index[mine, 0] - i0
            j = self.index[mine, 1] - j0
            block = (i >= b0) & (i < b1) & (j >= c0) & (j < c1)
            local = np.asarray(array.local)
            local[i[block] - b0, j[block] - c0] = values[mine][block]
            array.local[...] = local
        hierarchy.restrict(name)


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

    ``faces`` maps ``"x_lower"``, ``"x_upper"``, ``"y_lower"``, ``"y_upper"`` (and
    ``"inner"``, the edges of a domain that does not fill its rectangle) to a
    :class:`~patchwork.nodal.Dirichlet` or :class:`~patchwork.nodal.Neumann`
    condition; a face that is not listed is a grounded Dirichlet face. Periodic
    axes take none, and nor does the axis of a cylindrical domain
    (``"x_lower"`` when the radius starts at 0): no flux for a scalar, zero for
    an azimuthal component. A Neumann gradient is the outward normal derivative
    of ``u``; an azimuthal component takes only the natural (zero) condition on
    a radial face. ``embedded`` are bodies at fixed values
    (:class:`~patchwork.nodal.EmbeddedDirichlet`, the last one listed wins);
    ``min_fraction`` bounds how close to a free cell's centre their surface may
    be, as a fraction of the distance of the centres.

    Attributes:
        cells: The :class:`CompositeCells` of the hierarchy.
        solid: Whether a leaf cell is inside a body (fixed at its value).
        free: Leaf-cell numbers of the unknowns.
        matrix: The operator on the unknowns (CSC), scaled by the cell volumes:
            ``matrix @ u`` is ``V (-div(a grad u) + c u)``.
    """

    @sp.profile("patchwork.cell_operator_setup", tags=PROFILE_TAGS)
    def __init__(
        self,
        hierarchy: AMRHierarchy,
        *,
        diffusion: CellValue = 1.0,
        reaction: CellValue = 0.0,
        component: str = "scalar",
        faces: Mapping[str, Dirichlet | Neumann] | None = None,
        embedded: Sequence[EmbeddedDirichlet] = (),
        mean: str = "harmonic",
        coarse_fine: str = "symmetric",
        min_fraction: float = 1.0e-3,
        cells: CompositeCells | None = None,
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
        self._embedded = tuple(embedded)
        self._faces = self._face_conditions(hierarchy, faces)
        self.diffusion = self.cells.evaluate(diffusion)
        if np.any(self.diffusion <= 0.0):
            raise ValueError("the diffusion coefficient must be positive")
        self._classify_bodies()
        self._assemble()
        self.set_reaction(reaction)

    # Conditions ----------------------------------------------------------------#
    @staticmethod
    def _face_conditions(
        hierarchy: AMRHierarchy, faces: Mapping[str, Dirichlet | Neumann] | None
    ) -> dict[str, Dirichlet | Neumann]:
        unknown = set(faces or ()) - set(FACES) - {INNER}
        if unknown:
            raise ValueError(
                f"unknown faces {sorted(unknown)}; expected {[*FACES, INNER]}"
            )
        result: dict[str, Dirichlet | Neumann] = {}
        for face, (axis, side) in FACES.items():
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

    def _face_mean(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self._mean == "harmonic":
            return 2.0 * a * b / (a + b)
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
        # conservative coarse-fine fluxes: rows, columns and values
        self._unsymmetric: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for axis in (0, 1):
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
        inner[inside] = cells.owner[spot[inside, 0], spot[inside, 1]]
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

    def _interpolation(
        self, axis: int, coarse: np.ndarray, start: np.ndarray, count: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cells and weights interpolating coarse values to the middle of fine faces.

        Along the face (the other axis): quadratic through the coarse cell and
        its two neighbours of the same size, linear where one is missing (a
        wall, a finer patch, a body), the coarse value where both are.
        """
        cells = self.cells
        other = 1 - axis
        span = cells.span[coarse]
        lattice = cells.lattice[coarse]
        middle = start + 0.5 * count
        centre = lattice[:, other] + 0.5 * span
        delta = (middle - centre) / span
        neighbours = []
        for sign in (-1, 1):
            spot = lattice.copy()
            spot[:, other] += sign * span
            size = cells.owner.shape[other]
            if self.cells.hierarchy.periodic[other]:
                spot[:, other] %= size
                inside = np.ones(coarse.size, dtype=bool)
            else:
                inside = (spot[:, other] >= 0) & (spot[:, other] < size)
            found = np.full(coarse.size, -1, dtype=np.int64)
            found[inside] = cells.owner[spot[inside, 0], spot[inside, 1]]
            valid = found >= 0
            valid[valid] &= (cells.span[found[valid]] == span[valid]) & np.all(
                cells.lattice[found[valid]] == spot[valid], axis=1
            )
            valid[valid] &= ~self.solid[found[valid]]
            neighbours.append((np.where(valid, found, coarse), valid))
        (below, has_below), (above, has_above) = neighbours
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
            np.stack([w_centre, w_below, w_above], axis=1),
        )

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
        self._wall_cells.append(outside)
        self._wall_strength.append(strength * weight)
        self._wall_weight.append(wall_weight)
        self._wall_value.append(
            lambda bodies=bodies: np.array(
                [body.value for body in self._embedded], dtype=complex
            )[bodies]
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
                lo = np.zeros(start.shape[1])
                hi = np.ones(start.shape[1])
                for _ in range(48):
                    mid = 0.5 * (lo + hi)
                    hit = np.asarray(
                        body.inside(_wrapped(hierarchy, start + mid * step)), dtype=bool
                    )
                    hi = np.where(hit, mid, hi)
                    lo = np.where(hit, lo, mid)
                fraction = hi
            result[mine] = np.maximum(fraction, self._min_fraction)
        return result

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
        names = np.full(cell.size, INNER, dtype=object)
        for face, (face_axis, side) in FACES.items():
            if face_axis == axis:
                names[boundary["rectangle"] & (boundary["side"] == side)] = face
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
        coordinates = (
            (position, middle) if axis == 0 else (middle, position)
        )  # the middle of each face, (x, y)
        for face, condition in self._faces.items():
            mine = (names == face) & ~on_axis
            if not mine.any():
                continue
            x, y = coordinates[0][mine], coordinates[1][mine]
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
                    lambda face=face, x=x, y=y: _evaluate_complex(
                        self._faces[face].value,  # type: ignore[union-attr]
                        x,
                        y,
                    )
                )
            else:
                if self._azimuthal and axis == 0:
                    gradient = _evaluate_complex(condition.gradient, x, y)
                    if np.any(gradient != 0.0):
                        raise ValueError(
                            "an azimuthal component takes only the natural "
                            "(zero) condition on a radial face"
                        )
                    continue
                self._flux_cells.append(cell[mine])
                self._flux.append(
                    lambda face=face, x=x, y=y, k=kappa[mine], a=area[mine]: (
                        k
                        * a
                        * _evaluate_complex(
                            self._faces[face].gradient,  # type: ignore[union-attr]
                            x,
                            y,
                        )
                    )
                )

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
                [body.value for body in self._embedded], dtype=complex
            )
            values[self.solid] = body_values[self._owner[self.solid]]
        self._values = values

    def set_conditions(
        self,
        faces: Mapping[str, Dirichlet | Neumann] | None = None,
        body_values: Sequence[complex] | None = None,
    ) -> None:
        """Change prescribed values without assembling or factorizing again.

        ``faces`` replaces the values of the listed faces; each must keep its
        kind. ``body_values`` are the new values of the embedded bodies.
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
                EmbeddedDirichlet(body.inside, value, body.fraction)  # type: ignore[arg-type]
                for body, value in zip(self._embedded, body_values, strict=True)
            )
        self._update_values()

    @sp.profile("patchwork.cell_operator_factorize", tags=PROFILE_TAGS)
    def set_reaction(self, reaction: CellValue) -> None:
        """Use a new reaction coefficient ``c``: refactorize, without assembling.

        A complex ``c`` makes the operator complex symmetric.
        """
        c = self.cells.evaluate(reaction, dtype=complex)
        if np.all(np.imag(c) == 0.0):
            c = np.real(c)
        self.reaction = c
        self.matrix = (
            self._stiffness_free + sparse.diags(c[self.free] * self._volumes_free)
        ).tocsc()
        self._update_values()
        self._check_anchored(c)
        self._factorization = None
        if self.free.size:
            self._factorization = scipy.sparse.linalg.splu(
                self.matrix, permc_spec="MMD_AT_PLUS_A"
            )

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
        result = self._values.copy()
        if self._factorization is not None:
            result[self.free] = _solve(
                self._factorization, right_hand_side, np.iscomplexobj(self.matrix)
            )
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
        return coupled + self.reaction[free] * self._volumes_free * u[free]


def _wrapped(hierarchy: AMRHierarchy, points: np.ndarray) -> np.ndarray:
    """``points`` ``(2, k)`` moved into the domain along its periodic axes."""
    if not any(hierarchy.periodic):
        return points
    points = points.copy()
    for axis, (lower, upper) in enumerate(hierarchy.bounds):
        if hierarchy.periodic[axis]:
            points[axis] = lower + (points[axis] - lower) % (upper - lower)
    return points


def _evaluate_complex(value: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A face value at points, complex allowed."""
    result = value(x, y) if callable(value) else value
    dtype = complex if np.iscomplexobj(result) else float
    return np.broadcast_to(np.asarray(result, dtype=dtype), x.shape).copy()


def _solve(
    factorization: Any, right_hand_side: np.ndarray, complex_matrix: bool
) -> np.ndarray:
    """Solve with a real or complex LU for a real or complex right-hand side."""
    if not complex_matrix and np.iscomplexobj(right_hand_side):
        real = factorization.solve(np.ascontiguousarray(np.real(right_hand_side)))
        imag = factorization.solve(np.ascontiguousarray(np.imag(right_hand_side)))
        return real + 1j * imag
    return factorization.solve(right_hand_side)
