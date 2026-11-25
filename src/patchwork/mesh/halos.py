"""Halo exchange between patches, and restriction from fine patches to coarse ones.

Each patch field is an mpiarray array on the patch's own ranks, with a halo of
``halo`` cells (or nodes) around it. `HaloExchange` fills that halo from
the rest of the hierarchy:

1. from the patches of the same level that the halo overlaps, across periodic
   faces too (a patch along a whole periodic axis from itself);
2. where no patch of the same level is there, by linear (1D) or bilinear (2D)
   interpolation of the level below;
3. between the ranks inside each patch, with mpiarray's own ``update_halos``.

Halo points outside the domain are left as they are: they belong to the
boundary conditions. `Restriction` goes the other way and gives the
coarse points under a finer patch the fine values: the mean of the fine cells,
or the coinciding fine node.

Both are schedules, built once (collectively) for a field and executed as often
as needed. All transport is one ``Alltoallv`` per execution
(`mpiarray.CopyPlan`) over the hierarchy's communicator.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import mpiarray as mpa
import numpy as np
from methodcard import ConvergenceProblem, method

if TYPE_CHECKING:
    from collections.abc import Sequence

    from patchwork.mesh.hierarchy import AMRHierarchy, Patch

Box = tuple[tuple[int, int], ...]


def _overlap(a: Box, b: Box) -> Box | None:
    box = tuple(
        (max(a0, b0), min(a1, b1)) for (a0, a1), (b0, b1) in zip(a, b, strict=True)
    )
    return box if all(start < end for start, end in box) else None


def _shift(box: Box, by: tuple[int, ...]) -> Box:
    return tuple((start + d, end + d) for (start, end), d in zip(box, by, strict=True))


def _frame(own: Box, width: int) -> list[Box]:
    """The halo around ``own``: two boxes per axis, two in 1D and four in 2D.

    The two boxes of an axis span the halo along the axes before it and only
    ``own`` along the axes after it, so the corners are in those of the last.
    """
    boxes = []
    for axis, (lo, hi) in enumerate(own):
        for part in ((lo - width, lo), (hi, hi + width)):
            boxes.append(
                tuple(
                    part
                    if other == axis
                    else (o0 - width, o1 + width)
                    if other < axis
                    else (o0, o1)
                    for other, (o0, o1) in enumerate(own)
                )
            )
    return boxes


class _BoxIndex:
    """The boxes of one level in bins: which of them may overlap a query box.

    A bin is ``size`` points long per axis (the mean extent of the boxes), and
    holds the boxes that touch it, so that a query visits only the boxes near
    it: building the index and querying every box once is linear in the
    number of boxes, where testing every pair is quadratic.
    """

    def _keys(self, box: Box) -> itertools.product:
        return itertools.product(
            *(
                range(lo // size, (hi - 1) // size + 1)
                for (lo, hi), size in zip(box, self._size, strict=True)
            )
        )

    def __init__(self, boxes: list[Box]) -> None:
        self._boxes = boxes
        extents = np.array(
            [[hi - lo for lo, hi in box] for box in boxes], dtype=float
        ).reshape(len(boxes), -1)
        self._size = (
            tuple(max(1, int(np.ceil(m))) for m in extents.mean(axis=0))
            if boxes
            else ()
        )
        self._bins: dict[tuple[int, ...], list[int]] = {}
        for number, box in enumerate(boxes):
            for key in self._keys(box):
                self._bins.setdefault(key, []).append(number)

    def candidates(self, *boxes: Box) -> list[int]:
        """The numbers of the boxes that may overlap one of ``boxes``, in order."""
        found: set[int] = set()
        for box in boxes:
            if any(lo >= hi for lo, hi in box):
                continue
            for key in self._keys(box):
                found.update(self._bins.get(key, ()))
        return sorted(found)


class _Points:
    """The points a field of one centering has on a hierarchy, level by level."""

    def __init__(self, hierarchy: AMRHierarchy, nodal: bool) -> None:
        self.hierarchy = hierarchy
        self.nodal = nodal
        self.ratio = hierarchy.refinement_ratio
        self.periodic = hierarchy.periodic
        self._domains: dict[int, np.ndarray] = {}
        self._wholes: dict[int, bool] = {}
        self._indices: dict[int, tuple[tuple[Patch, ...], _BoxIndex]] = {}

    def own(self, patch: Patch) -> Box:
        """The patch's points as a half-open box of level indices."""
        extra = 1 if self.nodal else 0
        return tuple((lo, hi + extra) for lo, hi in patch.box)

    def near(self, level: int, box: Box) -> list[Patch]:
        """The patches of ``level`` that may overlap ``box`` or one of its periodic images.

        In the order of `AMRHierarchy.patches_at_level`; overlaps still need
        `_overlap` (the index only rules out the patches far away).
        """
        if level not in self._indices:
            patches = self.hierarchy.patches_at_level(level)
            self._indices[level] = (
                patches,
                _BoxIndex([self.own(patch) for patch in patches]),
            )
        patches, index = self._indices[level]
        images = [_shift(box, tuple(-d for d in shift)) for shift in self.shifts(level)]
        return [patches[number] for number in index.candidates(*images)]

    def shifts(self, level: int) -> list[tuple[int, ...]]:
        """The periodic images to look in: shifts by a period on periodic axes."""
        cells = self.hierarchy.level_shape(level)
        options = [
            (-n, 0, n) if periodic else (0,)
            for n, periodic in zip(cells, self.periodic, strict=True)
        ]
        return list(itertools.product(*options))

    def _domain(self, level: int) -> np.ndarray:
        """Which cells of a level are in the domain."""
        if level not in self._domains:
            scale = self.ratio**level
            cells = np.zeros(self.hierarchy.level_shape(level), dtype=bool)
            for patch in self.hierarchy.patches_at_level(0):
                cells[tuple(slice(lo * scale, hi * scale) for lo, hi in patch.box)] = (
                    True
                )
            self._domains[level] = cells
        return self._domains[level]

    def _whole(self, level: int) -> bool:
        """Whether the domain fills the box of the level (no notch)."""
        if level not in self._wholes:
            self._wholes[level] = bool(self._domain(level).all())
        return self._wholes[level]

    def in_domain(self, level: int, *index: np.ndarray) -> np.ndarray:
        """Whether points (level indices, possibly beyond the domain) are in it.

        ``index`` holds one array per axis. A cell is if it is a cell of the
        domain; a node if one of the cells around it is.
        """
        cells = self._domain(level)
        shape = cells.shape
        if self._whole(level):  # the domain fills its box: a bounds check
            inside = np.ones(index[0].shape, dtype=bool)
            top = 1 if self.nodal else 0
            for axis, values in enumerate(index):
                if not self.periodic[axis]:
                    inside &= (values >= 0) & (values < shape[axis] + top)
            return inside
        offsets = (
            itertools.product((0, -1), repeat=len(index))
            if self.nodal
            else [(0,) * len(index)]
        )
        inside = np.zeros(index[0].shape, dtype=bool)
        for offset in offsets:
            where, valid = [], np.ones(index[0].shape, dtype=bool)
            for axis, values in enumerate(index):
                values = values + offset[axis]
                if self.periodic[axis]:
                    values = values % shape[axis]
                else:
                    valid &= (values >= 0) & (values < shape[axis])
                    values = np.clip(values, 0, shape[axis] - 1)
                where.append(values)
            inside |= valid & cells[tuple(where)]
        return inside

    def stencil(
        self, fine: np.ndarray, axis: int, level: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """The two coarse points around fine points along one axis, and their weights.

        ``level`` is the coarse level. Returns the lower coarse index and the
        weight of the upper one. Next to a wall of a non-periodic axis, where
        one of the two would lie beyond it, the two points inside are taken and
        the value is extrapolated linearly (a weight outside [0, 1]).
        """
        if self.nodal:
            position = fine / self.ratio
        else:
            position = (fine + 0.5) / self.ratio - 0.5
        lower = np.floor(position).astype(np.int64)
        count = self.hierarchy.level_shape(level)[axis] + (1 if self.nodal else 0)
        if not self.periodic[axis] and count >= 2:
            lower = np.clip(lower, 0, count - 2)
        return lower, position - lower


def _coarse_buffer(
    hierarchy: AMRHierarchy,
    points: _Points,
    patch: Patch,
    field_of: Any,
    buffer_box: Box,
    dtype: Any,
    copies: list,
) -> tuple[Any, np.ndarray]:
    """A buffer on the patch's ranks for a box of the level below, and its copies.

    Appends to ``copies`` the copies of the coarse patches (across periodic
    faces too) into the buffer. Returns the buffer (``None`` where the patch is
    not held) and which of its points a coarse patch provides.
    """
    shape = tuple(hi - lo for lo, hi in buffer_box)
    available = np.zeros(shape, dtype=bool)
    buffer = (
        mpa.zeros(shape, dtype=dtype, split=None, comm=patch.comm)
        if patch.is_local
        else None
    )
    for coarse in points.near(patch.level - 1, buffer_box):
        coarse_origin = tuple(lo for lo, _ in points.own(coarse))
        for shift in points.shifts(patch.level - 1):
            part = _overlap(buffer_box, _shift(points.own(coarse), shift))
            if part is None:
                continue
            source = _shift(
                part,
                tuple(-d - o for d, o in zip(shift, coarse_origin, strict=True)),
            )
            target = _shift(part, tuple(-lo for lo, _ in buffer_box))
            copies.append((field_of(coarse), source, buffer, target))
            available[tuple(slice(s, e) for s, e in target)] = True
    return buffer, available


def _storage_index(
    array: Any, patch: Patch, index: tuple[np.ndarray, ...]
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Which level points ``index`` (one array per axis) of ``patch`` this rank stores, and where."""
    origin = tuple(lo for lo, _ in patch.box)
    layout = array.layout
    mine = np.ones(index[0].shape, dtype=bool)
    local = []
    for values, o, (b0, b1), h in zip(
        index, origin, layout.index_bounds, layout.halo, strict=True
    ):
        values = values - o
        mine &= (values >= b0 - h) & (values < b1 + h)
        local.append(values - b0 + h)
    return mine, tuple(values[mine] for values in local)


class _Bilinear:
    """Points of an array computed as weighted sums of corners of a buffer (2 or 4)."""

    def __init__(self, array: Any, index: tuple, buffer: Any, flat: Any, weights: Any):
        self.array, self.index, self.buffer = array, index, buffer
        self.flat, self.weights = flat, weights

    def apply(self) -> None:
        values = np.asarray(self.buffer.local_with_halos).reshape(-1)[self.flat]
        self.array.local_with_halos[self.index] = (values * self.weights).sum(axis=1)


def interpolate_from_below(
    hierarchy: AMRHierarchy,
    points: _Points,
    patch: Patch,
    field_of: Any,
    array: Any,
    index: tuple[np.ndarray, ...],
    dtype: Any,
    copies: list,
) -> _Bilinear | None:
    """(Bi)linear interpolation of the level below ``patch`` at its points ``index``.

    ``index`` holds the level indices of points of ``patch`` (own or halo), one array per axis,
    ``field_of(coarse)`` the array of a patch of the level below (``None``
    where this rank does not hold it) and ``array`` the patch's own array
    (``None`` likewise). Appends to ``copies`` the copies that bring the coarse
    values into a buffer on the patch's ranks; the weights leave out coarse
    points on no patch and are renormalised. Returns what to compute after the
    copies (see `apply_interpolations`) on the ranks that hold some of
    the points, else ``None``.
    """
    if not index[0].size:
        return None
    below = patch.level - 1
    stencils = [
        points.stencil(values, axis, below) for axis, values in enumerate(index)
    ]
    buffer_box = tuple(
        (int(lower.min()), int(lower.max()) + 2) for lower, _ in stencils
    )
    shape = tuple(hi - lo for lo, hi in buffer_box)
    buffer, available = _coarse_buffer(
        hierarchy, points, patch, field_of, buffer_box, dtype, copies
    )
    # the two or four coarse points of every point, without the unavailable ones
    corners = []
    for corner in itertools.product((0, 1), repeat=len(index)):
        weight = np.ones(index[0].shape)
        at = []
        for (lower, fraction), upper, (first, _) in zip(
            stencils, corner, buffer_box, strict=True
        ):
            weight = weight * (fraction if upper else 1.0 - fraction)
            at.append(lower + upper - first)
        weight = np.where(available[tuple(at)], weight, 0.0)
        corners.append((np.ravel_multi_index(tuple(at), shape), weight))
    total = np.sum([weight for _, weight in corners], axis=0)
    keep = total > 0.0
    if not keep.any() or array is None:
        return None
    flat = np.stack([where[keep] for where, _ in corners], axis=1)
    weights = np.stack([weight[keep] / total[keep] for _, weight in corners], axis=1)
    mine, storage = _storage_index(
        array, patch, tuple(values[keep] for values in index)
    )
    if not mine.any():
        return None
    return _Bilinear(array, storage, buffer, flat[mine], weights[mine])


def _minmod(*values: np.ndarray) -> np.ndarray:
    """The value of smallest magnitude where all have one sign, else zero."""
    stacked = np.stack(values)
    same = np.all(stacked > 0, axis=0) | np.all(stacked < 0, axis=0)
    smallest = np.min(np.abs(stacked), axis=0)
    return np.where(same, np.sign(stacked[0]) * smallest, 0.0)


class _Conservative:
    """Fine cells as their coarse cell's value plus limited slopes (see below)."""

    def __init__(
        self,
        array: Any,
        index: tuple,
        buffer: Any,
        flat: Any,
        available: Any,
        offsets: Any,
    ):
        self.array, self.index, self.buffer = array, index, buffer
        self.flat, self.available, self.offsets = flat, available, offsets

    def apply(self) -> None:
        values = np.asarray(self.buffer.local_with_halos).reshape(-1)[self.flat]
        centre = values[:, 0]
        result = centre.copy()
        for axis in range(self.offsets.shape[1]):
            below, above = values[:, 1 + 2 * axis], values[:, 2 + 2 * axis]
            has_below = self.available[:, 2 * axis]
            has_above = self.available[:, 2 * axis + 1]
            up, down = above - centre, centre - below
            # monotonized central (MC) where both neighbours exist; flat where
            # one is missing, since a one-sided slope can undershoot (e.g. go
            # negative) on the side without a neighbour
            slope = np.where(
                has_below & has_above,
                _minmod(2.0 * up, 2.0 * down, 0.5 * (up + down)),
                0.0,
            )
            result += slope * self.offsets[:, axis]
        self.array.local_with_halos[self.index] = result


@method(
    equations={
        "fill": r"u_f = u_c + \sum_a \sigma_a\, \xi_a, \qquad \xi = \frac{(i \bmod r) + \frac12}{r} - \frac12",
        "slope": r"\sigma = \mathrm{minmod}\Big(2\Delta_+,\ 2\Delta_-,\ \tfrac12(\Delta_+ + \Delta_-)\Big), \qquad \sigma = 0 \text{ without both neighbours}",
        "radial_shift": r"\xi_r \to \xi_r - \frac{\Delta r \sum_k \xi_k^2}{r\, r_c} \quad (r\text{–}z)",
        "mean": r"\frac{\sum_{f \subset c} V_f u_f}{\sum_{f \subset c} V_f} = u_c",
    },
    assumptions=(
        "cell-centered fields; the volume-weighted mean of the fine cells of a coarse cell is its value",
        "no new extrema (a positive density stays positive); a linear field is reproduced away from walls",
    ),
    references=(
        "[van Leer, J. Comput. Phys. 23, 276 (1977)](https://doi.org/10.1016/0021-9991(77)90095-X): the monotonized central limiter",
    ),
)
def conservative_from_below(
    hierarchy: AMRHierarchy,
    points: _Points,
    patch: Patch,
    field_of: Any,
    array: Any,
    index: tuple[np.ndarray, ...],
    dtype: Any,
    copies: list,
) -> _Conservative | None:
    """Conservative interpolation of the level below at cells ``index`` of ``patch``.

    For cell-centered fields: each fine cell takes the value of the coarse
    cell it lies in plus a slope along each axis times its offset from that
    cell's centre. The slope is the monotonized central difference of the two
    coarse neighbours, and zero where one of them is missing (at a wall, or
    at the edge of the level below). So the fine cells of a coarse cell
    average to its value (weighted by their volumes in r–z, where the radial
    offsets are measured from the cell's volume-weighted centre), no new
    extrema appear (a positive density stays positive), and a linear field is
    reproduced away from walls (on a graded axis, where the coarse neighbours
    have the coarse cell's size: the slope is per coarse cell).
    Arguments and result as for `interpolate_from_below`.
    """
    if not index[0].size:
        return None
    ratio = points.ratio
    parent = [values // ratio for values in index]
    offsets = np.stack(
        [((values % ratio) + 0.5) / ratio - 0.5 for values in index], axis=1
    )
    if hierarchy.cylindrical:
        # In r–z the fine cells weigh with their radius: measure the radial
        # offsets from the volume-weighted centre of the coarse cell, so that the
        # volume-weighted mean of the fine cells is the coarse value.
        spacing = hierarchy.cell_sizes(patch.level - 1, 0, parent[0])
        radius = hierarchy._mid_radius(patch.level - 1, parent[0])
        squares = np.sum((((np.arange(ratio) + 0.5) / ratio) - 0.5) ** 2)
        offsets[:, 0] -= spacing * squares / (ratio * radius)
    buffer_box = tuple((int(p.min()) - 1, int(p.max()) + 2) for p in parent)
    shape = tuple(hi - lo for lo, hi in buffer_box)
    buffer, available = _coarse_buffer(
        hierarchy, points, patch, field_of, buffer_box, dtype, copies
    )
    centre = [p - first for p, (first, _) in zip(parent, buffer_box, strict=True)]
    # the coarse cell, then its neighbours below and above along each axis
    stencil = [tuple(centre)]
    for axis in range(len(index)):
        for step in (-1, 1):
            neighbour = list(centre)
            neighbour[axis] = neighbour[axis] + step
            stencil.append(tuple(neighbour))
    flat = np.stack([np.ravel_multi_index(at, shape) for at in stencil], axis=1)
    there = np.stack([available[at] for at in stencil], axis=1)
    keep = there[:, 0]  # a fine cell needs its coarse cell
    if not keep.any() or array is None:
        return None
    mine, storage = _storage_index(
        array, patch, tuple(values[keep] for values in index)
    )
    if not mine.any():
        return None
    return _Conservative(
        array,
        storage,
        buffer,
        flat[keep][mine],
        there[keep][mine][:, 1:],
        offsets[keep][mine],
    )


def apply_interpolations(interpolations: list) -> None:
    """Compute the interpolated points from their filled buffers."""
    for interpolation in interpolations:
        interpolation.apply()


def _halo_problem() -> ConvergenceProblem:
    """A smooth cell field interpolated into the halo of a level-1 patch, maximum norm."""
    from patchwork.mesh.hierarchy import AMRHierarchy

    def function(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sin(np.pi * x) * np.cos(1.3 * y)

    def error(spacing: float) -> float:
        n = round(1.0 / spacing)
        mesh = AMRHierarchy(base_shape=(n, n))
        fine = mesh.add_patch(level=1, box=((n // 2, 3 * n // 2),) * 2)
        mesh.allocate_field("u", halo=1)
        exact = {}
        for patch in mesh.patches:
            (i0, i1), (j0, j1) = patch.box
            dx, dy = mesh.spacing(patch.level)
            i, j = np.meshgrid(
                np.arange(i0 - 1, i1 + 1), np.arange(j0 - 1, j1 + 1), indexing="ij"
            )
            exact[patch.level] = function((i + 0.5) * dx, (j + 0.5) * dy)
            storage = patch.field("u").local_with_halos
            storage[...] = 0.0
            storage[1:-1, 1:-1] = exact[patch.level][1:-1, 1:-1]
        mesh.exchange_halos("u")
        halo = np.ones(exact[1].shape, dtype=bool)
        halo[1:-1, 1:-1] = False
        values = np.asarray(fine.field("u").local_with_halos)
        return float(np.abs(values - exact[1])[halo].max())

    return ConvergenceProblem(error, steps=[1 / 16, 1 / 32, 1 / 64])


@method(
    equations={
        "same_level": r"u_\mathrm{halo} = u \text{ of a patch of the same level (periodic images included)}",
        "position": r"X = \frac{i_f + \frac12}{r} - \frac12 \ \text{(cells)}, \qquad X = \frac{i_f}{r} \ \text{(nodes)}",
        "bilinear": r"u_f = \sum_{c \in \{0,1\}^d} \frac{w_c\, u_c}{\sum_{c\ \text{available}} w_c}, \qquad w_c = \prod_a \big(c_a \lambda_a + (1 - c_a)(1 - \lambda_a)\big), \quad \lambda = X - \lfloor X \rfloor",
    },
    order=2,
    assumptions=(
        "coarse points on no patch are dropped and the weights renormalised; next to a non-periodic wall the stencil is clipped to the two inside points (linear extrapolation)",
        "halo points outside the domain are left to the boundary conditions",
        "then mpiarray's `update_halos` within a patch; one `Alltoallv` per execution",
        "the order is that of the interpolated halo values of a smooth field, maximum norm",
    ),
    problem=_halo_problem,
)
def _key(patch: Patch) -> tuple:
    """A patch as its level, box and ranks: the same in the hierarchies a regrid makes."""
    return (patch.level, patch.box, patch.ranks)


class _HaloRecord:
    """The halo geometry of one patch, its fields named by patch keys.

    ``copies`` are ``(source, box, target, box)`` with each array a slot:
    ``("field", key)``, ``("coarse", key)`` (the field interpolated from) or
    ``("buffer", shape)``, the patch's buffer of coarse values; ``None`` where
    this rank does not hold the array. ``interpolation`` is the buffer's shape,
    the storage index, the buffer points and the weights of the halo points
    interpolated from the level below, or ``None``.
    """

    def __init__(self, copies: list[tuple], interpolation: tuple | None) -> None:
        self.copies = copies
        self.interpolation = interpolation


class HaloExchange:
    """Fill the halo of every patch of field ``name``; see the module docstring.

    Collective over the hierarchy's communicator, as is `execute`.
    """

    def _field(self, patch: Patch) -> Any:
        return patch.field(self._name) if patch.is_local else None

    def _coarse_field(self, patch: Patch) -> Any:
        return patch.field(self._below) if patch.is_local else None

    def _pair(
        self, points: _Points, patch: Patch, other: Patch, extended: Box
    ) -> list[tuple]:
        """The copies from ``other`` into the halo of ``patch`` (same level), found once.

        ``(source box, target box, covered slices)`` for every part, across periodic
        faces too. They depend on the two boxes only, so they are kept by the pair
        on the hierarchy and handed on by `AMRHierarchy.regrid`.
        """
        key = (patch.level, patch.box, other.box)
        found = self._pairs_known.get(key)
        if found is None:
            own = points.own(patch)
            width = self._width
            origin = tuple(lo for lo, _ in own)
            other_origin = tuple(lo for lo, _ in points.own(other))
            found = []
            for shift in points.shifts(patch.level):
                if other.box == patch.box and not any(shift):
                    continue
                theirs = _shift(points.own(other), shift)
                for frame in _frame(own, width):
                    part = _overlap(frame, theirs)
                    if part is None:
                        continue
                    source = _shift(
                        part,
                        tuple(-d - o for d, o in zip(shift, other_origin, strict=True)),
                    )
                    target = _shift(part, tuple(-o for o in origin))
                    cover = tuple(
                        slice(s - e0, t - e0)
                        for (s, t), (e0, _) in zip(part, extended, strict=True)
                    )
                    found.append((source, target, cover))
        self._pairs_used[key] = found
        return found

    def _same_level(self, points: _Points, patch: Patch, copies: list) -> np.ndarray:
        """Copies from the patches of the same level; the halo points they cover."""
        own = points.own(patch)
        width = self._width
        extended = tuple((lo - width, hi + width) for lo, hi in own)
        covered = np.zeros([hi - lo for lo, hi in extended], dtype=bool)
        covered[(slice(width, -width),) * len(own)] = True
        for other in points.near(patch.level, extended):
            for source, target, cover in self._pair(points, patch, other, extended):
                copies.append((self._field(other), source, self._field(patch), target))
                covered[cover] = True
        return covered

    def _coarse_fine(
        self,
        points: _Points,
        patch: Patch,
        covered: np.ndarray,
        dtype: Any,
        copies: list,
    ) -> None:
        """Interpolation of the level below into the halo points no patch covers."""
        own = points.own(patch)
        width = self._width
        index = tuple(
            values + lo - width
            for values, (lo, _) in zip(np.nonzero(~covered), own, strict=True)
        )
        keep = points.in_domain(patch.level, *index)
        interpolation = interpolate_from_below(
            self._hierarchy,
            points,
            patch,
            self._coarse_field,
            self._field(patch),
            tuple(values[keep] for values in index),
            dtype,
            copies,
        )
        if interpolation is not None:
            self._interpolations.append(interpolation)

    def _signatures(self, points: _Points) -> dict[int, tuple]:
        """What every patch's halo geometry depends on: itself and the patches within reach.

        The patches of its level that overlap its halo, and those of the level
        below that may hold the coarse points of its interpolation (a margin of
        one coarse cell and the stencil's), each as ``(level, box, ranks)``, in the
        order of the patches. All patches of a level are tested against all of
        the level and the one below at once, periodic images included. Keyed by
        ``id`` of the patch.
        """
        hierarchy = self._hierarchy
        width = self._width
        ratio = points.ratio
        result: dict[int, tuple] = {}

        def boxes(patches: Sequence[Patch]) -> np.ndarray:
            return np.array([points.own(p) for p in patches], dtype=np.int64).reshape(
                len(patches), hierarchy.ndim, 2
            )

        def touching(regions: np.ndarray, level: int) -> tuple[list[Patch], np.ndarray]:
            """The patches of ``level`` and which of them overlap each region."""
            candidates = list(hierarchy.patches_at_level(level))
            theirs = boxes(candidates)
            hit = np.zeros((regions.shape[0], theirs.shape[0]), dtype=bool)
            for shift in points.shifts(level):
                moved = theirs + np.asarray(shift, dtype=np.int64)[None, :, None]
                lower = np.maximum(regions[:, None, :, 0], moved[None, :, :, 0])
                upper = np.minimum(regions[:, None, :, 1], moved[None, :, :, 1])
                hit |= np.all(lower < upper, axis=-1)
            return candidates, hit

        for level in range(hierarchy.max_level + 1):
            if self._level is not None and level != self._level:
                continue
            mine = hierarchy.patches_at_level(level)
            if not mine:
                continue
            own = boxes(mine)
            extended = own + np.array([-width, width], dtype=np.int64)
            same, hit = touching(extended, level)
            below: list[Patch] = []
            under = np.zeros((len(mine), 0), dtype=bool)
            if level > 0:
                coarse = np.stack(
                    [extended[..., 0] // ratio - 1, -(-extended[..., 1] // ratio) + 2],
                    axis=-1,
                )
                below, under = touching(coarse, level - 1)
            for n, patch in enumerate(mine):
                key = (
                    _key(patch),
                    tuple(_key(same[m]) for m in np.flatnonzero(hit[n]).tolist()),
                )
                if level > 0:
                    key += (
                        tuple(
                            _key(below[m]) for m in np.flatnonzero(under[n]).tolist()
                        ),
                    )
                result[id(patch)] = key
        return result

    def _record(self, points: _Points, patch: Patch, dtype: Any) -> _HaloRecord:
        """The halo geometry of one patch, with its fields as patch keys."""
        slots: dict[int, tuple] = {}
        own_field, coarse_field = self._field, self._coarse_field

        def field(other: Patch) -> Any:
            array = own_field(other)
            if array is not None:
                slots[id(array)] = ("field", _key(other))
            return array

        def coarse(other: Patch) -> Any:
            array = coarse_field(other)
            if array is not None:
                slots[id(array)] = ("coarse", _key(other))
            return array

        copies: list[tuple] = []
        self._field, self._coarse_field = field, coarse  # type: ignore[method-assign]
        try:
            covered = self._same_level(points, patch, copies)
            interpolations: list = []
            if patch.level > 0:
                before = self._interpolations
                self._interpolations = interpolations
                try:
                    self._coarse_fine(points, patch, covered, dtype, copies)
                finally:
                    self._interpolations = before
        finally:
            del self._field, self._coarse_field  # back to the methods
        buffer = None
        interpolation = None
        if interpolations:
            (found,) = interpolations
            buffer = found.buffer
            slots[id(buffer)] = ("buffer", tuple(buffer.shape))
            interpolation = (
                tuple(buffer.shape),
                found.index,
                found.flat,
                found.weights,
            )
        else:
            for _, _, target, _ in copies:  # a buffer on a rank without points
                if target is not None and id(target) not in slots:
                    buffer = target
                    slots[id(target)] = ("buffer", tuple(target.shape))
        template = [
            (
                None if source is None else slots[id(source)],
                source_box,
                None if target is None else slots[id(target)],
                target_box,
            )
            for source, source_box, target, target_box in copies
        ]
        return _HaloRecord(template, interpolation)

    def _bind(
        self, record: _HaloRecord, patch: Patch, patches: dict, dtype: Any
    ) -> None:
        """The copies and the interpolation of a record, on this hierarchy's arrays."""
        buffer = None

        def resolve(slot: tuple | None) -> Any:
            nonlocal buffer
            if slot is None:
                return None
            kind, key = slot
            if kind == "buffer":
                if buffer is None:
                    buffer = mpa.zeros(key, dtype=dtype, split=None, comm=patch.comm)
                return buffer
            other = patches[key]
            return self._field(other) if kind == "field" else self._coarse_field(other)

        for source, source_box, target, target_box in record.copies:
            self._copies.append(
                (resolve(source), source_box, resolve(target), target_box)
            )
        if record.interpolation is not None:
            shape, index, flat, weights = record.interpolation
            if buffer is None:
                buffer = mpa.zeros(shape, dtype=dtype, split=None, comm=patch.comm)
            self._interpolations.append(
                _Bilinear(self._field(patch), index, buffer, flat, weights)
            )

    def __init__(
        self,
        hierarchy: AMRHierarchy,
        name: str,
        *,
        level: int | None = None,
        below: str | None = None,
    ) -> None:
        """Find where every halo point comes from (one ``allgather``).

        With ``level``, only for the patches of that level; with ``below``,
        interpolating field ``below`` of the level below instead of ``name``.

        The geometry of each patch's halo (which points come from which patch,
        and the interpolation weights) is kept on the hierarchy, by centering,
        width and type, and handed on by `AMRHierarchy.regrid`: a patch with the
        same box and ranks, and the same patches within reach of its halo, reuses
        it instead of finding it again.
        """
        spec = hierarchy._field_specs[name]
        self._hierarchy = hierarchy
        self._name = name
        self._below = name if below is None else below
        self._level = level
        self._width = int(spec["halo"])
        nodal = spec["centering"] == "node"
        points = _Points(hierarchy, nodal)
        self._copies: list[tuple] = []
        self._interpolations: list[_Bilinear] = []
        dtype = spec["dtype"]
        kind = ("halo", nodal, self._width, np.dtype(dtype).str, repr(spec["split"]))
        known = hierarchy._halo_records.get(kind, {})
        self._pairs_known = hierarchy._halo_records.get(("pairs", *kind), {})
        self._pairs_used: dict[tuple, list] = {}
        used: dict[tuple, _HaloRecord] = {}
        patches = {_key(patch): patch for patch in hierarchy.patches}
        signatures = self._signatures(points) if self._width else {}
        for patch in hierarchy.patches if self._width else ():
            if level is not None and patch.level != level:
                continue
            signature = signatures[id(patch)]
            record = known.get(signature)
            if record is None:
                record = self._record(points, patch, dtype)
            used[signature] = record
            self._bind(record, patch, patches, dtype)
        # keep the records of this mesh (all of them after a full build)
        hierarchy._halo_records[kind] = used if level is None else {**known, **used}
        pairs = self._pairs_used
        if level is not None:
            pairs = {**self._pairs_known, **pairs}
        hierarchy._halo_records[("pairs", *kind)] = pairs
        del self._pairs_known, self._pairs_used
        self._plan = mpa.CopyPlan(self._copies, comm=hierarchy.comm)
        del self._copies

    def execute(self) -> None:
        """Fill the halos. Collective over the hierarchy's communicator."""
        self._plan.execute()
        apply_interpolations(self._interpolations)
        if self._width:
            for patch in self._hierarchy.local_patches:
                if self._level is None or patch.level == self._level:
                    patch.field(self._name).update_halos()


@method(
    equations={
        "cells": r"u_c = \frac{\sum_{f \subset c} V_f\, u_f}{\sum_{f \subset c} V_f}",
        "nodes": r"u_c = u_f \quad \text{at the coinciding fine node}",
    },
    assumptions=(
        "finest level first, or one level; volume-weighted in $(r, z)$, a plain mean in the plane",
        "only the points of each rank's block are written",
    ),
)
class Restriction:
    """Give the coarse points under finer patches the fine values of field ``name``.

    Cell-centered: the mean of the fine cells of each coarse cell, weighted by
    their volumes in r–z (where they grow with the radius).
    Node-centered: the fine node at each coarse node. Only the points of each
    rank's block are written; the halos follow with `HaloExchange`.
    Collective over the hierarchy's communicator, as is `execute`.
    """

    def _add(
        self,
        hierarchy: AMRHierarchy,
        points: _Points,
        fine: Patch,
        copies: list,
        writes: list,
    ) -> None:
        """The copies and writes that restrict one fine patch."""
        name = self._name
        nodal = self._nodal
        ratio = self._ratio
        spec = hierarchy._field_specs[name]
        under = tuple(
            (lo // ratio, hi // ratio + (1 if nodal else 0)) for lo, hi in fine.box
        )
        for coarse in points.near(fine.level - 1, under):
            # across a periodic face too: a node on it is also the one at the start
            for shift in points.shifts(fine.level - 1):
                part = _overlap(_shift(under, shift), points.own(coarse))
                if part is None:
                    continue
                back = _shift(part, tuple(-d for d in shift))
                if nodal:
                    region = tuple((s * ratio, (e - 1) * ratio + 1) for s, e in back)
                else:
                    region = tuple((s * ratio, e * ratio) for s, e in back)
                shape = tuple(e - s for s, e in region)
                buffer = (
                    mpa.zeros(shape, dtype=spec["dtype"], split=None, comm=coarse.comm)
                    if coarse.is_local
                    else None
                )
                source = _shift(region, tuple(-lo for lo, _ in fine.box))
                copies.append(
                    (
                        fine.field(name) if fine.is_local else None,
                        source,
                        buffer,
                        tuple((0, n) for n in shape),
                    )
                )
                if coarse.is_local:
                    target = _shift(part, tuple(-lo for lo, _ in coarse.box))
                    # the fine level and first fine index weight the r–z mean
                    rows = (fine.level, region[0][0])
                    writes.append((coarse.field(name), target, buffer, rows))

    def __init__(self, hierarchy: AMRHierarchy, name: str) -> None:
        """Find where every fine region goes (one ``allgather``)."""
        spec = hierarchy._field_specs[name]
        nodal = spec["centering"] == "node"
        points = _Points(hierarchy, nodal)
        ratio = hierarchy.refinement_ratio
        self._name = name
        self._nodal = nodal
        self._ratio = ratio
        self._hierarchy = hierarchy
        # one plan per fine level, finest first: a level gets its finer values
        # before it gives them to the one below
        self._steps: list[tuple[int, Any, list[tuple]]] = []
        for level in range(hierarchy.max_level, 0, -1):
            copies: list[tuple] = []
            writes: list[tuple] = []
            for fine in hierarchy.patches_at_level(level):
                self._add(hierarchy, points, fine, copies, writes)
            plan = mpa.CopyPlan(copies, comm=hierarchy.comm)
            self._steps.append((level, plan, writes))

    def _write(self, writes: list[tuple]) -> None:
        r = self._ratio
        hierarchy = self._hierarchy
        for array, target, buffer, (level, first) in writes:
            fine = np.asarray(buffer.local_with_halos)
            if self._nodal:
                coarse = fine[(slice(None, None, r),) * fine.ndim]
            else:
                counts = [size // r for size in fine.shape]
                blocks = [part for n in counts for part in (n, r)]
                # the axes of the fine cells within each coarse cell
                within = tuple(range(1, 2 * fine.ndim, 2))
                if hierarchy.cylindrical:
                    n, m = counts
                    # the fine cells of a coarse cell differ only in their radius
                    weight = hierarchy.cell_volumes(
                        level, first + np.arange(n * r), np.zeros(1, dtype=np.int64)
                    )
                    weighted = (fine * weight[:, None]).reshape(n, r, m, r)
                    # each coarse cell: r x r fine cells, r of each radial weight
                    total = r * weight.reshape(n, r).sum(axis=1)
                    coarse = weighted.sum(axis=(1, 3)) / total[:, None]
                else:
                    coarse = fine.reshape(blocks).mean(axis=within)
            # this rank's block only: the halo is the exchange's
            layout = array.layout
            blocked = tuple(
                (max(s, b0), min(e, b1))
                for (s, e), (b0, b1) in zip(target, layout.index_bounds, strict=True)
            )
            if any(s >= e for s, e in blocked):
                continue
            storage = tuple(
                slice(s - b0 + h, e - b0 + h)
                for (s, e), (b0, _), h in zip(
                    blocked, layout.index_bounds, layout.halo, strict=True
                )
            )
            values = coarse[
                tuple(
                    slice(s - t0, e - t0)
                    for (s, e), (t0, _) in zip(blocked, target, strict=True)
                )
            ]
            array.local_with_halos[storage] = values

    def execute(self, level: int | None = None) -> None:
        """Restrict onto every level, or onto ``level`` only from the next finer one.

        Collective over the hierarchy's communicator.
        """
        for fine_level, plan, writes in self._steps:
            if level is None or fine_level == level + 1:
                plan.execute()
                self._write(writes)
