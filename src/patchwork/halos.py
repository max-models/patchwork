"""Halo exchange between patches, and restriction from fine patches to coarse ones.

Each patch field is an mpiarray array on the patch's own ranks, with a halo of
``halo`` cells (or nodes) around it. :class:`HaloExchange` fills that halo from
the rest of the hierarchy:

1. from the patches of the same level that the halo overlaps, across periodic
   faces too (a patch along a whole periodic axis from itself);
2. where no patch of the same level is there, by linear (1D) or bilinear (2D)
   interpolation of the level below;
3. between the ranks inside each patch, with mpiarray's own ``update_halos``.

Halo points outside the domain are left as they are: they belong to the
boundary conditions. :class:`Restriction` goes the other way and gives the
coarse points under a finer patch the fine values: the mean of the fine cells,
or the coinciding fine node.

Both are schedules, built once (collectively) for a field and executed as often
as needed. All transport is one ``Alltoallv`` per execution
(:class:`mpiarray.CopyPlan`) over the hierarchy's communicator.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import mpiarray as mpa
import numpy as np

if TYPE_CHECKING:
    from patchwork.mesh import AMRHierarchy, Patch

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


class _Points:
    """The points a field of one centering has on a hierarchy, level by level."""

    def __init__(self, hierarchy: AMRHierarchy, nodal: bool) -> None:
        self.hierarchy = hierarchy
        self.nodal = nodal
        self.ratio = hierarchy.refinement_ratio
        self.periodic = hierarchy.periodic
        self._domains: dict[int, np.ndarray] = {}

    def own(self, patch: Patch) -> Box:
        """The patch's points as a half-open box of level indices."""
        extra = 1 if self.nodal else 0
        return tuple((lo, hi + extra) for lo, hi in patch.box)

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

    def in_domain(self, level: int, *index: np.ndarray) -> np.ndarray:
        """Whether points (level indices, possibly beyond the domain) are in it.

        ``index`` holds one array per axis. A cell is if it is a cell of the
        domain; a node if one of the cells around it is.
        """
        cells = self._domain(level)
        shape = cells.shape
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
    for coarse in hierarchy.patches_at_level(patch.level - 1):
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
    copies (see :func:`apply_interpolations`) on the ranks that hold some of
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
    reproduced away from walls.
    Arguments and result as for :func:`interpolate_from_below`.
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
        spacing = hierarchy.spacing(patch.level - 1)[0]
        radius = hierarchy.bounds[0][0] + (parent[0] + 0.5) * spacing
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


class HaloExchange:
    """Fill the halo of every patch of field ``name``; see the module docstring.

    Collective over the hierarchy's communicator, as is :meth:`execute`.
    """

    def __init__(self, hierarchy: AMRHierarchy, name: str) -> None:
        """Find where every halo point comes from (one ``allgather``)."""
        spec = hierarchy._field_specs[name]
        self._hierarchy = hierarchy
        self._name = name
        self._width = int(spec["halo"])
        points = _Points(hierarchy, spec["centering"] == "node")
        copies: list[tuple] = []
        self._interpolations: list[_Bilinear] = []
        for patch in hierarchy.patches if self._width else ():
            covered = self._same_level(points, patch, copies)
            if patch.level > 0:
                self._coarse_fine(points, patch, covered, spec["dtype"], copies)
        self._plan = mpa.CopyPlan(copies, comm=hierarchy.comm)

    def _field(self, patch: Patch) -> Any:
        return patch.field(self._name) if patch.is_local else None

    def _same_level(self, points: _Points, patch: Patch, copies: list) -> np.ndarray:
        """Copies from the patches of the same level; the halo points they cover."""
        own = points.own(patch)
        width = self._width
        extended = tuple((lo - width, hi + width) for lo, hi in own)
        covered = np.zeros([hi - lo for lo, hi in extended], dtype=bool)
        covered[(slice(width, -width),) * len(own)] = True
        origin = tuple(lo for lo, _ in own)
        for other in self._hierarchy.patches_at_level(patch.level):
            other_origin = tuple(lo for lo, _ in points.own(other))
            for shift in points.shifts(patch.level):
                if other is patch and not any(shift):
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
                    copies.append(
                        (self._field(other), source, self._field(patch), target)
                    )
                    covered[
                        tuple(
                            slice(s - e0, t - e0)
                            for (s, t), (e0, _) in zip(part, extended, strict=True)
                        )
                    ] = True
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
            self._field,
            self._field(patch),
            tuple(values[keep] for values in index),
            dtype,
            copies,
        )
        if interpolation is not None:
            self._interpolations.append(interpolation)

    def execute(self) -> None:
        """Fill the halos. Collective over the hierarchy's communicator."""
        self._plan.execute()
        apply_interpolations(self._interpolations)
        if self._width:
            for patch in self._hierarchy.local_patches:
                patch.field(self._name).update_halos()


class Restriction:
    """Give the coarse points under finer patches the fine values of field ``name``.

    Cell-centered: the mean of the fine cells of each coarse cell, weighted by
    their volumes in r–z (where they grow with the radius).
    Node-centered: the fine node at each coarse node. Only the points of each
    rank's block are written; the halos follow with :class:`HaloExchange`.
    Collective over the hierarchy's communicator, as is :meth:`execute`.
    """

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
        # one plan per level, finest first: a level gets its finer values
        # before it gives them to the one below
        self._steps: list[tuple[Any, list[tuple]]] = []
        for level in range(hierarchy.max_level, 0, -1):
            copies: list[tuple] = []
            writes: list[tuple] = []
            for fine in hierarchy.patches_at_level(level):
                self._add(hierarchy, points, fine, copies, writes)
            self._steps.append((mpa.CopyPlan(copies, comm=hierarchy.comm), writes))

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
        for coarse in hierarchy.patches_at_level(fine.level - 1):
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

    def execute(self) -> None:
        """Restrict. Collective over the hierarchy's communicator."""
        for plan, writes in self._steps:
            plan.execute()
            self._write(writes)

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
                    weight = hierarchy.cell_volumes(level, first + np.arange(n * r))
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
