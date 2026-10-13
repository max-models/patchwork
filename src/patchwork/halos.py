"""Halo exchange between patches, and restriction from fine patches to coarse ones.

Each patch field is an mpiarray array on the patch's own ranks, with a halo of
``halo`` cells (or nodes) around it. :class:`HaloExchange` fills that halo from
the rest of the hierarchy:

1. from the patches of the same level that the halo overlaps, across periodic
   faces too (a patch along a whole periodic axis from itself);
2. where no patch of the same level is there, by bilinear interpolation of the
   level below;
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
    """The halo around ``own``: four boxes, the corners in the first two."""
    (x0, x1), (y0, y1) = own
    w = width
    return [
        ((x0 - w, x0), (y0 - w, y1 + w)),
        ((x1, x1 + w), (y0 - w, y1 + w)),
        ((x0, x1), (y0 - w, y0)),
        ((x0, x1), (y1, y1 + w)),
    ]


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

    def shifts(self, level: int) -> list[tuple[int, int]]:
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
                (i0, i1), (j0, j1) = patch.box
                cells[i0 * scale : i1 * scale, j0 * scale : j1 * scale] = True
            self._domains[level] = cells
        return self._domains[level]

    def in_domain(self, level: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """Whether points (level indices, possibly beyond the domain) are in it.

        A cell is if it is a cell of the domain; a node if one of the cells
        around it is.
        """
        cells = self._domain(level)
        shape = cells.shape
        offsets = [(0, 0), (-1, 0), (0, -1), (-1, -1)] if self.nodal else [(0, 0)]
        inside = np.zeros(i.shape, dtype=bool)
        for di, dj in offsets:
            index, valid = [], np.ones(i.shape, dtype=bool)
            for axis, values in enumerate((i + di, j + dj)):
                if self.periodic[axis]:
                    values = values % shape[axis]
                else:
                    valid &= (values >= 0) & (values < shape[axis])
                    values = np.clip(values, 0, shape[axis] - 1)
                index.append(values)
            inside |= valid & cells[index[0], index[1]]
        return inside

    def stencil(self, fine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The two coarse points around fine points along one axis, and their weights.

        Returns the lower coarse index and the weight of the upper one.
        """
        if self.nodal:
            position = fine / self.ratio
        else:
            position = (fine + 0.5) / self.ratio - 0.5
        lower = np.floor(position).astype(np.int64)
        return lower, position - lower


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
        self._interpolations: list[tuple] = []
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
        covered[width:-width, width:-width] = True
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
        extended = tuple((lo - width, hi + width) for lo, hi in own)
        i, j = np.nonzero(~covered)
        i = i + extended[0][0]
        j = j + extended[1][0]
        keep = points.in_domain(patch.level, i, j)
        i, j = i[keep], j[keep]
        if not i.size:
            return
        (ci, wi), (cj, wj) = points.stencil(i), points.stencil(j)
        buffer_box = (
            (int(ci.min()), int(ci.max()) + 2),
            (int(cj.min()), int(cj.max()) + 2),
        )
        shape = tuple(hi - lo for lo, hi in buffer_box)
        available = np.zeros(shape, dtype=bool)
        buffer = (
            mpa.zeros(shape, dtype=dtype, split=None, comm=patch.comm)
            if patch.is_local
            else None
        )
        for coarse in self._hierarchy.patches_at_level(patch.level - 1):
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
                copies.append((self._field(coarse), source, buffer, target))
                available[tuple(slice(s, e) for s, e in target)] = True
        # the four coarse points of every halo point, without the unavailable ones
        corners = []
        for a, b in itertools.product((0, 1), repeat=2):
            weight = (wi if a else 1.0 - wi) * (wj if b else 1.0 - wj)
            bi = ci + a - buffer_box[0][0]
            bj = cj + b - buffer_box[1][0]
            weight = np.where(available[bi, bj], weight, 0.0)
            corners.append((bi * shape[1] + bj, weight))
        total = sum(weight for _, weight in corners)
        keep = total > 0.0
        if not keep.any() or not patch.is_local:
            return
        flat = np.stack([index[keep] for index, _ in corners], axis=1)
        weights = np.stack(
            [weight[keep] / total[keep] for _, weight in corners], axis=1
        )
        array = patch.field(self._name)
        i = i[keep] - own[0][0]
        j = j[keep] - own[1][0]
        layout = array.layout
        mine = np.ones(i.shape, dtype=bool)
        local = []
        for values, (b0, b1), h in zip(
            (i, j), layout.index_bounds, layout.halo, strict=True
        ):
            mine &= (values >= b0 - h) & (values < b1 + h)
            local.append(values - b0 + h)
        if mine.any():
            index = (local[0][mine], local[1][mine])
            self._interpolations.append(
                (array, index, buffer, flat[mine], weights[mine])
            )

    def execute(self) -> None:
        """Fill the halos. Collective over the hierarchy's communicator."""
        self._plan.execute()
        for array, index, buffer, flat, weights in self._interpolations:
            values = np.asarray(buffer.local_with_halos).reshape(-1)[flat]
            array.local_with_halos[index] = (values * weights).sum(axis=1)
        if self._width:
            for patch in self._hierarchy.local_patches:
                patch.field(self._name).update_halos()


class Restriction:
    """Give the coarse points under finer patches the fine values of field ``name``.

    Cell-centered: the mean of the fine cells of each coarse cell.
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
                    writes.append((coarse.field(name), target, buffer))

    def execute(self) -> None:
        """Restrict. Collective over the hierarchy's communicator."""
        for plan, writes in self._steps:
            plan.execute()
            self._write(writes)

    def _write(self, writes: list[tuple]) -> None:
        r = self._ratio
        for array, target, buffer in writes:
            fine = np.asarray(buffer.local_with_halos)
            if self._nodal:
                coarse = fine[::r, ::r]
            else:
                n, m = fine.shape[0] // r, fine.shape[1] // r
                coarse = fine.reshape(n, r, m, r).mean(axis=(1, 3))
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
