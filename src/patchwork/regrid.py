"""Regridding: patches where tagged cells say the mesh needs to be finer.

Three steps, each usable on its own:

1. **Tagging** marks the cells of each level that need the next finer level:
   :func:`gradient_tags` where a field changes by more than a threshold across a
   cell, :func:`function_tags` where a function of the position says so, or any
   boolean array of a level's shape.
2. **Clustering** (:func:`cluster`, the Berger-Rigoutsos algorithm) covers the
   tagged cells of a level with few boxes (intervals in 1D, rectangles in 2D), each at least ``efficiency``
   full of tagged cells and at most ``max_size`` cells long.
3. :meth:`AMRHierarchy.regrid <patchwork.mesh.AMRHierarchy.regrid>` builds the
   new hierarchy from the boxes, level by level from the coarsest, and moves the
   fields into it: copied where an old patch of the same level was, else
   interpolated from the new level below.

The new levels are properly nested: a patch keeps ``nesting`` cells of the
level below between itself and the edge of that level, except at the edge of
the domain, so that the halo of every patch can be interpolated from the level
below.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import mpiarray as mpa
import numpy as np
import scope_profiler as sp
from maybempi import MPI

from patchwork._profiling import PROFILE_TAGS
from patchwork.halos import (
    _overlap,
    _Points,
    _shift,
    apply_interpolations,
    conservative_from_below,
    interpolate_from_below,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from patchwork.mesh import AMRHierarchy, IndexBox

Box = tuple[tuple[int, int], ...]

# Tagging -------------------------------------------------------------------------------- #


def _centres(hierarchy: AMRHierarchy, level: int) -> list[np.ndarray]:
    """The cell centres of a whole level, one array of the level's shape per axis."""
    axes = [
        x0 + (np.arange(n) + 0.5) * h
        for (x0, _), n, h in zip(
            hierarchy.bounds,
            hierarchy.level_shape(level),
            hierarchy.spacing(level),
            strict=True,
        )
    ]
    return list(np.meshgrid(*axes, indexing="ij"))


def function_tags(
    hierarchy: AMRHierarchy, function: Callable[..., Any], levels: int
) -> dict[int, np.ndarray]:
    """Tag the cells of levels ``0 .. levels - 1`` where ``function`` is true.

    ``function(x)`` in 1D, ``function(x, y)`` in 2D, gets the coordinates of
    the cell centres of a whole level and returns booleans; cells outside the
    domain are never tagged. The tags of level ``k`` ask for patches on level
    ``k + 1``, so ``levels`` is the number of refined levels wanted.
    """
    tags = {}
    for level in range(levels):
        shape = hierarchy.level_shape(level)
        values = np.asarray(function(*_centres(hierarchy, level)), dtype=bool)
        tags[level] = np.broadcast_to(values, shape) & _domain_cells(hierarchy, level)
    return tags


@sp.profile("patchwork.gradient_tags", tags=PROFILE_TAGS)
def gradient_tags(
    hierarchy: AMRHierarchy, name: str, threshold: float
) -> dict[int, np.ndarray]:
    """Tag the cells across which field ``name`` changes by more than ``threshold``.

    Cell-centered: a cell is tagged where its value differs by more than
    ``threshold`` from one of its neighbours (two in 1D, four in 2D).
    Node-centered: where the values at its corners span more than
    ``threshold``. The change is per
    cell of each level, so a finer level, with smaller cells, sees a smaller
    change: tagging stops once the cells resolve the field.

    Every existing level is tagged, from the data each rank holds, comparing
    only points of the same patch: where a patch is split over ranks, the halos
    between its blocks are filled first (``update_halos``, collective over the
    patch's ranks). The tags are combined over the ranks by
    :meth:`~patchwork.mesh.AMRHierarchy.regrid`.
    """
    spec = hierarchy._field_specs.get(name)
    if spec is None:
        raise KeyError(f"hierarchy has no field {name!r}")
    nodal = spec["centering"] == "node"
    points = _Points(hierarchy, nodal)
    tags = {
        level: np.zeros(hierarchy.level_shape(level), dtype=bool)
        for level in range(hierarchy.max_level + 1)
    }
    for patch in hierarchy.local_patches:
        array = patch.field(name)
        layout = array.layout
        if layout.distributed and any(layout.halo):
            array.update_halos()  # the seams between this patch's ranks
        storage = np.asarray(array.local_with_halos)
        # level indices of the storage's points
        axes = [
            lo + np.arange(b0 - h, b1 + h)
            for (lo, _), (b0, b1), h in zip(
                patch.box, layout.index_bounds, layout.halo, strict=True
            )
        ]
        index = np.meshgrid(*axes, indexing="ij")
        # the patch's own points only (its halo beyond them is not filled)
        valid = np.ones(storage.shape, dtype=bool)
        for values, (lo, hi) in zip(index, points.own(patch), strict=True):
            valid &= (values >= lo) & (values < hi)
        if nodal:
            ndim = storage.ndim
            shifts = list(itertools.product((0, 1), repeat=ndim))

            def corner(array: np.ndarray, shift: tuple[int, ...]) -> np.ndarray:
                return array[
                    tuple(
                        slice(s, s + n - 1)
                        for s, n in zip(shift, array.shape, strict=True)
                    )
                ]

            corners = [corner(storage, shift) for shift in shifts]
            ok = np.ones(corners[0].shape, dtype=bool)
            for shift in shifts:
                ok &= corner(valid, shift)
            change = np.where(
                ok, np.max(corners, axis=0) - np.min(corners, axis=0), 0.0
            )
            cells = [corner(values, (0,) * ndim) for values in index]
        else:
            change = np.zeros(storage.shape)
            for axis in range(storage.ndim):
                for step in (1, -1):
                    neighbour = np.roll(storage, -step, axis=axis)
                    both = valid & np.roll(valid, -step, axis=axis)
                    # np.roll wraps around the storage: not a neighbour
                    edge = [slice(None)] * storage.ndim
                    edge[axis] = slice(-1, None) if step == 1 else slice(0, 1)
                    both[tuple(edge)] = False
                    change = np.maximum(
                        change, np.where(both, np.abs(storage - neighbour), 0.0)
                    )
            cells = index
        # the patch's own cells in this rank's block
        own = change > threshold
        for values, (x0, x1), (b0, b1) in zip(
            cells, patch.box, layout.index_bounds, strict=True
        ):
            own &= (values >= x0 + b0) & (values < x0 + b1) & (values < x1)
        tags[patch.level][tuple(values[own] for values in cells)] = True
    return tags


def _domain_cells(hierarchy: AMRHierarchy, level: int) -> np.ndarray:
    """Which cells of a level are in the domain."""
    scale = hierarchy.refinement_ratio**level
    cells = np.zeros(hierarchy.level_shape(level), dtype=bool)
    for patch in hierarchy.patches_at_level(0):
        cells[tuple(slice(lo * scale, hi * scale) for lo, hi in patch.box)] = True
    return cells


def _coarsen(mask: np.ndarray, ratio: int) -> np.ndarray:
    """Which coarse cells hold a true fine cell: ``mask`` over blocks of ``ratio``."""
    blocks = [part for n in mask.shape for part in (n // ratio, ratio)]
    return np.asarray(mask.reshape(blocks).any(axis=tuple(range(1, 2 * mask.ndim, 2))))


def _grow(
    mask: np.ndarray, width: int, periodic: tuple[bool, ...], outside: bool
) -> np.ndarray:
    """Dilate a mask by ``width`` cells (``outside``: the value beyond non-periodic edges)."""
    result = mask.copy()
    if width <= 0:
        return result
    for axis in range(mask.ndim):
        grown = result.copy()
        for step in range(1, width + 1):
            for sign in (1, -1):
                if periodic[axis]:
                    grown |= np.roll(result, sign * step, axis=axis)
                    continue
                shifted = np.full(result.shape, outside, dtype=bool)
                source = [slice(None)] * mask.ndim
                target = [slice(None)] * mask.ndim
                n = result.shape[axis]
                if step >= n:
                    grown |= shifted if outside else False
                    continue
                if sign == 1:
                    source[axis], target[axis] = slice(0, n - step), slice(step, n)
                else:
                    source[axis], target[axis] = slice(step, n), slice(0, n - step)
                shifted[tuple(target)] = result[tuple(source)]
                grown |= shifted
        result = grown
    return result


def _shrink(mask: np.ndarray, width: int, periodic: tuple[bool, ...]) -> np.ndarray:
    """Erode a mask by ``width`` cells; beyond non-periodic edges counts as inside."""
    return ~_grow(~mask, width, periodic, outside=False)


# Clustering ----------------------------------------------------------------------------- #


def cluster(
    tags: np.ndarray,
    efficiency: float = 0.7,
    max_size: int | None = None,
    allowed: np.ndarray | None = None,
    min_size: int = 1,
) -> list[IndexBox]:
    """Cover the tagged cells with boxes (the Berger-Rigoutsos algorithm).

    A box is kept when at least ``efficiency`` of its cells are tagged, it
    holds no cell that ``allowed`` forbids, and it is at most ``max_size`` cells
    long on each axis. Otherwise it is cut: at a gap in the tagged cells (a
    row or column without tags; a cell in 1D), else where the number of tagged
    cells per row or column bends most (a zero crossing of its second
    difference), else in the middle of its longest side. Each part shrinks to
    the tagged cells in it. In 1D the boxes are intervals.

    Args:
        tags: Booleans, one per cell: a 1D or 2D array.
        efficiency: The smallest fraction of tagged cells in a box, in (0, 1].
        max_size: The longest side of a box, in cells; default: no limit.
        allowed: Booleans of the shape of ``tags``: where boxes may go.
            Tagged cells that are not allowed are dropped.
        min_size: The shortest side a box should have, in cells. A shorter box
            is widened, one row or column at a time, into cells that are
            allowed and that no other box has, as far as there is room: it
            stays shorter where there is none (the shape of ``tags`` or the
            allowed region is narrower).

    Returns:
        Half-open boxes ``((i0, i1),)`` or ``((i0, i1), (j0, j1))`` that do not
        overlap and cover every allowed tagged cell.
    """
    tags = np.asarray(tags, dtype=bool)
    if tags.ndim not in (1, 2):
        raise ValueError("tags must be a 1D or 2D array")
    if not 0.0 < efficiency <= 1.0:
        raise ValueError(f"efficiency must be in (0, 1], got {efficiency}")
    if max_size is not None and max_size < 1:
        raise ValueError(f"max_size must be at least 1, got {max_size}")
    if min_size < 1:
        raise ValueError(f"min_size must be at least 1, got {min_size}")
    if max_size is not None and min_size > max_size:
        raise ValueError(f"min_size {min_size} exceeds max_size {max_size}")
    if allowed is not None:
        allowed = np.asarray(allowed, dtype=bool)
        if allowed.shape != tags.shape:
            raise ValueError("allowed must have the shape of tags")
        tags = tags & allowed
    boxes: list[IndexBox] = []
    pending: list[Box] = [tuple((0, n) for n in tags.shape)]
    while pending:
        box = _shrink_to_tags(tags, pending.pop())
        if box is None:
            continue
        where = tuple(slice(lo, hi) for lo, hi in box)
        inside = tags[where]
        fits = allowed is None or bool(allowed[where].all())
        if inside.mean() >= efficiency and fits:
            boxes.extend(_chop(box, max_size))
            continue
        axis, cut = _cut(inside)
        lo, hi = box[axis]
        first, second = list(box), list(box)
        first[axis] = (lo, lo + cut)
        second[axis] = (lo + cut, hi)
        pending.extend([tuple(second), tuple(first)])  # type: ignore[arg-type]
    if min_size > 1:
        boxes = _widen(boxes, tags.shape, allowed, min_size)
    return sorted(boxes)


def _widen(
    boxes: list[IndexBox],
    shape: tuple[int, ...],
    allowed: np.ndarray | None,
    min_size: int,
) -> list[IndexBox]:
    """Widen the boxes shorter than ``min_size`` into free, allowed cells."""
    taken = np.zeros(shape, dtype=bool)
    for box in boxes:
        taken[tuple(slice(lo, hi) for lo, hi in box)] = True
    result = []
    for box in boxes:
        lows = [lo for lo, _ in box]
        highs = [hi for _, hi in box]
        for axis in range(len(shape)):
            while highs[axis] - lows[axis] < min_size:
                grew = False
                for side in (-1, 1):  # the lower side first, then the upper one
                    cell = lows[axis] - 1 if side < 0 else highs[axis]
                    if not 0 <= cell < shape[axis]:
                        continue
                    index: list[Any] = [
                        slice(lo, hi) for lo, hi in zip(lows, highs, strict=True)
                    ]
                    index[axis] = cell
                    strip = tuple(index)
                    if taken[strip].any() or (
                        allowed is not None and not allowed[strip].all()
                    ):
                        continue
                    taken[strip] = True
                    if side < 0:
                        lows[axis] -= 1
                    else:
                        highs[axis] += 1
                    grew = True
                    break
                if not grew:
                    break
        result.append(tuple(zip(lows, highs, strict=True)))
    return result


def _others(ndim: int, axis: int) -> tuple[int, ...]:
    return tuple(other for other in range(ndim) if other != axis)


def _shrink_to_tags(tags: np.ndarray, box: Box) -> Box | None:
    inside = tags[tuple(slice(lo, hi) for lo, hi in box)]
    result = []
    for axis, (lo, _) in enumerate(box):
        hit = np.flatnonzero(inside.any(axis=_others(inside.ndim, axis)))
        if not hit.size:
            return None
        result.append((lo + int(hit[0]), lo + int(hit[-1]) + 1))
    return tuple(result)


def _cut(inside: np.ndarray) -> tuple[int, int]:
    """Where to cut a box: ``(axis, offset)`` with ``0 < offset < extent``."""
    signatures = [
        inside.sum(axis=_others(inside.ndim, axis)) for axis in range(inside.ndim)
    ]
    # 1. a row or column without tags, nearest the middle
    best = None
    for axis, signature in enumerate(signatures):
        n = signature.size
        for k in np.flatnonzero(signature == 0):
            score = abs(2 * int(k) - n)
            if best is None or score < best[0]:
                best = (score, axis, int(k))
    if best is not None:
        return best[1], best[2]
    # 2. the strongest zero crossing of the second difference
    best = None
    for axis, signature in enumerate(signatures):
        n = signature.size
        if n < 4:
            continue
        laplacian = np.diff(signature.astype(np.int64), 2)
        for k in range(laplacian.size - 1):
            if laplacian[k] * laplacian[k + 1] < 0:
                strength = abs(int(laplacian[k + 1] - laplacian[k]))
                cut = k + 2
                score = (-strength, abs(2 * cut - n))
                if best is None or score < best[0]:
                    best = (score, axis, cut)
    if best is not None:
        return best[1], best[2]
    # 3. the middle of the longest side
    axis = int(np.argmax(inside.shape))
    return axis, inside.shape[axis] // 2


def _chop(box: Box, max_size: int | None) -> list[Box]:
    """Cut a box into nearly equal parts of at most ``max_size`` cells per side."""
    if max_size is None:
        return [box]
    ranges = []
    for lo, hi in box:
        parts = -(-(hi - lo) // max_size)
        edges = [lo + (hi - lo) * k // parts for k in range(parts + 1)]
        ranges.append(list(zip(edges[:-1], edges[1:], strict=True)))
    return [tuple(parts) for parts in itertools.product(*ranges)]


# The new hierarchy ---------------------------------------------------------------------- #


def combine_tags(
    hierarchy: AMRHierarchy, tags: Mapping[int, Any]
) -> dict[int, np.ndarray]:
    """Check the tags of each level and combine them over the ranks (logical or)."""
    combined = {}
    for level, values in sorted(tags.items()):
        if int(level) != level or level < 0:
            raise ValueError(f"tags are given per level, not for {level!r}")
        shape = hierarchy.level_shape(int(level))
        array = np.ascontiguousarray(np.asarray(values, dtype=bool))
        if array.shape != shape:
            raise ValueError(
                f"tags of level {level} have the shape {array.shape}, not {shape}"
            )
        mine = array.astype(np.uint8)
        everyone = np.empty_like(mine)
        hierarchy.comm.Allreduce(mine, everyone, op=MPI.MAX)
        combined[int(level)] = everyone.astype(bool)
    return combined


def _forbidden_cells(
    hierarchy: AMRHierarchy, forbid: Any, level: int
) -> np.ndarray | None:
    """The cells of ``level`` that no patch may cover: a mask, or ``None`` for none."""
    if forbid is None:
        return None
    shape = hierarchy.level_shape(level)
    if callable(forbid):
        values = np.asarray(forbid(*_centres(hierarchy, level)), dtype=bool)
        return np.ascontiguousarray(np.broadcast_to(values, shape))
    if level not in forbid:
        return None
    values = np.asarray(forbid[level], dtype=bool)
    if values.shape != shape:
        raise ValueError(
            f"forbid of level {level} has the shape {values.shape}, not {shape}"
        )
    return values


def new_boxes(
    hierarchy: AMRHierarchy,
    tags: Mapping[int, np.ndarray],
    *,
    efficiency: float,
    buffer: int,
    nesting: int,
    max_patch_size: int | None,
    min_patch_size: int = 1,
    forbid: Any = None,
) -> dict[int, list[IndexBox]]:
    """The boxes of every refined level, from the combined tags of each level.

    ``min_patch_size`` is in cells of the new level, ``forbid`` as for
    :meth:`~patchwork.mesh.AMRHierarchy.regrid`.
    """
    ratio = hierarchy.refinement_ratio
    periodic = hierarchy.periodic
    if max_patch_size is not None and max_patch_size < ratio:
        raise ValueError(
            f"max_patch_size must be at least the refinement ratio {ratio}"
        )
    if buffer < 0 or nesting < 0:
        raise ValueError("buffer and nesting must not be negative")
    if min_patch_size < 1:
        raise ValueError("min_patch_size must be at least 1")
    if max_patch_size is not None and min_patch_size > max_patch_size:
        raise ValueError("min_patch_size must not exceed max_patch_size")
    levels = [level for level, values in tags.items() if values.any()]
    top = max(levels) + 1 if levels else 0
    # each level's tags, grown by the buffer, and the cells a finer level needs
    grown: dict[int, np.ndarray] = {}
    for level in range(top - 1, -1, -1):
        domain = _domain_cells(hierarchy, level)
        level_tags = tags.get(level, np.zeros(domain.shape, dtype=bool)).copy()
        if level + 1 in grown:
            # the next level's cells, with the margin of their own nesting
            finer = _grow(grown[level + 1], nesting, periodic, outside=False)
            level_tags |= _coarsen(finer, ratio)
        grown[level] = _grow(level_tags, buffer, periodic, outside=False) & domain
    boxes: dict[int, list[IndexBox]] = {}
    covered = _domain_cells(hierarchy, 0)
    for level in range(top):
        domain = _domain_cells(hierarchy, level)
        # inside the level, away from its coarse-fine edges (the domain's edges are fine)
        allowed = _shrink(covered | ~domain, nesting, periodic) & covered
        # a cell of this level may be refined only if none of its fine cells is forbidden
        forbidden = _forbidden_cells(hierarchy, forbid, level + 1)
        if forbidden is not None:
            allowed &= ~_coarsen(forbidden, ratio)
        max_size = None if max_patch_size is None else max_patch_size // ratio
        min_size = -(-min_patch_size // ratio)
        coarse = cluster(grown[level], efficiency, max_size, allowed, min_size)
        boxes[level + 1] = [
            tuple((lo * ratio, hi * ratio) for lo, hi in box) for box in coarse
        ]
        covered = np.zeros(hierarchy.level_shape(level + 1), dtype=bool)
        for box in boxes[level + 1]:
            covered[tuple(slice(lo, hi) for lo, hi in box)] = True
        if not boxes[level + 1]:
            del boxes[level + 1]
            break
    return boxes


@sp.profile("patchwork.regrid_transfer", tags=PROFILE_TAGS)
def transfer(old: AMRHierarchy, new: AMRHierarchy, name: str) -> None:
    """Fill field ``name`` of ``new`` from ``old``, level by level (collective).

    A point of a new patch takes the value of an old patch of the same level
    that has it, else the bilinear interpolation of the new level below (for a
    conservative field, the conservative one of
    :func:`~patchwork.halos.conservative_from_below`). Only the patches' own
    points are written; the halos stay zero.
    """
    spec = new._field_specs[name]
    nodal = spec["centering"] == "node"
    conservative = spec["conservative"]
    old_points = _Points(old, nodal)
    points = _Points(new, nodal)

    def old_field(patch: Any) -> Any:
        return patch.field(name) if patch.is_local else None

    def new_field(patch: Any) -> Any:
        return patch.field(name) if patch.is_local else None

    for level in range(new.max_level + 1):
        copies: list[tuple] = []
        interpolations = []
        for patch in new.patches_at_level(level):
            own = points.own(patch)
            origin = tuple(lo for lo, _ in own)
            covered = np.zeros([hi - lo for lo, hi in own], dtype=bool)
            for previous in (
                old.patches_at_level(level) if level <= old.max_level else ()
            ):
                theirs = old_points.own(previous)
                their_origin = tuple(lo for lo, _ in theirs)
                for shift in points.shifts(level):
                    part = _overlap(own, _shift(theirs, shift))
                    if part is None:
                        continue
                    source = _shift(
                        part,
                        tuple(-d - o for d, o in zip(shift, their_origin, strict=True)),
                    )
                    target = _shift(part, tuple(-o for o in origin))
                    copies.append(
                        (old_field(previous), source, new_field(patch), target)
                    )
                    covered[tuple(slice(s, e) for s, e in target)] = True
            if level == 0:
                continue
            index = tuple(
                values + o
                for values, o in zip(np.nonzero(~covered), origin, strict=True)
            )
            fill = conservative_from_below if conservative else interpolate_from_below
            interpolation = fill(
                new,
                points,
                patch,
                new_field,
                new_field(patch),
                index,
                spec["dtype"],
                copies,
            )
            if interpolation is not None:
                interpolations.append(interpolation)
        mpa.CopyPlan(copies, comm=new.comm).execute()
        apply_interpolations(interpolations)
