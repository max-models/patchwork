"""Regridding: patches where tagged cells say the mesh needs to be finer.

Three steps, each usable on its own:

1. **Tagging** marks the cells of each level that need the next finer level:
   `gradient_tags` where a field changes by more than a threshold across a
   cell, `function_tags` where a function of the position says so,
   `marker_tags` where a cell holds more than a number of markers, or any
   boolean array of a level's shape.
2. **Placing the patches**, in one of two ways:

   - clustering (`cluster`, the Berger-Rigoutsos algorithm) covers the
     tagged cells of a level with few boxes (intervals in 1D, rectangles in 2D),
     each at least ``efficiency`` full of tagged cells and at most ``max_size``
     cells long;
   - a quadtree (an octree in 3D, `quadtree_boxes`) cuts every level into
     aligned square blocks of ``block_size`` cells and refines each block that
     holds a tag: the blocks of level ``k + 1`` are the children of those of
     level ``k``, and every patch is one block.
3. `AMRHierarchy.regrid` builds the
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

import cunumpy as xp
import mpiarray as mpa
import numpy as np
import scope_profiler as sp
from maybempi import MPI
from methodcard import method

from patchwork._profiling import PROFILE_TAGS
from patchwork.mesh.halos import (
    _overlap,
    _Points,
    _shift,
    apply_interpolations,
    conservative_from_below,
    interpolate_from_below,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from patchwork.mesh.hierarchy import AMRHierarchy, IndexBox

Box = tuple[tuple[int, int], ...]

# Tagging -------------------------------------------------------------------------------- #


def _centres(hierarchy: AMRHierarchy, level: int) -> list[np.ndarray]:
    """The cell centres of a whole level, one array of the level's shape per axis."""
    axes = []
    for axis in range(hierarchy.ndim):
        nodes = hierarchy.node_positions(level, axis)
        axes.append(0.5 * (nodes[:-1] + nodes[1:]))
    return list(np.meshgrid(*axes, indexing="ij"))


def _domain_cells(hierarchy: AMRHierarchy, level: int) -> np.ndarray:
    """Which cells of a level are in the domain."""
    scale = hierarchy.refinement_ratio**level
    cells = np.zeros(hierarchy.level_shape(level), dtype=bool)
    for patch in hierarchy.patches_at_level(0):
        cells[tuple(slice(lo * scale, hi * scale) for lo, hi in patch.box)] = True
    return cells


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


def marker_counts(
    hierarchy: AMRHierarchy, *coordinates: Any, levels: int
) -> dict[int, np.ndarray]:
    """The number of markers in every cell of levels ``0 .. levels - 1``, over all ranks.

    ``marker_counts(x, ...)`` in 1D, ``marker_counts(x, y, ...)`` in 2D: the
    positions of this rank's markers (arrays of one shape). Every level is
    counted whole, whether a patch covers a cell or not, and the counts are
    added over the ranks (collective over ``comm``): every rank gets the same
    arrays, of the levels' shapes (`AMRHierarchy.level_shape`). Markers
    outside the domain (and in the notch of an L) are not counted. The counts
    make the tags of `marker_tags` and, scaled to the cost of a marker, the
    ``cost`` of `AMRHierarchy.regrid`.
    """
    if len(coordinates) != hierarchy.ndim:
        raise ValueError(
            f"marker counts take {hierarchy.ndim} coordinate arrays, "
            f"got {len(coordinates)}"
        )
    if levels < 0:
        raise ValueError("levels must not be negative")
    points = np.stack([np.ravel(np.asarray(c, dtype=float)) for c in coordinates])
    if any(hierarchy.periodic):
        points = points.copy()
        hierarchy.wrap(points)
    counts: dict[int, np.ndarray] = {}
    if not levels:
        return counts
    # one histogram on the finest level; a coarser cell is the sum of its fine cells
    top = levels - 1
    edges = [hierarchy.node_positions(top, axis) for axis in range(points.shape[0])]
    found, _ = np.histogramdd(points.T, bins=edges)
    mine = np.ascontiguousarray(found)
    everyone = np.empty_like(mine)
    hierarchy.comm.Allreduce(mine, everyone, op=MPI.SUM)
    ratio = hierarchy.refinement_ratio
    for level in range(top, -1, -1):
        if level < top:
            blocks = [part for n in everyone.shape for part in (n // ratio, ratio)]
            axes = tuple(range(1, 2 * everyone.ndim, 2))
            everyone = everyone.reshape(blocks).sum(axis=axes)
        counts[level] = np.where(_domain_cells(hierarchy, level), everyone, 0.0)
    return dict(sorted(counts.items()))


def marker_tags(
    hierarchy: AMRHierarchy, *coordinates: Any, per_cell: float, levels: int
) -> dict[int, np.ndarray]:
    """Tag the cells of levels ``0 .. levels - 1`` that hold more than ``per_cell`` markers.

    ``marker_tags(x, ...)`` in 1D, ``marker_tags(x, y, ...)`` in 2D, counted
    by `marker_counts` (collective over ``comm``, the same tags on every
    rank). Refinement driven by these tags stops where the cells of a level
    hold at most ``per_cell`` markers each, as a particle tree stops splitting
    at its leaf capacity.
    """
    counts = marker_counts(hierarchy, *coordinates, levels=levels)
    return {level: found > per_cell for level, found in counts.items()}


@sp.profile("patchwork.gradient_tags", tags=PROFILE_TAGS)
@method(
    equations={
        "cells": r"\text{tag } i \iff \max_{j \in N(i)} |u_i - u_j| > \tau \quad \text{(two neighbours in 1D, four in 2D)}",
        "nodes": r"\text{tag } K \iff \max_{p \in K} \phi_p - \min_{p \in K} \phi_p > \tau",
    },
    assumptions=(
        "the change per cell of each level: tagging stops once the cells resolve the field",
        "only points of one patch are compared; the tags are combined over the ranks by `regrid`",
    ),
)
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
    `AMRHierarchy.regrid`.
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


# Clustering ----------------------------------------------------------------------------- #


@method(
    equations={
        "efficiency": r"\text{keep } B \iff \frac{\#\{\text{tagged cells in } B\}}{|B|} \ge \texttt{efficiency} \text{ and } B \subset \mathcal{A}, \quad \text{chopped into } \lceil n / \texttt{max\_size} \rceil \text{ parts per axis}",
        "signature": r"\Sigma_k = \#\{\text{tagged cells in row (or column) } k\}",
        "inflection": r"\Delta^2_k = \Sigma_{k+2} - 2\Sigma_{k+1} + \Sigma_k, \qquad \text{cut at } k + 2 \text{ where } |\Delta^2_{k+1} - \Delta^2_k| \text{ is largest among sign changes}",
    },
    assumptions=(
        "cut at an empty row or column nearest the middle, else at the strongest inflection (sides of at least 4 cells), else in the middle of the longest side",
        "every part shrinks to its tagged cells; boxes shorter than `min_size` are widened into free allowed cells",
        "the boxes are disjoint and cover every allowed tagged cell",
    ),
    references=(
        "[Berger and Rigoutsos, IEEE Trans. Syst. Man Cybern. 21(5) (1991)](https://ieeexplore.ieee.org/document/120081/)",
    ),
)
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
    row, column or plane without tags; a cell in 1D), else where the number of tagged
    cells per row or column bends most (a zero crossing of its second
    difference), else in the middle of its longest side. Each part shrinks to
    the tagged cells in it. In 1D the boxes are intervals, in 3D boxes.

    Args:
        tags: Booleans, one per cell: a 1D, 2D or 3D array.
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
        Half-open boxes ``((i0, i1),)``, ``((i0, i1), (j0, j1))`` or
        ``((i0, i1), (j0, j1), (k0, k1))`` that do not overlap and cover every
        allowed tagged cell.
    """
    tags = np.asarray(tags, dtype=bool)
    if tags.ndim not in (1, 2, 3):
        raise ValueError("tags must be a 1D, 2D or 3D array")
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
    `AMRHierarchy.regrid`.
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


def _refine(mask: np.ndarray, ratio: int) -> np.ndarray:
    """The fine cells of every true coarse cell: ``mask`` repeated ``ratio`` times per axis."""
    for axis in range(mask.ndim):
        mask = np.repeat(mask, ratio, axis=axis)
    return mask


def _blocks(mask: np.ndarray, size: int) -> np.ndarray:
    """Which aligned blocks of ``size`` cells per axis hold a true cell (a partial last block counts)."""
    padded = np.zeros([-(-n // size) * size for n in mask.shape], dtype=bool)
    padded[tuple(slice(0, n) for n in mask.shape)] = mask
    return _coarsen(padded, size)


def _block_cells(blocks: np.ndarray, size: int, shape: tuple[int, ...]) -> np.ndarray:
    """The cells of the true blocks, cut to ``shape``."""
    cells = _refine(blocks, size)
    return cells[tuple(slice(0, n) for n in shape)]


@method(
    equations={
        "blocks": r"B^{k}_{\mathbf m} = \prod_{d} [\, m_d b,\ (m_d + 1) b \,) \text{ (cells of level } k\text{)}, \quad \text{refined} \iff B^{k}_{\mathbf m} \text{ holds a tag of level } k - 1",
        "children": r"B^{k+1}_{\mathbf m'} \subset B^{k}_{\mathbf m} \iff \lfloor \mathbf m' / r \rfloor = \mathbf m \quad (2^d \text{ children for } r = 2)",
    },
    assumptions=(
        "each patch is one block of `block_size` cells of its level, aligned to multiples of `block_size`; the last block of a level may be shorter, and a block that crosses the edge of the domain or a forbidden cell is cut into rectangles of its allowed cells",
        "a level is refined, with a margin of `nesting` cells, wherever a finer block needs it, so the blocks are properly nested (a balanced tree)",
    ),
    references=(
        "[MacNeice et al., Comput. Phys. Commun. 126, 330 (2000)](https://doi.org/10.1016/S0010-4655(99)00501-9)",
        "[Burstedde, Wilcox and Ghattas, SIAM J. Sci. Comput. 33, 1103 (2011)](https://doi.org/10.1137/100791634)",
    ),
)
def quadtree_boxes(
    hierarchy: AMRHierarchy,
    tags: Mapping[int, np.ndarray],
    *,
    block_size: int,
    buffer: int,
    nesting: int,
    min_patch_size: int = 1,
    forbid: Any = None,
) -> dict[int, list[IndexBox]]:
    """The boxes of every refined level as the blocks of a quadtree (an octree in 3D).

    Every level is cut into aligned blocks of ``block_size`` cells per axis.
    A block of level ``k + 1`` is a patch when a tag of level ``k``, grown by
    ``buffer`` cells, lies under it, or when a finer block needs it: each block
    keeps ``nesting`` cells of the level below between itself and that level's
    edge (the edge of the domain excepted), so the level below is refined
    around it too. With a refinement ratio of two, the blocks of level ``k + 1``
    under a block of level ``k`` are its four children (eight in 3D): the
    patches are the nodes of a tree, all of one size, and the tree is balanced.

    A block that crosses the edge of the domain (the notch of an L), a
    forbidden cell (``forbid`` as for `AMRHierarchy.regrid`) or the margin
    the level below needs is cut into rectangles of its allowed cells, which may
    then be narrower than ``block_size``; ``min_patch_size`` widens those
    pieces as `cluster` does. ``block_size`` is in cells of the new level and
    must be a multiple of the refinement ratio.
    """
    ratio = hierarchy.refinement_ratio
    periodic = hierarchy.periodic
    if int(block_size) != block_size or block_size < ratio or block_size % ratio:
        raise ValueError(
            f"block_size must be a multiple of the refinement ratio {ratio}, "
            f"got {block_size}"
        )
    if buffer < 0 or nesting < 0:
        raise ValueError("buffer and nesting must not be negative")
    if min_patch_size < 1:
        raise ValueError("min_patch_size must be at least 1")
    size = int(block_size) // ratio  # a block of level k + 1, in cells of level k
    levels = [level for level, values in tags.items() if values.any()]
    top = max(levels) + 1 if levels else 0
    # top down: the blocks each level needs, for its own tags and the finer blocks
    refined: dict[int, np.ndarray] = {}  # level k + 1 -> its blocks, over level k
    margin: np.ndarray | None = None  # cells of level k the finer blocks need
    for level in range(top - 1, -1, -1):
        domain = _domain_cells(hierarchy, level)
        need = _grow(
            tags.get(level, np.zeros(domain.shape, dtype=bool)),
            buffer,
            periodic,
            outside=False,
        )
        if margin is not None:
            need |= _coarsen(margin, ratio)
        need &= domain
        refined[level + 1] = _blocks(need, size)
        cells = _block_cells(refined[level + 1], size, domain.shape) & domain
        margin = _grow(cells, nesting, periodic, outside=False)
    # bottom up: each needed block, cut to the cells where its level may go
    boxes: dict[int, list[IndexBox]] = {}
    covered = _domain_cells(hierarchy, 0)
    min_size = -(-min_patch_size // ratio)
    for level in range(top):
        domain = _domain_cells(hierarchy, level)
        allowed = _shrink(covered | ~domain, nesting, periodic) & covered
        forbidden = _forbidden_cells(hierarchy, forbid, level + 1)
        if forbidden is not None:
            allowed &= ~_coarsen(forbidden, ratio)
        found: list[IndexBox] = []
        for index in zip(*np.nonzero(refined[level + 1]), strict=True):
            block = tuple(
                (int(m) * size, min((int(m) + 1) * size, n))
                for m, n in zip(index, domain.shape, strict=True)
            )
            inside = allowed[tuple(slice(lo, hi) for lo, hi in block)]
            if inside.all():
                parts = [block]
            elif inside.any():
                parts = [
                    tuple(
                        (lo + a, lo + b)
                        for (lo, _), (a, b) in zip(block, part, strict=True)
                    )
                    for part in cluster(inside, 1.0, None, inside, min_size)
                ]
            else:
                continue
            found.extend(parts)
        if not found:
            break
        boxes[level + 1] = [
            tuple((lo * ratio, hi * ratio) for lo, hi in box) for box in sorted(found)
        ]
        covered = np.zeros(hierarchy.level_shape(level + 1), dtype=bool)
        for box in boxes[level + 1]:
            covered[tuple(slice(lo, hi) for lo, hi in box)] = True
    return boxes


def _curve_keys(items: list[tuple], ratio: int, curve: str) -> list[int]:
    """The key of every ``(level, n, box)`` on a curve: its lower corner in finest cells."""
    top = max(level for level, _, _ in items)
    corners = np.array(
        [[lo * ratio ** (top - level) for lo, _ in box] for level, _, box in items],
        dtype=np.uint64,
    )
    if corners.shape[1] == 1:  # 1D: the curve is the axis
        return corners.reshape(-1).tolist()
    if curve == "morton":
        return xp.algorithms.morton_encode(*corners.T).tolist()
    levels = max(1, int(corners.max()).bit_length())
    return xp.algorithms.hilbert_encode(*corners.T, levels=levels).tolist()


def _costs(
    boxes: Mapping[int, Sequence[IndexBox]],
    items: list[tuple],
    weights: Mapping[int, Sequence[float]] | None,
) -> list[float]:
    """The cost of every item: its cells, or its weight."""
    if weights is None:
        return [float(np.prod([hi - lo for lo, hi in box])) for _, _, box in items]
    for level in boxes:
        if len(weights[level]) != len(boxes[level]):
            raise ValueError(f"weights of level {level} need one value per box")
    return [float(weights[level][n]) for level, n, _ in items]


@method(
    equations={
        "morton": r"k(B) = \sum_{b} \sum_{d=0}^{D-1} \mathrm{bit}_b\big(r^{L - \ell}\, i_d\big)\, 2^{bD + d} \quad \text{(the lower corner } \mathbf i \text{ of } B \text{ on level } \ell\text{, in cells of the finest level } L\text{)}",
        "hilbert": r"k(B) = H\big(r^{L - \ell}\, \mathbf i\big), \quad H \text{ the Hilbert index (Skilling's construction): consecutive keys are neighbouring cells}",
        "ranks": r"\text{along the curve, rank } q \text{ takes boxes while } w_q + c_B/2 \le T, \quad T = \frac{1}{R}\Big(\sum_q w_q^{0} + \sum_B c_B\Big)",
    },
    assumptions=(
        "the patches of all refined levels on one curve, a coarse patch before the finer ones at the same corner, so that a patch and the patches over it tend to share a rank",
        "the cost $c_B$ of a box is its cells, or its weight (`box_costs`); the load a rank already has (the base level) counts",
        "a patch stays on one rank",
    ),
    references=(
        "[Morton, IBM technical report (1966)](https://dominoweb.draco.res.ibm.com/reports/Morton1966.pdf)",
        "[Skilling, AIP Conf. Proc. 707, 381 (2004)](https://doi.org/10.1063/1.1751381)",
        "[Burstedde, Wilcox and Ghattas, SIAM J. Sci. Comput. 33, 1103 (2011)](https://doi.org/10.1137/100791634)",
    ),
)
def curve_ranks(
    boxes: Mapping[int, Sequence[IndexBox]],
    size: int,
    *,
    curve: str = "morton",
    ratio: int = 2,
    load: Sequence[float] | None = None,
    weights: Mapping[int, Sequence[float]] | None = None,
) -> dict[int, list[int]]:
    """The rank of each box: the boxes of all levels along a space-filling curve, cut into ``size`` runs.

    ``boxes`` maps a level to its boxes. Every box gets the key of its lower
    corner, in cells of the finest level, on the ``curve``: ``"morton"`` (the
    Z-order curve) or ``"hilbert"`` (without the Z's jumps, so that the run of
    a rank is more compact). Boxes close in space are close on the curve
    whatever their level, and a box comes before the finer boxes at the same
    corner. Walking the curve, a rank takes boxes until its cost, with the
    ``load`` it already has (the cells of the base level, say), reaches an equal
    share of all costs; then the next rank goes on. The cost of a box is its
    cells, or ``weights`` (per level, one per box, in the units of ``load``;
    see `box_costs`). Returns, per level, the ranks in the order of
    ``boxes[level]``.
    """
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")
    if curve not in ("morton", "hilbert"):
        raise ValueError(f"curve must be 'morton' or 'hilbert', got {curve!r}")
    loads = [0.0] * size if load is None else [float(w) for w in load]
    if len(loads) != size:
        raise ValueError(f"load needs {size} values, got {len(loads)}")
    items = [
        (level, n, box) for level in sorted(boxes) for n, box in enumerate(boxes[level])
    ]
    result: dict[int, list[int]] = {level: [0] * len(boxes[level]) for level in boxes}
    if not items:
        return result
    keys = _curve_keys(items, ratio, curve)
    costs = _costs(boxes, items, weights)
    share = (sum(loads) + sum(costs)) / size
    rank = 0
    for k in sorted(range(len(items)), key=lambda k: (keys[k], items[k][0])):
        while rank < size - 1 and loads[rank] + costs[k] / 2 > share:
            rank += 1
        level, n, _ = items[k]
        result[level][n] = rank
        loads[rank] += costs[k]
    return result


@method(
    equations={
        "greedy": r"c_{B_1} \ge c_{B_2} \ge \dots, \qquad r(B_k) = \arg\min_q w_q \text{ (the lowest such rank)}, \quad w_{r(B_k)} \mathrel{+}= c_{B_k}",
    },
    assumptions=(
        "the greedy (longest processing time first) solution of the knapsack-like partition: the largest cost at most 4/3 of the best possible",
        "no locality: neighbouring patches may land on any ranks",
    ),
    references=(
        "[Graham, SIAM J. Appl. Math. 17, 416 (1969)](https://doi.org/10.1137/0117039)",
    ),
)
def knapsack_ranks(
    boxes: Mapping[int, Sequence[IndexBox]],
    size: int,
    *,
    load: Sequence[float] | None = None,
    weights: Mapping[int, Sequence[float]] | None = None,
) -> dict[int, list[int]]:
    """The rank of each box: the largest cost first, each to the rank with the least so far.

    The greedy solution of cutting the boxes into ``size`` sets of equal cost,
    what ``placement="balanced"`` does with cells, here with the ``weights`` of
    `box_costs` and the ``load`` the ranks already have. Balances better than a
    curve, but neighbouring boxes may land on different ranks. Returns, per
    level, the ranks in the order of ``boxes[level]``.
    """
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")
    loads = [0.0] * size if load is None else [float(w) for w in load]
    if len(loads) != size:
        raise ValueError(f"load needs {size} values, got {len(loads)}")
    items = [
        (level, n, box) for level in sorted(boxes) for n, box in enumerate(boxes[level])
    ]
    result: dict[int, list[int]] = {level: [0] * len(boxes[level]) for level in boxes}
    costs = _costs(boxes, items, weights)
    for k in sorted(
        range(len(items)), key=lambda k: (-costs[k], items[k][0], items[k][2])
    ):
        rank = min(range(size), key=lambda q: (loads[q], q))
        level, n, _ = items[k]
        result[level][n] = rank
        loads[rank] += costs[k]
    return result


def box_costs(
    hierarchy: AMRHierarchy,
    boxes: Mapping[int, Sequence[IndexBox]],
    cost: Mapping[int, Any],
) -> tuple[dict[int, list[float]], list[float]]:
    """The cost of each new box and of each rank's base patches.

    ``cost`` is one of two things:

    - a mapping from a level to a per-cell cost of that level's shape, in units
      of one cell (``0.5 * marker_counts(...)[level]`` if a marker costs half a
      cell; a missing level costs nothing beyond its cells): a box costs its
      cells plus the cost of its leaf cells, the cells no finer box covers,
      since that is where their markers deposit and gather;
    - a mapping from a patch, ``(level, box)``, to its cost, e.g. the seconds
      measured with ``deposit(..., times=...)`` and ``interpolate(...,
      times=...)``: a box costs its own value, and a box without one the mean
      cost per cell of those that have one.

    ``hierarchy`` holds the base level (the patches of level 0 with their
    ranks). Returns the weights of ``boxes`` per level and the load of every
    rank from the base level, for `curve_ranks` and `knapsack_ranks`.
    """
    if any(isinstance(key, tuple) for key in cost):
        measured = {
            (int(level), tuple(tuple(side) for side in box)): float(value)
            for (level, box), value in cost.items()
        }
        cells = sum(float(np.prod([hi - lo for lo, hi in box])) for _, box in measured)
        per_cell = sum(measured.values()) / cells if cells else 0.0

        def measured_weight(level: int, box: IndexBox) -> float:
            found = measured.get((level, box))
            if found is not None:
                return found
            return per_cell * float(np.prod([hi - lo for lo, hi in box]))

        weights = {
            level: [measured_weight(level, box) for box in boxes[level]]
            for level in boxes
        }
        load = [0.0] * hierarchy.comm.Get_size()
        for patch in hierarchy.patches_at_level(0):
            for rank in patch.ranks:
                load[rank] += measured_weight(0, patch.box) / len(patch.ranks)
        return weights, load
    ratio = hierarchy.refinement_ratio
    top = max(boxes, default=0)
    leaf_cost = {}
    for level in range(top + 1):
        shape = hierarchy.level_shape(level)
        values = cost.get(level)
        if values is None:
            values = np.zeros(shape)
        values = np.asarray(values, dtype=float)
        if values.shape != shape:
            raise ValueError(
                f"cost of level {level} has the shape {values.shape}, not {shape}"
            )
        leaf = np.ones(shape, dtype=bool)
        for box in boxes.get(level + 1, ()):
            leaf[tuple(slice(lo // ratio, hi // ratio) for lo, hi in box)] = False
        leaf_cost[level] = np.where(leaf, values, 0.0)

    def weight(level: int, box: IndexBox) -> float:
        where = tuple(slice(lo, hi) for lo, hi in box)
        cells = float(np.prod([hi - lo for lo, hi in box]))
        return cells + float(leaf_cost[level][where].sum())

    weights = {level: [weight(level, box) for box in boxes[level]] for level in boxes}
    load = [0.0] * hierarchy.comm.Get_size()
    for patch in hierarchy.patches_at_level(0):
        for rank in patch.ranks:
            load[rank] += weight(0, patch.box) / len(patch.ranks)
    return weights, load


@sp.profile("patchwork.regrid_transfer", tags=PROFILE_TAGS)
def transfer(old: AMRHierarchy, new: AMRHierarchy, name: str) -> None:
    """Fill field ``name`` of ``new`` from ``old``, level by level (collective).

    A point of a new patch takes the value of an old patch of the same level
    that has it, else the bilinear interpolation of the new level below (for a
    conservative field, the conservative one of
    `conservative_from_below`). Only the patches' own
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
            # the old patches near this one (a bin index, not every pair)
            mine = []
            for previous in (
                old_points.near(level, own) if level <= old.max_level else ()
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
                    mine.append((old_field(previous), source, new_field(patch), target))
            # the last copy of a point wins: drop the copies that later ones overwrite
            # entirely (an old patch with the same box makes its neighbours' shared
            # nodes redundant), so that a kept patch costs one copy, not one per neighbour
            covered = np.zeros([hi - lo for lo, hi in own], dtype=bool)
            kept = []
            for copy in reversed(mine):
                where = tuple(slice(s, e) for s, e in copy[3])
                if not covered[where].all():
                    kept.append(copy)
                    covered[where] = True
            copies.extend(reversed(kept))
            if level == 0 or covered.all():
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
