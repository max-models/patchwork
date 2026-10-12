"""Composite finite-volume Poisson solve on a static patch hierarchy."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

if TYPE_CHECKING:
    from patchwork.mesh import AMRHierarchy, Patch


@dataclass(frozen=True)
class _Cell:
    """One active leaf cell in finest-grid integer coordinates."""

    patch: Patch
    i: int
    j: int
    lower: tuple[int, int]
    upper: tuple[int, int]
    spacing: tuple[float, float]
    volume: float
    rhs: float


def _parent_patch(hierarchy: AMRHierarchy, patch: Patch) -> Patch:
    """Find the coarser patch containing ``patch``."""
    ratio = hierarchy.refinement_ratio
    for parent in hierarchy.patches_at_level(patch.level - 1):
        if all(
            parent_lo * ratio <= child_lo and child_hi <= parent_hi * ratio
            for (parent_lo, parent_hi), (child_lo, child_hi) in zip(
                parent.box, patch.box, strict=True
            )
        ):
            return parent
    raise ValueError(f"patch at level {patch.level} has no parent")


def solve_poisson(
    hierarchy: AMRHierarchy,
    rhs: str,
    *,
    potential: str = "potential",
    boundary_value: float = 0.0,
) -> None:
    r"""Solve ``-laplace(phi) = rhs`` on the composite 2D mesh.

    This uses a cell-centered, conservative finite-volume operator. Coarse cells
    fully covered by a finer patch are omitted from the linear system; fluxes
    across coarse-fine faces use the exact overlap length and center distance.
    The homogeneous or constant Dirichlet value is applied on all four domain
    boundaries. The solve is replicated on each communicator rank after
    collectively gathering the patch right-hand sides, which keeps this initial
    implementation simple and useful for serial and one-rank runs.

    The solution is written to a field named by ``potential`` on every patch.
    Covered coarse cells are filled by volume averaging their fine children.
    All ranks must call this function in the same order.
    """
    if rhs not in hierarchy.field_names:
        raise KeyError(f"hierarchy has no right-hand-side field {rhs!r}")
    if potential not in hierarchy.field_names:
        hierarchy.allocate_field(potential)

    patches = hierarchy.patches
    right_sides = {id(p): p.field(rhs).to_numpy() for p in patches}
    covered: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for child in patches:
        if child.level == 0:
            continue
        parent = _parent_patch(hierarchy, child)
        ratio = hierarchy.refinement_ratio
        (i0, i1), (j0, j1) = child.box
        covered[id(parent)].update(
            (i, j)
            for i in range(i0 // ratio, i1 // ratio)
            for j in range(j0 // ratio, j1 // ratio)
        )

    finest_scale = hierarchy.refinement_ratio**hierarchy.max_level
    dx0, dy0 = hierarchy.spacing(0)
    cells: list[_Cell] = []
    for patch in patches:
        dx, dy = hierarchy.spacing(patch.level)
        scale = hierarchy.refinement_ratio ** (hierarchy.max_level - patch.level)
        values = right_sides[id(patch)]
        (i0, i1), (j0, j1) = patch.box
        for i in range(i0, i1):
            for j in range(j0, j1):
                if (i, j) in covered[id(patch)]:
                    continue
                index = len(cells)
                cells.append(
                    _Cell(
                        patch=patch,
                        i=i,
                        j=j,
                        lower=(i * scale, j * scale),
                        upper=((i + 1) * scale, (j + 1) * scale),
                        spacing=(dx, dy),
                        volume=dx * dy,
                        rhs=float(values[i - i0, j - j0]),
                    )
                )

    # Match opposing face segments in a common finest-level integer lattice.
    negative: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    positive: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for index, cell in enumerate(cells):
        x0, y0 = cell.lower
        x1, y1 = cell.upper
        negative[(0, x0)].append((index, y0, y1))
        positive[(0, x1)].append((index, y0, y1))
        negative[(1, y0)].append((index, x0, x1))
        positive[(1, y1)].append((index, x0, x1))

    diagonal = np.zeros(len(cells), dtype=float)
    rows: list[int] = []
    columns: list[int] = []
    entries: list[float] = []
    b = np.empty(len(cells), dtype=float)
    for index, cell in enumerate(cells):
        b[index] = cell.rhs * cell.volume

    def connect(a: tuple[int, int, int], c: tuple[int, int, int], axis: int) -> None:
        """Add one conservative face coupling for overlapping cell faces."""
        ia, a0, a1 = a
        ib, b0, b1 = c
        face_ticks = min(a1, b1) - max(a0, b0)
        if face_ticks <= 0:
            return
        face_length = face_ticks / finest_scale * (dy0 if axis == 0 else dx0)
        distance = 0.5 * (cells[ia].spacing[axis] + cells[ib].spacing[axis])
        conductance = face_length / distance
        diagonal[ia] += conductance
        diagonal[ib] += conductance
        rows.extend((ia, ib))
        columns.extend((ib, ia))
        entries.extend((-conductance, -conductance))

    for (axis, coordinate), high_faces in positive.items():
        low_faces = negative.get((axis, coordinate), ())
        for high in high_faces:
            for low in low_faces:
                connect(high, low, axis)

    boundary_rhs = float(boundary_value)
    for index, cell in enumerate(cells):
        (x0, y0), (x1, y1) = cell.lower, cell.upper
        dx, dy = cell.spacing
        if x0 == 0:
            conductance = dy / (0.5 * dx)
            diagonal[index] += conductance
            b[index] += conductance * boundary_rhs
        if x1 == finest_scale * hierarchy.base_shape[0]:
            conductance = dy / (0.5 * dx)
            diagonal[index] += conductance
            b[index] += conductance * boundary_rhs
        if y0 == 0:
            conductance = dx / (0.5 * dy)
            diagonal[index] += conductance
            b[index] += conductance * boundary_rhs
        if y1 == finest_scale * hierarchy.base_shape[1]:
            conductance = dx / (0.5 * dy)
            diagonal[index] += conductance
            b[index] += conductance * boundary_rhs

    rows.extend(range(len(cells)))
    columns.extend(range(len(cells)))
    entries.extend(diagonal.tolist())
    operator = sparse.coo_matrix(
        (entries, (rows, columns)), shape=(len(cells), len(cells))
    ).tocsr()
    solution = scipy.sparse.linalg.spsolve(operator, b)

    # Populate leaf-cell solutions, then restrict from fine to coarse levels.
    values_by_patch = {id(p): np.zeros(p.shape, dtype=float) for p in patches}
    for cell, value in zip(cells, solution, strict=True):
        (i0, _), (j0, _) = cell.patch.box
        values_by_patch[id(cell.patch)][cell.i - i0, cell.j - j0] = value
    ratio = hierarchy.refinement_ratio
    for level in range(hierarchy.max_level, 0, -1):
        for child in hierarchy.patches_at_level(level):
            parent = _parent_patch(hierarchy, child)
            (i0, i1), (j0, j1) = child.box
            child_values = values_by_patch[id(child)]
            parent_values = values_by_patch[id(parent)]
            reshaped = child_values.reshape(
                (child.shape[0] // ratio, ratio, child.shape[1] // ratio, ratio)
            )
            restricted = reshaped.mean(axis=(1, 3))
            pi0, _ = i0 // ratio, i1 // ratio
            pj0, _ = j0 // ratio, j1 // ratio
            parent_box_i0 = parent.box[0][0]
            parent_box_j0 = parent.box[1][0]
            parent_values[
                pi0 - parent_box_i0 : pi0 - parent_box_i0 + restricted.shape[0],
                pj0 - parent_box_j0 : pj0 - parent_box_j0 + restricted.shape[1],
            ] = restricted

    for patch in patches:
        array = patch.field(potential)
        (i0, i1), (j0, j1) = array.layout.index_bounds
        array.local[...] = values_by_patch[id(patch)][i0:i1, j0:j1]
