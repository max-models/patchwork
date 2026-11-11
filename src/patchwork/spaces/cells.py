"""The cell space of a patch hierarchy: the lowest-order L² (Q0) space.

One value per *leaf cell* (a cell of a patch that no finer patch covers).
`CompositeCells` holds the leaf cells, their volumes, the faces between them
(with their areas) and the maps to and from the patches, in the Cartesian plane
or the axisymmetric r–z plane; `cell_body` makes a body of leaf cells. The
operators on these cells are `patchwork.solvers.cell_operator.CellOperator`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from methodcard import method

from patchwork.spaces.boundaries import EmbeddedDirichlet, EmbeddedFloating

if TYPE_CHECKING:
    from patchwork.mesh.hierarchy import AMRHierarchy

#: A coefficient or source on the leaf cells: a number, an array over the cells,
#: or a function of the cell-centre coordinates: ``(x,)`` in 1D, ``(x, y)`` in 2D
#: (``(r, z)`` in r–z).
CellValue = float | complex | np.ndarray | Callable[..., Any]


@method(
    equations={
        "leaf_cells": r"\text{leaf} = \text{cells of a patch that no finer patch covers}, \qquad \text{span} = r^{L_\mathrm{max} - L} \text{ lattice cells}",
        "volumes": r"V_i = \Delta x\,\Delta y \ \text{(plane)}, \qquad V_i = 2\pi r_c\,\Delta r\,\Delta z \ \text{(}r\text{–}z\text{)}",
        "face_areas": r"A_f = t_1 - t_0 \ \text{(plane)}, \qquad A_{r} = 2\pi r_f (z_1 - z_0), \quad A_{z} = \pi (r_1^2 - r_0^2) \ \text{(}r\text{–}z\text{)}",
    },
    assumptions=(
        "a face is a run of finest-lattice faces between one pair of leaf cells",
        "in 1D a face is a point of area 1",
    ),
    components=("patchwork.mesh.hierarchy.AMRHierarchy",),
)
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
        """A `CellValue` on the leaf cells, ``(n,)``."""
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
        ``count`` as in `faces`; ``rectangle``, whether the face is on a
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
        fine cells (`AMRHierarchy.restrict`).
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


@method(
    equations={
        "fraction": r"\theta = \frac{h_\mathrm{out}}{h_\mathrm{out} + h_\mathrm{in}} \quad \text{(the surface on the shared face)}",
    },
    assumptions=("a body given by a mask of leaf cells, fixed or floating",),
)
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
    ``source`` of an `EmbeddedFloating` body.
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
