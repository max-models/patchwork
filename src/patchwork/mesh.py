"""Block-structured adaptive mesh hierarchy and mpiarray-backed patch fields."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mpiarray as mpa

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from mpiarray import DistributedArray

IndexBox = tuple[tuple[int, int], tuple[int, int]]
PhysicalBounds = tuple[tuple[float, float], tuple[float, float]]


def _box(value: Sequence[Sequence[int]]) -> IndexBox:
    """Normalize and validate a 2D half-open integer box."""
    if len(value) != 2 or any(len(axis) != 2 for axis in value):
        raise ValueError("a patch box must be ((i_lower, i_upper), (j_lower, j_upper))")
    result = tuple((int(axis[0]), int(axis[1])) for axis in value)
    if any(lo < 0 or hi <= lo for lo, hi in result):
        raise ValueError(
            f"patch bounds must be non-empty and non-negative, got {result}"
        )
    if any(int(axis[0]) != axis[0] or int(axis[1]) != axis[1] for axis in value):
        raise ValueError("patch bounds must be integers")
    return result  # type: ignore[return-value]


def _shared_cells(a: IndexBox, b: IndexBox) -> int:
    """Number of cells two half-open boxes have in common."""
    count = 1
    for (x0, x1), (y0, y1) in zip(a, b, strict=True):
        count *= max(min(x1, y1) - max(x0, y0), 0)
    return count


def _overlap(a: IndexBox, b: IndexBox) -> bool:
    """Return whether two half-open boxes overlap in their interiors."""
    return all(
        max(x0, y0) < min(x1, y1) for (x0, x1), (y0, y1) in zip(a, b, strict=True)
    )


@dataclass
class Patch:
    """One rectangular patch; its box uses level-local, half-open cell indices."""

    level: int
    box: IndexBox
    _fields: dict[str, DistributedArray] = field(default_factory=dict, repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        """Number of cells along each axis."""
        return tuple(hi - lo for lo, hi in self.box)  # type: ignore[return-value]

    @property
    def node_shape(self) -> tuple[int, int]:
        """Number of nodes (cell corners) along each axis: one more than cells."""
        return tuple(hi - lo + 1 for lo, hi in self.box)  # type: ignore[return-value]

    @property
    def field_names(self) -> tuple[str, ...]:
        """Names of allocated fields on this patch."""
        return tuple(self._fields)

    def field(self, name: str) -> DistributedArray:
        """Get an allocated field by name."""
        try:
            return self._fields[name]
        except KeyError as exc:
            raise KeyError(
                f"patch at level {self.level} has no field {name!r}"
            ) from exc

    def local_cell_centers(self, hierarchy: AMRHierarchy) -> tuple[Any, Any]:
        """Coordinates of this rank's cell centers for the first allocated field."""
        if not self._fields:
            raise RuntimeError("allocate a field before requesting local cell centers")
        layout = next(iter(self._fields.values())).layout
        (i0, _), (j0, _) = self.box
        (li0, li1), (lj0, lj1) = layout.index_bounds
        dx, dy = hierarchy.spacing(self.level)
        x0, y0 = hierarchy.lower_bound
        import numpy as np

        x = x0 + (i0 + np.arange(li0, li1) + 0.5) * dx
        y = y0 + (j0 + np.arange(lj0, lj1) + 0.5) * dy
        return np.meshgrid(x, y, indexing="ij")

    def node_coordinates(self, hierarchy: AMRHierarchy) -> tuple[Any, Any]:
        """Coordinates of the nodes of the whole patch along each axis (1D arrays)."""
        import numpy as np

        dx, dy = hierarchy.spacing(self.level)
        x0, y0 = hierarchy.lower_bound
        (i0, i1), (j0, j1) = self.box
        return x0 + np.arange(i0, i1 + 1) * dx, y0 + np.arange(j0, j1 + 1) * dy

    def physical_bounds(self, hierarchy: AMRHierarchy) -> PhysicalBounds:
        """Physical rectangle occupied by this patch."""
        dx, dy = hierarchy.spacing(self.level)
        x0, y0 = hierarchy.lower_bound
        (i0, i1), (j0, j1) = self.box
        return ((x0 + i0 * dx, x0 + i1 * dx), (y0 + j0 * dy, y0 + j1 * dy))


class AMRHierarchy:
    """A 2D nested patch hierarchy with mpiarray-backed fields.

    Patch coordinates are integer cell bounds on their own refinement level.
    A child patch's bounds must align with the parent level and lie within the
    patches of that level. The refinement ratio is two by default. Fields
    are cell-centered, or node-centered (one value per cell corner, see
    :meth:`allocate_field`). ``periodic`` flags the axes on which the last
    cell neighbours the first; the node-centered solver
    (:class:`~patchwork.nodal.NodalPoisson`) uses it. Without an MPI launcher,
    mpiarray supplies a serial communicator and the same interface works with a
    single process.

    The domain is the rectangle of ``base_shape`` cells, or with ``base_boxes``
    the union of those boxes of base cells: an L, a T, a rectangle with a notch.
    The boxes are the patches of level zero. Where the domain does not fill
    the rectangle, its edge is a boundary like the rectangle's own faces.
    """

    def __init__(
        self,
        base_shape: tuple[int, int],
        bounds: PhysicalBounds = ((0.0, 1.0), (0.0, 1.0)),
        *,
        refinement_ratio: int = 2,
        periodic: tuple[bool, bool] = (False, False),
        base_boxes: Sequence[Sequence[Sequence[int]]] | None = None,
        comm: Any = None,
    ) -> None:
        """Create the base level: ``base_shape`` cells over the physical ``bounds``.

        ``base_boxes`` are half-open boxes of base cells that do not overlap;
        their union is the domain. By default it is the whole rectangle.
        """
        if len(base_shape) != 2 or any(int(n) != n or n <= 0 for n in base_shape):
            raise ValueError("base_shape must contain two positive integer cell counts")
        if int(refinement_ratio) != refinement_ratio or refinement_ratio < 2:
            raise ValueError("refinement_ratio must be an integer of at least 2")
        if len(bounds) != 2 or any(len(axis) != 2 for axis in bounds):
            raise ValueError("bounds must be ((x_lower, x_upper), (y_lower, y_upper))")
        if any(hi <= lo for lo, hi in bounds):
            raise ValueError("each physical upper bound must exceed its lower bound")
        self.base_shape = tuple(int(n) for n in base_shape)
        self.bounds = tuple(tuple(float(x) for x in axis) for axis in bounds)
        self.refinement_ratio = int(refinement_ratio)
        if len(periodic) != 2:
            raise ValueError("periodic must have one flag per axis")
        self.periodic = (bool(periodic[0]), bool(periodic[1]))
        self.comm = mpa.default_comm() if comm is None else comm
        self._field_specs: dict[str, dict[str, Any]] = {}
        whole = ((0, self.base_shape[0]), (0, self.base_shape[1]))
        boxes = [whole] if base_boxes is None else [_box(box) for box in base_boxes]
        if not boxes:
            raise ValueError("base_boxes must hold at least one box")
        for number, box in enumerate(boxes):
            if any(hi > n for (_, hi), n in zip(box, self.base_shape, strict=True)):
                raise ValueError(f"base box {box} exceeds the base shape {base_shape}")
            if any(_overlap(box, other) for other in boxes[:number]):
                raise ValueError(f"base box {box} overlaps another base box")
        self._patches: list[Patch] = [Patch(level=0, box=box) for box in boxes]

    @property
    def lower_bound(self) -> tuple[float, float]:
        """Physical lower-left corner of the domain."""
        return self.bounds[0][0], self.bounds[1][0]

    @property
    def upper_bound(self) -> tuple[float, float]:
        """Physical upper-right corner of the domain."""
        return self.bounds[0][1], self.bounds[1][1]

    @property
    def patches(self) -> tuple[Patch, ...]:
        """All patches, ordered by level and then lower corner."""
        return tuple(sorted(self._patches, key=lambda p: (p.level, p.box)))

    @property
    def max_level(self) -> int:
        """Finest level currently present."""
        return max(p.level for p in self._patches)

    @property
    def field_names(self) -> tuple[str, ...]:
        """Names allocated throughout the hierarchy."""
        return tuple(self._field_specs)

    def spacing(self, level: int = 0) -> tuple[float, float]:
        """Cell spacing at a refinement level."""
        scale = self.refinement_ratio**level
        return tuple(
            (hi - lo) / (n * scale)
            for (lo, hi), n in zip(self.bounds, self.base_shape, strict=True)
        )  # type: ignore[return-value]

    def level_shape(self, level: int) -> tuple[int, int]:
        """Number of cells in the full domain at a refinement level."""
        scale = self.refinement_ratio**level
        return self.base_shape[0] * scale, self.base_shape[1] * scale

    def patches_at_level(self, level: int) -> tuple[Patch, ...]:
        """Patches on one level."""
        return tuple(p for p in self.patches if p.level == level)

    def add_patch(self, level: int, box: Sequence[Sequence[int]]) -> Patch:
        """Add a non-overlapping, properly nested fine patch.

        ``box`` is half-open and measured in cells at ``level``. Bounds must
        align to the cells of the level below, and the patch must lie within
        the patches of that level (it may span several that touch).
        """
        if level < 1 or int(level) != level:
            raise ValueError("fine patch level must be a positive integer")
        level = int(level)
        normalized = _box(box)
        shape = self.level_shape(level)
        if any(hi > n for (_, hi), n in zip(normalized, shape, strict=True)):
            raise ValueError(
                f"patch {normalized} exceeds level-{level} domain shape {shape}"
            )
        ratio = self.refinement_ratio
        if any(lo % ratio or hi % ratio for lo, hi in normalized):
            raise ValueError(
                f"level-{level} patch bounds must align to level-{level - 1} cells"
            )
        if any(p.level == level and _overlap(normalized, p.box) for p in self._patches):
            raise ValueError(
                f"patch {normalized} overlaps an existing level-{level} patch"
            )
        # every cell of the level below that the patch refines is in a patch there
        coarse = tuple((lo // ratio, hi // ratio) for lo, hi in normalized)
        covered = sum(
            _shared_cells(coarse, parent.box)
            for parent in self.patches_at_level(level - 1)
        )
        if covered != _shared_cells(coarse, coarse):
            raise ValueError(
                f"patch {normalized} is not contained in the level-{level - 1} patches"
            )
        patch = Patch(level=level, box=normalized)
        for name, spec in self._field_specs.items():
            patch._fields[name] = self._zeros(patch, spec)
        self._patches.append(patch)
        return patch

    def _zeros(self, patch: Patch, spec: dict[str, Any]) -> DistributedArray:
        """A zero array for one patch of a field with allocation ``spec``."""
        shape = patch.node_shape if spec["centering"] == "node" else patch.shape
        return mpa.zeros(
            shape,
            dtype=spec["dtype"],
            split=spec["split"],
            halo=spec["halo"],
            comm=self.comm,
        )

    def centering(self, name: str) -> str:
        """``"cell"`` or ``"node"``: where the values of field ``name`` sit."""
        try:
            return self._field_specs[name]["centering"]
        except KeyError as exc:
            raise KeyError(f"hierarchy has no field {name!r}") from exc

    def allocate_field(
        self,
        name: str,
        *,
        dtype: Any = float,
        halo: int = 1,
        split: Any = 0,
        centering: str = "cell",
    ) -> None:
        """Allocate a named zero-filled field on every existing patch.

        The default splits patch arrays over the first axis of ``comm``. Use
        ``split=None`` to replicate each patch on every rank. With
        ``centering="node"`` the values sit on the cell corners: a patch of
        ``(nx, ny)`` cells holds ``(nx + 1, ny + 1)`` of them, and patches that
        touch or nest share the nodes they have in common.
        """
        if not name or name in self._field_specs:
            raise ValueError(f"field name must be non-empty and unique, got {name!r}")
        if centering not in ("cell", "node"):
            raise ValueError(f"centering must be 'cell' or 'node', got {centering!r}")
        spec = {"dtype": dtype, "halo": halo, "split": split, "centering": centering}
        arrays = {id(p): self._zeros(p, spec) for p in self._patches}
        self._field_specs[name] = spec
        for patch in self._patches:
            patch._fields[name] = arrays[id(patch)]

    def __iter__(self) -> Iterator[Patch]:
        """Iterate through patches in level order."""
        return iter(self.patches)
