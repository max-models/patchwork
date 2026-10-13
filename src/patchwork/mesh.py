"""Block-structured adaptive mesh hierarchy and mpiarray-backed patch fields."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mpiarray as mpa
from maybempi import MPI

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

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


#: Tag of the messages that carry a gathered patch field to another rank.
_GATHER_TAG = 7311


@dataclass
class Patch:
    """One rectangular patch; its box uses level-local, half-open cell indices.

    A patch lives on some of the ranks of its hierarchy: ``ranks``, the same on
    every rank. On those ranks ``comm`` is the patch's own communicator, over
    which mpiarray splits its fields; on the others it is ``None`` and the
    patch holds no data (``is_local`` is false).
    """

    level: int
    box: IndexBox
    ranks: tuple[int, ...] = (0,)
    comm: Any = field(default=None, repr=False, compare=False)
    _fields: dict[str, DistributedArray] = field(default_factory=dict, repr=False)

    @property
    def is_local(self) -> bool:
        """Whether this rank holds a part of the patch."""
        return self.comm is not None

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
        """Get an allocated field by name, on a rank that holds the patch."""
        if not self.is_local:
            raise RuntimeError(
                f"this rank holds no part of the level-{self.level} patch {self.box}: "
                f"it is on ranks {self.ranks} (see AMRHierarchy.local_patches)"
            )
        try:
            return self._fields[name]
        except KeyError as exc:
            raise KeyError(
                f"patch at level {self.level} has no field {name!r}"
            ) from exc

    def local_node_coordinates(
        self, hierarchy: AMRHierarchy, name: str
    ) -> tuple[Any, Any]:
        """Coordinates of this rank's nodes of the node-centered field ``name``.

        Two arrays of the shape of ``field(name).local``, for filling it.
        """
        import numpy as np

        (i0, i1), (j0, j1) = self.field(name).layout.index_bounds
        x, y = self.node_coordinates(hierarchy)
        return np.meshgrid(x[i0:i1], y[j0:j1], indexing="ij")

    def local_cell_centers(
        self, hierarchy: AMRHierarchy, name: str | None = None
    ) -> tuple[Any, Any]:
        """Coordinates of this rank's cells of the cell-centered field ``name``.

        Two arrays of the shape of ``field(name).local``, for filling it.
        Without ``name``, of the first cell-centered field allocated.
        """
        if not self.is_local:
            raise RuntimeError(
                f"this rank holds no part of the level-{self.level} patch {self.box}: "
                f"it is on ranks {self.ranks} (see AMRHierarchy.local_patches)"
            )
        if name is None:
            cell_fields = [
                array
                for array in self._fields.values()
                if tuple(array.shape) == self.shape
            ]
            if not cell_fields:
                raise RuntimeError(
                    "allocate a cell-centered field before requesting local cell centers"
                )
            layout = cell_fields[0].layout
        else:
            layout = self.field(name).layout
            if tuple(layout.shape) != self.shape:
                raise ValueError(f"field {name!r} is not cell-centered")
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

    Under MPI every rank builds the same hierarchy, with the same calls in the
    same order. Each patch then lives on some of the ranks of ``comm`` and has
    a communicator of its own for them: a small patch on one rank, a large one
    on several, over which mpiarray splits its fields. ``placement`` says
    where a patch goes that is added without ``ranks``: ``"all"`` puts it on
    every rank, ``"balanced"`` on the one rank that holds the fewest cells so
    far.

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
        placement: str = "all",
        base_ranks: Sequence[int | Sequence[int] | None] | None = None,
        comm: Any = None,
    ) -> None:
        """Create the base level: ``base_shape`` cells over the physical ``bounds``.

        ``base_boxes`` are half-open boxes of base cells that do not overlap;
        their union is the domain. By default it is the whole rectangle.
        ``base_ranks`` gives the ranks of each base box, as ``ranks`` of
        :meth:`add_patch` does for a patch; by default ``placement`` decides.
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
        if placement not in ("all", "balanced"):
            raise ValueError(
                f"placement must be 'all' or 'balanced', got {placement!r}"
            )
        self.placement = placement
        # one communicator per group of ranks, shared by the patches on it (and
        # by the hierarchies regrid makes from this one, which set it beforehand)
        if not hasattr(self, "_communicators"):
            self._communicators: dict[tuple[int, ...], Any] = {}
        self._cells_on_rank = [0.0] * self.comm.Get_size()
        # halo exchanges and restrictions, by kind and field; rebuilt after add_patch
        self._schedules: dict[tuple[str, str], Any] = {}
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
        if base_ranks is None:
            base_ranks = [None] * len(boxes)
        if len(base_ranks) != len(boxes):
            raise ValueError("base_ranks needs one entry per base box")
        self._patches: list[Patch] = []
        for box, ranks in zip(boxes, base_ranks, strict=True):
            self._patches.append(self._new_patch(0, box, ranks))

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

    def _new_patch(
        self, level: int, box: IndexBox, ranks: int | Sequence[int] | None
    ) -> Patch:
        """A patch on ``ranks`` (or where ``placement`` puts it), with its communicator."""
        size = self.comm.Get_size()
        cells = (box[0][1] - box[0][0]) * (box[1][1] - box[1][0])
        if ranks is None:
            if self.placement == "all":
                chosen = tuple(range(size))
            else:
                chosen = (min(range(size), key=lambda r: (self._cells_on_rank[r], r)),)
        else:
            listed = [ranks] if isinstance(ranks, int) else list(ranks)
            chosen = tuple(sorted({int(rank) for rank in listed}))
            if not chosen or len(chosen) != len(listed):
                raise ValueError(f"ranks must be distinct and not empty, got {ranks!r}")
            if chosen[0] < 0 or chosen[-1] >= size:
                raise ValueError(
                    f"ranks {chosen} are not all ranks of a communicator of {size}"
                )
        for rank in chosen:
            self._cells_on_rank[rank] += cells / len(chosen)
        return Patch(
            level=level, box=box, ranks=chosen, comm=self._communicator(chosen)
        )

    def _communicator(self, ranks: tuple[int, ...]) -> Any:
        """The communicator of a group of ranks; ``None`` on a rank outside it.

        Collective for a group that is neither one rank nor all of them (a
        split of the hierarchy's communicator), the first time the group is used.
        """
        if len(ranks) == self.comm.Get_size():
            return self.comm
        if ranks not in self._communicators:
            rank = self.comm.Get_rank()
            if len(ranks) == 1:
                group = MPI.COMM_SELF if rank == ranks[0] else None
            else:
                member = rank in ranks
                group = self.comm.Split(color=0 if member else MPI.UNDEFINED, key=rank)
                if not member:
                    group = None
            self._communicators[ranks] = group
        return self._communicators[ranks]

    @property
    def local_patches(self) -> tuple[Patch, ...]:
        """The patches this rank holds a part of, ordered as :attr:`patches`."""
        return tuple(patch for patch in self.patches if patch.is_local)

    @property
    def cells_on_rank(self) -> tuple[float, ...]:
        """Cells of all patches held by every rank (a patch on several ranks is shared)."""
        return tuple(self._cells_on_rank)

    def gather(self, patch: Patch, name: str, root: int | None = None) -> Any:
        """The whole of field ``name`` of ``patch`` as a NumPy array.

        Collective over the hierarchy's communicator: every rank calls it,
        whether it holds the patch or not. The array is returned on every rank,
        or with ``root`` on that rank only (``None`` on the others).
        """
        size = self.comm.Get_size()
        rank = self.comm.Get_rank()
        # collective over the patch's own ranks: each of them gets the whole array
        data = patch.field(name).to_numpy() if patch.is_local else None
        if size == 1 or len(patch.ranks) == size:
            return data if root is None or rank == root else None
        holder = patch.ranks[0]
        if root is None:
            return self.comm.bcast(data if rank == holder else None, root=holder)
        if root in patch.ranks:
            return data if rank == root else None
        if rank == holder:
            self.comm.send(data, dest=root, tag=_GATHER_TAG)
        if rank == root:
            return self.comm.recv(source=holder, tag=_GATHER_TAG)
        return None

    def add_patch(
        self,
        level: int,
        box: Sequence[Sequence[int]],
        ranks: int | Sequence[int] | None = None,
    ) -> Patch:
        """Add a non-overlapping, properly nested fine patch.

        ``box`` is half-open and measured in cells at ``level``. Bounds must
        align to the cells of the level below, and the patch must lie within
        the patches of that level (it may span several that touch).

        ``ranks`` are the ranks of the hierarchy's communicator that hold the
        patch: one rank or several, over which its fields are split. Without
        it, ``placement`` decides. Under MPI every rank adds the same patches
        in the same order.
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
        (i0, i1), (j0, j1) = normalized
        coarse: IndexBox = ((i0 // ratio, i1 // ratio), (j0 // ratio, j1 // ratio))
        covered = sum(
            _shared_cells(coarse, parent.box)
            for parent in self.patches_at_level(level - 1)
        )
        if covered != _shared_cells(coarse, coarse):
            raise ValueError(
                f"patch {normalized} is not contained in the level-{level - 1} patches"
            )
        patch = self._new_patch(level, normalized, ranks)
        self._schedules.clear()
        if patch.is_local:
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
            comm=patch.comm,
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
        conservative: bool = False,
    ) -> None:
        """Allocate a named zero-filled field on every existing patch.

        A patch's arrays exist on the ranks that hold it. The default splits
        them over those ranks along the first axis (which then needs at least
        one cell per rank). Use ``split=None`` to replicate them instead. With
        ``centering="node"`` the values sit on the cell corners: a patch of
        ``(nx, ny)`` cells holds ``(nx + 1, ny + 1)`` of them, and patches that
        touch or nest share the nodes they have in common.

        A ``conservative`` field is an amount per area, such as a density, whose
        integral (:meth:`integral`) :meth:`regrid` keeps: it is restricted
        before a regrid, and the cells of new patches are filled from the level
        below so that the fine cells of a coarse cell keep its mean. It must be
        cell-centered.
        """
        if not name or name in self._field_specs:
            raise ValueError(f"field name must be non-empty and unique, got {name!r}")
        if centering not in ("cell", "node"):
            raise ValueError(f"centering must be 'cell' or 'node', got {centering!r}")
        if conservative and centering != "cell":
            raise ValueError("a conservative field must be cell-centered")
        spec = {
            "dtype": dtype,
            "halo": halo,
            "split": split,
            "centering": centering,
            "conservative": bool(conservative),
        }
        arrays = {id(p): self._zeros(p, spec) for p in self._patches if p.is_local}
        self._field_specs[name] = spec
        for patch in self._patches:
            if patch.is_local:
                patch._fields[name] = arrays[id(patch)]

    def exchange_halos(self, name: str) -> None:
        """Fill the halo of every patch of field ``name`` from the rest of the hierarchy.

        From the patches of the same level the halo overlaps (across periodic
        faces too), else by bilinear interpolation of the level below, and
        between the ranks of each patch. Halo points outside the domain are
        left for the boundary conditions. Collective over ``comm``; the
        schedule is built at the first call and kept until a patch is added.
        See :class:`~patchwork.halos.HaloExchange`.
        """
        from patchwork.halos import HaloExchange

        self._schedule("halo", name, HaloExchange).execute()

    def restrict(self, name: str) -> None:
        """Give the points of field ``name`` under finer patches the fine values.

        The mean of the fine cells (cell-centered) or the coinciding fine node
        (node-centered), finest level first. Collective over ``comm``; call
        :meth:`exchange_halos` after it for the halos. See
        :class:`~patchwork.halos.Restriction`.
        """
        from patchwork.halos import Restriction

        self._schedule("restrict", name, Restriction).execute()

    def regrid(
        self,
        tags: Mapping[int, Any],
        *,
        efficiency: float = 0.7,
        buffer: int = 1,
        nesting: int = 1,
        max_patch_size: int | None = None,
        placement: str | None = None,
    ) -> AMRHierarchy:
        """A new hierarchy with patches over the tagged cells, and the fields moved to it.

        ``tags`` maps a level to booleans of that level's shape
        (:meth:`level_shape`): the cells that need the next finer level. Make
        them with :func:`~patchwork.regrid.gradient_tags`,
        :func:`~patchwork.regrid.function_tags` or by hand; each rank may tag
        from the data it holds, and the tags are combined over the ranks.

        Level by level from the coarsest, the tagged cells, grown by
        ``buffer`` cells, are covered with boxes by
        :func:`~patchwork.regrid.cluster` (at least ``efficiency`` of their
        cells tagged, at most ``max_patch_size`` cells of the new level long).
        A level also covers, with a margin, what the finer levels need, and
        each patch keeps ``nesting`` cells of the level below between itself
        and that level's edge (the edge of the domain excepted). The base level
        stays as it is. The levels above the last tagged one are dropped.

        Every field is moved: a point takes the value of an old patch of the
        same level that has it, else the bilinear interpolation of the new
        level below. A conservative field (see :meth:`allocate_field`) is first
        restricted on this hierarchy, and its new cells are filled so that the
        fine cells of a coarse cell keep its mean: its :meth:`integral` is the
        same before and after. Halos are not filled; call :meth:`exchange_halos`.

        New patches are placed by ``placement`` (default: this hierarchy's),
        the largest first. Collective over ``comm``; this hierarchy is left as
        it is, except for the restriction of conservative fields.
        """
        from patchwork.regrid import combine_tags, new_boxes, transfer

        combined = combine_tags(self, tags)
        boxes = new_boxes(
            self,
            combined,
            efficiency=efficiency,
            buffer=buffer,
            nesting=nesting,
            max_patch_size=max_patch_size,
        )
        base = self.patches_at_level(0)
        new = AMRHierarchy.__new__(AMRHierarchy)
        new._communicators = self._communicators
        new.__init__(  # type: ignore[misc]
            self.base_shape,  # type: ignore[arg-type]
            self.bounds,  # type: ignore[arg-type]
            refinement_ratio=self.refinement_ratio,
            periodic=self.periodic,
            base_boxes=[patch.box for patch in base],
            base_ranks=[patch.ranks for patch in base],
            placement=self.placement if placement is None else placement,
            comm=self.comm,
        )
        for level in sorted(boxes):
            for box in sorted(boxes[level], key=lambda b: -_shared_cells(b, b)):
                new.add_patch(level, box)
        for name, spec in self._field_specs.items():
            if spec["conservative"]:
                self.restrict(name)
            new.allocate_field(name, **spec)
            transfer(self, new, name)
        return new

    def integral(self, name: str) -> float:
        """The integral of field ``name`` over the domain, from the finest values.

        Cell-centered: the sum of value times area over the cells that no finer
        patch covers. Node-centered: the trapezoidal rule over those cells.
        Collective over ``comm``; the same value on every rank.
        """
        import numpy as np

        if self.centering(name) == "node":
            from patchwork.nodal import CompositeNodes

            nodes = CompositeNodes(self)
            return float(nodes.volumes @ nodes.from_patches(name))
        ratio = self.refinement_ratio
        total = 0.0
        for patch in self.local_patches:
            array = patch.field(name)
            (b0, b1), (c0, c1) = array.layout.index_bounds
            (i0, _), (j0, _) = patch.box
            leaf = np.ones((b1 - b0, c1 - c0), dtype=bool)
            # this rank's block, in the patch's level indices
            block = ((i0 + b0, i0 + b1), (j0 + c0, j0 + c1))
            for finer in self.patches_at_level(patch.level + 1):
                (f0, f1), (g0, g1) = finer.box
                under = ((f0 // ratio, f1 // ratio), (g0 // ratio, g1 // ratio))
                lo = [max(a[0], b[0]) for a, b in zip(under, block, strict=True)]
                hi = [min(a[1], b[1]) for a, b in zip(under, block, strict=True)]
                if lo[0] < hi[0] and lo[1] < hi[1]:
                    leaf[
                        lo[0] - block[0][0] : hi[0] - block[0][0],
                        lo[1] - block[1][0] : hi[1] - block[1][0],
                    ] = False
            dx, dy = self.spacing(patch.level)
            total += float(np.asarray(array.local)[leaf].sum()) * dx * dy
        return float(self.comm.allreduce(total, op=MPI.SUM))

    def _schedule(self, kind: str, name: str, build: Any) -> Any:
        if name not in self._field_specs:
            raise KeyError(f"hierarchy has no field {name!r}")
        key = (kind, name)
        if key not in self._schedules:
            self._schedules[key] = build(self, name)
        return self._schedules[key]

    def __iter__(self) -> Iterator[Patch]:
        """Iterate through patches in level order."""
        return iter(self.patches)
