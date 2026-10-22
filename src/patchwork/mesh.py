"""Block-structured adaptive mesh hierarchy and mpiarray-backed patch fields."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mpiarray as mpa
import scope_profiler as sp
from maybempi import MPI

from patchwork._profiling import PROFILE_TAGS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from mpiarray import DistributedArray

IndexBox = tuple[tuple[int, int], tuple[int, int]]
PhysicalBounds = tuple[tuple[float, float], tuple[float, float]]

#: The geometries of a hierarchy: a Cartesian plane ``(x, y)``, or the
#: axisymmetric ``(r, z)`` plane of a body of revolution (``"cylindrical"``).
GEOMETRIES = ("cartesian", "cylindrical")


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


def _gather_blocks(
    hierarchy: AMRHierarchy, name: str
) -> dict[int, tuple[list[tuple[int, IndexBox]], bool]]:
    """Which block of field ``name`` every rank holds, by patch number (collective).

    For each patch (index in :attr:`AMRHierarchy.patches`): the ranks of the hierarchy's
    communicator that hold a part of it, each with its block ``((a0, a1), (b0, b1))`` in
    the patch's own indices, and whether the field is replicated (every rank of a patch
    on several ranks holds all of it).
    """
    mine = []
    rank = hierarchy.comm.Get_rank()
    for number, patch in enumerate(hierarchy.patches):
        if patch.is_local:
            mine.append((number, rank, tuple(patch.field(name).layout.index_bounds)))
    table: dict[int, list[tuple[int, IndexBox]]] = {}
    for part in hierarchy.comm.allgather(mine):
        for number, holder, bounds in part:
            table.setdefault(number, []).append((holder, bounds))
    result = {}
    for number in range(len(hierarchy.patches)):
        members = sorted(table[number])
        replicated = len(members) > 1 and all(
            bounds == members[0][1] for _, bounds in members
        )
        result[number] = (members, replicated)
    return result


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
    far, or a function ``placement(level, box)`` that returns the ranks (as
    ``ranks`` of :meth:`add_patch` takes them). With ``max_cells_per_rank``, a
    ``"balanced"`` patch with more cells than that is split over as many of
    the least loaded ranks as it needs (at most all of them).

    The domain is the rectangle of ``base_shape`` cells, or with ``base_boxes``
    the union of those boxes of base cells: an L, a T, a rectangle with a notch.
    The boxes are the patches of level zero. Where the domain does not fill
    the rectangle, its edge is a boundary like the rectangle's own faces.

    ``geometry`` is ``"cartesian"``, a plane ``(x, y)``, or ``"cylindrical"``,
    the axisymmetric plane ``(r, z)`` of a body of revolution: the first axis
    is the radius (``bounds[0][0] >= 0``, not periodic), and a cell is a ring
    of volume ``2 pi r dr dz`` (:meth:`cell_volumes`). :meth:`integral`,
    :meth:`restrict` and the conservative interpolation of :meth:`regrid` then
    weight cells by their volumes. Fields are per unit length along ``z`` in
    the Cartesian plane and totals over the ring in r–z.
    """

    def __init__(
        self,
        base_shape: tuple[int, int],
        bounds: PhysicalBounds = ((0.0, 1.0), (0.0, 1.0)),
        *,
        refinement_ratio: int = 2,
        periodic: tuple[bool, bool] = (False, False),
        base_boxes: Sequence[Sequence[Sequence[int]]] | None = None,
        placement: str | Callable[[int, IndexBox], Any] = "all",
        max_cells_per_rank: int | None = None,
        base_ranks: Sequence[int | Sequence[int] | None] | None = None,
        comm: Any = None,
        geometry: str = "cartesian",
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
        if geometry not in GEOMETRIES:
            raise ValueError(f"geometry must be one of {GEOMETRIES}, got {geometry!r}")
        if geometry == "cylindrical":
            if self.bounds[0][0] < 0.0:
                raise ValueError("a cylindrical domain needs a radius r >= 0")
            if self.periodic[0]:
                raise ValueError(
                    "the radius of a cylindrical domain cannot be periodic"
                )
        self.geometry = geometry
        self.comm = mpa.default_comm() if comm is None else comm
        if not callable(placement) and placement not in ("all", "balanced"):
            raise ValueError(
                f"placement must be 'all', 'balanced' or a function, got {placement!r}"
            )
        if max_cells_per_rank is not None and int(max_cells_per_rank) < 1:
            raise ValueError("max_cells_per_rank must be at least 1")
        self.placement = placement
        self.max_cells_per_rank = (
            None if max_cells_per_rank is None else int(max_cells_per_rank)
        )
        # one communicator per group of ranks, shared by the patches on it (and
        # by the hierarchies regrid makes from this one, which set it beforehand)
        if not hasattr(self, "_communicators"):
            self._communicators: dict[tuple[int, ...], Any] = {}
        self._cells_on_rank = [0.0] * self.comm.Get_size()
        # halo exchanges and restrictions, by kind and field; rebuilt after add_patch
        self._schedules: dict[tuple[str, str], Any] = {}
        # which patch owns each cell of a level, rebuilt after add_patch (see locate)
        self._locators: dict[int, tuple[tuple[int, int], Any]] = {}
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

    @property
    def cylindrical(self) -> bool:
        """Whether the hierarchy is the axisymmetric r–z plane."""
        return self.geometry == "cylindrical"

    @property
    def on_axis(self) -> bool:
        """Whether a cylindrical domain reaches the axis (its radius starts at 0)."""
        return self.cylindrical and self.bounds[0][0] == 0.0

    def cell_volumes(self, level: int, i: Any) -> Any:
        """Volumes of the cells of ``level`` with first index ``i`` (an array).

        ``dx * dy`` in the Cartesian plane; ``2 pi r dr dz`` in r–z, with ``r``
        the cell's mid-radius (the exact volume of the ring). The volume does
        not depend on the second index.
        """
        import numpy as np

        dx, dy = self.spacing(level)
        i = np.asarray(i, dtype=float)
        if not self.cylindrical:
            return np.full(i.shape, dx * dy)
        radius = self.bounds[0][0] + (i + 0.5) * dx
        return 2.0 * np.pi * radius * dx * dy

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
        if ranks is None and callable(self.placement):
            ranks = self.placement(level, box)
        if ranks is None:
            if self.placement == "all":
                chosen = tuple(range(size))
            else:
                limit = self.max_cells_per_rank
                wanted = 1 if limit is None else min(size, max(1, -(-cells // limit)))
                least_loaded = sorted(
                    range(size), key=lambda r: (self._cells_on_rank[r], r)
                )
                chosen = tuple(sorted(least_loaded[:wanted]))
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
        self._locators.clear()
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

    @sp.profile("patchwork.exchange_halos", tags=PROFILE_TAGS)
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

    @sp.profile("patchwork.restrict", tags=PROFILE_TAGS)
    def restrict(self, name: str) -> None:
        """Give the points of field ``name`` under finer patches the fine values.

        The mean of the fine cells (cell-centered) or the coinciding fine node
        (node-centered), finest level first. Collective over ``comm``; call
        :meth:`exchange_halos` after it for the halos. See
        :class:`~patchwork.halos.Restriction`.
        """
        from patchwork.halos import Restriction

        self._schedule("restrict", name, Restriction).execute()

    @sp.profile("patchwork.regrid", tags=PROFILE_TAGS)
    def regrid(
        self,
        tags: Mapping[int, Any],
        *,
        efficiency: float = 0.7,
        buffer: int = 1,
        nesting: int = 1,
        max_patch_size: int | None = None,
        min_patch_size: int = 1,
        forbid: Any = None,
        placement: str | Callable[[int, IndexBox], Any] | None = None,
        fill_halos: bool = False,
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
        same before and after. Halos are not filled unless ``fill_halos``
        (then :meth:`exchange_halos` runs for every field of the new
        hierarchy).

        ``min_patch_size`` is the shortest side a patch may have, in cells of
        its own level (rounded up to whole cells of the level below): a patch
        that would be narrower is widened where there is room, so that no
        patch is one cell wide. ``forbid`` keeps patches off places a code
        cannot handle, such as a periodic seam: a mapping from a level
        (1 and up) to booleans of that level's shape (:meth:`level_shape`),
        true where no patch of that level may cover a cell, or a function of
        the cell centres ``(x, y)`` of a level that returns such booleans and
        is used for every refined level. Tags that would need a forbidden cell
        are dropped, as are those that would need patches too close to it.

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
            min_patch_size=min_patch_size,
            forbid=forbid,
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
            max_cells_per_rank=self.max_cells_per_rank,
            comm=self.comm,
            geometry=self.geometry,
        )
        for level in sorted(boxes):
            for box in sorted(boxes[level], key=lambda b: -_shared_cells(b, b)):
                new.add_patch(level, box)
        for name, spec in self._field_specs.items():
            if spec["conservative"]:
                self.restrict(name)
            new.allocate_field(name, **spec)
            transfer(self, new, name)
        if fill_halos:
            for name in new.field_names:
                new.exchange_halos(name)
        return new

    def integral(self, name: str) -> float:
        """The integral of field ``name`` over the domain, from the finest values.

        Cell-centered: the sum of value times area (in r–z: times volume,
        :meth:`cell_volumes`) over the cells that no finer patch covers.
        Node-centered: the trapezoidal rule over those cells (Cartesian only).
        Collective over ``comm``; the same value on every rank.
        """
        import numpy as np

        if self.centering(name) == "node":
            if self.cylindrical:
                raise NotImplementedError(
                    "the integral of a node-centered field in r–z is not implemented"
                )
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
            volumes = self.cell_volumes(patch.level, np.arange(i0 + b0, i0 + b1))
            weighted = np.asarray(array.local) * volumes[:, None]
            total += float(weighted[leaf].sum())
        return float(self.comm.allreduce(total, op=MPI.SUM))

    def _locator(self, level: int) -> tuple[tuple[int, int], Any]:
        """Which patch holds each cell of ``level``: the corner of the table and the table.

        The table covers the bounding box of the level's patches and holds the index
        into :attr:`patches` of the patch over each cell, -1 where there is none.
        """
        import numpy as np

        if level not in self._locators:
            members = [
                (number, patch)
                for number, patch in enumerate(self.patches)
                if patch.level == level
            ]
            i_low = min(patch.box[0][0] for _, patch in members)
            i_high = max(patch.box[0][1] for _, patch in members)
            j_low = min(patch.box[1][0] for _, patch in members)
            j_high = max(patch.box[1][1] for _, patch in members)
            table = np.full((i_high - i_low, j_high - j_low), -1, dtype=np.intp)
            for number, patch in members:
                (i0, i1), (j0, j1) = patch.box
                table[i0 - i_low : i1 - i_low, j0 - j_low : j1 - j_low] = number
            self._locators[level] = ((i_low, j_low), table)
        return self._locators[level]

    def _cells(self, x: Any, y: Any) -> tuple[Any, Any, Any, tuple[Any, Any]]:
        """Where flat arrays of points are: patch, cell and wrapped coordinates.

        Returns, for each point, the index of the finest patch that holds it
        (-1 outside the domain), its cell on the level of that patch (level
        indices), and the coordinates wrapped into the domain on periodic axes.
        """
        import numpy as np

        coordinates = []
        inside = np.ones(x.size, dtype=bool)
        for axis, values in enumerate((x, y)):
            low, high = self.bounds[axis]
            if self.periodic[axis]:
                values = low + np.mod(values - low, high - low)
            inside &= (values >= low) & (values <= high)  # False for NaN
            coordinates.append(values)
        found = np.full(x.size, -1, dtype=np.intp)
        cell_i = np.zeros(x.size, dtype=np.intp)
        cell_j = np.zeros(x.size, dtype=np.intp)
        for level in range(self.max_level, -1, -1):
            pending = np.flatnonzero(inside & (found < 0))
            if pending.size == 0:
                break
            (i_low, j_low), table = self._locator(level)
            cells = []
            for axis in (0, 1):
                low = self.bounds[axis][0]
                spacing = self.spacing(level)[axis]
                count = self.level_shape(level)[axis]
                index = np.floor((coordinates[axis][pending] - low) / spacing)
                cells.append(np.minimum(index.astype(np.intp), count - 1))
            i = cells[0] - i_low
            j = cells[1] - j_low
            within = (i >= 0) & (i < table.shape[0]) & (j >= 0) & (j < table.shape[1])
            owner = np.full(pending.size, -1, dtype=np.intp)
            owner[within] = table[i[within], j[within]]
            hit = owner >= 0
            found[pending[hit]] = owner[hit]
            cell_i[pending[hit]] = cells[0][hit]
            cell_j[pending[hit]] = cells[1][hit]
        return found, cell_i, cell_j, (coordinates[0], coordinates[1])

    @staticmethod
    def _points(x: Any, y: Any) -> tuple[Any, Any, tuple[int, ...]]:
        """Flat float arrays of the coordinates of points, and their common shape."""
        import numpy as np

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.shape != y.shape:
            raise ValueError(
                f"x and y must have the same shape, got {x.shape} and {y.shape}"
            )
        return x.reshape(-1), y.reshape(-1), x.shape

    def locate(self, x: Any, y: Any) -> Any:
        """The finest patch that holds each point: its index in :attr:`patches`.

        ``x`` and ``y`` are arrays of the same shape (physical coordinates); the result
        has that shape and is -1 for a point outside the domain, such as one in the notch of
        an L. Cells are half open, so a point on the edge between two patches belongs to the
        one that starts there; the upper faces of the domain, though, belong to the last
        cell. On a periodic axis a point outside the domain is wrapped back into it first.
        Local to every rank (every rank holds the whole hierarchy); no communication.
        """
        x, y, shape = self._points(x, y)
        return self._cells(x, y)[0].reshape(shape)

    def _owners(self, found: Any, cell_i: Any, cell_j: Any, name: str) -> Any:
        """The rank that holds the block of field ``name`` where each marker's cell starts."""
        import numpy as np

        blocks = self._schedule("blocks", name, _gather_blocks)
        patches = self.patches
        result = np.full(found.shape, -1, dtype=np.int64)
        for number in np.unique(found[found >= 0]):
            members, replicated = blocks[int(number)]
            if replicated:
                raise ValueError(
                    f"field {name!r} is replicated (split=None) on the several ranks "
                    f"of the level-{patches[number].level} patch {patches[number].box}: "
                    "a point has no owner; allocate it with split=0"
                )
            chosen = np.flatnonzero(found == number)
            (i0, _), (j0, _) = patches[number].box
            a, b = cell_i[chosen] - i0, cell_j[chosen] - j0
            for rank, ((a0, a1), (b0, b1)) in members:
                inside = (a >= a0) & (a < a1) & (b >= b0) & (b < b1)
                result[chosen[inside]] = rank
        return result

    def owner(self, x: Any, y: Any, name: str) -> Any:
        """The rank that should hold each marker: -1 outside the domain.

        A marker belongs to the rank that holds the block of field ``name`` in which the
        lower left corner of its cell lies, in the finest patch that holds it (see
        :meth:`locate`). That block is the one :meth:`deposit` writes into: the other
        corners of the cell are in its halo, which :meth:`accumulate_halos` hands on. The
        same for a cell-centered and a node-centered field, since cell ``(i, j)`` starts at
        node ``(i, j)``. Collective the first time for a field (until a patch is added), as
        the ranks tell each other their blocks; the same result on every rank for the same
        points.
        """
        x, y, shape = self._points(x, y)
        found, cell_i, cell_j, _ = self._cells(x, y)
        return self._owners(found, cell_i, cell_j, name).reshape(shape)

    @sp.profile("patchwork.migrate", tags=PROFILE_TAGS)
    def migrate(
        self, x: Any, y: Any, *arrays: Any, name: str, outside: str = "error"
    ) -> tuple[Any, ...]:
        """Send markers to the ranks that hold their patches; returns what this rank gets.

        ``x`` and ``y`` are the positions of this rank's markers (1D) and ``arrays`` any
        other per-marker data, with one row per marker (velocities as ``v.T`` if they are
        stored ``(3, n)``). Each marker goes to :meth:`owner` of field ``name``, with
        :func:`mpiarray.migrate`. Returns ``(x, y, *arrays)`` of the markers now on this
        rank: those that stayed and those that arrived, ordered by the rank they came
        from. Call it again on the hierarchy :meth:`regrid` returns to move the markers
        whose patch changed ranks.

        A marker outside the domain raises ``ValueError`` on every rank with
        ``outside="error"``, or is dropped with ``outside="drop"``. Collective.
        """
        import numpy as np

        if outside not in ("error", "drop"):
            raise ValueError(f"outside must be 'error' or 'drop', got {outside!r}")
        x, y, shape = self._points(x, y)
        if len(shape) != 1:
            raise ValueError("x and y must be one-dimensional")
        for array in arrays:
            if np.shape(array)[:1] != shape:
                raise ValueError(
                    f"an array of shape {np.shape(array)} does not have one row per "
                    f"marker ({shape[0]})"
                )
        found, cell_i, cell_j, _ = self._cells(x, y)
        destinations = self._owners(found, cell_i, cell_j, name)
        lost = destinations < 0
        total = int(self.comm.allreduce(int(lost.sum()), op=MPI.SUM))
        if total and outside == "error":
            raise ValueError(f"{total} markers are outside the domain")
        keep = ~lost
        received = mpa.migrate(
            destinations[keep],
            x[keep],
            y[keep],
            *(np.asarray(array)[keep] for array in arrays),
            comm=self.comm,
        )
        return tuple(np.asarray(array) for array in received)

    @sp.profile("patchwork.deposit", tags=PROFILE_TAGS)
    def deposit(
        self, x: Any, y: Any, charge: Any, name: str, *, reset: bool = True
    ) -> None:
        """Deposit the charge of markers on the node-centered field ``name``, patch by patch.

        Each marker spreads its ``charge`` over the four corners of its cell, in the finest
        patch that holds it, with bilinear (cloud-in-cell) weights: the same weights as
        :meth:`CompositeNodes.deposit <patchwork.nodal.CompositeNodes.deposit>`, whose
        transposed interpolation the field gather is. The values are charges, not a density.
        The markers must be on the ranks that :meth:`owner` gives (use :meth:`migrate`),
        so that the corners fall in the block or its halo, and the field needs a halo of
        at least one node. The corners in the halo, and the nodes that several patches
        share, are not yet where the composite mesh wants them: follow with
        :meth:`accumulate_halos`. With ``reset`` the field is zeroed first, halos included.
        Collective (the check of the owners).
        """
        import numpy as np

        spec = self._field_specs.get(name)
        if spec is None:
            raise KeyError(f"hierarchy has no field {name!r}")
        if spec["centering"] != "node":
            raise ValueError("deposit is for node-centered fields")
        if spec["halo"] < 1:
            raise ValueError("deposit needs a halo of at least one node")
        x, y, shape = self._points(x, y)
        charge = np.asarray(charge, dtype=float).reshape(-1)
        if charge.size != x.size:
            raise ValueError(f"{x.size} points but {charge.size} charges")
        found, cell_i, cell_j, (xw, yw) = self._cells(x, y)
        wrong = (found >= 0) & (
            self._owners(found, cell_i, cell_j, name) != self.comm.Get_rank()
        )
        if self.comm.allreduce(int(wrong.sum()), op=MPI.SUM):
            raise ValueError(
                "markers must be on the rank that owns them (see owner and migrate)"
            )
        if reset:
            for patch in self.local_patches:
                patch.field(name).local_with_halos[...] = 0.0
        patches = self.patches
        for number in np.unique(found[found >= 0]):
            patch = patches[number]
            chosen = np.flatnonzero(found == number)
            array = patch.field(name)
            (b0, _), (c0, _) = array.layout.index_bounds
            h0, h1 = array.layout.halo
            (i0, _), (j0, _) = patch.box
            dx, dy = self.spacing(patch.level)
            x0, y0 = self.lower_bound
            ci, cj = cell_i[chosen], cell_j[chosen]
            fx = (xw[chosen] - x0) / dx - ci
            fy = (yw[chosen] - y0) / dy - cj
            storage = array.local_with_halos
            a = ci - i0 - b0 + h0
            b = cj - j0 - c0 + h1
            columns = storage.shape[1]
            flat, weights = [], []
            for da, wa in ((0, 1.0 - fx), (1, fx)):
                for db, wb in ((0, 1.0 - fy), (1, fy)):
                    flat.append((a + da) * columns + (b + db))
                    weights.append(wa * wb * charge[chosen])
            added = np.bincount(
                np.concatenate(flat),
                weights=np.concatenate(weights),
                minlength=storage.size,
            )
            storage += added.reshape(storage.shape)

    @sp.profile("patchwork.accumulate_halos", tags=PROFILE_TAGS)
    def accumulate_halos(self, name: str) -> None:
        """Bring deposited charge to the composite nodes: the scatter-add after :meth:`deposit`.

        For a node-centered field, in two steps. First each patch adds the halos of its
        blocks into the neighbouring blocks (mpiarray's ``accumulate_halos``; at the edges
        of the patch they are dropped). Then the charge of every copy of a node goes to
        one: the node is held by every patch that has it as a corner, and by the two
        copies of a node on a periodic seam (the upper one, at the end of the axis, is
        the owner there); the owner copy is that of the finest patch
        that holds it, the one :meth:`CompositeNodes.from_patches
        <patchwork.nodal.CompositeNodes.from_patches>` reads, and the last of the patches of
        that level in the order of :attr:`patches`. The other copies end up zero. So the
        total is kept, and ``from_patches`` returns what
        :meth:`CompositeNodes.deposit <patchwork.nodal.CompositeNodes.deposit>` gives for
        the same markers, to round-off.

        This is the whole of the scatter-add of a bilinear deposit: a marker's cell is in
        one patch, so its corners are nodes of that patch and no halo reaches over a
        coarse-fine edge. (The charge on a hanging node is passed to its coarse neighbours
        by :meth:`NodalPoisson.solve <patchwork.nodal.NodalPoisson.solve>`.) Cell-centered
        fields are not supported. Collective; the schedule is built at the first call and
        kept until a patch is added.
        """
        if self.centering(name) != "node":
            raise NotImplementedError(
                "accumulate_halos is for node-centered fields (the scatter-add of a deposit)"
            )
        from patchwork.accumulate import Accumulation

        self._schedule("accumulate", name, Accumulation).execute()

    def _schedule(self, kind: str, name: str, build: Any) -> Any:
        if name not in self._field_specs:
            raise KeyError(f"hierarchy has no field {name!r}")
        key = (kind, name)
        if key not in self._schedules:
            with sp.region("patchwork.build_schedule", tags=PROFILE_TAGS):
                self._schedules[key] = build(self, name)
        return self._schedules[key]

    def __iter__(self) -> Iterator[Patch]:
        """Iterate through patches in level order."""
        return iter(self.patches)
