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

#: A half-open box of cells, ``((i_lower, i_upper),)`` in 1D and
#: ``((i_lower, i_upper), (j_lower, j_upper))`` in 2D.
IndexBox = tuple[tuple[int, int], ...]
PhysicalBounds = tuple[tuple[float, float], ...]

#: The numbers of dimensions a hierarchy may have.
DIMENSIONS = (1, 2)

#: The geometries of a hierarchy: a Cartesian plane ``(x, y)``, or the
#: axisymmetric ``(r, z)`` plane of a body of revolution (``"cylindrical"``).
GEOMETRIES = ("cartesian", "cylindrical")


def _box(value: Sequence[Sequence[int]], ndim: int = 2) -> IndexBox:
    """Normalize and validate a half-open integer box of ``ndim`` axes."""
    if len(value) != ndim or any(len(axis) != 2 for axis in value):
        expected = ", ".join(f"({a}_lower, {a}_upper)" for a in "ij"[:ndim])
        raise ValueError(f"a patch box must be ({expected},)")
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
    def shape(self) -> tuple[int, ...]:
        """Number of cells along each axis."""
        return tuple(hi - lo for lo, hi in self.box)

    @property
    def node_shape(self) -> tuple[int, ...]:
        """Number of nodes (cell corners) along each axis: one more than cells."""
        return tuple(hi - lo + 1 for lo, hi in self.box)

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
    ) -> tuple[Any, ...]:
        """Coordinates of this rank's nodes of the node-centered field ``name``.

        One array per axis (``x``, or ``x`` and ``y``) of the shape of
        ``field(name).local``, for filling it.
        """
        import numpy as np

        bounds = self.field(name).layout.index_bounds
        axes = self.node_coordinates(hierarchy)
        return tuple(
            np.meshgrid(
                *(a[b0:b1] for a, (b0, b1) in zip(axes, bounds, strict=True)),
                indexing="ij",
            )
        )

    def local_cell_centers(
        self, hierarchy: AMRHierarchy, name: str | None = None
    ) -> tuple[Any, ...]:
        """Coordinates of this rank's cells of the cell-centered field ``name``.

        One array per axis of the shape of ``field(name).local``, for filling
        it. Without ``name``, of the first cell-centered field allocated.
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
        import numpy as np

        axes = [
            x0 + (i0 + np.arange(b0, b1) + 0.5) * h
            for (i0, _), (b0, b1), h, x0 in zip(
                self.box,
                layout.index_bounds,
                hierarchy.spacing(self.level),
                hierarchy.lower_bound,
                strict=True,
            )
        ]
        return tuple(np.meshgrid(*axes, indexing="ij"))

    def node_coordinates(self, hierarchy: AMRHierarchy) -> tuple[Any, ...]:
        """Coordinates of the nodes of the whole patch along each axis (1D arrays)."""
        import numpy as np

        return tuple(
            x0 + np.arange(i0, i1 + 1) * h
            for (i0, i1), h, x0 in zip(
                self.box,
                hierarchy.spacing(self.level),
                hierarchy.lower_bound,
                strict=True,
            )
        )

    def physical_bounds(self, hierarchy: AMRHierarchy) -> PhysicalBounds:
        """Physical interval (1D) or rectangle (2D) occupied by this patch."""
        return tuple(
            (x0 + i0 * h, x0 + i1 * h)
            for (i0, i1), h, x0 in zip(
                self.box,
                hierarchy.spacing(self.level),
                hierarchy.lower_bound,
                strict=True,
            )
        )


class AMRHierarchy:
    """A 1D or 2D nested patch hierarchy with mpiarray-backed fields.

    The number of dimensions is that of ``base_shape``: ``(nx,)`` cells for a
    line (a uniform 1D grid is a hierarchy of one level), ``(nx, ny)`` for a
    plane. Boxes, bounds, ``periodic`` and point coordinates then have one
    entry per axis, and everything below works the same way in both.

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
    the Cartesian plane and totals over the ring in r–z; in 1D they are per
    unit area of the planes normal to ``x``. The r–z plane is 2D only.
    """

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

    def _new_patch(
        self, level: int, box: IndexBox, ranks: int | Sequence[int] | None
    ) -> Patch:
        """A patch on ``ranks`` (or where ``placement`` puts it), with its communicator."""
        size = self.comm.Get_size()
        cells = _shared_cells(box, box)
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

    def __init__(
        self,
        base_shape: tuple[int, ...],
        bounds: PhysicalBounds | None = None,
        *,
        refinement_ratio: int = 2,
        periodic: Sequence[bool] | None = None,
        base_boxes: Sequence[Sequence[Sequence[int]]] | None = None,
        placement: str | Callable[[int, IndexBox], Any] = "all",
        max_cells_per_rank: int | None = None,
        base_ranks: Sequence[int | Sequence[int] | None] | None = None,
        comm: Any = None,
        geometry: str = "cartesian",
    ) -> None:
        """Create the base level: ``base_shape`` cells over the physical ``bounds``.

        ``bounds`` is ``((x_lower, x_upper),)`` in 1D and ``((x_lower,
        x_upper), (y_lower, y_upper))`` in 2D, the unit interval or square by
        default; ``periodic`` has one flag per axis, none periodic by default.
        ``base_boxes`` are half-open boxes of base cells that do not overlap;
        their union is the domain. By default it is the whole rectangle.
        ``base_ranks`` gives the ranks of each base box, as ``ranks`` of
        :meth:`add_patch` does for a patch; by default ``placement`` decides.
        """
        ndim = len(base_shape)
        if ndim not in DIMENSIONS or any(int(n) != n or n <= 0 for n in base_shape):
            raise ValueError(
                "base_shape must contain one or two positive integer cell counts"
            )
        if int(refinement_ratio) != refinement_ratio or refinement_ratio < 2:
            raise ValueError("refinement_ratio must be an integer of at least 2")
        if bounds is None:
            bounds = ((0.0, 1.0),) * ndim
        if len(bounds) != ndim or any(len(axis) != 2 for axis in bounds):
            expected = ", ".join(f"({a}_lower, {a}_upper)" for a in "xy"[:ndim])
            raise ValueError(f"bounds must be ({expected},), one pair per axis")
        if any(hi <= lo for lo, hi in bounds):
            raise ValueError("each physical upper bound must exceed its lower bound")
        self.base_shape = tuple(int(n) for n in base_shape)
        self.bounds = tuple(tuple(float(x) for x in axis) for axis in bounds)
        self.refinement_ratio = int(refinement_ratio)
        if periodic is None:
            periodic = (False,) * ndim
        if len(periodic) != ndim:
            raise ValueError("periodic must have one flag per axis")
        self.periodic = tuple(bool(flag) for flag in periodic)
        if geometry not in GEOMETRIES:
            raise ValueError(f"geometry must be one of {GEOMETRIES}, got {geometry!r}")
        if geometry == "cylindrical":
            if ndim != 2:
                raise ValueError("a cylindrical domain is the 2D r–z plane")
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
        # which patch owns each cell of every level, rebuilt after add_patch (locate)
        self._locators: tuple[Any, ...] | None = None
        self._field_specs: dict[str, dict[str, Any]] = {}
        whole = tuple((0, n) for n in self.base_shape)
        boxes = (
            [whole] if base_boxes is None else [_box(box, ndim) for box in base_boxes]
        )
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
    def ndim(self) -> int:
        """Number of dimensions: 1 or 2."""
        return len(self.base_shape)

    @property
    def lower_bound(self) -> tuple[float, ...]:
        """Physical lower (lower-left) corner of the domain."""
        return tuple(lo for lo, _ in self.bounds)

    @property
    def upper_bound(self) -> tuple[float, ...]:
        """Physical upper (upper-right) corner of the domain."""
        return tuple(hi for _, hi in self.bounds)

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

    def spacing(self, level: int = 0) -> tuple[float, ...]:
        """Cell spacing at a refinement level, per axis."""
        scale = self.refinement_ratio**level
        return tuple(
            (hi - lo) / (n * scale)
            for (lo, hi), n in zip(self.bounds, self.base_shape, strict=True)
        )

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

        ``dx`` on a line (per unit area), ``dx * dy`` in the Cartesian plane;
        ``2 pi r dr dz`` in r–z, with ``r`` the cell's mid-radius (the exact
        volume of the ring). The volume does not depend on the second index.
        """
        import numpy as np

        spacing = self.spacing(level)
        i = np.asarray(i, dtype=float)
        if not self.cylindrical:
            return np.full(i.shape, float(np.prod(spacing)))
        dx, dy = spacing
        radius = self.bounds[0][0] + (i + 0.5) * dx
        return 2.0 * np.pi * radius * dx * dy

    def face_areas(self, level: int, axis: int, i: Any) -> Any:
        """Areas of the faces of ``level`` normal to ``axis``, by first index ``i``.

        For ``axis=0``, ``i`` numbers the faces along the first axis (face ``i``
        is the lower face of cell ``i``); for another axis, it is the first index
        of the cells the faces belong to. ``1`` on a line (per unit area), ``dy``
        or ``dx`` in the Cartesian plane (per unit length out of it); in r–z,
        ``2 pi r dz`` for a radial face at ``r`` and ``2 pi r dr`` for an axial
        face of a cell of mid-radius ``r``, so that the faces of a cell close it
        (:meth:`cell_volumes`) and the faces of the fine cells over a coarse
        face add up to it. The area does not depend on the other indices.
        """
        import numpy as np

        if axis not in range(self.ndim):
            raise ValueError(
                f"axis must be in range({self.ndim}) for a {self.ndim}D hierarchy, "
                f"got {axis!r}"
            )
        spacing = self.spacing(level)
        i = np.asarray(i, dtype=float)
        if not self.cylindrical:
            others = [h for a, h in enumerate(spacing) if a != axis]
            return np.full(i.shape, float(np.prod(others)))
        dx, dy = spacing
        if axis == 0:
            return 2.0 * np.pi * (self.bounds[0][0] + i * dx) * dy
        return 2.0 * np.pi * (self.bounds[0][0] + (i + 0.5) * dx) * dx

    def level_shape(self, level: int) -> tuple[int, ...]:
        """Number of cells in the full domain at a refinement level."""
        scale = self.refinement_ratio**level
        return tuple(n * scale for n in self.base_shape)

    def patches_at_level(self, level: int) -> tuple[Patch, ...]:
        """Patches on one level."""
        return tuple(p for p in self.patches if p.level == level)

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
        normalized = _box(box, self.ndim)
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
        coarse: IndexBox = tuple((lo // ratio, hi // ratio) for lo, hi in normalized)
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
        self._locators = None
        if patch.is_local:
            for name, spec in self._field_specs.items():
                patch._fields[name] = self._zeros(patch, spec)
        self._patches.append(patch)
        return patch

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

    def _schedule(self, kind: str, name: str, build: Any) -> Any:
        if name not in self._field_specs:
            raise KeyError(f"hierarchy has no field {name!r}")
        key = (kind, name)
        if key not in self._schedules:
            with sp.region("patchwork.build_schedule", tags=PROFILE_TAGS):
                self._schedules[key] = build(self, name)
        return self._schedules[key]

    @sp.profile("patchwork.exchange_halos", tags=PROFILE_TAGS)
    def exchange_halos(
        self, name: str, *, level: int | None = None, below: str | None = None
    ) -> None:
        """Fill the halo of every patch of field ``name`` from the rest of the hierarchy.

        From the patches of the same level the halo overlaps (across periodic
        faces too), else by bilinear interpolation of the level below, and
        between the ranks of each patch. Halo points outside the domain are
        left for the boundary conditions. Collective over ``comm``; the
        schedule is built at the first call and kept until a patch is added.
        See :class:`~patchwork.halos.HaloExchange`.

        With ``level``, only the halos of the patches of that level are filled.
        With ``below``, the values interpolated from the level below are those
        of field ``below`` (of the same centering) instead: for a level that
        takes several time steps for one of the level below (subcycling), a
        field that holds the level below at the time in between.
        """
        from patchwork.halos import HaloExchange

        if level is None and below is None:
            self._schedule("halo", name, HaloExchange).execute()
            return
        if below is not None and self.centering(below) != self.centering(name):
            raise ValueError(
                f"field {below!r} is {self.centering(below)}-centered, "
                f"{name!r} is {self.centering(name)}-centered"
            )

        def build(hierarchy: AMRHierarchy, field: str) -> HaloExchange:
            return HaloExchange(hierarchy, field, level=level, below=below)

        self._schedule(f"halo/{level}/{below}", name, build).execute()

    @sp.profile("patchwork.restrict", tags=PROFILE_TAGS)
    def restrict(self, name: str, *, level: int | None = None) -> None:
        """Give the points of field ``name`` under finer patches the fine values.

        The mean of the fine cells (cell-centered) or the coinciding fine node
        (node-centered), finest level first. With ``level``, only the points
        of that level under the next finer one (a level that has caught up
        with the finer ones when they subcycle). Collective over ``comm``;
        call :meth:`exchange_halos` after it for the halos. See
        :class:`~patchwork.halos.Restriction`.
        """
        from patchwork.halos import Restriction

        self._schedule("restrict", name, Restriction).execute(level=level)

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
            # a replicated field (split=None): every rank of the patch holds it all
            if not array.layout.split and patch.comm.Get_rank() != 0:
                continue
            # this rank's block, in the patch's level indices
            block = tuple(
                (lo + b0, lo + b1)
                for (lo, _), (b0, b1) in zip(
                    patch.box, array.layout.index_bounds, strict=True
                )
            )
            leaf = np.ones([hi - lo for lo, hi in block], dtype=bool)
            for finer in self.patches_at_level(patch.level + 1):
                under = tuple((lo // ratio, hi // ratio) for lo, hi in finer.box)
                lo = [max(a[0], b[0]) for a, b in zip(under, block, strict=True)]
                hi = [min(a[1], b[1]) for a, b in zip(under, block, strict=True)]
                if all(a < b for a, b in zip(lo, hi, strict=True)):
                    leaf[
                        tuple(
                            slice(a - first, b - first)
                            for a, b, (first, _) in zip(lo, hi, block, strict=True)
                        )
                    ] = False
            volumes = self.cell_volumes(patch.level, np.arange(*block[0]))
            volumes = volumes.reshape((-1,) + (1,) * (self.ndim - 1))
            weighted = np.asarray(array.local) * volumes
            total += float(weighted[leaf].sum())
        return float(self.comm.allreduce(total, op=MPI.SUM))

    def _locator(self) -> tuple[Any, ...]:
        """The tables of :meth:`locate`: which patch holds each cell, level by level.

        Every level's table covers the bounding box of its patches and holds the
        index into :attr:`patches` of the patch over each cell, -1 where there is
        none. Returns the arguments of the ``locate_points`` kernel that describe
        the hierarchy, always for two axes (a 1D hierarchy gets one cell along
        ``y``): the domain's bounds and periodic axes, the spacing and cell count
        of the finest level, the span of a cell of each level on the finest one,
        each table's lower corner and shape, where it starts in the flat array of
        all tables, and that array. Built at the first call, kept until a patch is
        added.
        """
        import numpy as np

        if self._locators is None:
            pad = 2 - self.ndim
            finest = self.max_level
            corners, shapes, flat = [], [], []
            for level in range(finest + 1):
                members = [
                    (number, patch)
                    for number, patch in enumerate(self.patches)
                    if patch.level == level
                ]
                low = [
                    min(patch.box[axis][0] for _, patch in members)
                    for axis in range(self.ndim)
                ]
                high = [
                    max(patch.box[axis][1] for _, patch in members)
                    for axis in range(self.ndim)
                ]
                table = np.full(
                    [b - a for a, b in zip(low, high, strict=True)], -1, dtype=np.int32
                )
                for number, patch in members:
                    table[
                        tuple(
                            slice(lo - first, hi - first)
                            for (lo, hi), first in zip(patch.box, low, strict=True)
                        )
                    ] = number
                corners.append(low + [0] * pad)
                shapes.append(list(table.shape) + [1] * pad)
                flat.append(table.reshape(-1))
            offsets = np.zeros(len(flat) + 1, dtype=np.int64)
            np.cumsum([table.size for table in flat], out=offsets[1:])
            ratio = self.refinement_ratio
            self._locators = (
                np.asarray(list(self.lower_bound) + [0.0] * pad, dtype=float),
                np.asarray(list(self.upper_bound) + [1.0] * pad, dtype=float),
                np.asarray(list(self.periodic) + [False] * pad, dtype=np.int64),
                np.asarray(list(self.spacing(finest)) + [1.0] * pad, dtype=float),
                np.asarray(list(self.level_shape(finest)) + [1] * pad, dtype=np.int64),
                np.asarray(
                    [ratio ** (finest - level) for level in range(finest + 1)],
                    dtype=np.int64,
                ),
                np.asarray(corners, dtype=np.int64),
                np.asarray(shapes, dtype=np.int64),
                offsets,
                np.concatenate(flat),
            )
        return self._locators

    def _cells(self, coordinates: Sequence[Any]) -> tuple[Any, tuple[Any, ...], tuple]:
        """Where flat arrays of points are: patch, cell and wrapped coordinates.

        ``coordinates`` holds one flat array per axis. Returns, for each point,
        the index of the finest patch that holds it (-1 outside the domain),
        its cell on the level of that patch (level indices, one array per
        axis), and the coordinates wrapped into the domain on periodic axes.
        One pass of the compiled ``locate_points`` kernel.
        """
        import numpy as np

        from patchwork.kernels.locate_points import kernel as locate_points

        size = coordinates[0].size
        points = np.zeros((2, size))
        for axis, values in enumerate(coordinates):
            points[axis] = values
        found = np.empty(size, dtype=np.int64)
        cell = np.empty((2, size), dtype=np.int64)
        wrapped = np.empty((2, size), dtype=float)
        if size:  # the compiled kernel takes no empty arrays
            locate_points(points, *self._locator(), found, cell, wrapped)
        return found, tuple(cell[: self.ndim]), tuple(wrapped[: self.ndim])

    def _points(self, coordinates: Sequence[Any]) -> tuple[tuple, tuple[int, ...]]:
        """Flat float arrays of the coordinates of points, and their common shape."""
        import numpy as np

        if len(coordinates) != self.ndim:
            names = ", ".join("xy"[: self.ndim])
            raise ValueError(
                f"a {self.ndim}D hierarchy takes {self.ndim} coordinate arrays "
                f"({names}), got {len(coordinates)}"
            )
        arrays = [np.asarray(values, dtype=float) for values in coordinates]
        shape = arrays[0].shape
        if any(array.shape != shape for array in arrays):
            raise ValueError(
                "x and y must have the same shape, got "
                + " and ".join(str(array.shape) for array in arrays)
            )
        return tuple(array.reshape(-1) for array in arrays), shape

    def locate(self, *coordinates: Any) -> Any:
        """The finest patch that holds each point: its index in :attr:`patches`.

        ``locate(x)`` in 1D, ``locate(x, y)`` in 2D: arrays of the same shape
        (physical coordinates); the result has that shape and is -1 for a point
        outside the domain, such as one in the notch of an L. Cells are half
        open, so a point on the edge between two patches belongs to the one that
        starts there; the upper faces of the domain, though, belong to the last
        cell. On a periodic axis a point outside the domain is wrapped back into
        it first. Local to every rank (every rank holds the whole hierarchy); no
        communication.
        """
        points, shape = self._points(coordinates)
        return self._cells(points)[0].reshape(shape)

    def _owners(self, found: Any, cell: tuple[Any, ...], name: str) -> Any:
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
            local = [
                c[chosen] - lo
                for c, (lo, _) in zip(cell, patches[number].box, strict=True)
            ]
            for rank, bounds in members:
                inside = np.ones(chosen.size, dtype=bool)
                for values, (a0, a1) in zip(local, bounds, strict=True):
                    inside &= (values >= a0) & (values < a1)
                result[chosen[inside]] = rank
        return result

    def _split(
        self, arguments: tuple, name: str | None, extra: int, what: str
    ) -> tuple[tuple, tuple, str]:
        """``(coordinates, the next extra arguments, name)`` of a call by position.

        ``name`` is the last positional argument unless it is given by keyword.
        """
        if name is None:
            if not arguments or not isinstance(arguments[-1], str):
                raise TypeError(f"{what} needs the name of a field")
            arguments, name = arguments[:-1], arguments[-1]
        if len(arguments) != self.ndim + extra:
            names = ", ".join([*"xy"[: self.ndim], *(["charge"] if extra else [])])
            raise TypeError(
                f"{what} of a {self.ndim}D hierarchy takes ({names}, name), got "
                f"{len(arguments)} arrays"
            )
        return arguments[: self.ndim], arguments[self.ndim :], name

    def owner(self, *points: Any, name: str | None = None) -> Any:
        """The rank that should hold each marker: -1 outside the domain.

        ``owner(x, name)`` in 1D, ``owner(x, y, name)`` in 2D. A marker belongs
        to the rank that holds the block of field ``name`` in which the lower
        (left) corner of its cell lies, in the finest patch that holds it (see
        :meth:`locate`). That block is the one :meth:`deposit` writes into: the
        other corners of the cell are in its halo, which :meth:`accumulate_halos`
        hands on. The same for a cell-centered and a node-centered field, since
        cell ``(i, j)`` starts at node ``(i, j)``. Collective the first time for
        a field (until a patch is added), as the ranks tell each other their
        blocks; the same result on every rank for the same points.
        """
        coordinates, _, name = self._split(points, name, 0, "owner")
        flat, shape = self._points(coordinates)
        found, cell, _ = self._cells(flat)
        return self._owners(found, cell, name).reshape(shape)

    @sp.profile("patchwork.migrate", tags=PROFILE_TAGS)
    def migrate(
        self, *points_and_arrays: Any, name: str, outside: str = "error"
    ) -> tuple[Any, ...]:
        """Send markers to the ranks that hold their patches; returns what this rank gets.

        ``migrate(x, *arrays, name=...)`` in 1D, ``migrate(x, y, *arrays,
        name=...)`` in 2D: the positions of this rank's markers (1D arrays) and
        any other per-marker data, with one row per marker (velocities as
        ``v.T`` if they are stored ``(3, n)``). Each marker goes to
        :meth:`owner` of field ``name``, with :func:`mpiarray.migrate`. Returns
        the positions and ``arrays`` of the markers now on this rank, as given:
        those that stayed and those that arrived, ordered by the rank they came
        from. Call it again on the hierarchy :meth:`regrid` returns to move the
        markers whose patch changed ranks.

        A marker outside the domain raises ``ValueError`` on every rank with
        ``outside="error"``, or is dropped with ``outside="drop"``. Collective.
        """
        import numpy as np

        if outside not in ("error", "drop"):
            raise ValueError(f"outside must be 'error' or 'drop', got {outside!r}")
        coordinates = points_and_arrays[: self.ndim]
        arrays = points_and_arrays[self.ndim :]
        flat, shape = self._points(coordinates)
        if len(shape) != 1:
            raise ValueError("the positions must be one-dimensional")
        for array in arrays:
            if np.shape(array)[:1] != shape:
                raise ValueError(
                    f"an array of shape {np.shape(array)} does not have one row per "
                    f"marker ({shape[0]})"
                )
        found, cell, _ = self._cells(flat)
        destinations = self._owners(found, cell, name)
        lost = destinations < 0
        total = int(self.comm.allreduce(int(lost.sum()), op=MPI.SUM))
        if total and outside == "error":
            raise ValueError(f"{total} markers are outside the domain")
        keep = ~lost
        received = mpa.migrate(
            destinations[keep],
            *(values[keep] for values in flat),
            *(np.asarray(array)[keep] for array in arrays),
            comm=self.comm,
        )
        return tuple(np.asarray(array) for array in received)

    def _stored_block(self, patch: Patch, name: str) -> tuple[Any, Any]:
        """This rank's block of field ``name`` on ``patch``, halo included, and its grid.

        The grid is an :class:`mpifield.BlockGrid` of the stored nodes (the first one is
        the first halo node), so mpifield's weightings run on the block as on a field's.
        """
        from mpifield import BlockGrid

        array = patch.field(name)
        layout = array.layout
        spacing = self.spacing(patch.level)
        storage = array.local_with_halos
        first = [
            self.lower_bound[axis]
            + spacing[axis]
            * (patch.box[axis][0] + layout.index_bounds[axis][0] - layout.halo[axis])
            for axis in range(self.ndim)
        ]
        return storage, BlockGrid.from_geometry(tuple(first), spacing, storage.shape)

    @sp.profile("patchwork.deposit", tags=PROFILE_TAGS)
    def deposit(
        self, *arguments: Any, name: str | None = None, reset: bool = True
    ) -> None:
        """Deposit the charge of markers on the node-centered field ``name``, patch by patch.

        ``deposit(x, charge, name)`` in 1D, ``deposit(x, y, charge, name)`` in
        2D. Each marker spreads its ``charge`` over the corners of its cell (two
        in 1D, four in 2D), in the finest patch that holds it, with linear or
        bilinear (cloud-in-cell) weights: the same weights as
        :meth:`CompositeNodes.deposit <patchwork.nodal.CompositeNodes.deposit>`,
        whose transposed interpolation the field gather is, and of
        :meth:`interpolate`; mpifield's compiled kernels do the work. The values are
        charges, not a density. The markers must be on the ranks that
        :meth:`owner` gives (use :meth:`migrate`), so that the corners fall in
        the block or its halo, and the field needs a halo of at least one node.
        The corners in the halo, and the nodes that several patches share, are
        not yet where the composite mesh wants them: follow with
        :meth:`accumulate_halos`. With ``reset`` the field is zeroed first,
        halos included. Collective (the check of the owners).
        """
        import numpy as np

        coordinates, (charge,), name = self._split(arguments, name, 1, "deposit")
        spec = self._field_specs.get(name)
        if spec is None:
            raise KeyError(f"hierarchy has no field {name!r}")
        if spec["centering"] != "node":
            raise ValueError("deposit is for node-centered fields")
        if spec["halo"] < 1:
            raise ValueError("deposit needs a halo of at least one node")
        flat, _ = self._points(coordinates)
        size = flat[0].size
        charge = np.asarray(charge, dtype=float).reshape(-1)
        if charge.size != size:
            raise ValueError(f"{size} points but {charge.size} charges")
        found, cell, wrapped = self._cells(flat)
        wrong = (found >= 0) & (self._owners(found, cell, name) != self.comm.Get_rank())
        if self.comm.allreduce(int(wrong.sum()), op=MPI.SUM):
            raise ValueError(
                "markers must be on the rank that owns them (see owner and migrate)"
            )
        if reset:
            for patch in self.local_patches:
                patch.field(name).local_with_halos[...] = 0.0
        from mpifield import LINEAR

        patches = self.patches
        for number in np.unique(found[found >= 0]):
            chosen = np.flatnonzero(found == number)
            storage, grid = self._stored_block(patches[number], name)
            positions = np.array([w[chosen] for w in wrapped])
            # the kernels take the nodes component-last: a view with one component
            LINEAR.scatter_block(positions, charge[chosen], storage[..., None], grid)

    @sp.profile("patchwork.interpolate", tags=PROFILE_TAGS)
    def interpolate(self, *arguments: Any, name: str | None = None) -> Any:
        """Interpolate the node-centered field ``name`` at markers, each in its own patch.

        ``interpolate(x, name)`` in 1D, ``interpolate(x, y, name)`` in 2D. Each marker
        reads the corners of its cell in the finest patch that holds it, with the
        linear or bilinear weights of :meth:`deposit`, of which this is the transpose:
        for a field computed patch by patch, such as an electric field that jumps at a
        coarse-fine edge, every marker sees the field of its own patch. The markers
        must be on the ranks that :meth:`owner` gives (use :meth:`migrate`) and the
        halos filled (:meth:`exchange_halos`), since a cell's upper corners may be in
        the halo. A marker outside the domain gets zero. Returns an array of the shape
        of ``x``. Collective (the check of the owners).
        """
        import numpy as np
        from mpifield import LINEAR

        coordinates, _, name = self._split(arguments, name, 0, "interpolate")
        if self.centering(name) != "node":
            raise ValueError("interpolate is for node-centered fields")
        flat, shape = self._points(coordinates)
        found, cell, wrapped = self._cells(flat)
        wrong = (found >= 0) & (self._owners(found, cell, name) != self.comm.Get_rank())
        if self.comm.allreduce(int(wrong.sum()), op=MPI.SUM):
            raise ValueError(
                "markers must be on the rank that owns them (see owner and migrate)"
            )
        values = np.zeros(flat[0].size)
        patches = self.patches
        for number in np.unique(found[found >= 0]):
            chosen = np.flatnonzero(found == number)
            storage, grid = self._stored_block(patches[number], name)
            positions = np.array([w[chosen] for w in wrapped])
            result = np.zeros((1, chosen.size))
            LINEAR.gather_block(positions, storage[..., None], result, grid)
            values[chosen] = result[0]
        return values.reshape(shape)

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

    def __iter__(self) -> Iterator[Patch]:
        """Iterate through patches in level order."""
        return iter(self.patches)
