r"""Node-centered composite Poisson solve on a static patch hierarchy.

The unknowns sit on the cell corners of the composite mesh: the cells of every
patch that no finer patch covers (the *leaf cells*). Nodes that patches share
are one unknown. A node of a fine cell in the middle of a coarser cell's edge is
*hanging*: its value is the linear interpolant of that edge's ends, so the
potential is continuous across coarse-fine interfaces.

The operator is the five-point Laplacian wherever the mesh is uniform. It is
assembled cell by cell: every edge of a leaf cell couples its two ends with
half the conductance of the finite-difference stencil (the other half comes
from the cell on the other side), and a quarter of the cell's area goes to each
corner. A node on a domain face therefore gets a half control volume and no
flux through the face, the natural (Neumann) condition. Hanging nodes are
eliminated by the interpolation, which keeps the operator conservative and, in
the absence of embedded boundaries, symmetric.

In 1D the same assembly gives the three-point Laplacian (the linear finite
elements), with half of each cell's length at each end. A coarse–fine
interface is then a node of both levels: no node hangs, and the coupling
across it is the exact flux between the two cells.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sparse
import scope_profiler as sp

from patchwork._profiling import PROFILE_TAGS
from patchwork.backends import Backend, LinearSystem, backend_of

if TYPE_CHECKING:
    from patchwork.mesh import AMRHierarchy, Patch

#: The faces of the domain's bounding rectangle, as ``(axis, side)`` with side 0
#: the lower one. A 1D domain has the two ``x`` faces (see :func:`faces_of`).
FACES = {"x_lower": (0, 0), "x_upper": (0, 1), "y_lower": (1, 0), "y_upper": (1, 1)}


def faces_of(ndim: int) -> dict[str, tuple[int, int]]:
    """The faces of the bounding box of a domain of ``ndim`` dimensions."""
    return {face: where for face, where in FACES.items() if where[0] < ndim}


#: The rest of the boundary of a domain that does not fill its rectangle
#: (``AMRHierarchy(base_boxes=...)``): the edges inside the rectangle.
INNER = "inner"

Value = float | Callable[..., Any]

#: :class:`CompositeNodes` numbers its nodes with a table over the finest
#: lattice when the lattice has at most this many points per cell corner, and
#: by sorting the corners otherwise (a deep hierarchy with small fine patches).
_DENSE_TABLE_FACTOR = 16


@dataclass(frozen=True)
class Dirichlet:
    """A prescribed potential on a domain face: a number, ``value(x)`` in 1D or ``value(x, y)``."""

    value: Value = 0.0


@dataclass(frozen=True)
class Neumann:
    """A prescribed outward normal derivative on a domain face.

    ``gradient`` is a number or ``gradient(x, y)`` (``gradient(x)`` in 1D),
    evaluated at the middle of every cell edge on the face (at the face's node
    in 1D); zero is the natural condition.
    """

    gradient: Value = 0.0


@dataclass(frozen=True)
class EmbeddedDirichlet:
    """A body inside the domain held at one potential.

    ``inside(points)`` takes coordinates ``(ndim, n)`` inside the domain and returns
    which of them the body contains. Nodes inside are fixed at ``value``. On a
    mesh edge from a node outside to a node inside, the surface is located by
    bisection of ``inside`` (or by ``fraction(outside, inside)``, the fraction
    of the edge from the outside node at which the surface crosses it; across a
    periodic face ``inside`` is then the node's image beyond the face), and the
    equation of the outside node uses the distance to the surface instead of
    the mesh spacing: the Shortley-Weller stencil at nodes with four equal
    cells around them, a two-point flux to the surface elsewhere.
    """

    inside: Callable[[np.ndarray], Any]
    value: float = 0.0
    fraction: Callable[[np.ndarray, np.ndarray], Any] | None = None


def _evaluate(value: Value, *coordinates: np.ndarray) -> np.ndarray:
    """A face value at points (one array per axis): a number, or the result of a callable."""
    result = value(*coordinates) if callable(value) else value
    return np.broadcast_to(np.asarray(result, dtype=float), coordinates[0].shape).copy()


#: The halvings of an edge that locate an embedded surface on it: the fraction
#: is known to 2**-48 of the edge.
_SURFACE_BISECTIONS = 48

#: The cost of a call of a body's ``inside`` in points: a call with this many
#: points costs about twice an empty one. :func:`_surface_crossing` evaluates
#: several bisection steps in one call while the points are cheaper than the call.
_POINTS_PER_CALL = 10_000


def _section_bits(count: int) -> int:
    """The bisection steps per call of ``inside`` for ``count`` edges: a divisor of 48."""
    return min(
        (1, 2, 3, 4, 6, 8),
        key=lambda bits: (
            (_SURFACE_BISECTIONS // bits) * (_POINTS_PER_CALL + count * (2**bits - 1))
        ),
    )


def _surface_crossing(
    inside: Callable[[np.ndarray], Any],
    start: np.ndarray,
    step: np.ndarray,
    wrap: Callable[[np.ndarray], np.ndarray] = lambda points: points,
) -> np.ndarray:
    """Fraction of each segment ``start + t * step`` at which it enters a body.

    ``start`` and ``step`` are ``(ndim, k)``; ``start`` is outside the body,
    ``start + step`` inside. The result is that of :data:`_SURFACE_BISECTIONS`
    bisections of ``inside`` (the upper end of the last interval), but each
    call of ``inside`` evaluates the points of several bisection steps at once
    (all ``2**b - 1`` points that ``b`` steps might visit), which are then
    replayed: the same points and the same answer in a few calls instead of 48.
    ``wrap`` moves the points into the domain (along its periodic axes).
    """
    count = start.shape[1]
    bits = _section_bits(count)
    pieces = 2**bits
    low = np.zeros(count)
    high = np.ones(count)
    # the dyadic points of one round, exact in floating point: low + j / 2**r
    grid = np.arange(1, pieces) / pieces
    rows = np.arange(count)
    for _ in range(_SURFACE_BISECTIONS // bits):
        width = high - low
        middle = low[:, None] + width[:, None] * grid  # (k, pieces - 1)
        points = start[:, :, None] + middle[None] * step[:, :, None]
        hit = np.asarray(inside(wrap(points.reshape(start.shape[0], -1))), dtype=bool)
        # the ends of the interval: outside at 0, inside at ``pieces``
        hit = np.concatenate(
            (
                np.zeros((count, 1), dtype=bool),
                hit.reshape(count, pieces - 1),
                np.ones((count, 1), dtype=bool),
            ),
            axis=1,
        )
        lower = np.zeros(count, dtype=np.int64)
        upper = np.full(count, pieces, dtype=np.int64)
        for _ in range(bits):
            half = (lower + upper) // 2
            found = hit[rows, half]
            upper = np.where(found, half, upper)
            lower = np.where(found, lower, half)
        low, high = low + lower * (width / pieces), low + upper * (width / pieces)
    return high


def _corner_offsets(ndim: int) -> np.ndarray:
    """The corners of a cell, ``(2**ndim, ndim)``, the first axis varying fastest."""
    return np.array(
        [[(k >> axis) & 1 for axis in range(ndim)] for k in range(2**ndim)],
        dtype=np.int64,
    )


class CompositeNodes:
    """The nodes and leaf cells of a hierarchy, and the maps to its patches.

    Nodes are addressed on the *finest lattice*: the nodes of the finest level
    extended over the whole domain, ``lattice = level index * ratio**(finest -
    level)``. On a periodic axis the lattice wraps around, so the last node of
    the axis is the first.

    The arrays below have one column per axis (``ndim``, 1 or 2) and a cell has
    ``2**ndim`` corners.

    Attributes:
        hierarchy: The patch hierarchy.
        lattice: Finest-lattice coordinates of the nodes, ``(n, ndim)`` integers.
        coordinates: Physical coordinates of the nodes, ``(n, ndim)``.
        cell_corners: Node numbers of the corners of every leaf cell,
            ``(m, 2**ndim)``: lower, upper in 1D; lower-left, lower-right,
            upper-left, upper-right in 2D.
        cell_span: Edge length of every leaf cell on the finest lattice.
        cell_lower: Finest-lattice coordinates of the lower (lower-left) corners.
        cell_ranks: The rank of the hierarchy's communicator that holds every
            leaf cell: one of its patch's ranks, the rows of a patch on several
            ranks split over them as its arrays are.
        node_ranks: The rank that owns every node in a distributed solve: the
            lowest of the ranks of the leaf cells around it.
        volumes: Control volume of every node: a share of each cell around it
            (a half in 1D, a quarter in 2D).
        hanging: Whether a node is hanging (interpolated from other nodes).
        reduction: The interpolation ``T``: ``T @ u`` fills the hanging nodes of
            ``u`` from the others (which it leaves unchanged).
    """

    # Construction -------------------------------------------------------------#
    def _key(self, *index: np.ndarray) -> np.ndarray:
        """One integer per finest-lattice point, the same for periodic images.

        ``index`` holds one array of lattice coordinates per axis.
        """
        key = np.zeros(np.shape(index[0]), dtype=np.int64)
        for axis, values in enumerate(index):
            values = np.asarray(values, dtype=np.int64)
            if self._periodic[axis]:
                values = values % self._cells[axis]
            key = key * (int(self._cells[axis]) + 1) + values
        return key

    def _lookup(self, *index: np.ndarray) -> np.ndarray:
        """Node numbers of finest-lattice points, -1 where there is no node."""
        keys = self._key(*index)
        if self._table is not None:
            inside = (keys >= 0) & (keys < self._table.size)
            return np.where(inside, self._table[np.where(inside, keys, 0)], -1)
        position = np.searchsorted(self._keys, keys)
        position = np.minimum(position, self.num_nodes - 1)
        return np.where(self._keys[position] == keys, position, -1)

    def _number(self, keys: np.ndarray) -> None:
        """Number the distinct lattice keys of the cell corners in increasing order.

        Sets :attr:`num_nodes`, :attr:`cell_corners` and the sorted keys. Where
        the finest lattice is not much larger than the corners, a table over the
        lattice replaces the sort and makes :meth:`_lookup` a gather.
        """
        size = int(np.prod(self._cells + 1))
        self._table: np.ndarray | None = None
        if size <= _DENSE_TABLE_FACTOR * keys.size:
            present = np.zeros(size, dtype=bool)
            present[keys.reshape(-1)] = True
            self._keys = np.flatnonzero(present)
            table = np.full(size, -1, dtype=np.int64)
            table[self._keys] = np.arange(self._keys.size)
            self._table = table
            self.cell_corners = table[keys]
        else:
            self._keys, inverse = np.unique(keys, return_inverse=True)
            self.cell_corners = inverse.reshape(keys.shape)
        self.num_nodes = int(self._keys.size)

    def _find_hanging_nodes(self) -> None:
        """Find the nodes inside coarser cells' edges and their interpolation.

        In 1D a cell's faces are nodes, and no node hangs.
        """
        if self.ndim == 1:
            self.hanging = np.zeros(self.num_nodes, dtype=bool)
            self.reduction = sparse.csr_matrix(sparse.identity(self.num_nodes))
            return
        slaves, masters, weights, spans = [], [], [], []
        ll, lr, ul, ur = (self.cell_corners[:, k] for k in range(4))
        for span in np.unique(self.cell_span):
            if span == 1:
                continue
            cells = np.nonzero(self.cell_span == span)[0]
            x0, y0 = self.cell_lower[cells, 0], self.cell_lower[cells, 1]
            for k in range(1, int(span)):
                for i, j, first, second in (
                    (x0 + k, y0, ll, lr),
                    (x0 + k, y0 + span, ul, ur),
                    (x0, y0 + k, ll, ul),
                    (x0 + span, y0 + k, lr, ur),
                ):
                    node = self._lookup(i, j)
                    found = node >= 0
                    slaves.append(node[found])
                    masters.append(
                        np.column_stack((first[cells][found], second[cells][found]))
                    )
                    weights.append(np.full(int(found.sum()), k / span))
                    spans.append(np.full(int(found.sum()), span))
        self.hanging = np.zeros(self.num_nodes, dtype=bool)
        identity = sparse.csr_matrix(sparse.identity(self.num_nodes))
        if not slaves or not sum(part.size for part in slaves):
            self.reduction = identity
            return
        slave = np.concatenate(slaves)
        master = np.concatenate(masters)
        weight = np.concatenate(weights)
        span = np.concatenate(spans)
        # one constraint per node, from the coarsest edge it lies in
        order = np.lexsort((-span, slave))
        slave, master, weight = slave[order], master[order], weight[order]
        first = np.concatenate(([True], slave[1:] != slave[:-1]))
        slave, master, weight = slave[first], master[first], weight[first]
        self.hanging[slave] = True
        free = np.nonzero(~self.hanging)[0]
        reduction = sparse.csr_matrix(
            sparse.coo_matrix(
                (
                    np.concatenate((np.ones(free.size), 1.0 - weight, weight)),
                    (
                        np.concatenate((free, slave, slave)),
                        np.concatenate((free, master[:, 0], master[:, 1])),
                    ),
                ),
                shape=(self.num_nodes, self.num_nodes),
            )
        )
        # an end of the edge may itself hang on a still coarser cell
        for _ in range(self._finest + 1):
            if not reduction[:, slave].nnz:
                break
            reduction = sparse.csr_matrix(reduction @ reduction)
            reduction.eliminate_zeros()
        else:
            raise RuntimeError("the hanging-node constraints do not close")
        self.reduction = reduction

    def _leaf_cells(self) -> np.ndarray:
        """The leaf cell that covers every cell of the finest lattice."""
        if self._leaf_map is None:
            ndim = self.ndim
            leaf = np.full(tuple(self._cells), -1, dtype=np.int64)
            for span in np.unique(self.cell_span):
                cells = np.nonzero(self.cell_span == span)[0]
                offsets = np.arange(int(span))
                index = []
                for axis in range(ndim):
                    # the cells along the first axis, the span along each axis after
                    shape = [1] * (ndim + 1)
                    shape[axis + 1] = int(span)
                    index.append(
                        self.cell_lower[cells, axis].reshape(-1, *[1] * ndim)
                        + offsets.reshape(shape)
                    )
                leaf[tuple(index)] = cells.reshape(-1, *[1] * ndim)
            self._leaf_map = leaf
        return self._leaf_map

    # The boundary ----------------------------------------------------------------#
    def _find_boundary(self) -> None:
        """Find the cell edges on the boundary of the domain and name their face.

        An edge of a leaf cell is on the boundary if there is no cell beyond
        it: on a face of the bounding rectangle (``"x_lower"``, ...), or inside
        the rectangle where the domain does not fill it (``"inner"``). In 1D
        the "edge" is the cell's end node, of measure 1 (per unit area).
        """
        leaf = self._leaf_cells()
        faces = faces_of(self.ndim)
        edges = {face: ([], [], [], []) for face in (*faces, INNER)}
        offsets = _corner_offsets(self.ndim)
        centre = self.cell_lower + 0.5 * self.cell_span[:, None]
        for face, (axis, side) in faces.items():
            beyond = self.cell_lower[:, axis] + (self.cell_span if side else -1)
            at_rectangle = (beyond < 0) | (beyond >= self._cells[axis])
            if self._periodic[axis]:
                beyond = beyond % self._cells[axis]
                at_rectangle[:] = False
            where = [self.cell_lower[:, a] for a in range(self.ndim)]
            where[axis] = np.clip(beyond, 0, self._cells[axis] - 1)
            outside = at_rectangle | (leaf[tuple(where)] < 0)
            midpoint = centre.copy()
            midpoint[:, axis] += (0.5 if side else -0.5) * self.cell_span
            # the corners on that side of a cell: its two ends (lower-left
            # first), one node twice in 1D
            on_side = np.flatnonzero(offsets[:, axis] == side)
            first, second = on_side[0], on_side[-1]
            measure = np.prod(np.delete(self.cell_size, axis, axis=1), axis=1)
            for name, cells in (
                (face, np.nonzero(outside & at_rectangle)[0]),
                (INNER, np.nonzero(outside & ~at_rectangle)[0]),
            ):
                edges[name][0].append(self.cell_corners[cells, first])
                edges[name][1].append(self.cell_corners[cells, second])
                edges[name][2].append(measure[cells])
                edges[name][3].append(self._origin + midpoint[cells] * self._spacing)
        self._boundary = {
            face: (
                np.concatenate(first),
                np.concatenate(second),
                np.concatenate(length),
                np.concatenate(midpoint),
            )
            for face, (first, second, length, midpoint) in edges.items()
        }

    @sp.profile("patchwork.composite_nodes", tags=PROFILE_TAGS)
    def __init__(self, hierarchy: AMRHierarchy) -> None:
        """Number the nodes of ``hierarchy`` and find its leaf cells and hanging nodes."""
        self.hierarchy = hierarchy
        ratio = hierarchy.refinement_ratio
        finest = hierarchy.max_level
        self.ndim = ndim = hierarchy.ndim
        self._finest = finest
        self._cells = np.asarray(hierarchy.level_shape(finest), dtype=np.int64)
        self._periodic = tuple(hierarchy.periodic)
        self._spacing = np.asarray(hierarchy.spacing(finest), dtype=float)
        self._origin = np.asarray(hierarchy.lower_bound, dtype=float)

        lowers, spans, owners = [], [], []
        for patch in hierarchy.patches:
            span = ratio ** (finest - patch.level)
            covered = np.zeros(patch.shape, dtype=bool)
            for child in hierarchy.patches_at_level(patch.level + 1):
                # a finer patch may lie over several patches of this level
                covered[
                    tuple(
                        slice(
                            max(c0 // ratio, i0) - i0, max(min(c1 // ratio, i1) - i0, 0)
                        )
                        for (c0, c1), (i0, i1) in zip(child.box, patch.box, strict=True)
                    )
                ] = True
            index = np.nonzero(~covered)
            lowers.append(
                np.column_stack(
                    [
                        (values + i0) * span
                        for values, (i0, _) in zip(index, patch.box, strict=True)
                    ]
                )
            )
            spans.append(np.full(index[0].size, span, dtype=np.int64))
            # a patch on several ranks: its rows of cells split over them as
            # its arrays are (near-even, along the first axis)
            chunks = np.array_split(np.arange(patch.shape[0]), len(patch.ranks))
            row_rank = np.concatenate(
                [
                    np.full(chunk.size, rank, dtype=np.int64)
                    for chunk, rank in zip(chunks, patch.ranks, strict=True)
                ]
            )
            owners.append(row_rank[index[0]])
        self.cell_lower = np.concatenate(lowers)
        self.cell_span = np.concatenate(spans)
        self.cell_ranks = np.concatenate(owners)

        offsets = _corner_offsets(ndim)
        corners = (
            self.cell_lower[:, None, :] + self.cell_span[:, None, None] * offsets
        )  # (m, 2**ndim, ndim)
        self._number(self._key(*(corners[..., axis] for axis in range(ndim))))
        self.lattice = np.column_stack(
            np.unravel_index(self._keys, tuple(int(n) + 1 for n in self._cells))
        )
        self.coordinates = self._origin + self.lattice * self._spacing

        size = (
            self.cell_span[:, None] * self._spacing
        )  # (m, ndim) physical edge lengths
        self.cell_size = size
        self.node_ranks = np.full(self.num_nodes, np.iinfo(np.int64).max)
        np.minimum.at(
            self.node_ranks,
            self.cell_corners.reshape(-1),
            np.repeat(self.cell_ranks, 2**ndim),
        )
        self.volumes = np.bincount(
            self.cell_corners.reshape(-1),
            weights=np.repeat(np.prod(size, axis=1) / 2**ndim, 2**ndim),
            minlength=self.num_nodes,
        )
        self._find_hanging_nodes()
        self._leaf_map: np.ndarray | None = None
        self._patch_nodes: dict[tuple, np.ndarray] = {}
        self._reads: list | None = None
        self._outside: dict[int, np.ndarray | None] = {}
        self._find_boundary()

    # Patches -------------------------------------------------------------------#
    def patch_nodes(self, patch: Patch) -> np.ndarray:
        """Node numbers of the nodes of ``patch``, in its node shape (read-only).

        Found at the first call for a patch and kept: a patch's nodes are the same
        as long as the hierarchy is.
        """
        key = (patch.level, patch.box)
        nodes = self._patch_nodes.get(key)
        if nodes is None:
            span = self.hierarchy.refinement_ratio ** (self._finest - patch.level)
            index = np.meshgrid(
                *(np.arange(lo, hi + 1) * span for lo, hi in patch.box), indexing="ij"
            )
            nodes = self._lookup(*index)
            if (nodes < 0).any():
                raise RuntimeError("a patch node is not a node of the composite mesh")
            nodes.flags.writeable = False
            self._patch_nodes[key] = nodes
        return nodes

    def _patch_reads(self) -> list[tuple[np.ndarray, tuple[np.ndarray, ...]]]:
        """For every patch, the nodes whose value :meth:`from_patches` takes from it.

        One entry per patch of :attr:`AMRHierarchy.patches`: the node numbers and
        where those nodes are in the patch's node array (one index array per axis).
        A node shared by several patches is read from the finest, the last of that
        level in the order of the patches; a node on a periodic seam that a patch
        holds twice, from the upper copy. Built at the first call.
        """
        if self._reads is None:
            patches = self.hierarchy.patches
            nodes = [self.patch_nodes(patch).reshape(-1) for patch in patches]
            sizes = np.array([part.size for part in nodes])
            every = np.concatenate(nodes)  # coarse to fine, C order in a patch
            # the last copy of every node: the finest patch's, the upper on a seam
            last = every.size - 1 - np.unique(every[::-1], return_index=True)[1]
            if last.size != self.num_nodes:
                raise RuntimeError("a node of the composite mesh is in no patch")
            starts = np.concatenate(([0], np.cumsum(sizes)))
            holder = np.searchsorted(starts, last, side="right") - 1
            order = np.argsort(holder, kind="stable")
            bounds = np.searchsorted(holder[order], np.arange(len(patches) + 1))
            self._reads = []
            for number, box in enumerate(patch.box for patch in patches):
                chosen = last[order[bounds[number] : bounds[number + 1]]]
                shape = tuple(hi - lo + 1 for lo, hi in box)
                self._reads.append(
                    (every[chosen], np.unravel_index(chosen - starts[number], shape))
                )
        return self._reads

    def to_patches(self, values: np.ndarray, name: str) -> None:
        """Write node values ``(n,)`` into the node-centered field ``name``."""
        for patch in self.hierarchy.local_patches:
            array = patch.field(name)
            block = tuple(slice(b0, b1) for b0, b1 in array.layout.index_bounds)
            array.local[...] = values[self.patch_nodes(patch)][block]

    def from_patches(self, name: str) -> np.ndarray:
        """Node values ``(n,)`` of the node-centered field ``name``.

        Where patches share a node the finest one's value is taken. Collective:
        every patch array is gathered (on one rank, read in place). Which nodes come from which patch is found at the first call.
        """
        values = np.empty(self.num_nodes)
        hierarchy = self.hierarchy
        serial = hierarchy.comm.Get_size() == 1
        for patch, (nodes, index) in zip(
            hierarchy.patches, self._patch_reads(), strict=True
        ):
            if serial:
                data = patch.field(name).local  # the whole patch, without halos
            else:
                data = hierarchy.gather(patch, name)
            values[nodes] = data[index]
        return values

    def face_edges(
        self, face: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The cell edges on a part of the boundary.

        ``face`` is a face of the bounding rectangle (``"x_lower"``, ...) or
        ``"inner"``. Returns the two end nodes of every edge, its length and its
        midpoint ``(k, ndim)``. In 1D an edge is a node: both ends are that
        node and its length is 1.
        """
        return self._boundary[face]

    def face_nodes(self, face: str) -> np.ndarray:
        """Node numbers of the nodes on a part of the boundary (see :meth:`face_edges`)."""
        first, second, _, _ = self._boundary[face]
        return np.unique(np.concatenate((first, second)))

    def face_measure(self, face: str) -> np.ndarray:
        """Length of a part of the boundary that belongs to every node: zero away from it."""
        first, second, length, _ = self._boundary[face]
        measure = np.zeros(self.num_nodes)
        # the two ends of an edge share it
        np.add.at(measure, first, 0.5 * length)
        np.add.at(measure, second, 0.5 * length)
        return measure

    # Interpolation ---------------------------------------------------------------#
    def level_shape(self, level: int) -> tuple[int, ...]:
        """Number of nodes of a level's lattice over the whole domain, per axis.

        A periodic axis has as many nodes as cells: the last node is the first.
        """
        cells = self.hierarchy.level_shape(level)
        return tuple(
            n if periodic else n + 1
            for n, periodic in zip(cells, self._periodic, strict=True)
        )

    def _bilinear(
        self, lower: Sequence[np.ndarray], fraction: Sequence[np.ndarray], level: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Columns and weights ``(n, 2**ndim)`` of (bi)linear interpolation on a level lattice."""
        shape = self.level_shape(level)
        index, weight = [], []
        for axis in range(self.ndim):
            upper = lower[axis] + 1
            if self._periodic[axis]:
                upper = upper % shape[axis]
            else:
                upper = np.minimum(upper, shape[axis] - 1)
            index.append((lower[axis], upper))
            weight.append((1.0 - fraction[axis], fraction[axis]))
        corners = list(itertools.product((0, 1), repeat=self.ndim))
        columns = np.column_stack(
            [
                np.ravel_multi_index(
                    tuple(index[axis][c] for axis, c in enumerate(corner)), shape
                )
                for corner in corners
            ]
        )
        weights = np.column_stack(
            [
                np.prod([weight[axis][c] for axis, c in enumerate(corner)], axis=0)
                for corner in corners
            ]
        )
        return columns, weights

    def from_level(self, level: int) -> sparse.csr_matrix:
        """(Bi)linear interpolation from a level's lattice to the composite nodes.

        The matrix has shape ``(n, nodes of the level lattice)``, the lattice in
        C order with :meth:`level_shape`. It prolongs a field known on the
        uniform grid of ``level`` (e.g. the base grid) to every node.
        """
        span = self.hierarchy.refinement_ratio ** (self._finest - level)
        cells = self.hierarchy.level_shape(level)
        lower, fraction = [], []
        for axis in range(self.ndim):
            position = self.lattice[:, axis]
            index = np.minimum(position // span, cells[axis] - 1)
            lower.append(index)
            fraction.append((position - index * span) / span)
        columns, weights = self._bilinear(lower, fraction, level)
        rows = np.repeat(np.arange(self.num_nodes), 2**self.ndim)
        shape = self.level_shape(level)
        matrix = sparse.csr_matrix(
            sparse.coo_matrix(
                (weights.reshape(-1), (rows, columns.reshape(-1))),
                shape=(self.num_nodes, int(np.prod(shape))),
            )
        )
        matrix.eliminate_zeros()
        return matrix

    def evaluation(self, points: np.ndarray) -> sparse.csr_matrix:
        """The composite interpolant at ``points`` ``(ndim, k)``: a ``(k, n)`` matrix.

        Each point takes the linear (1D) or bilinear (2D) interpolant of the
        leaf cell it is in.
        Apply it to node values that have their hanging nodes filled (as
        :meth:`NodalPoisson.solve` returns them). A point outside the domain
        gets an empty row: the value zero.
        """
        points = np.asarray(points, dtype=float)
        corners, weights = self.cell_weights(points)
        rows = np.repeat(np.arange(points.shape[1]), corners.shape[1])
        matrix = sparse.csr_matrix(
            sparse.coo_matrix(
                (weights.reshape(-1), (rows, corners.reshape(-1))),
                shape=(points.shape[1], self.num_nodes),
            )
        )
        matrix.eliminate_zeros()
        return matrix

    def deposit(self, points: np.ndarray, charges: np.ndarray) -> np.ndarray:
        """The charges of point particles at ``points`` ``(ndim, k)`` on the nodes: ``(n,)``.

        Each charge is spread over the corners of its leaf cell with the
        (bi)linear weights of :meth:`evaluation`, whose transpose this is: the
        work a field interpolated with :meth:`evaluation` does on the particles
        is the work their deposited charge feels, and no particle pushes
        itself through the coupling. Hanging nodes receive charge like any
        other; :meth:`NodalPoisson.solve` with ``charge=`` hands it to the
        nodes they hang from with the transposed interpolation, so the total
        charge is kept. Points outside the domain deposit nothing. In 2D a compiled
        kernel (``deposit_nodes``) makes one pass over the points.
        """
        points = np.asarray(points, dtype=float)
        charges = np.asarray(charges, dtype=float).reshape(-1)
        if charges.shape[0] != points.shape[1]:
            raise ValueError(f"{points.shape[1]} points but {charges.shape[0]} charges")
        if self.ndim == 2 and points.shape[1]:
            # one pass of the compiled kernel, the same cells and weights
            from patchwork.kernels.deposit_nodes import kernel as deposit_nodes

            charge = np.zeros(self.num_nodes)
            deposit_nodes(
                np.ascontiguousarray(points),
                np.ascontiguousarray(charges),
                self._origin,
                self._spacing,
                self._cells,
                np.asarray(self._periodic, dtype=np.int64),
                self._leaf_cells(),
                self.cell_corners,
                self.cell_lower,
                self.cell_span,
                charge,
            )
            return charge
        corners, weights = self.cell_weights(points)
        return np.bincount(
            corners.reshape(-1),
            weights=(weights * charges[:, None]).reshape(-1),
            minlength=self.num_nodes,
        )

    def cell_weights(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Corner nodes and (bi)linear weights, ``(k, 2**ndim)``, of the leaf cell of each point.

        The corners are in the order of :attr:`cell_corners`; a point outside
        the domain has weights zero.
        """
        points = np.asarray(points, dtype=float)
        ndim = self.ndim
        if points.shape[0] != ndim:
            raise ValueError(
                f"points of a {ndim}D mesh are ({ndim}, k), got {points.shape}"
            )
        position = (points - self._origin[:, None]) / self._spacing[:, None]
        cell, local = [], []
        for axis in range(ndim):
            index = np.floor(position[axis]).astype(np.int64)
            if self._periodic[axis]:
                index = index % self._cells[axis]
                wrapped = position[axis] % self._cells[axis]
            else:
                index = np.clip(index, 0, self._cells[axis] - 1)
                wrapped = position[axis]
            cell.append(index)
            local.append(wrapped)
        leaf_map = self._leaf_cells()
        leaf = leaf_map[tuple(cell)]
        # a point on a lattice line may have its cell on the lower side only
        # (the edge of a domain that does not fill its rectangle)
        on_line = [local[axis] == np.floor(local[axis]) for axis in range(ndim)]
        steps = sorted(
            (step for step in itertools.product((0, 1), repeat=ndim) if any(step)),
            key=lambda step: (sum(step), step[::-1]),
        )
        for step in steps:
            missing = leaf < 0
            for axis in range(ndim):
                if step[axis]:
                    missing &= on_line[axis] & (cell[axis] > 0)
            if not missing.any():
                continue
            leaf[missing] = leaf_map[
                tuple(cell[axis][missing] - step[axis] for axis in range(ndim))
            ]
        inside = leaf >= 0
        for axis in range(ndim):
            if not self._periodic[axis]:
                # beyond a non-periodic face: no value (the clipped cell would
                # extrapolate)
                inside &= (position[axis] >= 0.0) & (
                    position[axis] <= self._cells[axis]
                )
        leaf = np.where(inside, leaf, 0)
        span = self.cell_span[leaf]
        fraction = [
            (local[axis] - self.cell_lower[leaf, axis]) / span for axis in range(ndim)
        ]
        weights = np.column_stack(
            [
                np.prod(
                    [
                        fraction[axis] if upper else 1.0 - fraction[axis]
                        for axis, upper in enumerate(corner)
                    ],
                    axis=0,
                )
                for corner in _corner_offsets(ndim)
            ]
        )
        weights[~inside] = 0.0  # outside the domain: no value
        return self.cell_corners[leaf], weights

    def to_level(self, level: int) -> sparse.csr_matrix:
        """The composite interpolant on a level's lattice over the whole domain.

        Shape ``(nodes of the level lattice, n)``, the lattice in C order with
        :meth:`level_shape`: the solution sampled on a uniform grid.
        """
        shape = self.level_shape(level)
        spacing = np.asarray(self.hierarchy.spacing(level))
        index = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
        points = (
            self._origin[:, None]
            + np.vstack([values.ravel() for values in index]) * spacing[:, None]
        )
        return self.evaluation(points)

    def _nodes_outside(self, level: int) -> np.ndarray | None:
        """The nodes of a level's lattice outside the domain; None if there are none."""
        if level not in self._outside:
            hierarchy = self.hierarchy
            scale = hierarchy.refinement_ratio**level
            cells = np.zeros(hierarchy.level_shape(level), dtype=bool)
            for patch in hierarchy.patches_at_level(0):
                cells[tuple(slice(lo * scale, hi * scale) for lo, hi in patch.box)] = (
                    True
                )
            if cells.all():
                self._outside[level] = None
            else:
                # a node is in the domain if one of the cells around it is
                around = cells
                for axis in range(self.ndim):
                    width = [(0, 0)] * self.ndim
                    width[axis] = (1, 1)
                    around = np.pad(
                        around,
                        width,
                        mode="wrap" if self._periodic[axis] else "constant",
                    )
                inside = np.zeros([n - 1 for n in around.shape], dtype=bool)
                for shift in itertools.product((0, 1), repeat=self.ndim):
                    inside |= around[
                        tuple(
                            slice(None, -1) if s == 0 else slice(1, None) for s in shift
                        )
                    ]
                shape = self.level_shape(level)
                self._outside[level] = ~inside[tuple(slice(0, n) for n in shape)]
        return self._outside[level]

    def _refine(self, array: np.ndarray, axis: int, ratio: int) -> np.ndarray:
        """Linear interpolation of a lattice to ``ratio`` times as many cells on one axis."""
        periodic = self._periodic[axis]
        count = array.shape[axis]
        shape = list(array.shape)
        shape[axis] = count * ratio if periodic else (count - 1) * ratio + 1
        fine = np.empty(shape, dtype=array.dtype)
        index: list[slice | int] = [slice(None)] * array.ndim

        def along(part: slice | int) -> tuple:
            index[axis] = part
            return tuple(index)

        fine[along(slice(None, None, ratio))] = array
        lower = array[along(slice(None, -1))]
        upper = array[along(slice(1, None))]
        for step in range(1, ratio):
            # written into the strided target, without temporaries
            target = fine[along(slice(step, (count - 1) * ratio, ratio))]
            if 2 * step == ratio:
                np.add(lower, upper, out=target)
                target *= 0.5
            else:
                weight = step / ratio
                np.multiply(lower, 1.0 - weight, out=target)
                target += weight * upper
            if periodic:
                # between the last node and the first
                weight = step / ratio
                fine[along((count - 1) * ratio + step)] = (1.0 - weight) * array[
                    along(count - 1)
                ] + weight * array[along(0)]
        return fine

    def _sample_scalar(self, arrays: Sequence[np.ndarray], level: int) -> np.ndarray:
        """:meth:`sample_patches` of one value per node."""
        ratio = self.hierarchy.refinement_ratio
        patches = self.hierarchy.patches
        # zero where the domain does not fill its rectangle
        current = np.zeros(self.level_shape(0), dtype=arrays[0].dtype)
        for depth in range(level + 1):
            if depth:
                for axis in range(self.ndim):
                    current = self._refine(current, axis, ratio)
            for patch, array in zip(patches, arrays, strict=True):
                if patch.level != depth:
                    continue
                where, source = [], []
                for axis, (lo, hi) in enumerate(patch.box):
                    # on a periodic axis the lattice has no copy of its first node
                    stop = min(hi + 1, current.shape[axis])
                    where.append(slice(lo, stop))
                    source.append(slice(0, stop - lo))
                current[tuple(where)] = array[tuple(source)]
        for patch, array in zip(patches, arrays, strict=True):
            if patch.level <= level:
                continue
            # a finer patch: its nodes that are nodes of this lattice too
            stride = ratio ** (patch.level - level)
            where, source = [], []
            for axis, (lo, hi) in enumerate(patch.box):
                first = (-lo) % stride
                start = (lo + first) // stride
                stop = min(hi // stride + 1, current.shape[axis])
                where.append(slice(start, stop))
                source.append(slice(first, first + (stop - start) * stride, stride))
            current[tuple(where)] = array[tuple(source)]
        outside = self._nodes_outside(level)
        if outside is not None:
            current[outside] = 0
        return current

    def sample_patches(
        self,
        arrays: Sequence[np.ndarray],
        level: int | None = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Patch-wise node fields on a level's lattice over the whole domain.

        For fields that are computed patch by patch and are not one composite
        function, such as a gradient. ``arrays`` holds one array per patch, in
        the order of ``hierarchy.patches``, each of the patch's ``node_shape``
        (further axes, e.g. vector components, are carried along). Every node
        of the lattice of ``level`` (the finest by default) takes its value from
        the finest patch whose nodes surround it, a patch's own boundary nodes
        included: by bilinear interpolation between the nodes of a coarser
        patch, and directly from the coinciding node of a finer one. Nodes
        outside a domain that does not fill its rectangle are zero.

        Returns an array of :meth:`level_shape` plus the further axes, ``out``
        if given.
        """
        level = self._finest if level is None else level
        ndim = self.ndim
        patches = self.hierarchy.patches
        if len(arrays) != len(patches):
            raise ValueError(f"expected {len(patches)} patch arrays, got {len(arrays)}")
        arrays = [np.asarray(array) for array in arrays]
        for patch, array in zip(patches, arrays, strict=True):
            if array.shape[:ndim] != patch.node_shape:
                raise ValueError(
                    f"a level-{patch.level} patch needs an array of "
                    f"{patch.node_shape} nodes, got {array.shape[:ndim]}"
                )
        extra = arrays[0].shape[ndim:]
        if out is None:
            out = np.empty((*self.level_shape(level), *extra), dtype=arrays[0].dtype)
        if not extra:
            out[...] = self._sample_scalar(arrays, level)
            return out
        # one component at a time: loops over a short last axis are slow
        components = out.reshape(*out.shape[:ndim], -1)
        for number in range(components.shape[-1]):
            components[..., number] = self._sample_scalar(
                [
                    array.reshape(*array.shape[:ndim], -1)[..., number]
                    for array in arrays
                ],
                level,
            )
        return out


class NodalPoisson:
    r"""Solve :math:`-\nabla^2\phi = f` on the nodes of a patch hierarchy.

    The operator is assembled and its backend set up once (factorized, by
    default); :meth:`solve` then costs two triangular solves. ``faces`` maps
    ``"x_lower"``, ``"x_upper"``, ``"y_lower"``, ``"y_upper"`` (only the
    first two in 1D) to a :class:`Dirichlet` or :class:`Neumann` condition; a
    face that is not listed is a grounded Dirichlet face, and the faces of a
    periodic axis take none. On a domain that does not fill its rectangle
    (``AMRHierarchy(base_boxes=...)``) these are the parts of its boundary on
    the rectangle's faces, and ``"inner"`` is the rest: the edges inside the
    rectangle. ``embedded`` are bodies at fixed potentials
    (:class:`EmbeddedDirichlet`); where they overlap the last one listed wins.

    ``backend`` solves the assembled operator (:mod:`patchwork.backends`): a
    :class:`~patchwork.backends.Backend` or its name. ``"lu"`` (the default)
    assembles the whole operator on every rank and factorizes it there
    (SuperLU; LAPACK's tridiagonal LU on a line). ``"petsc"`` (needs
    petsc4py; ``PetscBackend(rtol=..., options=...)`` for its tolerance and
    PETSc options) assembles on every rank only the rows of the unknowns it
    owns (:attr:`CompositeNodes.node_ranks`, from where the patches are), from
    the leaf cells around them, and solves them together with a PETSc Krylov
    method and an algebraic multigrid preconditioner (see
    :class:`PetscSystem`). ``"fft"`` solves a uniform mesh without bodies
    with fast transforms, also when nothing fixes the potential (every axis
    periodic or Neumann: the solution of zero mean); ``"auto"`` takes the LU
    on a line, the FFT where it applies and the LU elsewhere. All give the
    potential on all nodes on every rank.

    Attributes:
        nodes: The :class:`CompositeNodes` of the hierarchy.
        free: Node numbers of the unknowns: neither hanging nor fixed.
        fixed: Whether a node has a prescribed value (a face or a body).
        rows: Node numbers of the unknowns whose rows this rank holds: all of
            :attr:`free`, or those it owns with a distributed backend (in the
            order of the distributed system: by rank, then by node number).
        matrix: This rank's rows of the operator, over the columns of the
            unknowns (in :attr:`free` order, or in the distributed order).
        backend: The :class:`~patchwork.backends.Backend`.
        system: The :class:`~patchwork.backends.LinearSystem` it solves.
        solver: What the backend set up (an LU factorization, a
            :class:`PetscSystem`, an :class:`~patchwork.backends.FFTSolver`),
            ``None`` without unknowns.
    """

    # Assembly ------------------------------------------------------------------#
    def _wrapped(self, points: np.ndarray) -> np.ndarray:
        """``points`` ``(ndim, k)`` moved into the domain along its periodic axes."""
        hierarchy = self.nodes.hierarchy
        if not any(hierarchy.periodic):
            return points
        points = points.copy()
        for axis, (lower, upper) in enumerate(hierarchy.bounds):
            if hierarchy.periodic[axis]:
                points[axis] = lower + (points[axis] - lower) % (upper - lower)
        return points

    def _classify_bodies(self) -> None:
        """Which nodes are inside a body, and which one (the last listed)."""
        nodes = self.nodes
        self._owner = np.full(nodes.num_nodes, -1, dtype=np.int64)
        points = nodes.coordinates.T
        for number, body in enumerate(self._embedded):
            self._owner[np.asarray(body.inside(points), dtype=bool)] = number

    def _fractions(self, outside: np.ndarray, inside: np.ndarray) -> np.ndarray:
        """Fraction of each edge from an outside node at which a surface crosses it."""
        nodes = self.nodes
        hierarchy = nodes.hierarchy
        result = np.ones(outside.size)
        for number, body in enumerate(self._embedded):
            mine = self._owner[inside] == number
            if not mine.any():
                continue
            start = nodes.coordinates[outside[mine]].T
            step = nodes.coordinates[inside[mine]].T - start
            for axis, (lower, upper) in enumerate(hierarchy.bounds):
                if hierarchy.periodic[axis]:
                    # the edge through the periodic face, not the long way round
                    period = upper - lower
                    step[axis] -= period * np.round(step[axis] / period)
            if body.fraction is not None:
                fraction = np.asarray(body.fraction(start, start + step), dtype=float)
            else:
                fraction = _surface_crossing(body.inside, start, step, self._wrapped)
            result[mine] = np.maximum(fraction, self._min_fraction)
        return result

    def _shortley_weller(
        self, outside: np.ndarray, inside: np.ndarray, distance: np.ndarray
    ) -> tuple[np.ndarray, tuple, tuple] | None:
        """The Shortley-Weller rows of the regular nodes next to a body.

        A node is regular if four cells of one size meet at it (two in 1D). Its row is then
        the finite-difference Laplacian with unequal arms, times its control
        volume: second-order accurate up to the surface, as on a uniform grid.
        Returns those nodes and the ``(rows, columns, entries)`` of their rows
        of the operator and of the surface couplings, which replace the rows of
        the cell-by-cell assembly; ``None`` if there are none.
        """
        nodes = self.nodes
        n = nodes.num_nodes
        ndim = nodes.ndim
        corners = 2**ndim
        if not outside.size:
            return None
        flat = nodes.cell_corners.reshape(-1)
        count = np.bincount(flat, minlength=n)
        smallest = np.full(n, np.iinfo(np.int64).max)
        largest = np.zeros(n, dtype=np.int64)
        spans = np.repeat(nodes.cell_span, corners)
        np.minimum.at(smallest, flat, spans)
        np.maximum.at(largest, flat, spans)
        regular = (count == corners) & (smallest == largest) & ~nodes.hanging
        cut_nodes = np.unique(outside[regular[outside]])
        if not cut_nodes.size:
            return None

        span = smallest[cut_nodes]
        spacing = span[:, None] * nodes._spacing  # (k, ndim)
        # arm lengths and what each arm ends at: [axis][side] over the cut nodes
        arm = np.stack([np.stack([spacing[:, axis]] * 2) for axis in range(ndim)])
        body = np.full((ndim, 2, cut_nodes.size), -1, dtype=np.int64)
        neighbour = np.empty((ndim, 2, cut_nodes.size), dtype=np.int64)
        for axis in range(ndim):
            for side, sign in enumerate((-1, 1)):
                step = np.zeros((cut_nodes.size, ndim), dtype=np.int64)
                step[:, axis] = sign * span
                there = nodes.lattice[cut_nodes] + step
                neighbour[axis, side] = nodes._lookup(*there.T)
        # the cut edges from a regular node: the arm they shorten
        position = np.minimum(np.searchsorted(cut_nodes, outside), cut_nodes.size - 1)
        mine = cut_nodes[position] == outside
        position, in_node = position[mine], inside[mine]
        match = (neighbour[:, :, position] == in_node).reshape(2 * ndim, -1)
        if not match.any(axis=0).all():  # pragma: no cover
            raise RuntimeError("a cut edge of a regular node is not one of its arms")
        which = match.argmax(axis=0)
        arm[which // 2, which % 2, position] = distance[mine]
        body[which // 2, which % 2, position] = self._owner[in_node]

        rows, columns, entries = [], [], []
        wall_rows, wall_columns, wall_entries = [], [], []
        volume = np.prod(spacing, axis=1)
        diagonal = np.zeros(cut_nodes.size)
        for axis in range(ndim):
            total = arm[axis, 0] + arm[axis, 1]
            for side in range(2):
                coefficient = volume * 2.0 / (arm[axis, side] * total)
                diagonal += coefficient
                to_node = body[axis, side] < 0
                rows.append(cut_nodes[to_node])
                columns.append(neighbour[axis, side][to_node])
                entries.append(-coefficient[to_node])
                wall_rows.append(cut_nodes[~to_node])
                wall_columns.append(body[axis, side][~to_node])
                wall_entries.append(coefficient[~to_node])
        rows.append(cut_nodes)
        columns.append(cut_nodes)
        entries.append(diagonal)
        return (
            cut_nodes,
            tuple(np.concatenate(part) for part in (rows, columns, entries)),
            tuple(
                np.concatenate(part) for part in (wall_rows, wall_columns, wall_entries)
            ),
        )

    def _operator(
        self, cells: np.ndarray | None
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """The operator on all nodes and the surface couplings, from some leaf cells.

        The rows of nodes all of whose cells are among ``cells`` (all cells if
        ``None``) are complete; the others are not, and are left out by the caller.
        """
        nodes = self.nodes
        n = nodes.num_nodes
        solid = self._owner >= 0

        # Every leaf cell's edges (four in 2D, itself in 1D), each with its share
        # of the stencil's conductance: half in 2D, the whole in 1D.
        ndim = nodes.ndim
        corners = nodes.cell_corners if cells is None else nodes.cell_corners[cells]
        size = nodes.cell_size if cells is None else nodes.cell_size[cells]
        firsts, seconds, conductances, lengths = [], [], [], []
        for axis in range(ndim):
            cross = np.prod(np.delete(size, axis, axis=1), axis=1) / 2 ** (ndim - 1)
            for k in range(2**ndim):
                if (k >> axis) & 1:
                    continue
                firsts.append(corners[:, k])
                seconds.append(corners[:, k | (1 << axis)])
                conductances.append(cross / size[:, axis])
                lengths.append(size[:, axis])
        first = np.concatenate(firsts)
        second = np.concatenate(seconds)
        conductance = np.concatenate(conductances)
        length = np.concatenate(lengths)
        # A periodic axis one cell wide joins a node to itself: no coupling.
        keep = first != second
        first, second = first[keep], second[keep]
        conductance, length = conductance[keep], length[keep]

        # An edge that enters a body: the node outside sees the surface, at the
        # body's potential, instead of the node inside.
        cut_a = ~solid[first] & solid[second]
        cut_b = solid[first] & ~solid[second]
        outside = np.concatenate((first[cut_a], second[cut_b]))
        inside = np.concatenate((second[cut_a], first[cut_b]))
        cut_conductance = np.concatenate((conductance[cut_a], conductance[cut_b]))
        cut_length = np.concatenate((length[cut_a], length[cut_b]))
        fraction = self._fractions(outside, inside)
        plain = ~(cut_a | cut_b)
        first, second, conductance = first[plain], second[plain], conductance[plain]

        # the diagonal summed per node, the couplings once per edge half
        diagonal = np.bincount(
            np.concatenate((first, second, outside)),
            weights=np.concatenate(
                (conductance, conductance, cut_conductance / fraction)
            ),
            minlength=n,
        )
        on = np.flatnonzero(diagonal)
        rows = np.concatenate((first, second, on))
        columns = np.concatenate((second, first, on))
        entries = np.concatenate((-conductance, -conductance, diagonal[on]))
        # the surface values' share of the right-hand side, per body
        wall_rows, wall_columns = outside, self._owner[inside]
        wall_entries = cut_conductance / fraction

        stencils = self._shortley_weller(outside, inside, fraction * cut_length)
        if stencils is not None:
            # the rows of the regular nodes next to a body are theirs
            cut_nodes, replace, replace_wall = stencils
            assembled = np.ones(n, dtype=bool)
            assembled[cut_nodes] = False
            keep, keep_wall = assembled[rows], assembled[wall_rows]
            rows, columns, entries = (
                np.concatenate((part[keep], new))
                for part, new in zip((rows, columns, entries), replace, strict=True)
            )
            wall_rows, wall_columns, wall_entries = (
                np.concatenate((part[keep_wall], new))
                for part, new in zip(
                    (wall_rows, wall_columns, wall_entries), replace_wall, strict=True
                )
            )
        num_bodies = len(self._embedded)
        wall = sparse.csr_matrix(
            (wall_entries, (wall_rows, wall_columns)), shape=(n, max(num_bodies, 1))
        )
        operator = sparse.csr_matrix((entries, (rows, columns)), shape=(n, n))
        return operator, wall

    def _assemble(self) -> None:
        """Build this rank's rows of the operator, the solver and the right-hand-side maps."""
        nodes = self.nodes
        n = nodes.num_nodes
        self._classify_bodies()
        solid = self._owner >= 0

        self.fixed = np.zeros(n, dtype=bool)
        for face, condition in self._faces.items():
            if isinstance(condition, Dirichlet):
                self.fixed[nodes.face_nodes(face)] = True
        self.fixed |= solid
        self.fixed &= ~nodes.hanging
        self.free = np.nonzero(~self.fixed & ~nodes.hanging)[0]
        fixed = np.nonzero(self.fixed)[0]
        self._fixed_nodes = fixed

        reduction = nodes.reduction
        if self.backend.distributed:
            # the unknowns by owner: every rank's rows are contiguous
            comm = nodes.hierarchy.comm
            owner = nodes.node_ranks[self.free]
            self._unknowns = self.free[np.argsort(owner, kind="stable")]
            self.rows = self.free[owner == comm.Get_rank()]
            # the rows of the reduced operator of these unknowns take the rows
            # of the nodes that hang from them too, which need every leaf cell
            # around those nodes
            support = np.zeros(n, dtype=bool)
            support[sparse.csc_matrix(reduction[:, self.rows]).indices] = True
            cells = np.nonzero(support[nodes.cell_corners].any(axis=1))[0]
        else:
            self._unknowns = self.free
            self.rows = self.free
            cells = None
        operator, wall = self._operator(cells)

        # rows of T^T A T for the unknowns this rank holds
        pick = sparse.csr_matrix(reduction[:, self.rows].T)
        reduced = sparse.csr_matrix(pick @ operator @ reduction)
        self._wall = sparse.csr_matrix(pick @ wall)
        self._load = sparse.csr_matrix(pick @ sparse.diags(nodes.volumes))
        # a source already integrated over the control volumes (a deposited charge)
        self._integral_load = pick
        self._lift = reduced[:, fixed].tocsr()
        self._reduction_rows = reduction[np.nonzero(nodes.hanging)[0]].tocsr()
        self.matrix = reduced[:, self._unknowns].tocsr()
        self.system = LinearSystem(
            matrix=self.matrix,
            unknowns=self._unknowns,
            rows=self.rows,
            nodes=nodes,
            faces=dict(self._faces),
            bodies=len(self._embedded),
            singular=not self.fixed.any(),
            symmetric=not self._embedded,
            comm=nodes.hierarchy.comm,
        )
        reason = self.backend.unsupported(self.system)
        if reason is not None:
            raise ValueError(reason)
        self.solver = self.backend.setup(self.system) if self.free.size else None

    # Boundary values ---------------------------------------------------------------#
    def _update_values(self) -> None:
        """Evaluate the prescribed values and their share of the right-hand side."""
        nodes = self.nodes
        n = nodes.num_nodes
        values = np.zeros(n)
        coordinates = nodes.coordinates.T
        flux = np.zeros(n)
        # x faces before y faces, lower before upper: the first one decides a corner
        decided = np.zeros(n, dtype=bool)
        for face, condition in self._faces.items():
            if isinstance(condition, Dirichlet):
                on_face = nodes.face_nodes(face)
                on_face = on_face[~decided[on_face]]
                values[on_face] = _evaluate(condition.value, *coordinates[:, on_face])
                decided[on_face] = True
            else:
                # the gradient at the middle of every boundary edge, shared by its ends
                first, second, length, midpoint = nodes.face_edges(face)
                through = 0.5 * length * _evaluate(condition.gradient, *midpoint.T)
                np.add.at(flux, first, through)
                np.add.at(flux, second, through)
        body_values = np.array([body.value for body in self._embedded] or [0.0])
        solid = self._owner >= 0
        values[solid] = body_values[self._owner[solid]]
        self._values = values
        constant = self._wall @ body_values
        constant += self._integral_load @ flux
        constant -= self._lift @ values[self._fixed_nodes]
        self._constant = constant

    @sp.profile("patchwork.nodal_poisson_setup", tags=PROFILE_TAGS)
    def __init__(
        self,
        hierarchy: AMRHierarchy,
        *,
        faces: Mapping[str, Dirichlet | Neumann] | None = None,
        embedded: Sequence[EmbeddedDirichlet] = (),
        min_fraction: float = 1.0e-3,
        nodes: CompositeNodes | None = None,
        backend: str | Backend = "lu",
    ) -> None:
        """Assemble the operator and set up its backend.

        ``min_fraction`` is the smallest fraction of an edge between a node and
        an embedded surface: a surface closer than that (through the node, up to
        round-off) is moved there, which bounds the size of that node's row.
        """
        self.backend = backend_of(backend)
        self.nodes = CompositeNodes(hierarchy) if nodes is None else nodes
        self._embedded = tuple(embedded)
        self._min_fraction = float(min_fraction)
        known = faces_of(hierarchy.ndim)
        unknown = set(faces or ()) - set(known) - {INNER}
        if unknown:
            raise ValueError(
                f"unknown faces {sorted(unknown)}; expected {[*known, INNER]}"
            )
        self._faces: dict[str, Dirichlet | Neumann] = {}
        for face, (axis, _) in known.items():
            if hierarchy.periodic[axis]:
                if faces and face in faces:
                    raise ValueError(
                        f"{face} is on a periodic axis: it takes no condition"
                    )
                continue
            self._faces[face] = (faces or {}).get(face, Dirichlet(0.0))
        # after the faces of the rectangle, which decide the nodes they share
        self._faces[INNER] = (faces or {}).get(INNER, Dirichlet(0.0))
        self._assemble()
        self._update_values()

    def set_conditions(
        self,
        faces: Mapping[str, Dirichlet | Neumann] | None = None,
        body_values: Sequence[float] | None = None,
    ) -> None:
        """Change prescribed values without assembling or factorizing again.

        ``faces`` replaces the values of the listed faces; each must keep its
        kind (Dirichlet or Neumann). ``body_values`` are the new potentials of
        the embedded bodies, in their order.
        """
        for face, condition in (faces or {}).items():
            if face not in self._faces:
                raise ValueError(f"{face} takes no condition")
            if type(condition) is not type(self._faces[face]):
                raise ValueError(f"{face} cannot change its kind of condition")
            self._faces[face] = condition
        if body_values is not None:
            if len(body_values) != len(self._embedded):
                raise ValueError("one potential per embedded body is needed")
            self._embedded = tuple(
                EmbeddedDirichlet(body.inside, float(value), body.fraction)
                for body, value in zip(self._embedded, body_values, strict=True)
            )
        self._update_values()

    # Solving -------------------------------------------------------------------#
    @property
    def load(self) -> sparse.csr_matrix:
        """Map from a source on all nodes to the right-hand side of this rank's :attr:`rows`."""
        return self._load

    @property
    def integral_load(self) -> sparse.csr_matrix:
        """Map from a source integrated over each node's control volume to the right-hand side.

        The transposed interpolation of the hanging nodes, restricted to the
        unknowns: what a deposited charge (:meth:`CompositeNodes.deposit`)
        needs. ``load == integral_load @ diag(volumes)``.
        """
        return self._integral_load

    @property
    def constant(self) -> np.ndarray:
        """The share of the right-hand side that the prescribed values give."""
        return self._constant

    @sp.profile("patchwork.nodal_poisson_solve_reduced", tags=PROFILE_TAGS)
    def solve_reduced(self, right_hand_side: np.ndarray) -> np.ndarray:
        """Solve for a right-hand side of this rank's :attr:`rows`; the potential on all nodes.

        ``right_hand_side`` is ``load @ source + constant``, which callers that
        solve repeatedly with a source of fixed structure can precompute as one
        matrix. Hanging nodes are filled by interpolation. Collective with the
        PETSc backend.
        """
        nodes = self.nodes
        potential = self._values.copy()
        if self.solver is not None:
            potential[self._unknowns] = self.solver.solve(right_hand_side)
        if nodes.hanging.any():
            potential[nodes.hanging] = self._reduction_rows @ potential
        return potential

    @sp.profile("patchwork.nodal_poisson_solve", tags=PROFILE_TAGS)
    def solve(
        self,
        source: np.ndarray | str | None = None,
        potential: str | None = None,
        *,
        charge: np.ndarray | None = None,
    ) -> np.ndarray:
        r"""Solve for a source given on the nodes; the potential on all nodes.

        ``source`` is an array ``(n,)`` over :attr:`nodes`, or the name of a
        node-centered field of the hierarchy. Or ``charge`` is the source
        integrated over each node's control volume, ``(n,)``, as
        :meth:`CompositeNodes.deposit` gives it for point particles (divide a
        charge by the permittivity for :math:`-\nabla^2\phi = \rho /
        \varepsilon_0`); the charge on hanging nodes goes to the nodes they hang
        from, and the charge on fixed nodes is ignored. A source or charge array
        is the whole one on every rank (the sum over the ranks' deposits); each
        rank reads the part its rows need. With ``potential`` the result is also
        written to the node-centered field of that name (allocated if needed).
        """
        nodes = self.nodes
        if (source is None) == (charge is None):
            raise ValueError("give either a source or a charge")
        if charge is not None:
            charge = np.asarray(charge, dtype=float)
            if charge.shape != (nodes.num_nodes,):
                raise ValueError(
                    f"expected a charge on the {nodes.num_nodes} nodes, got {charge.shape}"
                )
            right_hand_side = self._integral_load @ charge + self._constant
        else:
            if isinstance(source, str):
                source = nodes.from_patches(source)
            source = np.asarray(source, dtype=float)
            if source.shape != (nodes.num_nodes,):
                raise ValueError(
                    f"expected a source on the {nodes.num_nodes} nodes, got {source.shape}"
                )
            right_hand_side = self._load @ source + self._constant
        result = self.solve_reduced(right_hand_side)
        if potential is not None:
            hierarchy = nodes.hierarchy
            if potential not in hierarchy.field_names:
                hierarchy.allocate_field(potential, centering="node")
            nodes.to_patches(result, potential)
        return result


def solve_nodal_poisson(
    hierarchy: AMRHierarchy,
    rhs: str,
    *,
    potential: str = "potential",
    faces: Mapping[str, Dirichlet | Neumann] | None = None,
    embedded: Sequence[EmbeddedDirichlet] = (),
    backend: str | Backend = "lu",
) -> NodalPoisson:
    r"""Solve :math:`-\nabla^2\phi = \mathrm{rhs}` once on the nodes of ``hierarchy``.

    ``rhs`` and ``potential`` name node-centered fields; see
    :class:`NodalPoisson` for the conditions and the backends. Returns the
    solver, which can solve again without assembling.
    """
    solver = NodalPoisson(hierarchy, faces=faces, embedded=embedded, backend=backend)
    solver.solve(rhs, potential=potential)
    return solver
