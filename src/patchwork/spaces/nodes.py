r"""The node space of a patch hierarchy: the lowest-order H¹ (Q1) space.

The unknowns sit on the cell corners of the composite mesh: the cells of every
patch that no finer patch covers (the *leaf cells*). Nodes that patches share
are one unknown. A node of a fine cell in the middle of a coarser cell's edge is
*hanging*: its value is the linear interpolant of that edge's ends, so a nodal
field is continuous across coarse-fine interfaces. In 1D a coarse–fine
interface is a node of both levels and no node hangs.

`CompositeNodes` numbers the nodes, finds the hanging ones and their
interpolation, the control volumes, the deposit of point charges (the transposed
interpolation) and the maps to and from the patches. The Poisson operator on
these nodes is `patchwork.solvers.nodal_poisson.NodalPoisson`.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sparse
import scope_profiler as sp
from methodcard import Rule, method

from patchwork._profiling import PROFILE_TAGS
from patchwork.spaces.boundaries import INNER, faces_of
from patchwork.spaces.mass import node_mass_blocks, scatter_blocks

if TYPE_CHECKING:
    from patchwork.mesh.hierarchy import AMRHierarchy, Patch

#: `CompositeNodes` numbers its nodes with a table over the finest
#: lattice when the lattice has at most this many points per cell corner, and
#: by sorting the corners otherwise (a deep hierarchy with small fine patches).
_DENSE_TABLE_FACTOR = 16


def _corner_offsets(ndim: int) -> np.ndarray:
    """The corners of a cell, ``(2**ndim, ndim)``, the first axis varying fastest."""
    return np.array(
        [[(k >> axis) & 1 for axis in range(ndim)] for k in range(2**ndim)],
        dtype=np.int64,
    )


def _deposit_rules() -> dict:
    """The SymPy rule of `CompositeNodes.deposit` (for the tests)."""
    import sympy

    from patchwork.mesh.hierarchy import AMRHierarchy

    x, y, q = sympy.symbols("x y q", real=True)
    nodes = CompositeNodes(AMRHierarchy(base_shape=(4, 4)))
    # the lower-left corner of the cell [1/4, 1/2]^2
    corner = int(np.argmin(np.abs(nodes.lattice - (1, 1)).sum(axis=1)))

    def charge(x: float, y: float, q: float) -> float:
        return float(nodes.deposit(np.array([[x], [y]]), np.array([q]))[corner])

    return {
        "bilinear": Rule(
            q * (2 - 4 * x) * (2 - 4 * y),
            implementation=charge,
            ranges={"x": (0.26, 0.49), "y": (0.26, 0.49), "q": (0.5, 2.0)},
        )
    }


@method(
    equations={
        "lattice": r"\mathbf{k} = \mathbf{i}_L\, r^{L_\mathrm{max} - L}",
        "hanging": r"\phi_\mathrm{hanging} = \Big(1 - \frac{k}{s}\Big)\phi_a + \frac{k}{s}\,\phi_b \quad \text{(from the ends } a, b \text{ of the coarsest edge it lies in, span } s\text{)}",
        "composition": r"T \leftarrow T\,T \quad \text{until no master hangs (at most } L_\mathrm{max} + 1 \text{ times)}",
        "control_volume": r"V_p = \sum_{K \ni p} \frac{|K|}{2^d}",
    },
    assumptions=(
        "nodes shared by patches are one unknown; a periodic axis wraps the lattice",
        "in 1D no node hangs: a coarse-fine interface is a node of both levels",
        "a node is owned by the lowest rank of its leaf cells",
    ),
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
        cell_size: Physical edge lengths of every leaf cell, ``(m, ndim)``.
        cell_centers: Physical coordinates of the centre of every leaf cell,
            ``(m, ndim)``: where a coefficient per leaf cell is evaluated.
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

        Sets `num_nodes`, `cell_corners` and the sorted keys. Where
        the finest lattice is not much larger than the corners, a table over the
        lattice replaces the sort and makes `_lookup` a gather.
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
        """Find the nodes on coarser cells' edges and faces and their interpolation.

        A node on the boundary of a coarser leaf cell (not one of its corners)
        hangs: it takes the multilinear interpolant of that cell's corners, from
        the coarsest such cell. In 2D that is the linear interpolant along the
        coarse edge; in 3D also the bilinear one on a coarse face. In 1D a
        cell's faces are nodes, and no node hangs.
        """
        if self.ndim == 1:
            self.hanging = np.zeros(self.num_nodes, dtype=bool)
            self.reduction = sparse.csr_matrix(sparse.identity(self.num_nodes))
            return
        ndim = self.ndim
        corners = _corner_offsets(ndim)  # (2**ndim, ndim)
        slaves, masters, weights, spans = [], [], [], []
        for span in np.unique(self.cell_span):
            if span == 1:
                continue
            span = int(span)
            # the lattice points on the cell's boundary that are not its corners
            grid = np.stack(
                np.meshgrid(*([np.arange(span + 1)] * ndim), indexing="ij"), axis=-1
            ).reshape(-1, ndim)
            ends = (grid == 0) | (grid == span)
            offsets = grid[ends.any(axis=1) & ~ends.all(axis=1)]
            # the multilinear weight of every corner at every such point
            fraction = offsets / span  # (p, ndim)
            weight = np.prod(
                np.where(
                    corners[None, :, :] == 1,
                    fraction[:, None, :],
                    1.0 - fraction[:, None, :],
                ),
                axis=2,
            )  # (p, 2**ndim)
            cells = np.nonzero(self.cell_span == span)[0]
            lower = self.cell_lower[cells]
            for offset, row in zip(offsets, weight, strict=True):
                point = lower + offset
                node = self._lookup(*(point[:, axis] for axis in range(ndim)))
                found = node >= 0
                slaves.append(node[found])
                masters.append(self.cell_corners[cells[found]])
                weights.append(np.broadcast_to(row, (int(found.sum()), row.size)))
                spans.append(np.full(int(found.sum()), span))
        self.hanging = np.zeros(self.num_nodes, dtype=bool)
        identity = sparse.csr_matrix(sparse.identity(self.num_nodes))
        if not slaves or not sum(part.size for part in slaves):
            self.reduction = identity
            return
        slave = np.concatenate(slaves)
        master = np.concatenate(masters)  # (k, 2**ndim)
        weight = np.concatenate(weights)  # (k, 2**ndim)
        span = np.concatenate(spans)
        # one constraint per node, from the coarsest cell on whose boundary it lies
        order = np.lexsort((-span, slave))
        slave, master, weight = slave[order], master[order], weight[order]
        first = np.concatenate(([True], slave[1:] != slave[:-1]))
        slave, master, weight = slave[first], master[first], weight[first]
        self.hanging[slave] = True
        free = np.nonzero(~self.hanging)[0]
        nonzero = weight != 0.0
        reduction = sparse.csr_matrix(
            sparse.coo_matrix(
                (
                    np.concatenate((np.ones(free.size), weight[nonzero])),
                    (
                        np.concatenate(
                            (
                                free,
                                np.broadcast_to(slave[:, None], weight.shape)[nonzero],
                            )
                        ),
                        np.concatenate((free, master[nonzero])),
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
        """Find the sides of leaf cells on the boundary of the domain and name their face.

        A side of a leaf cell (an edge in 2D, a face in 3D) is on the boundary if
        there is no cell beyond it: on a face of the bounding box (``"x_lower"``,
        ...), or inside the box where the domain does not fill it (``"inner"``).
        In 1D the side is the cell's end node, of measure 1 (per unit area).
        """
        leaf = self._leaf_cells()
        faces = faces_of(self.ndim)
        edges = {face: ([], [], []) for face in (*faces, INNER)}
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
            # the corners on that side of a cell, the lowest first: its two ends
            # in 2D (one node twice in 1D), the four corners of a face in 3D
            on_side = np.flatnonzero(offsets[:, axis] == side)
            if self.ndim == 1:
                on_side = np.repeat(on_side, 2)
            measure = np.prod(np.delete(self.cell_size, axis, axis=1), axis=1)
            for name, cells in (
                (face, np.nonzero(outside & at_rectangle)[0]),
                (INNER, np.nonzero(outside & ~at_rectangle)[0]),
            ):
                edges[name][0].append(self.cell_corners[cells][:, on_side])
                edges[name][1].append(measure[cells])
                edges[name][2].append(self._origin + midpoint[cells] * self._spacing)
        width = max(2, 2 ** (self.ndim - 1))
        self._boundary = {
            face: (
                np.concatenate(corners).reshape(-1, width),
                np.concatenate(measure),
                np.concatenate(midpoint).reshape(-1, self.ndim),
            )
            for face, (corners, measure, midpoint) in edges.items()
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
        self.cell_centers = self._origin + self.cell_lower * self._spacing + size / 2
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
        """For every patch, the nodes whose value `from_patches` takes from it.

        One entry per patch of `AMRHierarchy.patches`: the node numbers and
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

    def face_pieces(self, face: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The sides of leaf cells on a part of the boundary.

        ``face`` is a face of the bounding box (``"x_lower"``, ...) or
        ``"inner"``. Returns the corner nodes of every piece, ``(k, 2**(ndim-1))``
        (in 1D the node twice), its measure (length in 2D, area in 3D, 1 in 1D)
        and its midpoint ``(k, ndim)``.
        """
        return self._boundary[face]

    def face_edges(
        self, face: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The cell edges on a part of the boundary of a 1D or 2D mesh.

        Returns the two end nodes of every edge, its length and its midpoint
        ``(k, ndim)``. In 1D an edge is a node: both ends are that node and its
        length is 1. In 3D the boundary is made of faces: see `face_pieces`.
        """
        if self.ndim == 3:
            raise ValueError("a 3D boundary is made of faces: use face_pieces")
        corners, measure, midpoint = self._boundary[face]
        return corners[:, 0], corners[:, -1], measure, midpoint

    def face_nodes(self, face: str) -> np.ndarray:
        """Node numbers of the nodes on a part of the boundary (see `face_pieces`)."""
        return np.unique(self._boundary[face][0])

    def face_measure(self, face: str) -> np.ndarray:
        """Measure of a part of the boundary that belongs to every node: zero away from it.

        Each piece's measure is shared equally by its corners.
        """
        corners, measure, _ = self._boundary[face]
        share = np.zeros(self.num_nodes)
        np.add.at(
            share,
            corners.reshape(-1),
            np.repeat(measure / corners.shape[1], corners.shape[1]),
        )
        return share

    # Mass ------------------------------------------------------------------------#
    @method(
        equations={
            "mass": r"u^\top M_0 u = \int u_h^2\, dV, \qquad M_0 = \sum_K P_K^\top M_{0,K} P_K",
            "lumped": r"M_0 = \mathrm{diag}(V_p)",
        },
        assumptions=(
            "on all nodes, hanging ones included: for the free nodes use $T^\\top M_0 T$",
            "Cartesian measure, also on an $(r, z)$ hierarchy",
        ),
        components=("patchwork.spaces.mass.node_mass_blocks",),
    )
    def mass(
        self, lumped: bool = True, weight: np.ndarray | float | None = None
    ) -> sparse.csr_matrix:
        """The node mass matrix ``M0``: ``u^T M0 u`` is the integral of ``u_h**2``.

        ``u_h`` is the (bi)linear interpolant of the node values in every leaf
        cell. ``lumped`` takes the vertex quadrature: ``diag(volumes)``, the
        control volumes of the finite-difference scheme; otherwise the exact
        (consistent) Q1 mass. For values on the free nodes, use ``T^T M0 T``.
        ``weight`` is a coefficient per leaf cell (a number or an array).
        """
        if lumped and weight is None:
            return sparse.csr_matrix(sparse.diags(self.volumes))
        blocks = node_mass_blocks(self.cell_size, lumped=lumped)
        if weight is not None:
            weight = np.broadcast_to(np.asarray(weight), blocks.shape[:1])
            blocks = blocks * weight[:, None, None]
        return scatter_blocks(
            blocks,
            self.cell_corners,
            self.cell_corners,
            (self.num_nodes, self.num_nodes),
        )

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
        C order with `level_shape`. It prolongs a field known on the
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
        `NodalPoisson.solve` returns them). A point outside the domain
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

    @method(
        equations={
            "bilinear": r"Q_c = \sum_k q_k\, w_c(\mathbf{x}_k), \qquad w_{00} = (1-\xi)(1-\eta),\ w_{10} = \xi(1-\eta),\ w_{01} = (1-\xi)\eta,\ w_{11} = \xi\eta",
            "transpose": r"Q = E^\top q, \qquad E_{kc} = w_c(\mathbf{x}_k) \quad (\texttt{evaluation})",
            "hanging": r"Q_F = (T^\top Q)_F \quad \text{(in } \texttt{NodalPoisson.solve(charge=)}\text{): the total is kept}",
        },
        assumptions=(
            r"$(\xi, \eta)$ the fractions of the point in its leaf cell, from its lower-left corner; linear weights in 1D",
            "points outside the domain deposit nothing",
        ),
        symbolic=_deposit_rules,
    )
    def deposit(self, points: np.ndarray, charges: np.ndarray) -> np.ndarray:
        """The charges of point particles at ``points`` ``(ndim, k)`` on the nodes: ``(n,)``.

        Each charge is spread over the corners of its leaf cell with the
        (bi)linear weights of `evaluation`, whose transpose this is: the
        work a field interpolated with `evaluation` does on the particles
        is the work their deposited charge feels, and no particle pushes
        itself through the coupling. Hanging nodes receive charge like any
        other; `NodalPoisson.solve` with ``charge=`` hands it to the
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
                # a copy in C order: a (2, 1) slice is flagged C-contiguous but
                # can keep F strides, which the compiled kernel refuses
                np.array(points, order="C"),
                np.array(charges, order="C"),
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

    def point_cells(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        """The leaf cell of each point ``(ndim, k)``, where in it, and whether it is inside.

        Returns the leaf cell (0 for a point outside the domain), the fraction
        of the cell's span from its lower corner per axis (one array per axis),
        and whether the point is in the domain.
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
        return leaf, fraction, inside

    def cell_weights(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Corner nodes and (bi)linear weights, ``(k, 2**ndim)``, of the leaf cell of each point.

        The corners are in the order of `cell_corners`; a point outside
        the domain has weights zero.
        """
        leaf, fraction, inside = self.point_cells(points)
        weights = np.column_stack(
            [
                np.prod(
                    [
                        fraction[axis] if upper else 1.0 - fraction[axis]
                        for axis, upper in enumerate(corner)
                    ],
                    axis=0,
                )
                for corner in _corner_offsets(self.ndim)
            ]
        )
        weights[~inside] = 0.0  # outside the domain: no value
        return self.cell_corners[leaf], weights

    def to_level(self, level: int) -> sparse.csr_matrix:
        """The composite interpolant on a level's lattice over the whole domain.

        Shape ``(nodes of the level lattice, n)``, the lattice in C order with
        `level_shape`: the solution sampled on a uniform grid.
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
        """`sample_patches` of one value per node."""
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

        Returns an array of `level_shape` plus the further axes, ``out``
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
