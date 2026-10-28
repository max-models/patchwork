"""The lowest-order edge space of a 2D hierarchy, next to its nodes.

:class:`CompositeEdges` holds the edges of the leaf cells of
:class:`~patchwork.nodal.CompositeNodes`. A field ``e`` on them is the line
integral of the electric field along every edge, and inside a leaf cell the
field is the lowest-order (Whitney) edge interpolant: the x component varies
linearly in y between the cell's lower and upper x-edges, the y component
linearly in x between its left and right y-edges, and the tangential component
is constant along every edge. Then

- the discrete gradient :attr:`CompositeEdges.gradient` is the incidence of
  edges and nodes (end minus start), exact for any potential, and the field of
  ``gradient @ phi`` is the gradient of the bilinear interpolant of ``phi``;
- a coarse edge on a patch edge is two fine half-edges, which hang: the line
  integral of a constant tangential field over a half is half the coarse one
  (:attr:`CompositeEdges.reduction`), and the hanging nodes' interpolation
  commutes with the gradient (``G T = T1 G T``);
- the edge mass matrix (:meth:`CompositeEdges.mass`) is the field energy
  ``e^T M e`` of the cells (without ``eps0``); lumped, ``G^T M G`` is the
  operator of :class:`~patchwork.nodal.NodalPoisson` (its five-point stencil
  inside a patch).

This is the space of the implicit, energy-conserving PIC: the field gather
(:meth:`CompositeEdges.evaluation`) and the current deposit, its transpose
(:meth:`CompositeEdges.deposit`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sparse
import scope_profiler as sp

from patchwork._profiling import PROFILE_TAGS

if TYPE_CHECKING:
    from patchwork.nodal import CompositeNodes

#: The sides of a leaf cell, in the order of :attr:`CompositeEdges.cell_edges`:
#: direction of the edge, offset of its lower end from the cell's lower corner
#: (in spans, per axis), and the corners (of ``cell_corners``) at its two ends.
_SIDES = (
    (0, (0, 0), (0, 1)),  # lower x-edge
    (0, (0, 1), (2, 3)),  # upper x-edge
    (1, (0, 0), (0, 2)),  # left y-edge
    (1, (1, 0), (1, 3)),  # right y-edge
)


class CompositeEdges:
    """The edges of the leaf cells of a 2D hierarchy, and the operators on them.

    Edges are addressed on the finest lattice, as the nodes are: an edge is its
    direction (0 along x, 1 along y), the lattice point of its lower end and its
    span. A coarse edge and the fine edges that halve it are different edges;
    the fine ones hang.

    Attributes:
        nodes: The :class:`~patchwork.nodal.CompositeNodes` of the hierarchy.
        num_edges: Number of edges.
        direction: The axis along every edge, ``(n1,)``.
        lattice: Finest-lattice coordinates of the lower end of every edge,
            ``(n1, 2)``.
        span: Length of every edge on the finest lattice.
        length: Physical length of every edge.
        ends: Node numbers of the start and the end of every edge, ``(n1, 2)``.
        cell_edges: The edges of every leaf cell, ``(m, 4)``: lower and upper
            x-edge, left and right y-edge.
        hanging: Whether an edge hangs (lies in a coarser edge).
        reduction: The constraint ``T1``: ``T1 @ e`` fills the hanging edges of
            ``e`` from the others (which it leaves unchanged).
        gradient: The discrete gradient ``G``, ``(n1, n)``: the potential's
            difference along every edge.
    """

    # Construction -------------------------------------------------------------#
    def _keys(self, direction, lower, span) -> np.ndarray:
        """One integer per edge: direction, lower end (as a node key) and span."""
        node_key = self.nodes._key(lower[..., 0], lower[..., 1])
        return (node_key * 2 + direction) * self._max_span + span

    def _find(self, direction, lower, span) -> np.ndarray:
        """Edge numbers of edges, -1 where there is none."""
        keys = self._keys(direction, lower, span)
        position = np.minimum(np.searchsorted(self._sorted, keys), self.num_edges - 1)
        return np.where(self._sorted[position] == keys, position, -1)

    def _find_hanging_edges(self) -> None:
        """Find the edges inside coarser cells' edges and their constraints."""
        nodes = self.nodes
        ratio = nodes.hierarchy.refinement_ratio
        slaves, masters, weights, spans = [], [], [], []
        for coarse in np.unique(nodes.cell_span):
            cells = np.nonzero(nodes.cell_span == coarse)[0]
            fine = coarse // ratio
            while fine >= 1:
                count = int(coarse // fine)
                for side, (direction, offset, _) in enumerate(_SIDES):
                    master = self.cell_edges[cells, side]
                    for k in range(count):
                        lower = nodes.cell_lower[cells] + coarse * np.asarray(offset)
                        lower[:, direction] += k * fine
                        edge = self._find(direction, lower, fine)
                        found = edge >= 0
                        slaves.append(edge[found])
                        masters.append(master[found])
                        weights.append(np.full(int(found.sum()), fine / coarse))
                        spans.append(np.full(int(found.sum()), coarse))
                fine //= ratio
        self.hanging = np.zeros(self.num_edges, dtype=bool)
        identity = sparse.csr_matrix(sparse.identity(self.num_edges))
        if not slaves or not sum(part.size for part in slaves):
            self.reduction = identity
            return
        slave = np.concatenate(slaves)
        master = np.concatenate(masters)
        weight = np.concatenate(weights)
        span = np.concatenate(spans)
        # one constraint per edge, from the coarsest edge it lies in
        order = np.lexsort((-span, slave))
        slave, master, weight = slave[order], master[order], weight[order]
        first = np.concatenate(([True], slave[1:] != slave[:-1]))
        slave, master, weight = slave[first], master[first], weight[first]
        self.hanging[slave] = True
        free = np.nonzero(~self.hanging)[0]
        reduction = sparse.csr_matrix(
            sparse.coo_matrix(
                (
                    np.concatenate((np.ones(free.size), weight)),
                    (np.concatenate((free, slave)), np.concatenate((free, master))),
                ),
                shape=(self.num_edges, self.num_edges),
            )
        )
        # a coarse edge may itself lie in a still coarser one
        for _ in range(nodes._finest + 1):
            if not reduction[:, slave].nnz:
                break
            reduction = sparse.csr_matrix(reduction @ reduction)
            reduction.eliminate_zeros()
        else:
            raise RuntimeError("the hanging-edge constraints do not close")
        self.reduction = reduction

    @sp.profile("patchwork.composite_edges", tags=PROFILE_TAGS)
    def __init__(self, nodes: CompositeNodes) -> None:
        """Number the edges of the leaf cells of ``nodes`` and find the hanging ones."""
        if nodes.ndim != 2:
            raise NotImplementedError("the edge space is two-dimensional")
        self.nodes = nodes
        self._max_span = int(nodes.cell_span.max()) + 1
        directions, lowers, spans, ends = [], [], [], []
        for direction, offset, (start, end) in _SIDES:
            directions.append(np.full(nodes.cell_span.size, direction))
            lowers.append(
                nodes.cell_lower + nodes.cell_span[:, None] * np.asarray(offset)
            )
            spans.append(nodes.cell_span)
            ends.append(nodes.cell_corners[:, [start, end]])
        direction = np.stack(directions, axis=1)  # (m, 4)
        lower = np.stack(lowers, axis=1)  # (m, 4, 2)
        span = np.stack(spans, axis=1)
        keys = self._keys(direction, lower, span)
        self._sorted, first, inverse = np.unique(
            keys.reshape(-1), return_index=True, return_inverse=True
        )
        self.num_edges = int(self._sorted.size)
        self.cell_edges = inverse.reshape(keys.shape)
        self.direction = direction.reshape(-1)[first]
        self.lattice = lower.reshape(-1, 2)[first]
        for axis in range(2):
            if nodes._periodic[axis]:  # the seam's edges at the lower end
                self.lattice[:, axis] %= nodes._cells[axis]
        self.span = span.reshape(-1)[first]
        self.length = self.span * nodes._spacing[self.direction]
        self.ends = np.stack(ends, axis=1).reshape(-1, 2)[first]
        rows = np.repeat(np.arange(self.num_edges), 2)
        self.gradient = sparse.csr_matrix(
            sparse.coo_matrix(
                (
                    np.tile([-1.0, 1.0], self.num_edges),
                    (rows, self.ends.reshape(-1)),
                ),
                shape=(self.num_edges, nodes.num_nodes),
            )
        )
        # a periodic axis one cell wide joins a node to itself: no difference
        self.gradient.eliminate_zeros()
        self._find_hanging_edges()

    # Operators -------------------------------------------------------------------#
    def mass(self, lumped: bool = True) -> sparse.csr_matrix:
        """The edge mass matrix ``M1``: ``e^T M1 e`` is the integral of ``|E|^2``.

        Summed over the leaf cells, each with the edge interpolant of its own
        four edges. ``lumped`` takes the vertex (trapezoidal) quadrature, which
        makes it diagonal: ``h_perp / (2 h)`` per cell and edge, the conductance
        of :class:`~patchwork.nodal.NodalPoisson`'s stencil. Otherwise the exact
        integral, ``h_perp / h`` times ``[[1/3, 1/6], [1/6, 1/3]]`` for the two
        edges of a direction. Both are symmetric and positive definite on the
        free edges. For a field on hanging edges, use ``T1^T M1 T1``.
        """
        nodes = self.nodes
        size = nodes.cell_size  # (m, 2)
        rows, columns, entries = [], [], []
        for direction, (low, high) in ((0, (0, 1)), (1, (2, 3))):
            scale = size[:, 1 - direction] / size[:, direction]
            a, b = self.cell_edges[:, low], self.cell_edges[:, high]
            if lumped:
                pairs = ((a, a, 0.5), (b, b, 0.5))
            else:
                pairs = (
                    (a, a, 1.0 / 3.0),
                    (b, b, 1.0 / 3.0),
                    (a, b, 1.0 / 6.0),
                    (b, a, 1.0 / 6.0),
                )
            for row, column, factor in pairs:
                rows.append(row)
                columns.append(column)
                entries.append(factor * scale)
        return sparse.csr_matrix(
            sparse.coo_matrix(
                (
                    np.concatenate(entries),
                    (np.concatenate(rows), np.concatenate(columns)),
                ),
                shape=(self.num_edges, self.num_edges),
            )
        )

    def evaluation(self, points: np.ndarray) -> tuple[sparse.csr_matrix, ...]:
        """The field of an edge field at ``points`` ``(2, k)``: one ``(k, n1)`` matrix per axis.

        Each point takes the edge interpolant of the leaf cell it is in. Apply
        them to edge values with their hanging edges filled (``T1 @ e``). A point
        outside the domain gets empty rows: the field zero.
        """
        nodes = self.nodes
        leaf, fraction, inside = nodes.point_cells(points)
        size = nodes.cell_size[leaf]
        count = leaf.size
        rows = np.repeat(np.arange(count), 2)
        matrices = []
        for direction, (low, high) in ((0, (0, 1)), (1, (2, 3))):
            # the x component varies with y between the lower and upper x-edges
            across = fraction[1 - direction]
            weights = np.column_stack((1.0 - across, across)) / size[:, [direction]]
            weights[~inside] = 0.0
            columns = self.cell_edges[leaf][:, [low, high]]
            matrix = sparse.csr_matrix(
                sparse.coo_matrix(
                    (weights.reshape(-1), (rows, columns.reshape(-1))),
                    shape=(count, self.num_edges),
                )
            )
            matrix.eliminate_zeros()
            matrices.append(matrix)
        return tuple(matrices)

    def deposit(self, points: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Vectors ``(2, k)`` at ``points`` ``(2, k)`` on the edges: ``(n1,)``.

        The transpose of :meth:`evaluation`: ``deposit(points, v) @ e`` is the
        sum of ``v . E(points)`` for the field ``E`` of ``e``. With the
        velocities times the charges of markers, it is their current at those
        points (a path integral of it is the current along their paths).
        Hanging edges receive their share like any other; ``T1^T`` hands it to
        the edges they hang from.
        """
        vectors = np.asarray(vectors, dtype=float)
        x, y = self.evaluation(points)
        return x.T @ vectors[0] + y.T @ vectors[1]

    def _pieces(
        self, start: np.ndarray, end: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The straight paths from ``start`` to ``end`` ``(2, k)``, cut at the finest lattice.

        Returns the path of every piece and its two ends ``(2, pieces)``. A
        piece lies in one cell of the finest lattice, and so in one leaf cell.
        """
        nodes = self.nodes
        origin = nodes._origin[:, None]
        spacing = nodes._spacing[:, None]
        lower = (start - origin) / spacing
        upper = (end - origin) / spacing
        count = start.shape[1]
        paths, times = [np.arange(count)] * 2, [np.zeros(count), np.ones(count)]
        for axis in range(2):
            a, b = lower[axis], upper[axis]
            first = np.floor(np.minimum(a, b)).astype(np.int64) + 1
            last = np.ceil(np.maximum(a, b)).astype(np.int64) - 1
            crossings = np.maximum(last - first + 1, 0)
            path = np.repeat(np.arange(count), crossings)
            offset = np.repeat(np.cumsum(crossings) - crossings, crossings)
            line = first[path] + np.arange(path.size) - offset
            paths.append(path)
            times.append((line - a[path]) / (b - a)[path])
        path = np.concatenate(paths)
        time = np.concatenate(times)
        order = np.lexsort((time, path))
        path, time = path[order], time[order]
        # consecutive times of one path bound a piece
        same = path[1:] == path[:-1]
        path, begin, finish = path[1:][same], time[:-1][same], time[1:][same]
        step = end - start
        return (
            path,
            start[:, path] + begin * step[:, path],
            start[:, path] + finish * step[:, path],
        )

    def deposit_paths(
        self, start: np.ndarray, end: np.ndarray, charges: np.ndarray
    ) -> np.ndarray:
        """The current of charges moving along straight paths, integrated over the paths: ``(n1,)``.

        Every charge goes from ``start`` to ``end`` (``(2, k)``, not wrapped into
        the domain on a periodic axis); its path is split at every cell face it
        crosses, patch edges included, and each piece is integrated exactly
        against the edge functions of its leaf cell: ``q * integral of w . dl``.
        Then the charge is conserved exactly: ``(G T)^T j`` is the change of the
        charge on the nodes, ``T^T`` of :meth:`CompositeNodes.deposit
        <patchwork.nodal.CompositeNodes.deposit>` at the ends minus at the
        starts, and ``j @ e`` is the work of the field ``e`` on the charges. The
        current density times the time step, summed over a step's paths, is
        ``j``. Pieces outside the domain deposit nothing. A NumPy reference,
        one pass over all pieces.
        """
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        charges = np.asarray(charges, dtype=float).reshape(-1)
        if start.shape != end.shape or start.shape != (2, charges.size):
            raise ValueError(
                f"paths from {start.shape} to {end.shape} for {charges.size} charges"
            )
        path, a, b = self._pieces(start, end)
        nodes = self.nodes
        # the edge functions are linear in a leaf cell: the midpoint integrates them
        leaf, fraction, inside = nodes.point_cells(0.5 * (a + b))
        size = nodes.cell_size[leaf]
        charge = np.where(inside, charges[path], 0.0)
        step = b - a
        current = np.zeros(self.num_edges)
        for direction, (low, high) in ((0, (0, 1)), (1, (2, 3))):
            across = fraction[1 - direction]
            along = charge * step[direction] / size[:, direction]
            current += np.bincount(
                self.cell_edges[leaf, low],
                weights=(1.0 - across) * along,
                minlength=self.num_edges,
            )
            current += np.bincount(
                self.cell_edges[leaf, high],
                weights=across * along,
                minlength=self.num_edges,
            )
        return current
