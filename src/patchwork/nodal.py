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
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

if TYPE_CHECKING:
    from patchwork.mesh import AMRHierarchy, Patch

#: The faces of the domain's bounding rectangle, as ``(axis, side)`` with side 0
#: the lower one.
FACES = {"x_lower": (0, 0), "x_upper": (0, 1), "y_lower": (1, 0), "y_upper": (1, 1)}

#: The rest of the boundary of a domain that does not fill its rectangle
#: (``AMRHierarchy(base_boxes=...)``): the edges inside the rectangle.
INNER = "inner"

Value = float | Callable[[np.ndarray, np.ndarray], Any]


@dataclass(frozen=True)
class Dirichlet:
    """A prescribed potential on a domain face: a number or ``value(x, y)``."""

    value: Value = 0.0


@dataclass(frozen=True)
class Neumann:
    """A prescribed outward normal derivative on a domain face.

    ``gradient`` is a number or ``gradient(x, y)``, evaluated at the middle of
    every cell edge on the face; zero is the natural condition.
    """

    gradient: Value = 0.0


@dataclass(frozen=True)
class EmbeddedDirichlet:
    """A body inside the domain held at one potential.

    ``inside(points)`` takes coordinates ``(2, n)`` inside the domain and returns
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


def _evaluate(value: Value, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A face value at points: a number, or the result of a callable."""
    result = value(x, y) if callable(value) else value
    return np.broadcast_to(np.asarray(result, dtype=float), x.shape).copy()


class CompositeNodes:
    """The nodes and leaf cells of a hierarchy, and the maps to its patches.

    Nodes are addressed on the *finest lattice*: the nodes of the finest level
    extended over the whole domain, ``lattice = level index * ratio**(finest -
    level)``. On a periodic axis the lattice wraps around, so the last node of
    the axis is the first.

    Attributes:
        hierarchy: The patch hierarchy.
        lattice: Finest-lattice coordinates of the nodes, ``(n, 2)`` integers.
        coordinates: Physical coordinates of the nodes, ``(n, 2)``.
        cell_corners: Node numbers of the corners of every leaf cell, ``(m, 4)``
            in the order lower-left, lower-right, upper-left, upper-right.
        cell_span: Edge length of every leaf cell on the finest lattice.
        cell_lower: Finest-lattice coordinates of the lower-left corners.
        volumes: Control volume of every node: a quarter of each cell around it.
        hanging: Whether a node is hanging (interpolated from other nodes).
        reduction: The interpolation ``T``: ``T @ u`` fills the hanging nodes of
            ``u`` from the others (which it leaves unchanged).
    """

    def __init__(self, hierarchy: AMRHierarchy) -> None:
        """Number the nodes of ``hierarchy`` and find its leaf cells and hanging nodes."""
        self.hierarchy = hierarchy
        ratio = hierarchy.refinement_ratio
        finest = hierarchy.max_level
        self._finest = finest
        self._cells = np.asarray(hierarchy.level_shape(finest), dtype=np.int64)
        self._periodic = tuple(hierarchy.periodic)
        self._spacing = np.asarray(hierarchy.spacing(finest), dtype=float)
        self._origin = np.asarray(hierarchy.lower_bound, dtype=float)

        lowers, spans = [], []
        for patch in hierarchy.patches:
            span = ratio ** (finest - patch.level)
            (i0, _), (j0, _) = patch.box
            (_, i1), (_, j1) = patch.box
            covered = np.zeros(patch.shape, dtype=bool)
            for child in hierarchy.patches_at_level(patch.level + 1):
                # a finer patch may lie over several patches of this level
                (c0, c1), (d0, d1) = child.box
                covered[
                    max(c0 // ratio, i0) - i0 : max(min(c1 // ratio, i1) - i0, 0),
                    max(d0 // ratio, j0) - j0 : max(min(d1 // ratio, j1) - j0, 0),
                ] = True
            i, j = np.nonzero(~covered)
            lowers.append(np.column_stack(((i + i0) * span, (j + j0) * span)))
            spans.append(np.full(i.size, span, dtype=np.int64))
        self.cell_lower = np.concatenate(lowers)
        self.cell_span = np.concatenate(spans)

        offsets = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int64)
        corners = (
            self.cell_lower[:, None, :] + self.cell_span[:, None, None] * offsets
        )  # (m, 4, 2)
        keys = self._key(corners[..., 0], corners[..., 1])
        self._keys, self.cell_corners = np.unique(keys, return_inverse=True)
        self.cell_corners = self.cell_corners.reshape(keys.shape)
        self.num_nodes = int(self._keys.size)
        stride = int(self._cells[1]) + 1
        self.lattice = np.column_stack((self._keys // stride, self._keys % stride))
        self.coordinates = self._origin + self.lattice * self._spacing

        size = self.cell_span[:, None] * self._spacing  # (m, 2) physical edge lengths
        self.cell_size = size
        self.volumes = np.zeros(self.num_nodes)
        np.add.at(
            self.volumes,
            self.cell_corners.reshape(-1),
            np.repeat(0.25 * size[:, 0] * size[:, 1], 4),
        )
        self._find_hanging_nodes()
        self._leaf_map: np.ndarray | None = None
        self._outside: dict[int, np.ndarray | None] = {}
        self._find_boundary()

    # Construction -------------------------------------------------------------#
    def _key(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """One integer per finest-lattice point, the same for periodic images."""
        i = np.asarray(i, dtype=np.int64)
        j = np.asarray(j, dtype=np.int64)
        if self._periodic[0]:
            i = i % self._cells[0]
        if self._periodic[1]:
            j = j % self._cells[1]
        return i * (int(self._cells[1]) + 1) + j

    def _lookup(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """Node numbers of finest-lattice points, -1 where there is no node."""
        keys = self._key(i, j)
        position = np.searchsorted(self._keys, keys)
        position = np.minimum(position, self.num_nodes - 1)
        return np.where(self._keys[position] == keys, position, -1)

    def _find_hanging_nodes(self) -> None:
        """Find the nodes inside coarser cells' edges and their interpolation."""
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

    # Patches -------------------------------------------------------------------#
    def patch_nodes(self, patch: Patch) -> np.ndarray:
        """Node numbers of the nodes of ``patch``, in its node shape."""
        span = self.hierarchy.refinement_ratio ** (self._finest - patch.level)
        (i0, i1), (j0, j1) = patch.box
        i, j = np.meshgrid(
            np.arange(i0, i1 + 1) * span, np.arange(j0, j1 + 1) * span, indexing="ij"
        )
        nodes = self._lookup(i, j)
        if (nodes < 0).any():
            raise RuntimeError("a patch node is not a node of the composite mesh")
        return nodes

    def to_patches(self, values: np.ndarray, name: str) -> None:
        """Write node values ``(n,)`` into the node-centered field ``name``."""
        for patch in self.hierarchy.local_patches:
            array = patch.field(name)
            (i0, i1), (j0, j1) = array.layout.index_bounds
            array.local[...] = values[self.patch_nodes(patch)][i0:i1, j0:j1]

    def from_patches(self, name: str) -> np.ndarray:
        """Node values ``(n,)`` of the node-centered field ``name``.

        Where patches share a node the finest one's value is taken. Collective:
        every patch array is gathered.
        """
        values = np.zeros(self.num_nodes)
        for patch in self.hierarchy.patches:  # coarse to fine: the finest wins
            values[self.patch_nodes(patch)] = self.hierarchy.gather(patch, name)
        return values

    # The boundary ----------------------------------------------------------------#
    def _find_boundary(self) -> None:
        """Find the cell edges on the boundary of the domain and name their face.

        An edge of a leaf cell is on the boundary if there is no cell beyond
        it: on a face of the bounding rectangle (``"x_lower"``, ...), or inside
        the rectangle where the domain does not fill it (``"inner"``).
        """
        leaf = self._leaf_cells()
        edges = {face: ([], [], [], []) for face in (*FACES, INNER)}
        # the corners of the edge on each side of a cell: lower-left first
        corners = {(0, 0): (0, 2), (0, 1): (1, 3), (1, 0): (0, 1), (1, 1): (2, 3)}
        centre = self.cell_lower + 0.5 * self.cell_span[:, None]
        for face, (axis, side) in FACES.items():
            other = 1 - axis
            beyond = self.cell_lower[:, axis] + (self.cell_span if side else -1)
            at_rectangle = (beyond < 0) | (beyond >= self._cells[axis])
            if self._periodic[axis]:
                beyond = beyond % self._cells[axis]
                at_rectangle[:] = False
            clipped = np.clip(beyond, 0, self._cells[axis] - 1)
            along = self.cell_lower[:, other]
            rows, columns = (clipped, along) if axis == 0 else (along, clipped)
            outside = at_rectangle | (leaf[rows, columns] < 0)
            midpoint = centre.copy()
            midpoint[:, axis] += (0.5 if side else -0.5) * self.cell_span
            first, second = corners[axis, side]
            for name, cells in (
                (face, np.nonzero(outside & at_rectangle)[0]),
                (INNER, np.nonzero(outside & ~at_rectangle)[0]),
            ):
                edges[name][0].append(self.cell_corners[cells, first])
                edges[name][1].append(self.cell_corners[cells, second])
                edges[name][2].append(self.cell_size[cells, other])
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

    def face_edges(
        self, face: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The cell edges on a part of the boundary.

        ``face`` is a face of the bounding rectangle (``"x_lower"``, ...) or
        ``"inner"``. Returns the two end nodes of every edge, its length and its
        midpoint ``(k, 2)``.
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
    def level_shape(self, level: int) -> tuple[int, int]:
        """Number of nodes of a level's lattice over the whole domain, per axis.

        A periodic axis has as many nodes as cells: the last node is the first.
        """
        cells = self.hierarchy.level_shape(level)
        return tuple(
            n if periodic else n + 1
            for n, periodic in zip(cells, self._periodic, strict=True)
        )  # type: ignore[return-value]

    def _bilinear(
        self, lower: Sequence[np.ndarray], fraction: Sequence[np.ndarray], level: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Columns and weights (n, 4) of bilinear interpolation on a level lattice."""
        shape = self.level_shape(level)
        index, weight = [], []
        for axis in range(2):
            upper = lower[axis] + 1
            if self._periodic[axis]:
                upper = upper % shape[axis]
            else:
                upper = np.minimum(upper, shape[axis] - 1)
            index.append((lower[axis], upper))
            weight.append((1.0 - fraction[axis], fraction[axis]))
        columns = np.column_stack(
            [index[0][a] * shape[1] + index[1][b] for a in (0, 1) for b in (0, 1)]
        )
        weights = np.column_stack(
            [weight[0][a] * weight[1][b] for a in (0, 1) for b in (0, 1)]
        )
        return columns, weights

    def from_level(self, level: int) -> sparse.csr_matrix:
        """Bilinear interpolation from a level's lattice to the composite nodes.

        The matrix has shape ``(n, nodes of the level lattice)``, the lattice in
        C order with :meth:`level_shape`. It prolongs a field known on the
        uniform grid of ``level`` (e.g. the base grid) to every node.
        """
        span = self.hierarchy.refinement_ratio ** (self._finest - level)
        cells = self.hierarchy.level_shape(level)
        lower, fraction = [], []
        for axis in range(2):
            position = self.lattice[:, axis]
            index = np.minimum(position // span, cells[axis] - 1)
            lower.append(index)
            fraction.append((position - index * span) / span)
        columns, weights = self._bilinear(lower, fraction, level)
        rows = np.repeat(np.arange(self.num_nodes), 4)
        shape = self.level_shape(level)
        matrix = sparse.csr_matrix(
            sparse.coo_matrix(
                (weights.reshape(-1), (rows, columns.reshape(-1))),
                shape=(self.num_nodes, shape[0] * shape[1]),
            )
        )
        matrix.eliminate_zeros()
        return matrix

    def _leaf_cells(self) -> np.ndarray:
        """The leaf cell that covers every cell of the finest lattice."""
        if self._leaf_map is None:
            leaf = np.full(tuple(self._cells), -1, dtype=np.int64)
            for span in np.unique(self.cell_span):
                cells = np.nonzero(self.cell_span == span)[0]
                offsets = np.arange(int(span))
                i = self.cell_lower[cells, 0][:, None, None] + offsets[None, :, None]
                j = self.cell_lower[cells, 1][:, None, None] + offsets[None, None, :]
                leaf[i, j] = cells[:, None, None]
            self._leaf_map = leaf
        return self._leaf_map

    def evaluation(self, points: np.ndarray) -> sparse.csr_matrix:
        """The composite interpolant at ``points`` ``(2, k)``: a ``(k, n)`` matrix.

        Each point takes the bilinear interpolant of the leaf cell it is in.
        Apply it to node values that have their hanging nodes filled (as
        :meth:`NodalPoisson.solve` returns them). A point outside the domain
        gets an empty row: the value zero.
        """
        points = np.asarray(points, dtype=float)
        corners, weights = self.cell_weights(points)
        rows = np.repeat(np.arange(points.shape[1]), 4)
        matrix = sparse.csr_matrix(
            sparse.coo_matrix(
                (weights.reshape(-1), (rows, corners.reshape(-1))),
                shape=(points.shape[1], self.num_nodes),
            )
        )
        matrix.eliminate_zeros()
        return matrix

    def deposit(self, points: np.ndarray, charges: np.ndarray) -> np.ndarray:
        """The charges of point particles at ``points`` ``(2, k)`` on the nodes: ``(n,)``.

        Each charge is spread over the corners of its leaf cell with the
        bilinear weights of :meth:`evaluation`, whose transpose this is: the
        work a field interpolated with :meth:`evaluation` does on the particles
        is the work their deposited charge feels, and no particle pushes
        itself through the coupling. Hanging nodes receive charge like any
        other; :meth:`NodalPoisson.solve` with ``charge=`` hands it to the
        nodes they hang from with the transposed interpolation, so the total
        charge is kept. Points outside the domain deposit nothing.
        """
        points = np.asarray(points, dtype=float)
        charges = np.asarray(charges, dtype=float).reshape(-1)
        if charges.shape[0] != points.shape[1]:
            raise ValueError(f"{points.shape[1]} points but {charges.shape[0]} charges")
        corners, weights = self.cell_weights(points)
        return np.bincount(
            corners.reshape(-1),
            weights=(weights * charges[:, None]).reshape(-1),
            minlength=self.num_nodes,
        )

    def cell_weights(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Corner nodes ``(k, 4)`` and bilinear weights ``(k, 4)`` of the leaf cell of each point.

        The corners are in the order of :attr:`cell_corners`; a point outside
        the domain has weights zero.
        """
        points = np.asarray(points, dtype=float)
        position = (points - self._origin[:, None]) / self._spacing[:, None]
        cell, local = [], []
        for axis in range(2):
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
        leaf = leaf_map[cell[0], cell[1]]
        # a point on a lattice line may have its cell on the lower side only
        # (the edge of a domain that does not fill its rectangle)
        on_line = [local[axis] == np.floor(local[axis]) for axis in (0, 1)]
        for step in ((1, 0), (0, 1), (1, 1)):
            missing = leaf < 0
            for axis in (0, 1):
                if step[axis]:
                    missing &= on_line[axis] & (cell[axis] > 0)
            if not missing.any():
                continue
            leaf[missing] = leaf_map[
                cell[0][missing] - step[0], cell[1][missing] - step[1]
            ]
        inside = leaf >= 0
        for axis in range(2):
            if not self._periodic[axis]:
                # beyond a non-periodic face: no value (the clipped cell would
                # extrapolate)
                inside &= (position[axis] >= 0.0) & (
                    position[axis] <= self._cells[axis]
                )
        leaf = np.where(inside, leaf, 0)
        span = self.cell_span[leaf]
        fraction = [
            (local[axis] - self.cell_lower[leaf, axis]) / span for axis in (0, 1)
        ]
        weights = np.column_stack(
            [
                (fraction[0] if a else 1.0 - fraction[0])
                * (fraction[1] if b else 1.0 - fraction[1])
                for b in (0, 1)
                for a in (0, 1)
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
        i, j = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
        points = (
            self._origin[:, None] + np.vstack((i.ravel(), j.ravel())) * spacing[:, None]
        )
        return self.evaluation(points)

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
        patches = self.hierarchy.patches
        if len(arrays) != len(patches):
            raise ValueError(f"expected {len(patches)} patch arrays, got {len(arrays)}")
        arrays = [np.asarray(array) for array in arrays]
        for patch, array in zip(patches, arrays, strict=True):
            if array.shape[:2] != patch.node_shape:
                raise ValueError(
                    f"a level-{patch.level} patch needs an array of "
                    f"{patch.node_shape} nodes, got {array.shape[:2]}"
                )
        extra = arrays[0].shape[2:]
        if out is None:
            out = np.empty((*self.level_shape(level), *extra), dtype=arrays[0].dtype)
        if not extra:
            out[...] = self._sample_scalar(arrays, level)
            return out
        # one component at a time: loops over a short last axis are slow
        components = out.reshape(*out.shape[:2], -1)
        for number in range(components.shape[2]):
            components[:, :, number] = self._sample_scalar(
                [array.reshape(*array.shape[:2], -1)[:, :, number] for array in arrays],
                level,
            )
        return out

    def _sample_scalar(self, arrays: Sequence[np.ndarray], level: int) -> np.ndarray:
        """:meth:`sample_patches` of one value per node."""
        ratio = self.hierarchy.refinement_ratio
        patches = self.hierarchy.patches
        # zero where the domain does not fill its rectangle
        current = np.zeros(self.level_shape(0), dtype=arrays[0].dtype)
        for depth in range(level + 1):
            if depth:
                for axis in range(2):
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

    def _nodes_outside(self, level: int) -> np.ndarray | None:
        """The nodes of a level's lattice outside the domain; None if there are none."""
        if level not in self._outside:
            hierarchy = self.hierarchy
            scale = hierarchy.refinement_ratio**level
            cells = np.zeros(hierarchy.level_shape(level), dtype=bool)
            for patch in hierarchy.patches_at_level(0):
                (i0, i1), (j0, j1) = patch.box
                cells[i0 * scale : i1 * scale, j0 * scale : j1 * scale] = True
            if cells.all():
                self._outside[level] = None
            else:
                # a node is in the domain if one of the cells around it is
                around = cells
                for axis in range(2):
                    width = [(0, 0), (0, 0)]
                    width[axis] = (1, 1)
                    around = np.pad(
                        around,
                        width,
                        mode="wrap" if self._periodic[axis] else "constant",
                    )
                inside = (
                    around[:-1, :-1]
                    | around[1:, :-1]
                    | around[:-1, 1:]
                    | around[1:, 1:]
                )
                shape = self.level_shape(level)
                self._outside[level] = ~inside[: shape[0], : shape[1]]
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


class NodalPoisson:
    r"""Solve :math:`-\nabla^2\phi = f` on the nodes of a patch hierarchy.

    The operator is assembled and factorized once; :meth:`solve` then costs two
    triangular solves. ``faces`` maps ``"x_lower"``, ``"x_upper"``,
    ``"y_lower"``, ``"y_upper"`` to a :class:`Dirichlet` or :class:`Neumann`
    condition; a face that is not listed is a grounded Dirichlet face, and the
    faces of a periodic axis take none. On a domain that does not fill its
    rectangle (``AMRHierarchy(base_boxes=...)``) these are the parts of its
    boundary on the rectangle's faces, and ``"inner"`` is the rest: the edges
    inside the rectangle. ``embedded`` are bodies at fixed potentials
    (:class:`EmbeddedDirichlet`); where they overlap the last one listed wins.

    Attributes:
        nodes: The :class:`CompositeNodes` of the hierarchy.
        free: Node numbers of the unknowns: neither hanging nor fixed.
        fixed: Whether a node has a prescribed value (a face or a body).
    """

    def __init__(
        self,
        hierarchy: AMRHierarchy,
        *,
        faces: Mapping[str, Dirichlet | Neumann] | None = None,
        embedded: Sequence[EmbeddedDirichlet] = (),
        min_fraction: float = 1.0e-3,
        nodes: CompositeNodes | None = None,
    ) -> None:
        """Assemble and factorize the operator.

        ``min_fraction`` is the smallest fraction of an edge between a node and
        an embedded surface: a surface closer than that (through the node, up to
        round-off) is moved there, which bounds the size of that node's row.
        """
        self.nodes = CompositeNodes(hierarchy) if nodes is None else nodes
        self._embedded = tuple(embedded)
        self._min_fraction = float(min_fraction)
        unknown = set(faces or ()) - set(FACES) - {INNER}
        if unknown:
            raise ValueError(
                f"unknown faces {sorted(unknown)}; expected {[*FACES, INNER]}"
            )
        self._faces: dict[str, Dirichlet | Neumann] = {}
        for face, (axis, _) in FACES.items():
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

    # Assembly ------------------------------------------------------------------#
    def _wrapped(self, points: np.ndarray) -> np.ndarray:
        """``points`` ``(2, k)`` moved into the domain along its periodic axes."""
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
                low = np.zeros(start.shape[1])
                high = np.ones(start.shape[1])
                for _ in range(48):
                    middle = 0.5 * (low + high)
                    hit = np.asarray(
                        body.inside(self._wrapped(start + middle * step)), dtype=bool
                    )
                    high = np.where(hit, middle, high)
                    low = np.where(hit, low, middle)
                fraction = high
            result[mine] = np.maximum(fraction, self._min_fraction)
        return result

    def _assemble(self) -> None:
        """Build the operator, its factorization and the right-hand-side maps."""
        nodes = self.nodes
        n = nodes.num_nodes
        self._classify_bodies()
        solid = self._owner >= 0

        # Every leaf cell's four edges, with half the stencil's conductance each.
        ll, lr, ul, ur = (nodes.cell_corners[:, k] for k in range(4))
        hx, hy = nodes.cell_size[:, 0], nodes.cell_size[:, 1]
        first = np.concatenate((ll, ul, ll, lr))
        second = np.concatenate((lr, ur, ul, ur))
        conductance = np.concatenate(
            (0.5 * hy / hx, 0.5 * hy / hx, 0.5 * hx / hy, 0.5 * hx / hy)
        )
        length = np.concatenate((hx, hx, hy, hy))
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

        rows = np.concatenate((first, second, first, second, outside))
        columns = np.concatenate((first, second, second, first, outside))
        entries = np.concatenate(
            (
                conductance,
                conductance,
                -conductance,
                -conductance,
                cut_conductance / fraction,
            )
        )
        num_bodies = len(self._embedded)
        # the surface values' share of the right-hand side, per body
        wall = sparse.csr_matrix(
            sparse.coo_matrix(
                (cut_conductance / fraction, (outside, self._owner[inside])),
                shape=(n, max(num_bodies, 1)),
            )
        )
        operator = sparse.csr_matrix(
            sparse.coo_matrix((entries, (rows, columns)), shape=(n, n))
        )

        operator, wall = self._shortley_weller(
            operator, wall, outside, inside, fraction * cut_length
        )

        reduction = nodes.reduction
        reduced = sparse.csr_matrix(reduction.T @ operator @ reduction)
        self._wall = sparse.csr_matrix(reduction.T @ wall)
        self._load = sparse.csr_matrix(reduction.T @ sparse.diags(nodes.volumes))
        # a source already integrated over the control volumes (a deposited charge)
        self._integral_load = sparse.csr_matrix(reduction.T)

        self.fixed = np.zeros(n, dtype=bool)
        for face, condition in self._faces.items():
            if isinstance(condition, Dirichlet):
                self.fixed[nodes.face_nodes(face)] = True
        self.fixed |= solid
        self.fixed &= ~nodes.hanging
        self.free = np.nonzero(~self.fixed & ~nodes.hanging)[0]
        fixed = np.nonzero(self.fixed)[0]
        self._fixed_nodes = fixed
        self._lift = reduced[self.free][:, fixed].tocsr()
        self._load = self._load[self.free]
        self._integral_load = self._integral_load[self.free]
        self._wall = self._wall[self.free]
        self._reduction_rows = reduction[np.nonzero(nodes.hanging)[0]].tocsr()
        matrix = reduced[self.free][:, self.free].tocsc()
        self.matrix = matrix
        if not self.fixed.any():
            raise ValueError(
                "the Poisson operator is singular: without a Dirichlet face or "
                "an embedded body the potential is only known up to a constant"
            )
        self._factorization = None
        if self.free.size:
            # minimum degree on the pattern of A + A^T: the mesh's own graph,
            # about half the fill of the default column ordering
            self._factorization = scipy.sparse.linalg.splu(
                matrix, permc_spec="MMD_AT_PLUS_A"
            )

    def _shortley_weller(
        self,
        operator: sparse.csr_matrix,
        wall: sparse.csr_matrix,
        outside: np.ndarray,
        inside: np.ndarray,
        distance: np.ndarray,
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """Replace the rows of regular nodes next to a body by Shortley-Weller rows.

        A node is regular if four cells of one size meet at it. Its row is then
        the finite-difference Laplacian with unequal arms, times its control
        volume: second-order accurate up to the surface, as on a uniform grid.
        """
        nodes = self.nodes
        n = nodes.num_nodes
        if not outside.size:
            return operator, wall
        count = np.zeros(n, dtype=np.int64)
        np.add.at(count, nodes.cell_corners.reshape(-1), 1)
        smallest = np.full(n, np.iinfo(np.int64).max)
        largest = np.zeros(n, dtype=np.int64)
        spans = np.repeat(nodes.cell_span, 4)
        np.minimum.at(smallest, nodes.cell_corners.reshape(-1), spans)
        np.maximum.at(largest, nodes.cell_corners.reshape(-1), spans)
        regular = (count == 4) & (smallest == largest) & ~nodes.hanging
        cut_nodes = np.unique(outside[regular[outside]])
        if not cut_nodes.size:
            return operator, wall

        span = smallest[cut_nodes]
        spacing = span[:, None] * nodes._spacing  # (k, 2)
        position = {int(node): k for k, node in enumerate(cut_nodes)}
        # arm lengths and what each arm ends at: [axis][side] over the cut nodes
        arm = np.stack([np.stack([spacing[:, axis]] * 2) for axis in range(2)])
        body = np.full((2, 2, cut_nodes.size), -1, dtype=np.int64)
        neighbour = np.empty((2, 2, cut_nodes.size), dtype=np.int64)
        for axis in range(2):
            for side, sign in enumerate((-1, 1)):
                step = np.zeros((cut_nodes.size, 2), dtype=np.int64)
                step[:, axis] = sign * span
                there = nodes.lattice[cut_nodes] + step
                neighbour[axis, side] = nodes._lookup(there[:, 0], there[:, 1])
        for out_node, in_node, d in zip(outside, inside, distance, strict=True):
            k = position.get(int(out_node))
            if k is None:
                continue
            axis, side = np.argwhere(neighbour[:, :, k] == in_node)[0]
            arm[axis, side, k] = d
            body[axis, side, k] = self._owner[in_node]

        rows, columns, entries = [], [], []
        wall_rows, wall_columns, wall_entries = [], [], []
        volume = spacing[:, 0] * spacing[:, 1]
        diagonal = np.zeros(cut_nodes.size)
        for axis in range(2):
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

        keep = np.ones(n)
        keep[cut_nodes] = 0.0
        keep = sparse.diags(keep)
        operator = keep @ operator + sparse.coo_matrix(
            (np.concatenate(entries), (np.concatenate(rows), np.concatenate(columns))),
            shape=(n, n),
        )
        wall = keep @ wall + sparse.coo_matrix(
            (
                np.concatenate(wall_entries),
                (np.concatenate(wall_rows), np.concatenate(wall_columns)),
            ),
            shape=wall.shape,
        )
        return sparse.csr_matrix(operator), sparse.csr_matrix(wall)

    # Boundary values ---------------------------------------------------------------#
    def _update_values(self) -> None:
        """Evaluate the prescribed values and their share of the right-hand side."""
        nodes = self.nodes
        n = nodes.num_nodes
        values = np.zeros(n)
        x, y = nodes.coordinates[:, 0], nodes.coordinates[:, 1]
        flux = np.zeros(n)
        # x faces before y faces, lower before upper: the first one decides a corner
        decided = np.zeros(n, dtype=bool)
        for face, condition in self._faces.items():
            if isinstance(condition, Dirichlet):
                on_face = nodes.face_nodes(face)
                on_face = on_face[~decided[on_face]]
                values[on_face] = _evaluate(condition.value, x[on_face], y[on_face])
                decided[on_face] = True
            else:
                # the gradient at the middle of every boundary edge, shared by its ends
                first, second, length, midpoint = nodes.face_edges(face)
                through = (
                    0.5
                    * length
                    * _evaluate(condition.gradient, midpoint[:, 0], midpoint[:, 1])
                )
                np.add.at(flux, first, through)
                np.add.at(flux, second, through)
        body_values = np.array([body.value for body in self._embedded] or [0.0])
        solid = self._owner >= 0
        values[solid] = body_values[self._owner[solid]]
        self._values = values
        constant = self._wall @ body_values
        constant += (nodes.reduction.T @ flux)[self.free]
        constant -= self._lift @ values[self._fixed_nodes]
        self._constant = constant

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
        """Map from a source on all nodes to the right-hand side of the unknowns."""
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

    def solve_reduced(self, right_hand_side: np.ndarray) -> np.ndarray:
        """Solve for a right-hand side of the unknowns; the potential on all nodes.

        ``right_hand_side`` is ``load @ source + constant``, which callers that
        solve repeatedly with a source of fixed structure can precompute as one
        matrix. Hanging nodes are filled by interpolation.
        """
        nodes = self.nodes
        potential = self._values.copy()
        if self._factorization is not None:
            potential[self.free] = self._factorization.solve(right_hand_side)
        if nodes.hanging.any():
            potential[nodes.hanging] = self._reduction_rows @ potential
        return potential

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
        from, and the charge on fixed nodes is ignored. With ``potential`` the
        result is also written to the node-centered field of that name
        (allocated if needed).
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
) -> NodalPoisson:
    r"""Solve :math:`-\nabla^2\phi = \mathrm{rhs}` once on the nodes of ``hierarchy``.

    ``rhs`` and ``potential`` name node-centered fields; see
    :class:`NodalPoisson` for the conditions. Returns the solver, which can
    solve again without assembling.
    """
    solver = NodalPoisson(hierarchy, faces=faces, embedded=embedded)
    solver.solve(rhs, potential=potential)
    return solver
