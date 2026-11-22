"""The lowest-order edge space of a 2D or 3D hierarchy, next to its nodes.

`CompositeEdges` holds the edges of the leaf cells of
`CompositeNodes`. A field ``e`` on them is the line
integral of the electric field along every edge, and inside a leaf cell the
field is the lowest-order (Whitney, Nédélec first kind) edge interpolant: the
component along an axis varies multilinearly across the cell between the cell's
edges along that axis (linearly in y between the lower and upper x-edges in 2D,
bilinearly in y and z between the four x-edges in 3D), and the tangential
component is constant along every edge. Then

- the discrete gradient `CompositeEdges.gradient` is the incidence of
  edges and nodes (end minus start), exact for any potential, and the field of
  ``gradient @ phi`` is the gradient of the bilinear interpolant of ``phi``;
- a coarse edge on a patch edge is ``r`` fine edges, which hang: the line
  integral of a constant tangential field over a piece is its share of the
  coarse one (`CompositeEdges.reduction`); in 3D a fine edge inside a coarse
  face hangs too, and takes the line integral of the coarse face's
  interpolant; the hanging nodes' interpolation commutes with the gradient
  (``G T = T1 G T``);
- the edge mass matrix (`CompositeEdges.mass`) is the field energy
  ``e^T M e`` of the cells (without ``eps0``); ``G^T M G`` is the operator of
  `NodalPoisson`: its five-point stencil with the lumped mass (inside a patch),
  the Q1 finite-element stiffness with the consistent one
  (``NodalPoisson(mass="consistent")``). Both come from the per-cell blocks of
  `patchwork.spaces.mass`.

This is the space of the implicit, energy-conserving PIC: the field gather
(`CompositeEdges.evaluation`, compiled in `CompositeEdges.gather`) and the current
deposit, its transpose (`CompositeEdges.deposit`, along paths
`CompositeEdges.deposit_paths`). The compiled ones run mpifield's block kernels
(`mpifield.WHITNEY`) on every patch.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sparse
import scope_profiler as sp
from methodcard import Rule, method

from patchwork._profiling import PROFILE_TAGS
from patchwork.spaces.mass import (
    corner_bits,
    edge_corners,
    edge_mass_blocks,
    scatter_blocks,
)

if TYPE_CHECKING:
    from patchwork.spaces.nodes import CompositeNodes


def _weighted(blocks: np.ndarray, weight: np.ndarray | float | None) -> np.ndarray:
    """Per-cell blocks ``(m, a, b)`` times a coefficient per cell (none: one)."""
    if weight is None:
        return blocks
    weight = np.broadcast_to(np.asarray(weight), blocks.shape[:1])
    return blocks * weight[:, None, None]


def _sides(ndim: int) -> list[tuple[int, int, int]]:
    """The edges of a cell, in the order of `CompositeEdges.cell_edges`.

    For each axis, the edges along it in the order of `edge_corners`: the
    direction and the corners (of ``cell_corners``) at the two ends. In 2D: the
    lower and upper x-edge, then the left and right y-edge.
    """
    return [
        (axis, start, end)
        for axis in range(ndim)
        for start, end in edge_corners(ndim, axis)
    ]


def _axis_columns(ndim: int, axis: int) -> list[int]:
    """The columns of `CompositeEdges.cell_edges` that hold the edges along ``axis``."""
    per_axis = 2 ** (ndim - 1)
    return list(range(axis * per_axis, (axis + 1) * per_axis))


def _clip(
    a: np.ndarray, step: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The times in ``[0, 1]`` at which the paths ``a + t step`` ``(ndim, k)`` enter and leave a box.

    ``lower`` and ``upper`` are the box's corners ``(ndim,)``. A path that misses the
    box gets an exit before its entry. Every path is clipped with the same arithmetic,
    so two boxes that share a face give their common path point the same time.
    """
    enter = np.zeros(a.shape[1])
    leave = np.ones(a.shape[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        for axis in range(a.shape[0]):
            d = step[axis]
            low = (lower[axis] - a[axis]) / d
            high = (upper[axis] - a[axis]) / d
            moving = d != 0.0
            inside = (a[axis] >= lower[axis]) & (a[axis] <= upper[axis])
            enter = np.where(moving, np.maximum(enter, np.minimum(low, high)), enter)
            leave = np.where(moving, np.minimum(leave, np.maximum(low, high)), leave)
            leave = np.where(moving | inside, leave, -1.0)
    return enter, leave


def _shifts(
    a: np.ndarray,
    b: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    cells: np.ndarray,
    periodic: tuple[bool, ...],
) -> list[np.ndarray]:
    """The shifts by whole periods that bring part of some path into a box.

    On a periodic axis paths are not wrapped: a box at ``[lower, upper]`` meets them
    at ``[lower, upper] + k cells`` for every ``k`` within their extent.
    """
    ranges = []
    for axis, wraps in enumerate(periodic):
        if not wraps or a.shape[1] == 0:
            ranges.append([0])
            continue
        period = float(cells[axis])
        smallest = float(min(a[axis].min(), b[axis].min()))
        largest = float(max(a[axis].max(), b[axis].max()))
        first = int(np.ceil((smallest - upper[axis]) / period))
        last = int(np.floor((largest - lower[axis]) / period))
        ranges.append([k * period for k in range(first, last + 1)])
    return [np.array(shift) for shift in itertools.product(*ranges)]


@dataclass(frozen=True)
class _PatchBlock:
    """A patch as the block kernels see it, on the finest lattice (its unit is one cell).

    Attributes:
        grid: The `mpifield.BlockGrid` of the patch's nodes, in finest-lattice units
            (lower corner, span, nodes), bounded on every axis.
        lower: Finest-lattice coordinates of the patch's lower corner.
        upper: Finest-lattice coordinates of its upper corner.
        children: The boxes ``(c, 2, ndim)`` of the patches one level finer that
            overlap it: their cells are not leaf cells of this patch.
        edge_map: The composite edge of every edge of the block, ``num_nodes +
            (ndim,)``; -1 for an edge of no leaf cell of the patch.
    """

    grid: Any
    lower: np.ndarray
    upper: np.ndarray
    children: np.ndarray
    edge_map: np.ndarray


def _uniform_edges(hx: float, hy: float, cells: int = 4) -> CompositeEdges:
    """The edges of ``cells`` by ``cells`` uniform cells of ``hx`` by ``hy``."""
    from patchwork.mesh.hierarchy import AMRHierarchy
    from patchwork.spaces.nodes import CompositeNodes

    bounds = ((0.0, cells * hx), (0.0, cells * hy))
    mesh = AMRHierarchy(base_shape=(cells, cells), bounds=bounds)
    return CompositeEdges(CompositeNodes(mesh))


def _interior_x_edge(edges: CompositeEdges) -> int:
    """The x-edge from lattice point (1, 2): two cells share it."""
    lattice = edges.lattice
    found = (edges.direction == 0) & (lattice[:, 0] == 1) & (lattice[:, 1] == 2)
    return int(np.flatnonzero(found)[0])


def _mass_rules() -> dict:
    """The SymPy rules of `CompositeEdges.mass` (for the tests)."""
    import sympy

    hx, hy = sympy.symbols("hx hy", positive=True)
    ranges = {"hx": (0.05, 1.0), "hy": (0.05, 1.0)}

    def entry(hx: float, hy: float, lumped: bool) -> float:
        edges = _uniform_edges(hx, hy)
        edge = _interior_x_edge(edges)
        return float(edges.mass(lumped)[edge, edge])

    return {
        "lumped": Rule(
            hy / hx,
            implementation=lambda hx, hy: entry(hx, hy, True),
            ranges=ranges,
        ),
        "consistent": Rule(
            2 * hy / (3 * hx),
            implementation=lambda hx, hy: entry(hx, hy, False),
            ranges=ranges,
        ),
    }


def _whitney_rules() -> dict:
    """The SymPy rule of `CompositeEdges.evaluation` (for the tests)."""
    import sympy

    hx, eta, lower, upper = sympy.symbols("hx eta e_lower e_upper", real=True)

    def field(hx: float, eta: float, e_lower: float, e_upper: float) -> float:
        edges = _uniform_edges(hx, 1.0, cells=2)
        cell = int(np.flatnonzero((edges.nodes.cell_lower == 0).all(axis=1))[0])
        e = np.zeros(edges.num_edges)
        e[edges.cell_edges[cell, 0]] = e_lower
        e[edges.cell_edges[cell, 1]] = e_upper
        # E_x does not vary along x in a cell: any point of the row
        x, _ = edges.evaluation(np.array([[0.3 * hx], [eta]]))
        return float((x @ e)[0])

    return {
        "whitney": Rule(
            ((1 - eta) * lower + eta * upper) / hx,
            implementation=field,
            ranges={
                "hx": (0.1, 1.0),
                "eta": (0.01, 0.99),
                "e_lower": (0.5, 2.0),
                "e_upper": (0.5, 2.0),
            },
        )
    }


def _continuity_rules() -> dict:
    """The SymPy rule of `CompositeEdges.deposit_paths` (for the tests)."""
    import sympy

    from patchwork.mesh.hierarchy import AMRHierarchy
    from patchwork.spaces.nodes import CompositeNodes

    x0, y0, q = sympy.symbols("x0 y0 q", real=True)
    mesh = AMRHierarchy(base_shape=(8, 8))
    mesh.add_patch(level=1, box=((8, 16), (8, 16)))
    nodes = CompositeNodes(mesh)
    edges = CompositeEdges(nodes)
    divergence = (edges.gradient @ nodes.reduction).T
    # the lower-left corner (1/8, 1/8) of the coarse cell where the paths start
    corner = int(np.argmin(np.abs(nodes.lattice - (2, 2)).sum(axis=1)))

    def change(x0: float, y0: float, q: float) -> float:
        start = np.array([[x0], [y0]])
        # into the fine patch, across coarse faces and the patch's edge
        end = start + np.array([[0.55], [0.6]])
        current = edges.deposit_paths(start, end, np.array([q]))
        return float((divergence @ current)[corner])

    xi, eta = 8 * x0 - 1, 8 * y0 - 1
    return {
        "continuity": Rule(
            -q * (1 - xi) * (1 - eta),
            implementation=change,
            ranges={"x0": (0.13, 0.2), "y0": (0.13, 0.2), "q": (0.5, 2.0)},
        )
    }


@method(
    equations={
        "gradient": r"(G\phi)_e = \phi_{\mathrm{end}(e)} - \phi_{\mathrm{start}(e)}",
        "hanging_edges": r"(T_1 e)_f = \frac{s_f}{s_c} \sum_j w_j(\boldsymbol{\xi}_f)\, e_{c,j} \quad \text{over the coarse cell's edges } j \text{ parallel to } f \text{ (one on a coarse edge, two in a coarse face)}",
        "commuting": r"G\,T = T_1\,G\,T",
        "nodal_operator": r"\big(T^\top G^\top M_1\, G\, T\big)_{FF} = A_\mathrm{NodalPoisson} \quad \text{(lumped: the five-point stencil; consistent: the Q1 stiffness)}",
    },
    assumptions=(
        "2D and 3D Cartesian; edges addressed on the finest lattice",
        "a coarse edge and the fine edges that tile it are different edges; the fine ones hang, as do (3D) fine edges inside a coarse face",
    ),
    references=(
        "H. Whitney, Geometric Integration Theory, Princeton University Press (1957)",
        "[Bossavit, IEE Proc. A 135, 493 (1988)](https://doi.org/10.1049/ip-a-1.1988.0077)",
    ),
    components=("patchwork.spaces.nodes.CompositeNodes",),
)
class CompositeEdges:
    """The edges of the leaf cells of a 1D, 2D or 3D hierarchy, and the operators on them.

    Edges are addressed on the finest lattice, as the nodes are: an edge is its
    direction (0 along x, 1 along y), the lattice point of its lower end and its
    span. A coarse edge and the fine edges that halve it are different edges;
    the fine ones hang. On a line the edges are the leaf cells, and none hangs.

    Attributes:
        nodes: The `CompositeNodes` of the hierarchy.
        num_edges: Number of edges.
        direction: The axis along every edge, ``(n1,)``.
        lattice: Finest-lattice coordinates of the lower end of every edge,
            ``(n1, ndim)``.
        span: Length of every edge on the finest lattice.
        length: Physical length of every edge.
        ends: Node numbers of the start and the end of every edge, ``(n1, 2)``.
        cell_edges: The edges of every leaf cell, ``(m, ndim 2**(ndim-1))``:
            for each axis the edges along it in the order of their start corners
            (in 2D the lower and upper x-edge, then the left and right y-edge).
        hanging: Whether an edge hangs (lies in a coarser edge).
        reduction: The constraint ``T1``: ``T1 @ e`` fills the hanging edges of
            ``e`` from the others (which it leaves unchanged).
        gradient: The discrete gradient ``G``, ``(n1, n)``: the potential's
            difference along every edge.
    """

    # Construction -------------------------------------------------------------#
    def _keys(self, direction, lower, span) -> np.ndarray:
        """One integer per edge: direction, lower end (as a node key) and span."""
        ndim = self.nodes.ndim
        node_key = self.nodes._key(*(lower[..., axis] for axis in range(ndim)))
        return (node_key * ndim + direction) * self._max_span + span

    def _find(self, direction, lower, span) -> np.ndarray:
        """Edge numbers of edges, -1 where there is none."""
        keys = self._keys(direction, lower, span)
        position = np.minimum(np.searchsorted(self._sorted, keys), self.num_edges - 1)
        return np.where(self._sorted[position] == keys, position, -1)

    def _find_hanging_edges(self) -> None:
        """Find the edges on coarser cells' edges and faces and their constraints.

        A fine edge on the boundary of a coarser leaf cell hangs: its value is
        the line integral of that cell's edge interpolant along it, from the
        cell's edges parallel to it, with the multilinear weights of its
        position across the cell times the ratio of the spans. On a coarse edge
        that is the share of that edge (2D and 3D); inside a coarse face (3D)
        it mixes the face's two parallel edges.
        """
        nodes = self.nodes
        ndim = nodes.ndim
        ratio = nodes.hierarchy.refinement_ratio
        bits = corner_bits(ndim)
        slaves, masters, weights, spans = [], [], [], []
        for coarse in np.unique(nodes.cell_span):
            coarse = int(coarse)
            cells = np.nonzero(nodes.cell_span == coarse)[0]
            lower_corner = nodes.cell_lower[cells]
            fine = coarse // ratio
            while fine >= 1:
                count = coarse // fine
                for axis in range(ndim):
                    others = [b for b in range(ndim) if b != axis]
                    columns = _axis_columns(ndim, axis)
                    starts = [start for start, _ in edge_corners(ndim, axis)]
                    grids = [
                        np.arange(count) * fine
                        if b == axis
                        else np.arange(count + 1) * fine
                        for b in range(ndim)
                    ]
                    offsets = np.stack(
                        np.meshgrid(*grids, indexing="ij"), axis=-1
                    ).reshape(-1, ndim)
                    on_boundary = np.zeros(len(offsets), dtype=bool)
                    for b in others:
                        on_boundary |= (offsets[:, b] == 0) | (offsets[:, b] == coarse)
                    offsets = offsets[on_boundary]
                    # the weight of each parallel edge of the cell at each offset
                    fraction = offsets / coarse
                    weight = np.ones((len(offsets), len(starts))) * (fine / coarse)
                    for j, start in enumerate(starts):
                        for b in others:
                            weight[:, j] *= np.where(
                                bits[start, b] == 1,
                                fraction[:, b],
                                1.0 - fraction[:, b],
                            )
                    for offset, row in zip(offsets, weight, strict=True):
                        edge = self._find(
                            np.full(cells.size, axis), lower_corner + offset, fine
                        )
                        found = edge >= 0
                        slaves.append(edge[found])
                        masters.append(self.cell_edges[cells[found]][:, columns])
                        weights.append(
                            np.broadcast_to(row, (int(found.sum()), row.size))
                        )
                        spans.append(np.full(int(found.sum()), coarse))
                fine //= ratio
        self.hanging = np.zeros(self.num_edges, dtype=bool)
        identity = sparse.csr_matrix(sparse.identity(self.num_edges))
        if not slaves or not sum(part.size for part in slaves):
            self.reduction = identity
            return
        slave = np.concatenate(slaves)
        master = np.concatenate(masters)  # (k, 2**(ndim-1))
        weight = np.concatenate(weights)
        span = np.concatenate(spans)
        # one constraint per edge, from the coarsest cell on whose boundary it lies
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
        if nodes.ndim not in (1, 2, 3):
            raise NotImplementedError(
                "the edge space is one-, two- or three-dimensional"
            )
        self.nodes = nodes
        ndim = nodes.ndim
        offsets = corner_bits(ndim)
        self._max_span = int(nodes.cell_span.max()) + 1
        directions, lowers, spans, ends = [], [], [], []
        for direction, start, end in _sides(ndim):
            directions.append(np.full(nodes.cell_span.size, direction))
            lowers.append(nodes.cell_lower + nodes.cell_span[:, None] * offsets[start])
            spans.append(nodes.cell_span)
            ends.append(nodes.cell_corners[:, [start, end]])
        direction = np.stack(directions, axis=1)  # (m, edges per cell)
        lower = np.stack(lowers, axis=1)  # (m, edges per cell, ndim)
        span = np.stack(spans, axis=1)
        keys = self._keys(direction, lower, span)
        self._sorted, first, inverse = np.unique(
            keys.reshape(-1), return_index=True, return_inverse=True
        )
        self.num_edges = int(self._sorted.size)
        self.cell_edges = inverse.reshape(keys.shape)
        self.direction = direction.reshape(-1)[first]
        self.lattice = lower.reshape(-1, ndim)[first]
        for axis in range(ndim):
            if nodes._periodic[axis]:  # the seam's edges at the lower end
                self.lattice[:, axis] %= nodes._cells[axis]
        self.span = span.reshape(-1)[first]
        self.ends = np.stack(ends, axis=1).reshape(-1, 2)[first]
        # physical lengths: the edge's run of finest-lattice cells along its direction
        self.length = np.empty(self.num_edges)
        for axis in range(ndim):
            mine = self.direction == axis
            self.length[mine] = nodes.arm_lengths(
                self.ends[mine, 0], axis, self.span[mine]
            )
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
        self._blocks: list[_PatchBlock] | None = None
        self._cell_block = np.zeros(0, dtype=np.int64)

    # Operators -------------------------------------------------------------------#
    @method(
        equations={
            "energy": r"e^\top M e = \int |\mathbf{E}|^2\, dA \quad \text{(no } \varepsilon_0\text{)}",
            "lumped": r"M_{ee} = \sum_{K \ni e} \frac{h_\perp}{2 h_\parallel}",
            "consistent": r"M_K = \frac{h_\perp}{h_\parallel} \begin{pmatrix} 1/3 & 1/6 \\ 1/6 & 1/3 \end{pmatrix} \quad \text{per cell and direction}",
        },
        assumptions=(
            "lumped: the vertex (trapezoidal) quadrature, diagonal; its entries are the edge conductances of the nodal stencil",
            "consistent: the exact integral of the products of the linear profiles",
            r"for a field on hanging edges use $T_1^\top M T_1$",
        ),
        symbolic=_mass_rules,
        components=("patchwork.spaces.mass.edge_mass_blocks",),
    )
    def mass(
        self, lumped: bool = True, weight: np.ndarray | float | None = None
    ) -> sparse.csr_matrix:
        """The edge mass matrix ``M1``: ``e^T M1 e`` is the integral of ``|E|^2``.

        Summed over the leaf cells, each with the edge interpolant of its own
        four edges. ``lumped`` takes the vertex (trapezoidal) quadrature, which
        makes it diagonal: ``h_perp / (2 h)`` per cell and edge, the conductance
        of `NodalPoisson`'s stencil. Otherwise the exact
        integral, ``h_perp / h`` times ``[[1/3, 1/6], [1/6, 1/3]]`` for the two
        edges of a direction. Both are symmetric and positive definite on the
        free edges. For a field on hanging edges, use ``T1^T M1 T1``.
        ``weight`` is a coefficient per leaf cell (a number, or an array in the
        order of the leaf cells), e.g. a permittivity or a conductivity.
        """
        size = self.nodes.cell_size  # (m, ndim)
        ndim = self.nodes.ndim
        shape = (self.num_edges, self.num_edges)
        mass = sparse.csr_matrix(shape)
        for axis in range(ndim):
            # the cell's edges along the axis, in the order of `edge_corners`
            edges = self.cell_edges[:, _axis_columns(ndim, axis)]
            mass = mass + scatter_blocks(
                _weighted(edge_mass_blocks(size, axis, lumped), weight),
                edges,
                edges,
                shape,
            )
        return sparse.csr_matrix(mass)

    def _weights(
        self, leaf: np.ndarray, fraction: list[np.ndarray]
    ) -> list[np.ndarray]:
        """The multilinear weights of a cell's edges along each axis at points in it.

        ``fraction`` holds the points' fractions across their leaf cells, one
        array per axis. Returns, per axis, ``(k, 2**(ndim-1))`` in the order of
        `edge_corners`.
        """
        ndim = self.nodes.ndim
        bits = corner_bits(ndim)
        result = []
        for axis in range(ndim):
            starts = [start for start, _ in edge_corners(ndim, axis)]
            weight = np.ones((leaf.size, len(starts)))
            for j, start in enumerate(starts):
                for other in range(ndim):
                    if other != axis:
                        weight[:, j] *= (
                            fraction[other]
                            if bits[start, other]
                            else 1.0 - fraction[other]
                        )
            result.append(weight)
        return result

    @method(
        equations={
            "whitney": r"E_x = \frac{(1-\eta)\,e_\mathrm{lower} + \eta\,e_\mathrm{upper}}{h_x}, \qquad E_y = \frac{(1-\xi)\,e_\mathrm{left} + \xi\,e_\mathrm{right}}{h_y} \quad \text{(2D; in 3D bilinear across the four parallel edges)}",
            "deposit": r"j = E_x^\top v_x + E_y^\top v_y \quad (\texttt{deposit}\text{, the transpose})",
        },
        assumptions=(
            r"$(\xi, \eta)$ the fractions from the lower-left corner of the leaf cell; the tangential component is constant along every edge",
            r"apply to edge values with their hanging edges filled ($T_1 e$); a point outside the domain gets zero",
        ),
        references=(
            "[Bossavit, IEE Proc. A 135, 493 (1988)](https://doi.org/10.1049/ip-a-1.1988.0077)",
        ),
        symbolic=_whitney_rules,
    )
    def evaluation(self, points: np.ndarray) -> tuple[sparse.csr_matrix, ...]:
        """The field of an edge field at ``points`` ``(ndim, k)``: one ``(k, n1)`` matrix per axis.

        Each point takes the edge interpolant of the leaf cell it is in. Apply
        them to edge values with their hanging edges filled (``T1 @ e``). A point
        outside the domain gets empty rows: the field zero.
        """
        nodes = self.nodes
        leaf, fraction, inside = nodes.point_cells(points)
        weights = self._weights(leaf, fraction)
        count = leaf.size
        matrices = []
        for axis, weight in enumerate(weights):
            weight = weight / nodes.cell_size[leaf, axis][:, None]
            weight[~inside] = 0.0
            columns = self.cell_edges[leaf][:, _axis_columns(nodes.ndim, axis)]
            rows = np.repeat(np.arange(count), columns.shape[1])
            matrix = sparse.csr_matrix(
                sparse.coo_matrix(
                    (weight.reshape(-1), (rows, columns.reshape(-1))),
                    shape=(count, self.num_edges),
                )
            )
            matrix.eliminate_zeros()
            matrices.append(matrix)
        return tuple(matrices)

    def deposit(self, points: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Vectors ``(ndim, k)`` at ``points`` ``(ndim, k)`` on the edges: ``(n1,)``.

        The transpose of `evaluation`: ``deposit(points, v) @ e`` is the
        sum of ``v . E(points)`` for the field ``E`` of ``e``. With the
        velocities times the charges of markers, it is their current at those
        points (a path integral of it is the current along their paths).
        Hanging edges receive their share like any other; ``T1^T`` hands it to
        the edges they hang from.
        """
        vectors = np.asarray(vectors, dtype=float)
        return sum(
            (
                matrix.T @ vectors[axis]
                for axis, matrix in enumerate(self.evaluation(points))
            ),
            np.zeros(self.num_edges),
        )

    def _physical_unwrapped(self, logical: np.ndarray) -> np.ndarray:
        """Physical points ``(ndim, k)`` of logical ones, beyond periodic faces too."""
        hierarchy = self.nodes.hierarchy
        return np.array(
            [
                hierarchy.to_physical(axis, values)
                for axis, values in enumerate(logical)
            ],
            dtype=float,
        ).reshape(logical.shape)

    def _lattice_unwrapped(self, points: np.ndarray) -> np.ndarray:
        """``points`` ``(ndim, k)`` in units of the finest lattice, beyond periodic faces too."""
        nodes = self.nodes
        hierarchy = nodes.hierarchy
        logical = np.array(
            [hierarchy.to_logical(axis, values) for axis, values in enumerate(points)],
            dtype=float,
        ).reshape(points.shape)
        return (logical - nodes._origin[:, None]) / nodes._spacing[:, None]

    def _logical_paths(
        self, start: np.ndarray, end: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The paths from ``start`` to ``end`` ``(ndim, k)`` as straight parts on the finest lattice.

        On a graded axis the map from logical to physical coordinates is linear only
        within a base cell, so a straight physical path is cut where it crosses a
        base plane of a graded axis; each part is then straight in logical
        coordinates. Returns the path of every part and its ends in finest-lattice
        units from the origin, not wrapped. Without a graded axis, one part per path.
        """
        nodes = self.nodes
        hierarchy = nodes.hierarchy
        count = start.shape[1]
        if not hierarchy.graded:
            return (
                np.arange(count),
                self._lattice_unwrapped(start),
                self._lattice_unwrapped(end),
            )
        paths, times = [np.arange(count)] * 2, [np.zeros(count), np.ones(count)]
        step = end - start
        for axis in range(nodes.ndim):
            if hierarchy.axis_nodes[axis] is None:
                continue
            lower = hierarchy.bounds[axis][0]
            base = hierarchy.spacing(0)[axis]
            a = (hierarchy.to_logical(axis, start[axis]) - lower) / base
            b = (hierarchy.to_logical(axis, end[axis]) - lower) / base
            first = np.floor(np.minimum(a, b)).astype(np.int64) + 1
            last = np.ceil(np.maximum(a, b)).astype(np.int64) - 1
            crossings = np.maximum(last - first + 1, 0)
            path = np.repeat(np.arange(count), crossings)
            offset = np.repeat(np.cumsum(crossings) - crossings, crossings)
            plane = first[path] + np.arange(path.size) - offset
            at = hierarchy.to_physical(axis, lower + plane * base)
            paths.append(path)
            times.append((at - start[axis, path]) / step[axis, path])
        path = np.concatenate(paths)
        time = np.clip(np.concatenate(times), 0.0, 1.0)
        order = np.lexsort((time, path))
        path, time = path[order], time[order]
        same = path[1:] == path[:-1]
        path, begin, finish = path[1:][same], time[:-1][same], time[1:][same]
        keep = finish > begin
        path, begin, finish = path[keep], begin[keep], finish[keep]
        a = start[:, path] + begin * step[:, path]
        b = start[:, path] + finish * step[:, path]
        # the ends of a path exactly as given, so that consecutive parts meet
        is_first = np.ones(path.size, dtype=bool)
        is_first[1:] = path[1:] != path[:-1]
        is_last = np.ones(path.size, dtype=bool)
        is_last[:-1] = path[1:] != path[:-1]
        a[:, is_first] = start[:, path[is_first]]
        b[:, is_last] = end[:, path[is_last]]
        return path, self._lattice_unwrapped(a), self._lattice_unwrapped(b)

    def _uniform_pieces(
        self, start: np.ndarray, end: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`_pieces` of paths straight in logical coordinates, given as logical points."""
        nodes = self.nodes
        origin = nodes._origin[:, None]
        spacing = nodes._spacing[:, None]
        lower = (start - origin) / spacing
        upper = (end - origin) / spacing
        count = start.shape[1]
        paths, times = [np.arange(count)] * 2, [np.zeros(count), np.ones(count)]
        for axis in range(nodes.ndim):
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

    def _pieces(
        self, start: np.ndarray, end: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The straight paths from ``start`` to ``end`` ``(ndim, k)``, cut at the finest lattice.

        Returns the path of every piece and its two ends ``(ndim, pieces)``. A
        piece lies in one cell of the finest lattice, and so in one leaf cell.
        """
        nodes = self.nodes
        if nodes.hierarchy.graded:
            # straight in logical coordinates between the base planes: cut there,
            # then at the finest lattice, and map the pieces back
            outer, a, b = self._logical_paths(start, end)
            spacing = nodes._spacing[:, None]
            origin = nodes._origin[:, None]
            inner, a, b = self._uniform_pieces(
                origin + a * spacing, origin + b * spacing
            )
            return (
                outer[inner],
                self._physical_unwrapped(a),
                self._physical_unwrapped(b),
            )
        return self._uniform_pieces(start, end)

    def _deposit_pieces(
        self, start: np.ndarray, end: np.ndarray, charges: np.ndarray
    ) -> np.ndarray:
        """`deposit_paths` in NumPy, in any dimension: one pass over all pieces."""
        nodes = self.nodes
        path, a, b = self._pieces(start, end)
        leaf, fraction, inside = nodes.point_cells(0.5 * (a + b))
        size = nodes.cell_size[leaf]
        charge = np.where(inside, charges[path], 0.0)
        step = b - a
        # the fractions at the piece's ends, in the leaf cell of its midpoint
        half = [0.5 * step[axis] / size[:, axis] for axis in range(nodes.ndim)]
        at_start = [fraction[axis] - half[axis] for axis in range(nodes.ndim)]
        at_end = [fraction[axis] + half[axis] for axis in range(nodes.ndim)]
        # along a piece an edge function is a product of linear functions, of
        # degree ndim - 1: Simpson's rule integrates it exactly
        weights = [
            (w0 + 4.0 * w1 + w2) / 6.0
            for w0, w1, w2 in zip(
                self._weights(leaf, at_start),
                self._weights(leaf, fraction),
                self._weights(leaf, at_end),
                strict=True,
            )
        ]
        current = np.zeros(self.num_edges)
        for axis, weight in enumerate(weights):
            along = charge * step[axis] / size[:, axis]
            columns = self.cell_edges[leaf][:, _axis_columns(nodes.ndim, axis)]
            current += np.bincount(
                columns.reshape(-1),
                weights=(weight * along[:, None]).reshape(-1),
                minlength=self.num_edges,
            )
        return current

    # The blocks of the patches ------------------------------------------------------#
    def _patch_blocks(self) -> tuple[list[_PatchBlock], np.ndarray]:
        """Every patch as a block of the kernels, and the block of every leaf cell.

        Built on first use. A leaf cell's edges are the edges of its patch's block at
        the cell's corners; the edges of the cells that finer patches cover are not in
        the space (-1 in the block's ``edge_map``).
        """
        if self._blocks is not None:
            return self._blocks, self._cell_block
        from mpifield import BlockGrid

        nodes = self.nodes
        ndim = nodes.ndim
        hierarchy = nodes.hierarchy
        ratio = hierarchy.refinement_ratio
        finest = hierarchy.max_level
        bits = corner_bits(ndim)
        blocks = []
        cell_block = np.full(nodes.cell_span.size, -1, dtype=np.int64)
        for patch in hierarchy.patches:
            span = ratio ** (finest - patch.level)
            lower = np.array([lo for lo, _ in patch.box], dtype=np.int64) * span
            upper = np.array([hi for _, hi in patch.box], dtype=np.int64) * span
            children = [
                (
                    np.array([lo for lo, _ in child.box]) * (span // ratio),
                    np.array([hi for _, hi in child.box]) * (span // ratio),
                )
                for child in hierarchy.patches_at_level(patch.level + 1)
            ]
            children = np.array(
                [
                    (low, high)
                    for low, high in children
                    if np.all(low < upper) and np.all(high > lower)
                ],
                dtype=float,
            ).reshape(-1, 2, ndim)
            node_shape = tuple(int(n) + 1 for n in (upper - lower) // span)
            edge_map = np.full((*node_shape, ndim), -1, dtype=np.int64)
            cells = np.flatnonzero(
                (nodes.cell_span == span)
                & np.all(nodes.cell_lower >= lower, axis=1)
                & np.all(nodes.cell_lower < upper, axis=1)
            )
            cell_block[cells] = len(blocks)
            local = (nodes.cell_lower[cells] - lower) // span
            for axis in range(ndim):
                starts = [start for start, _ in edge_corners(ndim, axis)]
                columns = _axis_columns(ndim, axis)
                for column, start in zip(columns, starts, strict=True):
                    index = local + bits[start]
                    edge_map[(*index.T, axis)] = self.cell_edges[cells, column]
            grid = BlockGrid.from_geometry(
                lower.astype(float), (float(span),) * ndim, node_shape
            )
            blocks.append(
                _PatchBlock(
                    grid, lower.astype(float), upper.astype(float), children, edge_map
                )
            )
        self._blocks, self._cell_block = blocks, cell_block
        return blocks, cell_block

    def _leaf_paths(
        self, block: _PatchBlock, a: np.ndarray, b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The parts of the paths from ``a`` to ``b`` in the leaf cells of a patch.

        ``a`` and ``b`` are on the finest lattice ``(ndim, k)``, not wrapped. Returns
        the path of every part and its two ends, shifted by whole periods into the
        patch's frame: the paths clipped to the patch, without the boxes of its
        children, for every shift that brings them into it.
        """
        nodes = self.nodes
        step = b - a
        smallest, largest = np.minimum(a, b), np.maximum(a, b)
        lower, upper = block.lower[:, None], block.upper[:, None]
        paths, starts, ends = [], [], []
        for shift in _shifts(
            a, b, block.lower, block.upper, nodes._cells, nodes._periodic
        ):
            low, high = smallest - shift[:, None], largest - shift[:, None]
            near = np.all((high >= lower) & (low <= upper), axis=0)
            # a path inside the patch that comes nowhere near a child is one part
            whole = near & np.all((low >= lower) & (high <= upper), axis=0)
            for child_low, child_high in block.children:
                whole &= ~np.all(
                    (high >= child_low[:, None]) & (low <= child_high[:, None]), axis=0
                )
            inside = np.flatnonzero(whole)
            paths.append(inside)
            starts.append(a[:, inside] - shift[:, None])
            ends.append(b[:, inside] - shift[:, None])
            candidates = np.flatnonzero(near & ~whole)
            if candidates.size == 0:
                continue
            shifted = a[:, candidates] - shift[:, None]
            enter, leave = _clip(shifted, step[:, candidates], block.lower, block.upper)
            local = np.flatnonzero(leave > enter)
            if local.size == 0:
                continue
            hit = candidates[local]
            a_hit, step_hit = shifted[:, local], step[:, hit]
            enter, leave = enter[local], leave[local]
            # the times the paths enter and leave the children, within the patch
            cuts = [enter[:, None], leave[:, None]]
            covered = []
            for low, high in block.children:
                into, out = _clip(a_hit, step_hit, low, high)
                into = np.clip(into, enter, leave)
                out = np.clip(out, enter, leave)
                cuts += [into[:, None], out[:, None]]
                covered.append((into, out))
            times = np.sort(np.hstack(cuts), axis=1)
            first, last = times[:, :-1], times[:, 1:]
            keep = last > first
            middle = 0.5 * (first + last)
            for into, out in covered:
                keep &= ~((middle > into[:, None]) & (middle < out[:, None]))
            row, column = np.nonzero(keep)
            begin, finish = first[row, column], last[row, column]
            paths.append(hit[row])
            starts.append(a_hit[:, row] + begin * step_hit[:, row])
            ends.append(a_hit[:, row] + finish * step_hit[:, row])
        if not paths:
            empty = np.zeros((a.shape[0], 0))
            return np.zeros(0, dtype=np.int64), empty, empty
        return np.concatenate(paths), np.hstack(starts), np.hstack(ends)

    def _lattice(self, points: np.ndarray) -> np.ndarray:
        """``points`` ``(ndim, k)`` in units of the finest lattice, from the origin (logical)."""
        nodes = self.nodes
        return (nodes._logical(points) - nodes._origin[:, None]) / nodes._spacing[
            :, None
        ]

    def _lattice_sizes(self, lattice: np.ndarray) -> np.ndarray:
        """Physical sizes ``(ndim, k)`` of the finest-lattice cells at lattice points ``(ndim, k)``."""
        nodes = self.nodes
        hierarchy = nodes.hierarchy
        sizes = np.empty_like(lattice)
        for axis in range(nodes.ndim):
            if hierarchy.axis_nodes[axis] is None:
                sizes[axis] = nodes._spacing[axis]
                continue
            positions = hierarchy.node_positions(nodes._finest, axis)
            cell = np.clip(
                np.floor(lattice[axis]).astype(np.int64), 0, int(nodes._cells[axis]) - 1
            )
            sizes[axis] = positions[cell + 1] - positions[cell]
        return sizes

    @method(
        equations={
            "pieces": r"j_\mathrm{lower} \mathrel{+}= q\,\frac{\Delta x}{h_x}(1 - \eta_\mathrm{mid}), \qquad j_\mathrm{upper} \mathrel{+}= q\,\frac{\Delta x}{h_x}\,\eta_\mathrm{mid}, \qquad \text{likewise for the } y\text{-edges}",
            "continuity": r"(G T)^\top j = T^\top\big(Q(\mathbf{x}_1) - Q(\mathbf{x}_0)\big)",
            "product": r"a_1 b_1 - a_0 b_0 = \Delta a\,\bar b + \bar a\,\Delta b \quad \text{for linear } a(t), b(t)",
            "work": r"j^\top e = \sum_k q_k \int_{\mathbf{x}_0}^{\mathbf{x}_1} \mathbf{E}\cdot d\mathbf{l}",
        },
        assumptions=(
            "straight paths, clipped to the leaf cells of every patch and integrated on the patch's block by mpifield's compiled path deposit, which cuts them at its cell faces and integrates the edge functions (linear along a piece in 2D, quadratic in 3D) exactly; `_deposit_pieces` (Simpson's rule on the finest-lattice pieces) is the NumPy reference",
            r"$Q$ the bilinear node deposit (`CompositeNodes.deposit`); conserved across patch edges and periodic seams to round-off",
            "pieces outside the domain deposit nothing",
        ),
        references=(
            "[Villasenor and Buneman, Comput. Phys. Commun. 69, 306 (1992)](https://doi.org/10.1016/0010-4655(92)90169-Y)",
        ),
        components=(
            "patchwork.spaces.nodes.CompositeNodes.deposit",
            "mpifield.edges.EdgeWeighting",
        ),
        symbolic=_continuity_rules,
    )
    def deposit_paths(
        self, start: np.ndarray, end: np.ndarray, charges: np.ndarray
    ) -> np.ndarray:
        """The current of charges moving along straight paths, integrated over the paths: ``(n1,)``.

        Every charge goes from ``start`` to ``end`` (``(ndim, k)``, not wrapped into
        the domain on a periodic axis); its path is split at every cell face it
        crosses, patch edges included, and each piece is integrated exactly
        against the edge functions of its leaf cell: ``q * integral of w . dl``.
        Then the charge is conserved exactly: ``(G T)^T j`` is the change of the
        charge on the nodes, ``T^T`` of `CompositeNodes.deposit` at the ends minus at the
        starts, and ``j @ e`` is the work of the field ``e`` on the charges. The
        current density times the time step, summed over a step's paths, is
        ``j``. Pieces outside the domain deposit nothing. The paths are clipped to
        the leaf cells of every patch (its box without its children's, shifted by
        whole periods on a periodic axis), and mpifield's compiled path deposit
        (`mpifield.WHITNEY`) integrates each part on the patch's block, whose edges
        are those of its leaf cells; `_deposit_pieces` is the NumPy reference.
        """
        from mpifield import WHITNEY

        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        charges = np.asarray(charges, dtype=float).reshape(-1)
        if start.shape != end.shape or start.shape != (self.nodes.ndim, charges.size):
            raise ValueError(
                f"paths from {start.shape} to {end.shape} for {charges.size} charges"
            )
        current = np.zeros(self.num_edges)
        if charges.size == 0:
            return current
        path_of, a, b = self._logical_paths(start, end)
        charges = charges[path_of]
        for axis, wraps in enumerate(self.nodes._periodic):
            if wraps:  # starts into the first period: fewer images to clip
                period = float(self.nodes._cells[axis])
                offset = np.floor(a[axis] / period) * period
                a[axis] -= offset
                b[axis] -= offset
        blocks, _ = self._patch_blocks()
        for block in blocks:
            path, part_start, part_end = self._leaf_paths(block, a, b)
            if path.size == 0:
                continue
            block_current = np.zeros(block.edge_map.shape)
            WHITNEY.deposit_paths_block(
                part_start, part_end, charges[path], block_current, block.grid
            )
            known = block.edge_map >= 0
            current += np.bincount(
                block.edge_map[known],
                weights=block_current[known],
                minlength=self.num_edges,
            )
        return current

    def gather(self, points: np.ndarray, values: np.ndarray) -> np.ndarray:
        """The field of the edge values ``values`` at ``points`` ``(ndim, k)``: ``(ndim, k)``.

        `evaluation` applied to ``values``, with mpifield's compiled edge gather
        (`mpifield.WHITNEY`) on the block of each point's patch. Apply it to values
        with their hanging edges filled (``T1 @ e``); a point outside the domain gets
        zero.
        """
        from mpifield import WHITNEY

        points = np.asarray(points, dtype=float)
        values = np.asarray(values, dtype=float)
        nodes = self.nodes
        blocks, cell_block = self._patch_blocks()
        leaf, _, inside = nodes.point_cells(points)
        lattice = self._lattice(points)
        for axis in range(nodes.ndim):
            if nodes._periodic[axis]:
                lattice[axis] %= nodes._cells[axis]
        field = np.zeros_like(points)
        # the gathered field is per lattice unit: divide by the lattice cell's size
        sizes = self._lattice_sizes(lattice)
        owner = np.where(inside, cell_block[leaf], -1)
        for number in np.unique(owner[owner >= 0]):
            block = blocks[number]
            mine = np.flatnonzero(owner == number)
            known = block.edge_map >= 0
            block_values = np.zeros(block.edge_map.shape)
            block_values[known] = values[block.edge_map[known]]
            result = np.zeros((nodes.ndim, mine.size))
            WHITNEY.gather_block(
                np.ascontiguousarray(lattice[:, mine]), block_values, result, block.grid
            )
            field[:, mine] = result / sizes[:, mine]
        return field
