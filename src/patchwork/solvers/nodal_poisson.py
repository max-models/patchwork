r"""Node-centered composite Poisson solve on a static patch hierarchy.

The unknowns are those of `patchwork.spaces.nodes.CompositeNodes`.

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

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sparse
import scope_profiler as sp
from methodcard import ConvergenceProblem, Rule, method

from patchwork._profiling import PROFILE_TAGS
from patchwork.solvers.backends import Backend, LinearSystem, backend_of
from patchwork.spaces.boundaries import (
    INNER,
    Dirichlet,
    EmbeddedDirichlet,
    Neumann,
    _evaluate,
    _surface_crossing,
    faces_of,
)
from patchwork.spaces.mass import edge_corners, edge_mass_blocks
from patchwork.spaces.nodes import CompositeNodes

if TYPE_CHECKING:
    from patchwork.mesh.hierarchy import AMRHierarchy

#: The mass matrices `NodalPoisson` assembles with: the finite-difference
#: (lumped) or the finite-element (consistent) one.
MASSES = ("lumped", "consistent")


def _diagonal(solver: NodalPoisson, lattice: tuple[int, ...]) -> float:
    """The diagonal entry of the operator at the node of a finest-lattice point."""
    node = int(np.argmin(np.abs(solver.nodes.lattice - lattice).sum(axis=1)))
    row = int(np.flatnonzero(solver.free == node)[0])
    return float(solver.matrix[row, row])


def _five_point(hx: float, hy: float) -> float:
    """The diagonal of the operator at the centre of four cells ``hx`` by ``hy``."""
    from patchwork.mesh.hierarchy import AMRHierarchy

    mesh = AMRHierarchy(base_shape=(4, 4), bounds=((0.0, 4.0 * hx), (0.0, 4.0 * hy)))
    return _diagonal(NodalPoisson(mesh), (2, 2))


def _cut_arm(theta: float) -> float:
    """The diagonal of a regular node whose right arm a wall cuts at ``theta h``."""
    from patchwork.mesh.hierarchy import AMRHierarchy

    wall = 0.5 + theta / 8.0
    body = EmbeddedDirichlet(
        lambda points: points[0] > wall,
        1.0,
        fraction=lambda start, end: (wall - start[0]) / (end[0] - start[0]),
    )
    solver = NodalPoisson(AMRHierarchy(base_shape=(8, 8)), embedded=(body,))
    return _diagonal(solver, (4, 4))


def _nodal_rules() -> dict:
    """The SymPy rules of `NodalPoisson` (for the tests)."""
    import sympy

    hx, hy, theta = sympy.symbols("hx hy theta", positive=True)
    return {
        "stencil": Rule(
            2 * hy / hx + 2 * hx / hy,
            implementation=_five_point,
            ranges={"hx": (0.05, 1.0), "hy": (0.05, 1.0)},
        ),
        "shortley_weller": Rule(
            2 + 2 / theta, implementation=_cut_arm, ranges={"theta": (0.05, 0.95)}
        ),
    }


def _nodal_problem() -> ConvergenceProblem:
    """A manufactured solution on two nested patches, in the volume-weighted L2 norm."""
    from patchwork.mesh.hierarchy import AMRHierarchy

    def error(spacing: float) -> float:
        n = round(1.0 / spacing)
        mesh = AMRHierarchy(base_shape=(n, n))
        mesh.add_patch(level=1, box=((n // 2, 3 * n // 2),) * 2)
        mesh.add_patch(level=2, box=((3 * n // 2, 5 * n // 2),) * 2)
        solver = NodalPoisson(mesh)
        x, y = solver.nodes.coordinates.T
        exact = np.sin(np.pi * x) * np.sin(np.pi * y)
        difference = solver.solve(2.0 * np.pi**2 * exact) - exact
        return float(np.sqrt(np.sum(difference**2 * solver.nodes.volumes)))

    return ConvergenceProblem(error, steps=[1 / 16, 1 / 32, 1 / 64])


@method(
    equations={
        "problem": r"-\nabla^2 \phi = f \ \text{in } \Omega, \qquad \phi = \phi_D \ \text{or} \ \partial_n \phi = g \ \text{on faces}, \qquad \phi = U_k \ \text{in bodies } B_k",
        "stencil": r"A_{pp} \mathrel{+}= g,\ A_{qq} \mathrel{+}= g,\ A_{pq} = A_{qp} \mathrel{-}= g \quad \text{per leaf-cell edge } pq \text{ along } a, \qquad g = \frac{h_\perp}{2 h_a} \ \text{(2D)}, \ \frac{1}{h} \ \text{(1D)}",
        "uniform": r"A_{pp} = \frac{2h_y}{h_x} + \frac{2h_x}{h_y}, \qquad A_{p,x\pm} = -\frac{h_y}{h_x}, \qquad A_{p,y\pm} = -\frac{h_x}{h_y}",
        "reduced": r"(T^\top A T)_{FF}\,\phi_F = T^\top V f - (T^\top A T)_{FD}\,\phi_D + T^\top b_N + W U, \qquad V = \mathrm{diag}(V_p)",
        "charge": r"T^\top V f \to T^\top q \quad (\texttt{solve(charge=q)})",
        "consistent": r"A = T^\top G^\top M_1 G\, T, \qquad V \to M_0 \quad \text{(consistent mass: the Q1 elements; lumped } M_1 \text{ and } V \text{: the stencil above)}",
        "neumann": r"b_{N,p} \mathrel{+}= \tfrac12 |e|\, g \quad \text{for each end } p \text{ of a boundary edge } e",
        "cut_edge": r"A_{pp} \mathrel{+}= \frac{g}{\theta}, \qquad b_p \mathrel{+}= \frac{g}{\theta}\, U_k, \qquad \theta = \max(\theta_\mathrm{found}, \theta_\mathrm{min})",
        "shortley_weller": r"A_{pp} = h_x h_y \sum_\mathrm{axes} \Big(\frac{2}{a_-(a_- + a_+)} + \frac{2}{a_+(a_- + a_+)}\Big), \qquad A_{p,\pm} = -\,h_x h_y\,\frac{2}{a_\pm(a_- + a_+)}",
    },
    order=2,
    assumptions=(
        r"unlisted faces are grounded; no permittivity: for a charge pass $\rho / \varepsilon_0$",
        "lumped and consistent mass share one assembly from the edge mass blocks (`patchwork.spaces.mass`); the consistent one takes no bodies and keeps the lumped Neumann load",
        "Cartesian control volumes and conductances, also on an $(r, z)$ hierarchy",
        r"the Shortley-Weller rows at regular cut nodes (four equal cells, not hanging), the diagonal-only cut elsewhere; they make the matrix unsymmetric",
        r"$\theta$ by 48 bisections of `inside`, or the body's `fraction`; $\theta_\mathrm{min}$ = `min_fraction` ($10^{-3}$)",
        r"the order is in the volume-weighted $L_2$ norm; in the maximum norm about 1.86 on nested patches (an $O(1)$ truncation error on the interface nodes)",
    ),
    references=(
        "[Shortley and Weller, J. Appl. Phys. 9, 334 (1938)](https://doi.org/10.1063/1.1710426)",
        "[Gibou, Fedkiw, Cheng and Kang, J. Comput. Phys. 176, 205 (2002)](https://doi.org/10.1006/jcph.2001.6977): the diagonal-only cut edge",
    ),
    components=(
        CompositeNodes,
        "patchwork.spaces.mass.edge_mass_blocks",
        "patchwork.spaces.nodes.CompositeNodes.mass",
    ),
    symbolic=_nodal_rules,
    problem=_nodal_problem,
)
class NodalPoisson:
    r"""Solve $-\nabla^2\phi = f$ on the nodes of a patch hierarchy.

    The operator is assembled and its backend set up once (factorized, by
    default); `solve` then costs two triangular solves. ``faces`` maps
    ``"x_lower"``, ``"x_upper"``, ``"y_lower"``, ``"y_upper"`` (only the
    first two in 1D) to a `Dirichlet` or `Neumann` condition; a
    face that is not listed is a grounded Dirichlet face, and the faces of a
    periodic axis take none. On a domain that does not fill its rectangle
    (``AMRHierarchy(base_boxes=...)``) these are the parts of its boundary on
    the rectangle's faces, and ``"inner"`` is the rest: the edges inside the
    rectangle. ``embedded`` are bodies at fixed potentials
    (`EmbeddedDirichlet`); where they overlap the last one listed wins.

    ``backend`` solves the assembled operator (`patchwork.solvers.backends`): a
    `Backend` or its name. ``"lu"`` (the default)
    assembles the whole operator on every rank and factorizes it there
    (SuperLU; LAPACK's tridiagonal LU on a line). ``"petsc"`` (needs
    petsc4py; ``PetscBackend(rtol=..., options=...)`` for its tolerance and
    PETSc options) assembles on every rank only the rows of the unknowns it
    owns (`CompositeNodes.node_ranks`, from where the patches are), from
    the leaf cells around them, and solves them together with a PETSc Krylov
    method and an algebraic multigrid preconditioner (see
    `PetscSystem`). ``"fft"`` solves a uniform mesh without bodies
    with fast transforms, also when nothing fixes the potential (every axis
    periodic or Neumann: the solution of zero mean); ``"auto"`` takes the LU
    on a line, the FFT where it applies and the LU elsewhere. All give the
    potential on all nodes on every rank.

    ``mass`` chooses the scheme: ``"lumped"`` (the default) is the finite
    differences above, the five-point stencil with the control volumes;
    ``"consistent"`` is the Q1 finite elements, the operator ``T^T G^T M1 G T``
    of the consistent edge mass (`CompositeEdges.mass`) and the load
    ``T^T M0`` of the consistent node mass (`CompositeNodes.mass`). Both are
    assembled from the same per-cell blocks (`patchwork.spaces.mass`). The
    consistent scheme is exact for bilinear potentials, also across
    coarse–fine interfaces, takes no embedded bodies (their cut edges are
    finite differences), and is solved by LU or PETSc (``"auto"`` takes the
    LU): the FFT diagonalizes only the five-point stencil.

    Attributes:
        nodes: The `CompositeNodes` of the hierarchy.
        free: Node numbers of the unknowns: neither hanging nor fixed.
        fixed: Whether a node has a prescribed value (a face or a body).
        rows: Node numbers of the unknowns whose rows this rank holds: all of
            `free`, or those it owns with a distributed backend (in the
            order of the distributed system: by rank, then by node number).
        matrix: This rank's rows of the operator, over the columns of the
            unknowns (in `free` order, or in the distributed order).
        backend: The `Backend`.
        mass: ``"lumped"`` or ``"consistent"``.
        system: The `LinearSystem` it solves.
        solver: What the backend set up (an LU factorization, a
            `PetscSystem`, an `FFTSolver`),
            ``None`` without unknowns.
    """

    # Assembly ------------------------------------------------------------------#
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
                fraction = _surface_crossing(
                    body.inside, start, step, self.nodes.hierarchy.wrapped
                )
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

    @staticmethod
    def _consistent_couplings(
        corners: np.ndarray, size: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The off-diagonal entries of the consistent edge mass in ``G^T M1 G``.

        Two parallel edges ``(a0, a1)`` and ``(b0, b1)`` of a cell with mass
        entry ``m`` add ``m (u_a1 - u_a0)(u_b1 - u_b0)`` to the energy (from
        each of the two orderings): ``+m`` at ``(a1, b1)`` and ``(a0, b0)``,
        ``-m`` at ``(a1, b0)`` and ``(a0, b1)``.
        """
        ndim = size.shape[1]
        rows, columns, entries = [], [], []
        for axis in range(ndim):
            blocks = edge_mass_blocks(size, axis, lumped=False)
            pairs = edge_corners(ndim, axis)
            for i, (a0, a1) in enumerate(pairs):
                for j, (b0, b1) in enumerate(pairs):
                    if i == j:
                        continue
                    coupling = blocks[:, i, j]
                    for row, column, sign in (
                        (a1, b1, 1.0),
                        (a0, b0, 1.0),
                        (a1, b0, -1.0),
                        (a0, b1, -1.0),
                    ):
                        rows.append(corners[:, row])
                        columns.append(corners[:, column])
                        entries.append(sign * coupling)
        if not rows:  # 1D: one edge per cell
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty, np.zeros(0)
        return np.concatenate(rows), np.concatenate(columns), np.concatenate(entries)

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
        # of the stencil's conductance: the diagonal of the edge mass block
        # (half in 2D, the whole in 1D, lumped). A consistent mass also couples
        # the parallel edges of a cell (`_consistent_couplings`).
        ndim = nodes.ndim
        corners = nodes.cell_corners if cells is None else nodes.cell_corners[cells]
        size = nodes.cell_size if cells is None else nodes.cell_size[cells]
        lumped = self.mass == "lumped"
        firsts, seconds, conductances, lengths = [], [], [], []
        for axis in range(ndim):
            blocks = edge_mass_blocks(size, axis, lumped)
            for k, (start, end) in enumerate(edge_corners(ndim, axis)):
                firsts.append(corners[:, start])
                seconds.append(corners[:, end])
                conductances.append(blocks[:, k, k])
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
        if not lumped:
            extra = self._consistent_couplings(corners, size)
            rows, columns, entries = (
                np.concatenate((part, more))
                for part, more in zip((rows, columns, entries), extra, strict=True)
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
        lumped = self.mass == "lumped"
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
        self._load = sparse.csr_matrix(pick @ nodes.mass(lumped=lumped))
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
            mass=self.mass,
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
        mass: str = "lumped",
    ) -> None:
        """Assemble the operator and set up its backend.

        ``min_fraction`` is the smallest fraction of an edge between a node and
        an embedded surface: a surface closer than that (through the node, up to
        round-off) is moved there, which bounds the size of that node's row.
        ``mass`` is ``"lumped"`` (the finite-difference scheme) or
        ``"consistent"`` (the Q1 finite elements; no embedded bodies).
        """
        if mass not in MASSES:
            raise ValueError(f"mass must be one of {MASSES}, got {mass!r}")
        if mass == "consistent" and embedded:
            raise ValueError(
                "the embedded bodies' cut edges are finite differences: "
                "they take mass='lumped'"
            )
        self.mass = mass
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
        """Map from a source on all nodes to the right-hand side of this rank's `rows`."""
        return self._load

    @property
    def integral_load(self) -> sparse.csr_matrix:
        """Map from a source integrated over each node's control volume to the right-hand side.

        The transposed interpolation of the hanging nodes, restricted to the
        unknowns: what a deposited charge (`CompositeNodes.deposit`)
        needs. ``load == integral_load @ diag(volumes)``.
        """
        return self._integral_load

    @property
    def constant(self) -> np.ndarray:
        """The share of the right-hand side that the prescribed values give."""
        return self._constant

    @sp.profile("patchwork.nodal_poisson_solve_reduced", tags=PROFILE_TAGS)
    def solve_reduced(self, right_hand_side: np.ndarray) -> np.ndarray:
        """Solve for a right-hand side of this rank's `rows`; the potential on all nodes.

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

        ``source`` is an array ``(n,)`` over `nodes`, or the name of a
        node-centered field of the hierarchy. Or ``charge`` is the source
        integrated over each node's control volume, ``(n,)``, as
        `CompositeNodes.deposit` gives it for point particles (divide a
        charge by the permittivity for $-\nabla^2\phi = \rho /
        \varepsilon_0$); the charge on hanging nodes goes to the nodes they hang
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
    r"""Solve $-\nabla^2\phi = \mathrm{rhs}$ once on the nodes of ``hierarchy``.

    ``rhs`` and ``potential`` name node-centered fields; see
    `NodalPoisson` for the conditions and the backends. Returns the
    solver, which can solve again without assembling.
    """
    solver = NodalPoisson(hierarchy, faces=faces, embedded=embedded, backend=backend)
    solver.solve(rhs, potential=potential)
    return solver
