r"""Geometric multigrid over the patches of a hierarchy (`MultigridBackend`).

The composite node spaces of a hierarchy are nested: dropping its finest level
leaves a hierarchy whose leaf cells are unions of the leaf cells of the full
one, and whose continuous, piecewise (bi-, tri-)linear functions (hanging nodes
interpolated) are functions of the full space too. Dropping levels one by one
down to the base level, and then coarsening the base level by two along every
axis while its boxes allow it, gives a sequence of spaces

$$
V_0 \subset V_1 \subset \dots \subset V_L,
$$

each the `CompositeNodes` of a hierarchy (`coarser_hierarchies`). The
interpolation from one to the next evaluates the coarse function at the fine
unknowns (`CompositeNodes.evaluation` after the coarse hanging nodes are
filled), the coarse operators are the Galerkin products $P^\top A P$, so a
variable coefficient, Neumann faces, embedded bodies and the r-z metric need
no stencils of their own: whatever the finest operator is, the coarse ones
are its restrictions. Each level is smoothed by Chebyshev iteration with Jacobi
preconditioning, the coarsest is solved by LU; the V-cycle preconditions
conjugate gradients (GMRES for a non-symmetric operator, with embedded bodies).

A refined patch is a level of the multigrid: the refinement is what the
coarse levels remove first, so the number of iterations does not grow with
the number of refinement levels or with the cell count (`test_multigrid.py`).
The smoother works on all unknowns of a level, also those outside the newest
patch: a level costs the composite mesh's unknowns, not only the patch's.

The whole system is on every rank, as with ``"lu"``; the distributed
version (rows by patch rank, smoothing with halo exchanges) is the
implicit PIC's next step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
import scope_profiler as sp
from methodcard import method

from patchwork._profiling import PROFILE_TAGS
from patchwork.solvers.backends import BACKENDS, Backend, LinearSystem
from patchwork.spaces.boundaries import Dirichlet
from patchwork.spaces.nodes import CompositeNodes

if TYPE_CHECKING:
    from patchwork.mesh.hierarchy import AMRHierarchy

#: Stop coarsening the base level once a level has at most this many unknowns.
COARSEST_UNKNOWNS = 1000

#: Power iterations for the largest eigenvalue of the Jacobi-preconditioned operator.
POWER_ITERATIONS = 20


def _rebuilt(
    hierarchy: AMRHierarchy,
    base_shape: tuple[int, ...],
    base: list[tuple],
    patches: list[tuple],
) -> AMRHierarchy:
    """A hierarchy on the same domain and ranks as ``hierarchy``, without fields.

    ``base`` holds the base boxes and their ranks, ``patches`` the level, box
    and ranks of the finer patches. The communicators of ``hierarchy`` are
    reused, so building it is not collective.
    """
    from patchwork.mesh.hierarchy import AMRHierarchy

    new = AMRHierarchy.__new__(AMRHierarchy)
    new._communicators = hierarchy._communicators
    new.__init__(
        base_shape,
        tuple((lower, upper) for lower, upper in hierarchy.bounds),
        refinement_ratio=hierarchy.refinement_ratio,
        periodic=hierarchy.periodic,
        base_boxes=[box for box, _ in base],
        base_ranks=[ranks for _, ranks in base],
        comm=hierarchy.comm,
        geometry=hierarchy.geometry,
    )
    for level, box, ranks in patches:
        new.add_patch(level, box, ranks=ranks)
    return new


def _halved(hierarchy: AMRHierarchy) -> AMRHierarchy | None:
    """The base level of ``hierarchy`` coarsened by two; ``None`` if its boxes do not allow it.

    Every bound of every base box must be even, and every axis must keep at
    least two cells.
    """
    boxes = [patch.box for patch in hierarchy.patches_at_level(0)]
    shape = hierarchy.base_shape
    if any(n % 2 or n < 4 for n in shape):
        return None
    if any(bound % 2 for box in boxes for pair in box for bound in pair):
        return None
    base = [
        (tuple((lo // 2, hi // 2) for lo, hi in patch.box), patch.ranks)
        for patch in hierarchy.patches_at_level(0)
    ]
    return _rebuilt(hierarchy, tuple(n // 2 for n in shape), base, [])


def coarser_hierarchies(
    hierarchy: AMRHierarchy, unknowns: int = COARSEST_UNKNOWNS
) -> list[AMRHierarchy]:
    """The hierarchies of the coarse spaces of the multigrid, from the finest down.

    First ``hierarchy`` without its finest level, then without the next, down
    to its base level; then the base level coarsened by two (`_halved`)
    while it has more than ``unknowns`` nodes and its boxes allow it.
    ``hierarchy`` itself is not in the list.
    """
    base = [(patch.box, patch.ranks) for patch in hierarchy.patches_at_level(0)]
    result = []
    for top in range(hierarchy.max_level - 1, -1, -1):
        patches = [
            (patch.level, patch.box, patch.ranks)
            for patch in hierarchy.patches
            if 0 < patch.level <= top
        ]
        result.append(_rebuilt(hierarchy, hierarchy.base_shape, base, patches))
    current = result[-1] if result else hierarchy
    while int(np.prod([n + 1 for n in current.base_shape])) > unknowns:
        coarser = _halved(current)
        if coarser is None:
            break
        result.append(coarser)
        current = coarser
    return result


def _unknowns(nodes: CompositeNodes, system: LinearSystem) -> np.ndarray:
    """The unknowns of a coarse space: neither hanging nor on a Dirichlet face of ``system``."""
    keep = ~nodes.hanging
    for face, condition in system.faces.items():
        if isinstance(condition, Dirichlet):
            keep[nodes.face_nodes(face)] = False
    return np.nonzero(keep)[0]


def prolongation(
    coarse: CompositeNodes,
    coarse_unknowns: np.ndarray,
    fine: CompositeNodes,
    fine_unknowns: np.ndarray,
) -> sparse.csr_matrix:
    """The interpolation from the coarse space's unknowns to the fine one's.

    The coarse function (hanging nodes filled by its `CompositeNodes.reduction`,
    prescribed nodes zero) evaluated at the fine unknowns:
    ``(fine unknowns, coarse unknowns)``.
    """
    points = fine.coordinates[fine_unknowns].T
    matrix = coarse.evaluation(points) @ coarse.reduction[:, coarse_unknowns]
    matrix = sparse.csr_matrix(matrix)
    matrix.eliminate_zeros()
    return matrix


def _unit_diagonal_where_empty(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """``matrix`` with a one on the diagonal of its rows without a diagonal entry."""
    empty = matrix.diagonal() == 0.0
    if not empty.any():
        return matrix
    return sparse.csr_matrix(matrix + sparse.diags(empty.astype(float)))


class _Level:
    """One level of the V-cycle: its operator and its Chebyshev smoother."""

    def __init__(self, matrix: sparse.csr_matrix, steps: int, seed: int) -> None:
        """Estimate the largest eigenvalue of the Jacobi-preconditioned operator."""
        size = int(np.shape(matrix)[0])
        self.matrix = matrix
        self.inverse_diagonal = 1.0 / matrix.diagonal()
        self.steps = int(steps)
        vector = np.random.default_rng(seed).standard_normal(size)
        largest = 1.0
        for _ in range(POWER_ITERATIONS):
            vector /= np.linalg.norm(vector)
            vector = self.inverse_diagonal * (matrix @ vector)
            largest = float(np.linalg.norm(vector))
        # PETSc's default interval for a Chebyshev smoother: [0.1, 1.1] x the estimate
        self.lower, self.upper = 0.1 * largest, 1.1 * largest

    def smooth(self, x: np.ndarray, b: np.ndarray) -> np.ndarray:
        """``steps`` Chebyshev iterations on ``A x = b`` from ``x``."""
        theta = 0.5 * (self.upper + self.lower)
        delta = 0.5 * (self.upper - self.lower)
        sigma = theta / delta
        rho = 1.0 / sigma
        residual = b - self.matrix @ x
        step = self.inverse_diagonal * residual / theta
        for k in range(self.steps):
            x = x + step
            if k == self.steps - 1:
                break
            residual = residual - self.matrix @ step
            rho_next = 1.0 / (2.0 * sigma - rho)
            step = (
                rho_next * rho * step
                + 2.0 * rho_next / delta * self.inverse_diagonal * residual
            )
            rho = rho_next
        return x


@method(
    equations={
        "spaces": r"V_0 \subset \dots \subset V_L: \ \text{the composite nodes of the hierarchy without its finest levels, then of its base level coarsened by } 2^k",
        "interpolation": r"P_k = E_{k}\big(\mathbf{x}^{(k+1)}_\mathrm{free}\big)\, T_k \Big|_{\mathrm{free}}, \qquad A_{k} = P_k^\top A_{k+1} P_k \ (\text{Galerkin})",
        "cycle": r"\text{V-cycle: Chebyshev/Jacobi smoothing on } [0.1, 1.1]\,\lambda_\max(D^{-1}A), \text{ LU on } V_0; \ \text{CG (GMRES if } A \ne A^\top\text{)}, \ \|r\| \le \texttt{rtol}\,\|b\|",
        "singular": r"b \to b - \bar b, \qquad \sum_p V_p (T\phi)_p = 0 \quad \text{(no Dirichlet face, no body; all nodes, the hanging ones too)}",
    },
    assumptions=(
        "the coarse spaces are nested in the fine one: a refinement level is a multigrid level",
        "the base level halves while every bound of its boxes is even and every axis keeps two cells; else its LU is the coarsest solve",
        "the whole system on every rank (like 'lu')",
        "point (Jacobi) smoothing: a strongly anisotropic coefficient (magnetized susceptibility) slows it down",
    ),
    references=(
        "W. L. Briggs, V. E. Henson and S. F. McCormick, A Multigrid Tutorial, 2nd ed., SIAM (2000)",
        "[McCormick and Thomas, Math. Comp. 46, 439 (1986)](https://doi.org/10.1090/S0025-5718-1986-0829618-0): the fast adaptive composite grid method",
        "[Adams, Brezina, Hu and Tuminaro, J. Comput. Phys. 188, 593 (2003)](https://doi.org/10.1016/S0021-9991(03)00194-3): Chebyshev smoothers",
    ),
    components=(CompositeNodes,),
)
class CompositeMultigrid:
    """A geometric multigrid V-cycle over the levels of a hierarchy, preconditioning CG.

    See the module docstring. `solve` runs the Krylov iteration to ``rtol``;
    `cycle` applies one V-cycle (the preconditioner alone, e.g. for an outer
    nonlinear iteration's own Krylov method); `update` takes a new operator on
    the same unknowns (a new coefficient) and keeps the interpolations.

    Attributes:
        num_levels: The levels of the V-cycle (1: a direct solve).
        interpolations: ``P_k`` from each level to the next finer one, the
            finest first.
        hierarchies: The hierarchies of the coarse spaces, the finest first.
        iterations: The Krylov iterations of the last solve.
        residuals: The relative residual after each iteration of the last solve.
    """

    @sp.profile("patchwork.multigrid_setup", tags=PROFILE_TAGS)
    def __init__(
        self,
        system: LinearSystem,
        rtol: float = 1.0e-10,
        max_iterations: int = 200,
        smoothing_steps: int = 2,
        coarsest_unknowns: int = COARSEST_UNKNOWNS,
    ) -> None:
        """Build the coarse spaces and their interpolations, then the levels."""
        self.rtol = float(rtol)
        self.max_iterations = int(max_iterations)
        self.smoothing_steps = int(smoothing_steps)
        fine = system.nodes
        fine_unknowns = np.asarray(system.unknowns)
        self.hierarchies = coarser_hierarchies(fine.hierarchy, coarsest_unknowns)
        self.interpolations = []
        for hierarchy in self.hierarchies:
            coarse = CompositeNodes(hierarchy)
            coarse_unknowns = _unknowns(coarse, system)
            if not coarse_unknowns.size:
                break
            self.interpolations.append(
                prolongation(coarse, coarse_unknowns, fine, fine_unknowns)
            )
            fine, fine_unknowns = coarse, coarse_unknowns
        self.num_levels = len(self.interpolations) + 1
        # the control volumes of the hanging nodes go to the nodes they hang from:
        # sum_p V_p phi_p over all nodes, phi on the unknowns
        nodes = system.nodes
        self._volumes = (nodes.reduction.T @ nodes.volumes)[system.unknowns]
        self.iterations = 0
        self.residuals: list[float] = []
        self.update(system)

    @sp.profile("patchwork.multigrid_update", tags=PROFILE_TAGS)
    def update(self, system: LinearSystem) -> None:
        """Take the operator of ``system`` (same unknowns): the Galerkin products and smoothers."""
        operator = sparse.csr_matrix(system.matrix)
        self.symmetric = system.symmetric
        self.singular = system.singular
        self.matrix = operator
        self.levels = [_Level(operator, self.smoothing_steps, 0)]
        for depth, interpolation in enumerate(self.interpolations[:-1]):
            operator = _unit_diagonal_where_empty(
                sparse.csr_matrix(interpolation.T @ operator @ interpolation)
            )
            self.levels.append(_Level(operator, self.smoothing_steps, depth + 1))
        if self.interpolations:
            last = self.interpolations[-1]
            operator = _unit_diagonal_where_empty(
                sparse.csr_matrix(last.T @ operator @ last)
            )
        # a singular coarsest operator (the constants): pin its first unknown
        start = 1 if self.singular else 0
        self._coarsest = scipy.sparse.linalg.splu(
            sparse.csc_matrix(operator[start:, start:]), permc_spec="MMD_AT_PLUS_A"
        )

    def _coarsest_solve(self, b: np.ndarray) -> np.ndarray:
        """The coarsest level's direct solve (up to a constant if singular)."""
        if not self.singular:
            return self._coarsest.solve(b)
        result = np.zeros(b.size)
        result[1:] = self._coarsest.solve(b[1:] - b.mean())
        return result

    def _cycle(self, depth: int, b: np.ndarray) -> np.ndarray:
        """A V-cycle from level ``depth`` (0 the finest) on ``A x = b`` from zero."""
        if depth == self.num_levels - 1:
            return self._coarsest_solve(b)
        level = self.levels[depth]
        interpolation = self.interpolations[depth]
        x = level.smooth(np.zeros(b.size), b)
        residual = b - level.matrix @ x
        x = x + interpolation @ self._cycle(depth + 1, interpolation.T @ residual)
        return level.smooth(x, b)

    def cycle(self, right_hand_side: np.ndarray) -> np.ndarray:
        """One V-cycle from zero: the preconditioner applied to ``right_hand_side``."""
        return self._cycle(0, np.asarray(right_hand_side, dtype=float))

    def as_linear_operator(self) -> scipy.sparse.linalg.LinearOperator:
        """The V-cycle as a SciPy `LinearOperator` (a preconditioner ``M``)."""
        size = int(np.shape(self.matrix)[0])
        return scipy.sparse.linalg.LinearOperator(
            (size, size),
            matvec=self.cycle,  # pyright: ignore[reportCallIssue]
        )

    def _conjugate_gradients(self, b: np.ndarray) -> np.ndarray:
        """Preconditioned CG from zero."""
        norm = np.linalg.norm(b)
        x = np.zeros(b.size)
        self.residuals = []
        if norm == 0.0:
            self.iterations = 0
            return x
        residual = b.copy()
        z = self.cycle(residual)
        direction = z.copy()
        rz = residual @ z
        for iteration in range(1, self.max_iterations + 1):
            image = self.matrix @ direction
            alpha = rz / (direction @ image)
            x += alpha * direction
            residual -= alpha * image
            relative = float(np.linalg.norm(residual) / norm)
            self.residuals.append(relative)
            if relative <= self.rtol:
                self.iterations = iteration
                return x
            z = self.cycle(residual)
            rz_next = residual @ z
            direction = z + (rz_next / rz) * direction
            rz = rz_next
        raise RuntimeError(
            f"the multigrid solve did not converge in {self.max_iterations} "
            f"iterations (relative residual {self.residuals[-1]:.2e})"
        )

    def _gmres(self, b: np.ndarray) -> np.ndarray:
        """Right-preconditioned GMRES (SciPy) from zero, for a non-symmetric operator."""
        norm = np.linalg.norm(b)
        self.residuals = []
        if norm == 0.0:
            self.iterations = 0
            return np.zeros(b.size)
        x, info = scipy.sparse.linalg.gmres(
            self.matrix,
            b,
            rtol=self.rtol,
            atol=0.0,
            restart=50,
            maxiter=self.max_iterations,
            M=self.as_linear_operator(),
            callback=self.residuals.append,
            callback_type="pr_norm",
        )
        self.iterations = len(self.residuals)
        if info != 0:
            raise RuntimeError(
                f"the multigrid solve did not converge in {self.max_iterations} "
                f"iterations (GMRES info {info})"
            )
        return x

    @sp.profile("patchwork.multigrid_solve", tags=PROFILE_TAGS)
    def solve(self, right_hand_side: np.ndarray) -> np.ndarray:
        """The unknowns for the right-hand side (of all rows), to ``rtol``.

        A singular operator (no Dirichlet face, no body) gets the right-hand side
        without its mean and gives the solution of zero volume-weighted mean.
        """
        b = np.asarray(right_hand_side, dtype=float)
        if self.singular:
            b = b - b.mean()
        x = self._conjugate_gradients(b) if self.symmetric else self._gmres(b)
        if self.singular:
            x -= (self._volumes @ x) / self._volumes.sum()
        return x


class MultigridBackend(Backend):
    """Geometric multigrid over the patches (`CompositeMultigrid`), on every rank.

    Any hierarchy, conditions, embedded bodies (GMRES) and the singular
    operators (the solution of zero volume-weighted mean). The tolerance is
    relative: a direct solver's answer to about ``rtol`` times the condition
    number. With `NodalPoisson.set_coefficient` the interpolations are kept
    and only the Galerkin operators are built again.
    """

    name = "multigrid"

    def __init__(
        self,
        rtol: float = 1.0e-10,
        max_iterations: int = 200,
        smoothing_steps: int = 2,
        coarsest_unknowns: int = COARSEST_UNKNOWNS,
    ) -> None:
        """The options of `CompositeMultigrid`."""
        self.options: dict[str, Any] = {
            "rtol": rtol,
            "max_iterations": max_iterations,
            "smoothing_steps": smoothing_steps,
            "coarsest_unknowns": coarsest_unknowns,
        }

    def unsupported(self, system: LinearSystem) -> str | None:
        """Every system: also the singular ones."""
        return None

    def setup(self, system: LinearSystem) -> CompositeMultigrid:
        """Build the levels."""
        return CompositeMultigrid(system, **self.options)

    def __repr__(self) -> str:
        """The backend and its options."""
        options = ", ".join(f"{key}={value!r}" for key, value in self.options.items())
        return f"MultigridBackend({options})"


BACKENDS[MultigridBackend.name] = MultigridBackend
