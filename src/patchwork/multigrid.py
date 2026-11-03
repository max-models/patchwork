r"""Geometric multigrid for the composite operator of a uniform mesh (`MultigridBackend`).

A mesh of one level that fills its box is a lattice: ``n + 1`` nodes along an
axis of ``n`` cells, ``n`` along a periodic one. It coarsens by taking every
other node along each axis, and its interpolation is the tensor product of 1D
linear interpolations (`prolongation_1d`), so the hierarchy is built from the
lattice shape alone. Any number of nodes works: on a non-periodic axis whose
cell count is odd the last coarse cell is one fine cell wide; a periodic axis
with an odd number of nodes, or one that is already short, is not coarsened
further (semi-coarsening).

The unknowns are only part of the finest lattice (Dirichlet faces and embedded
bodies are not unknowns), so the finest interpolation is the lattice one
restricted to the rows of the unknowns: the coarse-grid correction leaves the
prescribed nodes alone. The coarse levels are whole lattices with Galerkin
operators ``P^T A P``, so bodies, Neumann faces and an r-z metric need no
stencils of their own; a coarse node that no unknown interpolates from (deep
inside a body) gets a unit diagonal.

Each level is smoothed by Chebyshev iteration with Jacobi preconditioning, the
coarsest one solved directly; PETSc's PCMG runs the V-cycle as the
preconditioner of conjugate gradients (GMRES for a non-symmetric operator).
Ported from picnic's geometric multigrid, which this replaces.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sparse
import scope_profiler as sp
from methodcard import method

from patchwork._profiling import PROFILE_TAGS
from patchwork.distributed import petsc

if TYPE_CHECKING:
    from patchwork.backends import LinearSystem

#: Stop coarsening once a level has at most this many nodes.
COARSEST_NODES = 2000

#: An axis with at most this many nodes is not coarsened further.
MIN_AXIS_NODES = 3

_PREFIXES = itertools.count()


def prolongation_1d(num_nodes: int, periodic: bool) -> sparse.csr_matrix | None:
    """Linear interpolation from every other node to all ``num_nodes`` nodes of an axis.

    Returns the ``(num_nodes, num_coarse)`` matrix, or ``None`` if the axis is
    not coarsened (too few nodes, or a periodic axis with an odd node count).
    The first and the last node of a non-periodic axis are coarse nodes.
    """
    if num_nodes <= MIN_AXIS_NODES:
        return None
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    if periodic:
        if num_nodes % 2:
            return None
        num_coarse = num_nodes // 2
        for fine in range(num_nodes):
            if fine % 2 == 0:
                rows.append(fine)
                cols.append(fine // 2)
                vals.append(1.0)
            else:
                rows += [fine, fine]
                cols += [fine // 2, (fine // 2 + 1) % num_coarse]
                vals += [0.5, 0.5]
        return sparse.csr_matrix((vals, (rows, cols)), shape=(num_nodes, num_coarse))
    coarse_nodes = list(range(0, num_nodes, 2))
    if coarse_nodes[-1] != num_nodes - 1:
        coarse_nodes.append(num_nodes - 1)
    for coarse, (left, right) in enumerate(itertools.pairwise(coarse_nodes)):
        rows.append(left)
        cols.append(coarse)
        vals.append(1.0)
        for fine in range(left + 1, right):
            weight = (fine - left) / (right - left)
            rows += [fine, fine]
            cols += [coarse, coarse + 1]
            vals += [1.0 - weight, weight]
    rows.append(num_nodes - 1)
    cols.append(len(coarse_nodes) - 1)
    vals.append(1.0)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(num_nodes, len(coarse_nodes)))


def prolongations(
    shape: tuple[int, ...], periodic: tuple[bool, ...]
) -> list[sparse.csr_matrix]:
    """The interpolations of the lattice hierarchy, from the finest level down.

    Entry ``k`` maps level ``k + 1`` (coarser) to level ``k``, rows and columns
    in C order: the Kronecker product of the axes' `prolongation_1d` (the
    identity on an axis that is not coarsened). Empty if the lattice cannot be
    coarsened or has at most `COARSEST_NODES` nodes.
    """
    result = []
    shape = tuple(int(n) for n in shape)
    while int(np.prod(shape)) > COARSEST_NODES:
        factors = [prolongation_1d(n, p) for n, p in zip(shape, periodic, strict=True)]
        if all(factor is None for factor in factors):
            break
        factors = [
            sparse.identity(n, format="csr") if factor is None else factor
            for n, factor in zip(shape, factors, strict=True)
        ]
        prolongation = factors[0]
        for factor in factors[1:]:
            prolongation = sparse.kron(prolongation, factor, format="csr")
        result.append(prolongation)
        shape = tuple(factor.shape[1] for factor in factors)
    return result


def lattice_shape(system: LinearSystem) -> tuple[int, ...]:
    """The nodes per axis of the one-level lattice of ``system``'s mesh."""
    hierarchy = system.nodes.hierarchy
    cells = hierarchy.level_shape(0)
    return tuple(
        int(n) if periodic else int(n) + 1
        for n, periodic in zip(cells, hierarchy.periodic, strict=True)
    )


def _unit_diagonal_where_empty(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """``matrix`` with a one on the diagonal of its rows without a diagonal entry."""
    empty = matrix.diagonal() == 0.0
    if not empty.any():
        return matrix
    return sparse.csr_matrix(matrix + sparse.diags(empty.astype(float)))


def _petsc_matrix(matrix: sparse.csr_matrix, comm: Any) -> Any:
    """A sequential PETSc AIJ matrix of ``matrix``."""
    PETSc = petsc()  # noqa: N806 - PETSc's own name
    matrix = sparse.csr_matrix(matrix)
    matrix.sum_duplicates()
    matrix.sort_indices()
    result = PETSc.Mat().createAIJ(
        size=matrix.shape,
        csr=(
            matrix.indptr.astype(PETSc.IntType),
            matrix.indices.astype(PETSc.IntType),
            matrix.data.astype(PETSc.ScalarType),
        ),
        comm=comm,
    )
    result.assemble()
    return result


@method(
    equations={
        "coarsening": r"P_k = \bigotimes_a P^{(a)}_k, \qquad A_{k+1} = P_k^\top A_k P_k \ (\text{Galerkin})",
        "finest": r"P_0 = (P^\mathrm{lattice}_0)_{\text{rows of the unknowns}, \,:}",
        "cycle": r"\text{V-cycle: Chebyshev/Jacobi smoothing, LU on the coarsest level; CG (GMRES if } A \ne A^\top\text{)}, \ \|r\| \le \texttt{rtol}\,\|b\|",
    },
    assumptions=(
        "one level that fills its box (a lattice), any number of nodes per axis; periodic axes with an odd count or few nodes are not coarsened",
        "the whole system on every rank (like 'lu'); coarse nodes without fine unknowns get a unit diagonal",
        "the previous solution is the initial guess; a nonsingular operator",
    ),
    references=(
        "W. L. Briggs, V. E. Henson and S. F. McCormick, A Multigrid Tutorial, 2nd ed., SIAM (2000)",
        "[PCMG](https://petsc.org/main/manualpages/PC/PCMG)",
    ),
)
class MultigridSolver:
    """A geometric multigrid V-cycle preconditioning CG, for one system (see module docstring).

    Attributes:
        num_levels: The levels of the hierarchy (1: too small to coarsen, a direct solve).
        iterations: The Krylov iterations of the last solve.
    """

    @sp.profile("patchwork.multigrid_setup", tags=PROFILE_TAGS)
    def __init__(
        self,
        system: LinearSystem,
        rtol: float = 1.0e-10,
        smoothing_steps: int = 1,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        """Build the hierarchy, the Galerkin operators and the PCMG preconditioner."""
        PETSc = petsc()  # noqa: N806 - PETSc's own name
        comm = PETSc.COMM_SELF
        shape = lattice_shape(system)
        nodes = system.nodes
        lattice = nodes.lattice[system.unknowns]
        fine_rows = np.ravel_multi_index(tuple(lattice.T), shape)
        hierarchy = prolongations(shape, tuple(nodes.hierarchy.periodic))
        if hierarchy:  # the unknowns' rows of the finest interpolation
            hierarchy[0] = sparse.csr_matrix(hierarchy[0][fine_rows])

        self.num_levels = len(hierarchy) + 1
        operator = sparse.csr_matrix(system.matrix)
        self.matrix = _petsc_matrix(operator, comm)
        if system.symmetric:
            self.matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        operators = [self.matrix]
        interpolations = []
        for prolongation in hierarchy:
            operator = _unit_diagonal_where_empty(
                sparse.csr_matrix(prolongation.T @ operator @ prolongation)
            )
            operators.append(_petsc_matrix(operator, comm))
            interpolations.append(_petsc_matrix(prolongation, comm))

        prefix = f"patchwork_multigrid{next(_PREFIXES)}_"
        settings = {
            "ksp_type": "cg" if system.symmetric else "gmres",
            "ksp_rtol": rtol,
            "ksp_atol": 0.0,
            "ksp_max_it": 500,
            "mg_levels_ksp_type": "chebyshev",
            "mg_levels_ksp_chebyshev_esteig": "0,0.1,0,1.1",
            "mg_levels_ksp_max_it": int(smoothing_steps),
            "mg_levels_pc_type": "jacobi",
            "mg_coarse_ksp_type": "preonly",
            "mg_coarse_pc_type": "lu",
            **dict(options or {}),
        }
        # the level solvers read their options when PCMG creates them
        database = PETSc.Options()
        for key, value in settings.items():
            database.setValue(prefix + key, value)
        ksp = PETSc.KSP().create(comm=comm)
        ksp.setOptionsPrefix(prefix)
        ksp.setOperators(self.matrix)
        pc = ksp.getPC()
        pc.setType("mg")
        pc.setMGLevels(self.num_levels)
        pc.setMGType(PETSc.PC.MGType.MULTIPLICATIVE)
        pc.setMGCycleType(PETSc.PC.MGCycleType.V)
        for depth, interpolation in enumerate(interpolations):
            level = self.num_levels - 1 - depth  # PCMG counts from the coarsest
            pc.setMGInterpolation(level, interpolation)
            # an explicit transpose: a plain product is cheaper than MatMultTranspose
            pc.setMGRestriction(level, interpolation.transpose(PETSc.Mat()))
        for depth, level_operator in enumerate(operators):
            level = self.num_levels - 1 - depth
            solver = pc.getMGCoarseSolve() if level == 0 else pc.getMGSmoother(level)
            solver.setOperators(level_operator)
        ksp.setFromOptions()
        ksp.setInitialGuessNonzero(True)
        ksp.setUp()  # PCMG reads the level options here
        for key in settings:
            database.delValue(prefix + key)
        self.ksp = ksp
        self._x, self._b = self.matrix.createVecs()
        self._x.set(0.0)
        self.iterations = 0

    def reset(self) -> None:
        """Start the next solve from zero instead of the previous solution."""
        self._x.set(0.0)

    @sp.profile("patchwork.multigrid_solve", tags=PROFILE_TAGS)
    def solve(self, right_hand_side: np.ndarray) -> np.ndarray:
        """The unknowns for the right-hand side (of all rows)."""
        self._b.getArray()[...] = np.asarray(right_hand_side, dtype=float)
        self.ksp.solve(self._b, self._x)
        reason = self.ksp.getConvergedReason()
        if reason < 0:
            raise RuntimeError(
                f"the multigrid solve did not converge (reason {reason})"
            )
        self.iterations = self.ksp.getIterationNumber()
        return self._x.getArray().copy()
