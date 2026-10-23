"""A sparse system distributed by rows over the ranks, solved with PETSc.

The composite solvers (``NodalPoisson(backend="petsc")``) assemble on every
rank only the rows of the unknowns that rank owns, in a global numbering in
which each rank's unknowns are contiguous, and hand them to a
:class:`PetscSystem`: a PETSc ``MPIAIJ`` matrix with exactly those rows and a
Krylov solver with an algebraic multigrid preconditioner (BoomerAMG from hypre
when PETSc has it, PETSc's own GAMG otherwise). The solution is then gathered to
every rank, as the gathered (SuperLU) path returns it.

The solver reads PETSc's options database under its prefix (``patchwork_``
by default), so ``-patchwork_ksp_type`` and the like on the command line, or
``options`` here, override the defaults. If the Krylov method does not
converge, the solve is repeated once with GMRES and block Jacobi (ILU on each
rank's block), which is slower but robust; if that fails too, it raises.

petsc4py is an optional dependency (``pip install patchwork-amr[petsc]``);
this module imports it only when a system is built.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import scipy.sparse as sparse
import scope_profiler as sp
from maybempi import petsc_comm

from patchwork._profiling import PROFILE_TAGS

#: Options of the fallback solve, after the first one diverged.
FALLBACK = {
    "ksp_type": "gmres",
    "ksp_gmres_restart": 100,
    "pc_type": "bjacobi",
    "sub_pc_type": "ilu",
    "ksp_max_it": 10000,
}


def petsc() -> Any:
    """``petsc4py.PETSc``, initialized; an ImportError that says how to get it."""
    try:
        from petsc4py import PETSc  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise ImportError(
            'backend="petsc" needs petsc4py: `pip install patchwork-amr[petsc]` '
            "(or `conda install -c conda-forge petsc4py`)"
        ) from error
    return PETSc


def has_petsc() -> bool:
    """Whether petsc4py can be imported."""
    try:
        petsc()
    except ImportError:
        return False
    return True


class PetscSystem:
    """Rows of a square sparse system on every rank, solved together with PETSc.

    ``rows`` holds this rank's rows, ``(k, N)`` with columns in the global
    numbering, where rank ``r`` owns rows ``offsets[r]:offsets[r + 1]`` (its
    ``k``). ``comm`` is the communicator of all ranks that take part (an mpi4py
    communicator or maybempi's serial one). ``symmetric`` picks conjugate
    gradients, otherwise GMRES. ``options`` are PETSc options without the
    leading dash and prefix, e.g. ``{"pc_type": "gamg", "ksp_rtol": 1e-8}``.
    Collective over ``comm``.

    Attributes:
        iterations: The Krylov iterations of the last solve.
        fell_back: Whether the last solve needed the fallback solver.
    """

    @sp.profile("patchwork.petsc_setup", tags=PROFILE_TAGS)
    def __init__(
        self,
        rows: sparse.csr_matrix,
        comm: Any,
        *,
        symmetric: bool = False,
        options: Mapping[str, Any] | None = None,
        prefix: str = "patchwork_",
        rtol: float = 1.0e-10,
    ) -> None:
        """Build the matrix and the solver; the preconditioner is set up here too."""
        PETSc = petsc()  # noqa: N806 - PETSc's own name
        self.comm = petsc_comm(comm)  # COMM_SELF for maybempi's serial stand-in
        size = self.comm.getSize()
        rows = sparse.csr_matrix(rows)
        rows.sum_duplicates()
        rows.sort_indices()
        local, total = np.shape(rows)
        counts = comm.allgather(local) if size > 1 else [local]
        self.offsets = np.concatenate(([0], np.cumsum(counts)))
        if self.offsets[-1] != total:
            raise ValueError(
                f"the ranks hold {self.offsets[-1]} rows of a system of {total} unknowns"
            )
        integer, scalar = PETSc.IntType, PETSc.ScalarType
        self.matrix = PETSc.Mat().createAIJ(
            size=((local, total), (local, total)),
            csr=(
                rows.indptr.astype(integer),
                rows.indices.astype(integer),
                rows.data.astype(scalar),
            ),
            comm=self.comm,
        )
        self.matrix.assemble()
        if symmetric:
            self.matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        self._x, self._b = self.matrix.createVecs()
        self._x.set(0.0)
        self._gather, self._everywhere = PETSc.Scatter.toAll(self._x)

        defaults: dict[str, Any] = {
            "ksp_type": "cg" if symmetric else "gmres",
            "ksp_rtol": rtol,
            "ksp_atol": 0.0,
            "ksp_max_it": 2000,
            "pc_type": "hypre" if _has_package(PETSc, "hypre") else "gamg",
        }
        if defaults["pc_type"] == "hypre":
            defaults["pc_hypre_type"] = "boomeramg"
        self._rtol = rtol
        self._prefix = prefix
        self.ksp = self._solver(defaults, dict(options or {}))
        self.iterations = 0
        self.fell_back = False

    def _solver(self, defaults: dict[str, Any], options: dict[str, Any]) -> Any:
        """A KSP from the defaults, the options database and ``options``, set up."""
        PETSc = petsc()  # noqa: N806
        database = PETSc.Options()
        added = []
        for key, value in {**defaults, **options}.items():
            name = self._prefix + key
            # the command line overrides the defaults, `options` override both
            if key in options or not database.hasName(name):
                database.setValue(name, value)
                added.append(name)
        ksp = PETSc.KSP().create(comm=self.comm)
        ksp.setOptionsPrefix(self._prefix)
        ksp.setOperators(self.matrix)
        ksp.setFromOptions()
        for name in added:
            database.delValue(name)
        ksp.setInitialGuessNonzero(True)
        ksp.setUp()
        return ksp

    def reset(self) -> None:
        """Start the next solve from zero instead of the previous solution."""
        self._x.set(0.0)

    @sp.profile("patchwork.petsc_solve", tags=PROFILE_TAGS)
    def solve(self, right_hand_side: np.ndarray) -> np.ndarray:
        """Solve for this rank's rows of the right-hand side; the solution on every rank.

        The previous solution is the initial guess. Collective.
        """
        b = self._b.getArray()
        b[...] = np.asarray(right_hand_side, dtype=float)
        self.fell_back = False
        self.ksp.solve(self._b, self._x)
        if self.ksp.getConvergedReason() < 0:
            self.fell_back = True
            reason = self.ksp.getConvergedReason()
            fallback = self._solver({**FALLBACK, "ksp_rtol": self._rtol}, {})
            self._x.set(0.0)
            fallback.solve(self._b, self._x)
            if fallback.getConvergedReason() < 0:
                raise RuntimeError(
                    f"PETSc did not converge (reason {reason}, then "
                    f"{fallback.getConvergedReason()} with GMRES and block Jacobi)"
                )
            self.iterations = fallback.getIterationNumber()
        else:
            self.iterations = self.ksp.getIterationNumber()
        self._gather.scatter(self._x, self._everywhere)
        return self._everywhere.getArray().copy()


def _has_package(PETSc: Any, name: str) -> bool:  # noqa: N803
    """Whether PETSc was built with an external package (e.g. ``"hypre"``)."""
    try:
        return bool(PETSc.Sys.hasExternalPackage(name))
    except AttributeError:  # an older petsc4py
        return False
