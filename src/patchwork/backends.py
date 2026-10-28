r"""Solver backends: one assembled composite operator, solved in different ways.

A composite solve (:class:`~patchwork.NodalPoisson`) assembles its operator
once, as a :class:`LinearSystem`: the rows of the unknowns, the mesh they live
on and the conditions they were assembled with. A backend turns that system
into a solver whose ``solve(right_hand_side)`` returns the unknowns. Every
backend solves the same operator, so on a problem that two of them support they
give the same answer to round-off (or to the tolerance of an iterative one).

``backend=`` of a solve takes a backend or its name:

- ``"lu"`` (:class:`LUBackend`): a direct LU factorization, on every rank
  (LAPACK's for the tridiagonal operator of a line, SuperLU otherwise). Any
  mesh, any conditions, embedded bodies; not a singular operator.
- ``"petsc"`` (:class:`PetscBackend`): the rows distributed over the ranks, a
  PETSc Krylov method with algebraic multigrid (needs petsc4py).
- ``"fft"`` (:class:`FFTBackend`): fast transforms on a uniform mesh (a
  hierarchy of one level that fills its box) without embedded bodies, every
  axis periodic or with the same kind of condition, Dirichlet or Neumann, on
  both of its faces. Also the singular operators: all axes periodic or Neumann.
- ``"auto"`` (:class:`AutoBackend`): the tridiagonal LU on a line, the FFT
  where it applies, the LU elsewhere.

A backend states what it cannot solve (:meth:`Backend.unsupported`), and the
solve refuses such a system with that reason. A new backend subclasses
:class:`Backend`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import numpy as np
import scipy.fft
import scipy.linalg
import scipy.sparse as sparse
import scipy.sparse.linalg

if TYPE_CHECKING:
    from patchwork.nodal import CompositeNodes, Dirichlet, Neumann


class Solver(Protocol):
    """What :meth:`Backend.setup` returns: a solver of one system."""

    def solve(self, right_hand_side: np.ndarray) -> np.ndarray:
        """The unknowns for this rank's rows of the right-hand side, on every rank."""
        ...


@dataclass(frozen=True)
class LinearSystem:
    """An assembled composite operator, as a backend receives it.

    Attributes:
        matrix: This rank's rows of the operator over the columns of the
            unknowns: all rows unless the backend is distributed.
        unknowns: Node numbers of the columns, in their order.
        rows: Node numbers of this rank's rows.
        nodes: The :class:`~patchwork.nodal.CompositeNodes` of the mesh.
        faces: The condition of every part of the boundary (as
            :class:`~patchwork.NodalPoisson` keeps them, with ``"inner"``).
        bodies: The number of embedded bodies.
        singular: Whether nothing fixes the potential (no Dirichlet face, no
            body): the operator then has the constants as its null space.
        symmetric: Whether the matrix is symmetric.
        comm: The communicator of the hierarchy.
    """

    matrix: sparse.csr_matrix
    unknowns: np.ndarray
    rows: np.ndarray
    nodes: CompositeNodes
    faces: Mapping[str, Dirichlet | Neumann]
    bodies: int
    singular: bool
    symmetric: bool
    comm: Any


class Backend:
    """A way to solve a :class:`LinearSystem`; subclasses implement :meth:`setup`.

    ``distributed`` backends get only the rows of the unknowns their rank owns
    (:attr:`~patchwork.nodal.CompositeNodes.node_ranks`), in a numbering in
    which every rank's unknowns are contiguous; the others get all rows on
    every rank. A backend is a configuration: one instance can set up any
    number of systems (e.g. one per regrid).
    """

    name: ClassVar[str] = ""
    distributed: ClassVar[bool] = False

    def unsupported(self, system: LinearSystem) -> str | None:
        """Why this backend cannot solve ``system``; ``None`` if it can.

        By default every system but a singular one.
        """
        if system.singular:
            return (
                "the Poisson operator is singular: without a Dirichlet face or "
                "an embedded body the potential is only known up to a constant "
                f"(the {self.name!r} backend needs a nonsingular operator; "
                "'fft' solves for the solution of zero mean on a uniform mesh)"
            )
        return None

    def setup(self, system: LinearSystem) -> Solver:
        """Factorize or otherwise prepare the solve of ``system``."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """The backend's name."""
        return f"{type(self).__name__}()"


class TridiagonalLU:
    """LAPACK's factorization of a tridiagonal matrix: LDL^T if symmetric, else LU.

    The operator of a line (1D, no periodic axis) is tridiagonal; the
    symmetric one (no embedded body) is positive definite, and its LDL^T solve
    (``pttrs``) is about three times as fast as the pivoted LU (``gttrs``).
    """

    def __init__(self, matrix: sparse.spmatrix) -> None:
        """Factorize the three diagonals of ``matrix``."""
        matrix = sparse.dia_matrix(matrix)
        lower, diagonal, upper = (matrix.diagonal(k).copy() for k in (-1, 0, 1))
        self.symmetric = bool(np.array_equal(lower, upper))
        factorize, self._solve = scipy.linalg.get_lapack_funcs(
            ("pttrf", "pttrs") if self.symmetric else ("gttrf", "gttrs"),
            dtype=np.float64,
        )
        if self.symmetric:
            *self._factors, info = factorize(diagonal, upper)
        else:
            *self._factors, info = factorize(lower, diagonal, upper)
        if info != 0:  # pragma: no cover - the operator is nonsingular
            raise np.linalg.LinAlgError(
                f"the tridiagonal factorization failed ({info})"
            )

    def solve(self, right_hand_side: np.ndarray) -> np.ndarray:
        """The solution for ``right_hand_side``."""
        solution, _ = self._solve(*self._factors, right_hand_side)
        return solution


def _tridiagonal(matrix: sparse.spmatrix) -> bool:
    """Whether every entry of ``matrix`` is on the diagonal or next to it."""
    matrix = sparse.coo_matrix(matrix)
    return bool(np.all(np.abs(matrix.row - matrix.col) <= 1))


class LUBackend(Backend):
    """A direct LU factorization of the whole operator, on every rank.

    A tridiagonal operator (a line without a periodic seam) is factorized with
    LAPACK (:class:`TridiagonalLU`), any other with SciPy's SuperLU. Its
    ordering is minimum degree on the pattern of ``A + A^T``, the mesh's own
    graph, with about half the fill of the default column ordering; the
    symmetric mode keeps the diagonal pivots that ordering assumes (the
    operator is diagonally dominant), a fifth faster than threshold pivoting.
    """

    name = "lu"

    def setup(self, system: LinearSystem) -> Solver:
        """Factorize the matrix."""
        if _tridiagonal(system.matrix):
            return TridiagonalLU(system.matrix)
        return scipy.sparse.linalg.splu(
            sparse.csc_matrix(system.matrix),
            permc_spec="MMD_AT_PLUS_A",
            options={"SymmetricMode": True},
        )


class PetscBackend(Backend):
    """The rows distributed over the ranks and solved with PETSc (:class:`~patchwork.PetscSystem`).

    Conjugate gradients for a symmetric operator, GMRES otherwise, with an
    algebraic multigrid preconditioner, to the relative residual ``rtol``.
    ``options`` are PETSc options without the dash and the prefix, e.g.
    ``{"pc_type": "gamg"}``; ``-<prefix>ksp_type`` and the like on the command
    line apply as well. Needs petsc4py.
    """

    name = "petsc"
    distributed = True

    def __init__(
        self,
        rtol: float = 1.0e-10,
        options: Mapping[str, Any] | None = None,
        prefix: str = "patchwork_nodal_",
    ) -> None:
        """Keep the tolerance and the options."""
        self.rtol = float(rtol)
        self.options = dict(options or {})
        self.prefix = prefix

    def setup(self, system: LinearSystem) -> Solver:
        """Build the distributed matrix and the Krylov solver."""
        from patchwork.distributed import PetscSystem

        return PetscSystem(
            system.matrix,
            system.comm,
            symmetric=system.symmetric,
            options=self.options,
            prefix=self.prefix,
            rtol=self.rtol,
        )

    def __repr__(self) -> str:
        """The backend with its tolerance and options."""
        return f"PetscBackend(rtol={self.rtol!r}, options={self.options!r})"


def _axis_kind(system: LinearSystem, axis: int) -> str | None:
    """``"periodic"``, ``"dirichlet"`` or ``"neumann"``; ``None`` for mixed faces."""
    from patchwork.nodal import FACES, Dirichlet

    hierarchy = system.nodes.hierarchy
    if hierarchy.periodic[axis]:
        return "periodic"
    lower, upper = (
        system.faces[face] for face, where in FACES.items() if where[0] == axis
    )
    if type(lower) is not type(upper):
        return None
    return "dirichlet" if isinstance(lower, Dirichlet) else "neumann"


def _uniform_kinds(system: LinearSystem) -> tuple[str, ...]:
    """The kind of every axis, for a system with no mixed faces."""
    kinds = []
    for axis in range(system.nodes.ndim):
        kind = _axis_kind(system, axis)
        if kind is None:  # pragma: no cover - FFTBackend.unsupported refuses it
            raise ValueError(f"axis {axis} has different conditions on its faces")
        kinds.append(kind)
    return tuple(kinds)


class FFTSolver:
    """The FFT solve of one uniform system (see :class:`FFTBackend`).

    The operator divided by the control volumes is the standard difference
    Laplacian, a sum over the axes of second differences, which the transforms
    of the axes diagonalize: the discrete Fourier transform on a periodic axis,
    the sine transform (DST-I) between Dirichlet faces and the cosine
    transform (DCT-I) between Neumann faces (whose nodes have half control
    volumes). ``solve`` divides the right-hand side by the volumes, transforms,
    divides by the eigenvalues and transforms back. A zero eigenvalue (the
    constants, when nothing fixes the potential) is dropped: the solution of
    zero mean, weighted by the volumes, for the right-hand side without its
    mean.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        kinds: tuple[str, ...],
        cells: tuple[int, ...],
        spacing: tuple[float, ...],
        volumes: np.ndarray,
        spectral: bool = False,
        workers: int | None = None,
    ) -> None:
        """Precompute the inverse eigenvalues and the volumes."""
        self.shape = shape
        self.kinds = kinds
        self.workers = workers
        periodic = [axis for axis, kind in enumerate(kinds) if kind == "periodic"]
        # real data stays real through the sine and cosine transforms and the
        # first periodic one (rfft); the other periodic axes are complex
        self._real_axis = periodic[-1] if periodic else None
        self._complex_axes = periodic[:-1]
        eigenvalues = np.zeros(())
        for axis, (kind, n, h) in enumerate(zip(kinds, cells, spacing, strict=True)):
            if kind == "periodic":
                if spectral:
                    values = (2.0 * np.pi * scipy.fft.fftfreq(n, d=h)) ** 2
                else:
                    values = (4.0 / h**2) * np.sin(np.pi * np.arange(n) / n) ** 2
                if axis == self._real_axis:
                    values = values[: n // 2 + 1]
            else:
                # n - 1 interior nodes (sine) or n + 1 nodes (cosine)
                k = np.arange(1, n) if kind == "dirichlet" else np.arange(n + 1)
                values = (4.0 / h**2) * np.sin(0.5 * np.pi * k / n) ** 2
            index: list[slice | None] = [None] * len(kinds)
            index[axis] = slice(None)
            eigenvalues = eigenvalues + values[tuple(index)]
        inverse = np.zeros(eigenvalues.shape)
        np.divide(1.0, eigenvalues, out=inverse, where=eigenvalues > 0.0)
        self._inverse = inverse
        self._inverse_volumes = (1.0 / volumes).reshape(shape)

    def solve(self, right_hand_side: np.ndarray) -> np.ndarray:
        """The unknowns for the right-hand side (of all rows)."""
        workers = self.workers
        data: Any = np.asarray(right_hand_side, dtype=float).reshape(self.shape)
        data = data * self._inverse_volumes
        for axis, kind in enumerate(self.kinds):
            if kind == "dirichlet":
                data = scipy.fft.dst(data, type=1, axis=axis, workers=workers)
            elif kind == "neumann":
                data = scipy.fft.dct(data, type=1, axis=axis, workers=workers)
        if self._real_axis is not None:
            data = scipy.fft.rfft(data, axis=self._real_axis, workers=workers)
        if self._complex_axes:
            data = scipy.fft.fftn(data, axes=self._complex_axes, workers=workers)
        data *= self._inverse
        if self._complex_axes:
            data = scipy.fft.ifftn(data, axes=self._complex_axes, workers=workers)
        if self._real_axis is not None:
            data = scipy.fft.irfft(
                data,
                n=self.shape[self._real_axis],
                axis=self._real_axis,
                workers=workers,
            )
        for axis, kind in enumerate(self.kinds):
            if kind == "dirichlet":
                data = scipy.fft.idst(data, type=1, axis=axis, workers=workers)
            elif kind == "neumann":
                data = scipy.fft.idct(data, type=1, axis=axis, workers=workers)
        return data.reshape(-1)


class FFTBackend(Backend):
    """Fast transforms on a uniform mesh: :class:`FFTSolver`.

    The mesh is one level that fills its box (a uniform grid), without
    embedded bodies; on every axis that is not periodic, both faces are
    Dirichlet or both are Neumann (any values). The singular operators (every
    axis periodic or Neumann) are solved too, for the solution of zero mean.
    The cost of a solve is ``O(N log N)`` and there is no factorization.

    ``spectral=True`` (every axis periodic) replaces the eigenvalues of the
    difference Laplacian by those of the exact one, ``|k|**2``: a spectral
    solve, which is not the assembled operator. ``workers`` is passed to
    :mod:`scipy.fft` (threads per transform).
    """

    name = "fft"

    def __init__(self, spectral: bool = False, workers: int | None = None) -> None:
        """Keep the options."""
        self.spectral = bool(spectral)
        self.workers = workers

    def unsupported(self, system: LinearSystem) -> str | None:
        """A non-uniform mesh, a body, mixed faces or a distributed system."""
        nodes = system.nodes
        hierarchy = nodes.hierarchy
        if system.bodies:
            return "the 'fft' backend takes no embedded bodies"
        if hierarchy.max_level > 0:
            return "the 'fft' backend needs a uniform mesh: one level, no patches"
        if nodes.cell_span.size != int(np.prod(hierarchy.level_shape(0))):
            return "the 'fft' backend needs a domain that fills its box"
        if getattr(hierarchy, "geometry", "cartesian") != "cartesian":
            return "the 'fft' backend needs Cartesian coordinates"
        kinds = [_axis_kind(system, axis) for axis in range(nodes.ndim)]
        if None in kinds:
            return (
                "the 'fft' backend needs the same kind of condition (Dirichlet "
                "or Neumann) on both faces of an axis that is not periodic"
            )
        if self.spectral and set(kinds) != {"periodic"}:
            return "spectral=True needs every axis periodic"
        return None

    def setup(self, system: LinearSystem) -> Solver:
        """The eigenvalues of the operator and the layout of the unknowns."""
        nodes = system.nodes
        hierarchy = nodes.hierarchy
        cells = tuple(int(n) for n in hierarchy.level_shape(0))
        kinds = _uniform_kinds(system)
        # the unknowns of every axis on the lattice, in C order: a periodic
        # axis has no copy of its first node, a Dirichlet face no unknown
        bounds = {"periodic": (0, 0), "dirichlet": (1, 0), "neumann": (0, 1)}
        ranges = [
            np.arange(bounds[kind][0], n + bounds[kind][1])
            for kind, n in zip(kinds, cells, strict=True)
        ]
        expected = np.stack(
            [axis.ravel() for axis in np.meshgrid(*ranges, indexing="ij")], axis=1
        )
        found = nodes.lattice[system.unknowns]
        if not np.array_equal(found, expected):  # pragma: no cover - by numbering
            raise RuntimeError("the unknowns are not the nodes of a uniform box")
        return FFTSolver(
            shape=tuple(axis.size for axis in ranges),
            kinds=kinds,
            cells=cells,
            spacing=tuple(float(h) for h in hierarchy.spacing(0)),
            volumes=nodes.volumes[system.unknowns],
            spectral=self.spectral,
            workers=self.workers,
        )

    def __repr__(self) -> str:
        """The backend with its options."""
        return f"FFTBackend(spectral={self.spectral!r}, workers={self.workers!r})"


class AutoBackend(Backend):
    """The fastest exact backend: the LU on a line, else the FFT where it applies, else the LU.

    On a line (a tridiagonal operator) the LU's solve costs a few operations
    per node, less than the transforms; on a uniform mesh in more dimensions,
    the transforms cost less than the triangular solves of the sparse LU.
    """

    name = "auto"

    def __init__(self) -> None:
        """The two candidates."""
        self.fft = FFTBackend()
        self.lu = LUBackend()

    def choose(self, system: LinearSystem) -> Backend:
        """The backend ``system`` gets."""
        if self.lu.unsupported(system) is None and _tridiagonal(system.matrix):
            return self.lu
        return self.fft if self.fft.unsupported(system) is None else self.lu

    def unsupported(self, system: LinearSystem) -> str | None:
        """What the chosen backend cannot solve."""
        return self.choose(system).unsupported(system)

    def setup(self, system: LinearSystem) -> Solver:
        """Set up the chosen backend."""
        return self.choose(system).setup(system)


#: The backends by name, as ``backend=`` takes them.
BACKENDS: dict[str, type[Backend]] = {
    backend.name: backend
    for backend in (LUBackend, PetscBackend, FFTBackend, AutoBackend)
}


def backend_of(backend: str | Backend) -> Backend:
    """A backend from its name (with its default options), or the backend itself."""
    if isinstance(backend, Backend):
        return backend
    if backend not in BACKENDS:
        raise ValueError(
            f"backend must be one of {sorted(BACKENDS)} or a Backend, got {backend!r}"
        )
    return BACKENDS[backend]()
