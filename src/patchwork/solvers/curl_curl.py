r"""The curl–curl operator on the edge space of a 2D or 3D hierarchy.

`CurlCurl` solves

$$
\nabla\times(\nu\,\nabla\times\mathbf{A}) + \beta\,\mathbf{A} = \mathbf{J}
$$

for a vector potential (or an electric field) on the edges of the leaf cells:
the magnetoquasistatic field of a current (``beta = 0``, or ``beta = i omega
sigma`` in a conductor), or a time step of an implicit Maxwell solver. The
discrete operator is assembled from the spaces of `patchwork.spaces`,

$$
K = T_1^\top\big(C^\top M_2(\nu)\, C + M_1(\beta)\big) T_1,
$$

with the curl ``C`` from the edges to the faces (in 2D to the leaf cells), the
face mass ``M2`` weighted by ``nu`` and the edge mass ``M1`` by ``beta``, both
lumped (the finite-integration / Yee scheme) or consistent (Nédélec elements),
and the hanging edges eliminated by their constraints ``T1``. Faces of the
domain are ``"electric"`` (tangential ``A`` zero, a perfect conductor) or
``"magnetic"`` (the natural condition, tangential ``nu curl A`` zero);
periodic axes take none. ``C G = 0`` holds exactly, so the gradients are the
null space of the curl term: without ``beta`` the operator is singular, and
either a small ``gauge`` term (a weak Coulomb gauge) or hypre's AMS, which
solves the singular system for a divergence-free current, makes it solvable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
import scope_profiler as sp
from methodcard import method

from patchwork._profiling import PROFILE_TAGS
from patchwork.spaces.boundaries import faces_of
from patchwork.spaces.edges import CompositeEdges
from patchwork.spaces.faces import CompositeFaces
from patchwork.spaces.nodes import CompositeNodes

if TYPE_CHECKING:
    from patchwork.mesh.hierarchy import AMRHierarchy

#: A coefficient on the leaf cells: a number, an array over the leaf cells (in the
#: order of `CompositeNodes.cell_corners`) or a function of the cell centres.
Coefficient = float | complex | np.ndarray | Callable[..., Any]

#: The conditions of a face of the domain.
CONDITIONS = ("electric", "magnetic")

#: The ways `CurlCurl` solves its system.
SOLVERS = ("lu", "ams", "auto")


def has_ams() -> bool:
    """Whether petsc4py is installed with hypre (for ``solver="ams"``)."""
    from patchwork.solvers.distributed import _has_package, has_petsc, petsc

    return has_petsc() and _has_package(petsc(), "hypre")


class AmsSolver:
    """A real symmetric edge system solved with conjugate gradients and hypre's AMS.

    ``matrix`` is the whole system on every rank; the rows are split evenly
    over the ranks of ``comm`` for PETSc. ``gradient`` maps the interior nodes
    to the unknown edges (AMS's discrete gradient) and ``constants`` are the
    edge values of the uniform fields along each axis. With ``singular`` (no
    ``beta`` term) AMS solves the curl–curl alone, for a right-hand side
    orthogonal to the gradients.

    Attributes:
        iterations: The conjugate gradient iterations of the last solve.
    """

    @sp.profile("patchwork.ams_setup", tags=PROFILE_TAGS)
    def __init__(
        self,
        matrix: sparse.csr_matrix,
        gradient: sparse.csr_matrix,
        constants: list[np.ndarray],
        comm: Any,
        *,
        singular: bool,
        rtol: float = 1.0e-10,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        """Build the PETSc matrices, the AMS preconditioner and the Krylov solver."""
        from maybempi import petsc_comm

        from patchwork.solvers.distributed import petsc

        PETSc = petsc()  # noqa: N806 - PETSc's own name
        self.comm = petsc_comm(comm)
        size, rank = self.comm.getSize(), self.comm.getRank()
        n, nodes = int(np.shape(matrix)[0]), int(np.shape(gradient)[1])

        def split(count: int) -> tuple[int, int]:
            bounds = np.linspace(0, count, size + 1).astype(int)
            return int(bounds[rank]), int(bounds[rank + 1])

        self._rows = split(n)
        node_rows = split(nodes)
        integer, scalar = PETSc.IntType, PETSc.ScalarType

        def aij(rows: sparse.csr_matrix, shape, columns_local: int) -> Any:
            rows = sparse.csr_matrix(rows)
            rows.sum_duplicates()
            rows.sort_indices()
            mat = PETSc.Mat().createAIJ(
                size=((int(np.shape(rows)[0]), shape[0]), (columns_local, shape[1])),
                csr=(
                    rows.indptr.astype(integer),
                    rows.indices.astype(integer),
                    rows.data.astype(scalar),
                ),
                comm=self.comm,
            )
            mat.assemble()
            return mat

        lo, hi = self._rows
        local = hi - lo
        self.matrix = aij(matrix[lo:hi], (n, n), local)
        self.matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        discrete_gradient = aij(
            gradient[lo:hi], (n, nodes), node_rows[1] - node_rows[0]
        )
        self._x, self._b = self.matrix.createVecs()
        vectors = []
        for constant in constants:
            vector = self.matrix.createVecLeft()
            vector.getArray()[...] = constant[lo:hi]
            vectors.append(vector)
        self._gather, self._everywhere = PETSc.Scatter.toAll(self._x)

        ksp = PETSc.KSP().create(comm=self.comm)
        ksp.setOptionsPrefix("patchwork_ams_")
        ksp.setOperators(self.matrix)
        ksp.setType("cg")
        ksp.setTolerances(rtol=rtol, atol=0.0, max_it=1000)
        pc = ksp.getPC()
        pc.setType("hypre")
        pc.setHYPREType("ams")
        pc.setHYPREDiscreteGradient(discrete_gradient)
        pc.setHYPRESetEdgeConstantVectors(*vectors)
        if singular:
            pc.setHYPRESetBetaPoissonMatrix(None)
        database = PETSc.Options()
        for key, value in (options or {}).items():
            database.setValue("patchwork_ams_" + key, value)
        ksp.setFromOptions()
        for key in options or {}:
            database.delValue("patchwork_ams_" + key)
        ksp.setUp()
        self.ksp = ksp
        self.iterations = 0
        # keep the PETSc objects the preconditioner refers to alive
        self._keep = (discrete_gradient, vectors)

    @sp.profile("patchwork.ams_solve", tags=PROFILE_TAGS)
    def solve(self, right_hand_side: np.ndarray) -> np.ndarray:
        """Solve; the solution on every rank. Collective."""
        lo, hi = self._rows
        self._b.getArray()[...] = np.asarray(right_hand_side, dtype=float)[lo:hi]
        self._x.set(0.0)
        self.ksp.solve(self._b, self._x)
        reason = self.ksp.getConvergedReason()
        if reason < 0:
            raise RuntimeError(f"AMS did not converge (reason {reason})")
        self.iterations = self.ksp.getIterationNumber()
        self._gather.scatter(self._x, self._everywhere)
        return self._everywhere.getArray().copy()


@method(
    equations={
        "problem": r"\nabla\times(\nu\,\nabla\times\mathbf{A}) + \beta\,\mathbf{A} = \mathbf{J}, \qquad \mathbf{A}\times\mathbf{n} = 0 \ \text{(electric faces)}",
        "operator": r"K = T_1^\top\big(C^\top M_2(\nu)\,C + M_1(\beta) + \epsilon\, M_1(\nu / h^2)\big)\,T_1",
        "load": r"b = T_1^\top \ell, \qquad \ell_e = \int \mathbf{J}\cdot\mathbf{w}_e\, dV \quad (\ell = M_1 j \text{ for edge values } j \text{ of } \mathbf{J})",
        "null_space": r"C\,G = 0: \quad K\,G_r = \beta\text{-term only}",
    },
    assumptions=(
        "the domain fills its box; electric faces fix the tangential edges on them to zero",
        r"$\epsilon$ = `gauge`: a weak Coulomb gauge, needed by the LU when $\beta = 0$",
        r"AMS (hypre) for real coefficients; the LU also for complex $\beta = i\omega\sigma$",
    ),
    references=(
        "[Hiptmair and Xu, SIAM J. Numer. Anal. 45, 2483 (2007)](https://doi.org/10.1137/060660588): the auxiliary space preconditioner",
        "[Kolev and Vassilevski, J. Comput. Math. 27, 604 (2009)](https://www.jstor.org/stable/43693530): AMS",
        "[Weiland, AEÜ 31, 116 (1977)](https://ui.adsabs.harvard.edu/abs/1977ArElU..31..116W): finite integration",
    ),
    components=(
        "patchwork.spaces.faces.CompositeFaces",
        "patchwork.spaces.edges.CompositeEdges.mass",
    ),
)
class CurlCurl:
    r"""Solve $\nabla\times(\nu\nabla\times\mathbf{A}) + \beta\mathbf{A} = \mathbf{J}$ on the edges.

    ``reluctivity`` (``nu``) and ``beta`` are coefficients on the leaf cells:
    numbers, arrays over the leaf cells or functions of the cell centres
    (``beta`` may be complex with the LU). ``faces`` maps the faces of the
    box (``"x_lower"``, ...) to ``"electric"`` (the default) or
    ``"magnetic"``. ``mass`` is ``"lumped"`` (finite integration, the
    staggered Yee scheme) or ``"consistent"`` (lowest-order Nédélec elements).
    ``gauge`` adds ``gauge * M1(nu / h**2)``, a weak Coulomb gauge that makes
    the operator nonsingular without ``beta``; it shifts the solution by
    ``O(gauge)``. ``solver`` is ``"lu"`` (SuperLU, every rank),
    ``"ams"`` (PETSc's conjugate gradients with hypre's AMS, rows split over
    the ranks; real coefficients) or ``"auto"`` (AMS where it applies, else
    the LU).

    Attributes:
        edges: The `CompositeEdges`.
        faces: The `CompositeFaces`.
        free: Edge numbers of the unknowns: neither hanging nor on an electric face.
        fixed: Whether an edge lies on an electric face.
        matrix: The operator on the unknowns.
        solver: ``"lu"`` or ``"ams"``, as chosen.
        iterations: The AMS iterations of the last solve (0 with the LU).
    """

    # Assembly ------------------------------------------------------------------#
    def _cells(self, value: Coefficient) -> np.ndarray:
        """A coefficient on the leaf cells, ``(m,)``."""
        nodes = self.nodes
        count = nodes.cell_span.size
        if callable(value):
            centre = (
                nodes._origin
                + (nodes.cell_lower + 0.5 * nodes.cell_span[:, None]) * nodes._spacing
            )
            value = value(*centre.T)
        array = np.asarray(value)
        if array.ndim and array.shape != (count,):
            raise ValueError(f"expected a coefficient on the {count} leaf cells")
        return np.broadcast_to(array, (count,)).copy()

    def _fixed_edges(self) -> np.ndarray:
        """Whether every edge lies on an electric face of the box."""
        edges, nodes = self.edges, self.nodes
        fixed = np.zeros(edges.num_edges, dtype=bool)
        for face, (axis, side) in faces_of(nodes.ndim).items():
            if nodes._periodic[axis] or self.conditions[face] != "electric":
                continue
            position = 0 if side == 0 else nodes._cells[axis]
            fixed |= (edges.direction != axis) & (edges.lattice[:, axis] == position)
        return fixed

    @sp.profile("patchwork.curl_curl_setup", tags=PROFILE_TAGS)
    def __init__(
        self,
        hierarchy: AMRHierarchy,
        *,
        reluctivity: Coefficient = 1.0,
        beta: Coefficient = 0.0,
        faces: Mapping[str, str] | None = None,
        mass: str = "lumped",
        gauge: float = 0.0,
        solver: str = "lu",
        edges: CompositeEdges | None = None,
        rtol: float = 1.0e-10,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        """Assemble the operator and set up its solver."""
        if hierarchy.ndim not in (2, 3):
            raise ValueError("the curl-curl operator is 2D or 3D")
        if mass not in ("lumped", "consistent"):
            raise ValueError(f"mass must be 'lumped' or 'consistent', got {mass!r}")
        if solver not in SOLVERS:
            raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}")
        known = faces_of(hierarchy.ndim)
        unknown = set(faces or ()) - set(known)
        if unknown:
            raise ValueError(f"unknown faces {sorted(unknown)}; expected {list(known)}")
        for face, condition in (faces or {}).items():
            if condition not in CONDITIONS:
                raise ValueError(
                    f"{face}: the condition must be one of {CONDITIONS}, got {condition!r}"
                )
            if hierarchy.periodic[known[face][0]]:
                raise ValueError(f"{face} is on a periodic axis: it takes no condition")
        self.conditions = {face: (faces or {}).get(face, "electric") for face in known}
        self.edges = (
            CompositeEdges(CompositeNodes(hierarchy)) if edges is None else edges
        )
        self.nodes = self.edges.nodes
        if self.nodes.face_pieces("inner")[0].size:
            raise NotImplementedError(
                "the curl-curl operator needs a domain that fills its box"
            )
        self.faces = CompositeFaces(self.edges)
        lumped = mass == "lumped"
        self.mass = mass
        nu = self._cells(reluctivity)
        beta_cells = self._cells(beta)
        if np.iscomplexobj(nu) or np.any(np.real(nu) <= 0):
            raise ValueError("the reluctivity must be real and positive")
        edges, nodes = self.edges, self.nodes
        if nodes.ndim == 3:
            curl = self.faces.curl
            curl_mass = self.faces.mass(lumped, weight=nu)
        else:
            curl = self.faces.cell_curl()
            area = np.prod(nodes.cell_size, axis=1)
            curl_mass = sparse.diags(nu / area)
        self.curl = curl
        operator = curl.T @ curl_mass @ curl
        self._edge_mass = edges.mass(lumped)
        self._beta_zero = not np.any(beta_cells)
        if not self._beta_zero:
            operator = operator + edges.mass(lumped, weight=beta_cells)
        if gauge:
            size = np.prod(nodes.cell_size, axis=1) ** (1.0 / nodes.ndim)
            operator = operator + edges.mass(lumped, weight=gauge * nu / size**2)
        reduction = edges.reduction
        self.fixed = self._fixed_edges()
        self.free = np.nonzero(~self.fixed & ~edges.hanging)[0]
        full = sparse.csr_matrix(reduction.T @ operator @ reduction)
        self.matrix = full[self.free][:, self.free].tocsr()
        real = not (np.iscomplexobj(beta_cells) and np.any(np.imag(beta_cells)))
        if solver == "auto":
            solver = "ams" if real and has_ams() else "lu"
        if solver == "ams":
            if not real:
                raise ValueError("AMS takes real coefficients; use solver='lu'")
            if not has_ams():
                raise ValueError("solver='ams' needs petsc4py built with hypre")
        elif self._beta_zero and not gauge:
            raise ValueError(
                "without beta the curl-curl operator is singular: give a gauge > 0 "
                "or use solver='ams'"
            )
        self.solver = solver
        self.iterations = 0
        if solver == "lu":
            self._lu = scipy.sparse.linalg.splu(sparse.csc_matrix(self.matrix))
        else:
            # the interior nodes: their gradients are the null space of the curl term
            interior = np.ones(nodes.num_nodes, dtype=bool)
            interior[nodes.hanging] = False
            for face, (axis, _) in faces_of(nodes.ndim).items():
                if not nodes._periodic[axis] and self.conditions[face] == "electric":
                    interior[nodes.face_nodes(face)] = False
            gradient = sparse.csr_matrix(edges.gradient @ nodes.reduction)
            discrete = gradient[self.free][:, np.flatnonzero(interior)]
            constants = [
                (gradient @ nodes.coordinates[:, axis])[self.free]
                for axis in range(nodes.ndim)
            ]
            self._ams = AmsSolver(
                self.matrix,
                discrete,
                constants,
                hierarchy.comm,
                singular=self._beta_zero and not gauge,
                rtol=rtol,
                options=options,
            )

    # Solving -------------------------------------------------------------------#
    def load(self, current: np.ndarray) -> np.ndarray:
        """The load ``M1 j`` of a current with edge values ``j`` (line integrals of ``J``)."""
        return self._edge_mass @ np.asarray(current)

    @sp.profile("patchwork.curl_curl_solve", tags=PROFILE_TAGS)
    def solve(self, load: np.ndarray) -> np.ndarray:
        """The potential on all edges for a load ``(n1,)``; hanging edges filled.

        ``load`` is the current integrated against the edge functions,
        ``integral of J . w_e``: what `CompositeEdges.deposit` and
        `CompositeEdges.deposit_paths` give for markers, or `load` of edge
        values of ``J``.
        """
        load = np.asarray(load)
        edges = self.edges
        if load.shape != (edges.num_edges,):
            raise ValueError(f"expected a load on the {edges.num_edges} edges")
        right_hand_side = (edges.reduction.T @ load)[self.free]
        if self.solver == "lu":
            values = self._lu.solve(right_hand_side.astype(self._lu.U.dtype))
        else:
            values = self._ams.solve(right_hand_side)
            self.iterations = self._ams.iterations
        potential = np.zeros(edges.num_edges, dtype=values.dtype)
        potential[self.free] = values
        return edges.reduction @ potential

    def flux(self, potential: np.ndarray) -> np.ndarray:
        """The curl of an edge field: fluxes through the faces (3D) or per leaf cell (2D)."""
        result = self.curl @ potential
        if self.nodes.ndim == 3:
            result = self.faces.reduction @ result
        return result
