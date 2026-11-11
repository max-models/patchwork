"""The solver layer: operators assembled on the spaces and their linear solves.

`nodal_poisson` (`NodalPoisson` on the node space), `cell_operator`
(`CellOperator` on the cell space), `curl_curl` (`CurlCurl` on the edge space,
with hypre's AMS), `backends` (the linear solvers behind them:
LU, PETSc, FFT), `multigrid` (geometric multigrid over the patches), `distributed` (the PETSc system distributed by rows) and
`poisson` (the legacy cell-centred Poisson solve).
"""

from patchwork.solvers.backends import (
    AutoBackend,
    Backend,
    FFTBackend,
    LinearSystem,
    LUBackend,
    PetscBackend,
)
from patchwork.solvers.cell_operator import CellOperator
from patchwork.solvers.curl_curl import CurlCurl
from patchwork.solvers.distributed import PetscSystem
from patchwork.solvers.multigrid import CompositeMultigrid, MultigridBackend
from patchwork.solvers.nodal_poisson import NodalPoisson, solve_nodal_poisson
from patchwork.solvers.poisson import solve_poisson

__all__ = [
    "AutoBackend",
    "Backend",
    "CellOperator",
    "CompositeMultigrid",
    "CurlCurl",
    "FFTBackend",
    "LUBackend",
    "LinearSystem",
    "MultigridBackend",
    "NodalPoisson",
    "PetscBackend",
    "PetscSystem",
    "solve_nodal_poisson",
    "solve_poisson",
]
