"""Patchwork: block-structured adaptive meshes backed by mpiarray.

Three layers: `patchwork.mesh` (the hierarchy, halos, regridding, refluxing),
`patchwork.spaces` (the lowest-order de Rham complex on it: nodes, edges, cells)
and `patchwork.solvers` (operators on the spaces and their backends). The
public names of all three are also exported here.
"""

import scope_profiler as sp

from patchwork._profiling import PROFILE_TAGS
from patchwork.mesh import (
    AMRHierarchy,
    FluxRegister,
    IndexBox,
    Patch,
    cluster,
    function_tags,
    gradient_tags,
    plot_patches,
)
from patchwork.solvers import (
    AutoBackend,
    Backend,
    CellOperator,
    CurlCurl,
    FFTBackend,
    LinearSystem,
    LUBackend,
    NodalPoisson,
    PetscBackend,
    PetscSystem,
    solve_nodal_poisson,
    solve_poisson,
)
from patchwork.spaces import (
    CompositeCells,
    CompositeEdges,
    CompositeFaces,
    CompositeNodes,
    Dirichlet,
    EmbeddedDirichlet,
    EmbeddedFloating,
    Neumann,
    cell_body,
)

# patchwork's profiling regions are off unless an application enables the tag
sp.register_opt_in_tag(*PROFILE_TAGS)

__version__ = "0.2.0"

__all__ = [
    "AMRHierarchy",
    "AutoBackend",
    "Backend",
    "CellOperator",
    "CurlCurl",
    "CompositeCells",
    "CompositeEdges",
    "CompositeFaces",
    "CompositeNodes",
    "Dirichlet",
    "EmbeddedDirichlet",
    "EmbeddedFloating",
    "FFTBackend",
    "FluxRegister",
    "IndexBox",
    "LUBackend",
    "LinearSystem",
    "Neumann",
    "NodalPoisson",
    "Patch",
    "PetscBackend",
    "PetscSystem",
    "__version__",
    "cell_body",
    "cluster",
    "function_tags",
    "gradient_tags",
    "plot_patches",
    "solve_nodal_poisson",
    "solve_poisson",
]
