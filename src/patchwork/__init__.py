"""Patchwork: block-structured adaptive meshes backed by mpiarray."""

import scope_profiler as sp

from patchwork._profiling import PROFILE_TAGS
from patchwork.backends import (
    AutoBackend,
    Backend,
    FFTBackend,
    LinearSystem,
    LUBackend,
    PetscBackend,
)
from patchwork.cells import CellOperator, CompositeCells, EmbeddedFloating, cell_body
from patchwork.distributed import PetscSystem
from patchwork.mesh import AMRHierarchy, IndexBox, Patch
from patchwork.nodal import (
    CompositeNodes,
    Dirichlet,
    EmbeddedDirichlet,
    Neumann,
    NodalPoisson,
    solve_nodal_poisson,
)
from patchwork.plotting import plot_patches
from patchwork.poisson import solve_poisson
from patchwork.reflux import FluxRegister
from patchwork.regrid import cluster, function_tags, gradient_tags

# patchwork's profiling regions are off unless an application enables the tag
sp.register_opt_in_tag(*PROFILE_TAGS)

__version__ = "0.2.0"

__all__ = [
    "AMRHierarchy",
    "AutoBackend",
    "Backend",
    "CellOperator",
    "CompositeCells",
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
