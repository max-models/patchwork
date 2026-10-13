"""Patchwork: block-structured adaptive meshes backed by mpiarray."""

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
from patchwork.regrid import cluster, function_tags, gradient_tags

__version__ = "0.1.0"

__all__ = [
    "AMRHierarchy",
    "CompositeNodes",
    "Dirichlet",
    "EmbeddedDirichlet",
    "IndexBox",
    "Neumann",
    "NodalPoisson",
    "Patch",
    "__version__",
    "cluster",
    "function_tags",
    "gradient_tags",
    "plot_patches",
    "solve_nodal_poisson",
    "solve_poisson",
]
