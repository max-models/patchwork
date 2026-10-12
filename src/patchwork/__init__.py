"""Patchwork: block-structured adaptive meshes backed by mpiarray."""

from patchwork.mesh import AMRHierarchy, IndexBox, Patch
from patchwork.plotting import plot_patches
from patchwork.poisson import solve_poisson

__version__ = "0.1.0"

__all__ = ["AMRHierarchy", "IndexBox", "Patch", "__version__", "plot_patches", "solve_poisson"]
