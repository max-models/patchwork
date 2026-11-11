"""The mesh layer: the patch hierarchy and what moves data between its patches.

`hierarchy` (`AMRHierarchy`, `Patch`, `IndexBox`: patches, fields, marker location
and migration), `halos` (halo exchange and restriction), `regrid` (tagging,
clustering, regridding), `accumulate` (deposit accumulation across patch edges),
`reflux` (flux registers) and `plotting`. It knows nothing of the spaces
(`patchwork.spaces`) or the solvers (`patchwork.solvers`) built on it.
"""

from patchwork.mesh.hierarchy import AMRHierarchy, IndexBox, Patch
from patchwork.mesh.plotting import plot_patches
from patchwork.mesh.reflux import FluxRegister
from patchwork.mesh.regrid import cluster, function_tags, gradient_tags

__all__ = [
    "AMRHierarchy",
    "FluxRegister",
    "IndexBox",
    "Patch",
    "cluster",
    "function_tags",
    "gradient_tags",
    "plot_patches",
]
