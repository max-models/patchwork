"""The space layer: the lowest-order de Rham complex on a patch hierarchy.

`nodes` (H¹, Q1: `CompositeNodes`), `edges` (H(curl), Whitney edges:
`CompositeEdges`), `faces` (H(div), Raviart–Thomas faces: `CompositeFaces`)
and `cells` (L², Q0: `CompositeCells`), each numbering its
unknowns on the leaf cells with the constraints across coarse–fine edges, and
`boundaries`, the conditions on the domain's faces and the bodies inside it.
The spaces build on `patchwork.mesh`; the solvers (`patchwork.solvers`) on them.
"""

from patchwork.spaces.boundaries import (
    Dirichlet,
    EmbeddedDirichlet,
    EmbeddedFloating,
    Neumann,
)
from patchwork.spaces.cells import CompositeCells, cell_body
from patchwork.spaces.edges import CompositeEdges
from patchwork.spaces.faces import CompositeFaces
from patchwork.spaces.nodes import CompositeNodes

__all__ = [
    "CompositeCells",
    "CompositeEdges",
    "CompositeFaces",
    "CompositeNodes",
    "Dirichlet",
    "EmbeddedDirichlet",
    "EmbeddedFloating",
    "Neumann",
    "cell_body",
]
