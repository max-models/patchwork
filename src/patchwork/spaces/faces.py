r"""The lowest-order face space of a 2D or 3D hierarchy: Raviart–Thomas faces.

`CompositeFaces` holds the faces of the leaf cells of
`CompositeNodes` (the sides of a cell: edges in 2D, faces in 3D). A field
``f`` on them is the flux of a vector field through every face, and inside a
leaf cell the field is the lowest-order Raviart–Thomas interpolant: the
component normal to an axis varies linearly along that axis between the cell's
two faces normal to it, and is constant across each face. Then

- the divergence `CompositeFaces.divergence` is the incidence of faces and
  leaf cells (the net outward flux of every cell), exact for any field;
- the curl `CompositeFaces.curl` (3D) maps edge values (line integrals) to
  face values (fluxes): the circulation around every face, Stokes' theorem on
  the lattice, so ``C G = 0`` and ``D C = 0`` hold exactly;
- a coarse face on a patch face is ``r**(ndim-1)`` fine faces, which hang: the
  flux of a field of constant normal component through a piece is its share of
  the coarse flux (`CompositeFaces.reduction`), and the constraints commute with
  the curl (``C T1 = T2 C T1``);
- the face mass matrix (`CompositeFaces.mass`) is ``f^T M2 f``, the integral of
  ``|B|^2``, lumped or consistent from the blocks of `patchwork.spaces.mass`.

In 2D a face is a cell edge and its value a flux through it: the edge space
rotated by a quarter turn. The 2D curl of an edge field is a value per leaf
cell (`CompositeFaces.curl` needs 3D; see `cell_curl`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sparse
import scope_profiler as sp
from methodcard import method

from patchwork._profiling import PROFILE_TAGS
from patchwork.spaces.edges import _weighted
from patchwork.spaces.mass import (
    corner_bits,
    edge_corners,
    face_mass_blocks,
    scatter_blocks,
)

if TYPE_CHECKING:
    from patchwork.spaces.edges import CompositeEdges


def _cyclic(axis: int) -> tuple[int, int]:
    """The two tangential axes of a face normal to ``axis``, in right-handed order."""
    return (axis + 1) % 3, (axis + 2) % 3


@method(
    equations={
        "divergence": r"(D f)_K = \sum_{a} \big(f_{K, a, \mathrm{upper}} - f_{K, a, \mathrm{lower}}\big)",
        "curl": r"(C e)_F = e_{b}(\text{at lower } c) + e_{c}(\text{at upper } b) - e_{b}(\text{at upper } c) - e_{c}(\text{at lower } b), \quad (a, b, c) \text{ cyclic}",
        "hanging_faces": r"(T_2 f)_g = \Big(\frac{s_g}{s_F}\Big)^{d-1} f_F \quad \text{for a fine face } g \text{ of span } s_g \text{ in a coarse face } F",
        "exact": r"C\,G = 0, \qquad D\,C = 0, \qquad C\,T_1 = T_2\,C\,T_1",
        "whitney": r"B_a = \frac{(1 - \xi_a)\, f_\mathrm{lower} + \xi_a\, f_\mathrm{upper}}{\prod_{i \ne a} h_i}",
    },
    assumptions=(
        "2D and 3D Cartesian; faces addressed on the finest lattice like the edges",
        "a coarse face and the fine faces that tile it are different faces; the fine ones hang",
        "cells are the leaf cells of `CompositeNodes`, in the order of its `cell_corners`",
    ),
    references=(
        "[Raviart and Thomas, Lecture Notes in Math. 606, 292 (1977)](https://doi.org/10.1007/BFb0064470)",
        "[Bossavit, IEE Proc. A 135, 493 (1988)](https://doi.org/10.1049/ip-a-1.1988.0077)",
    ),
    components=("patchwork.spaces.edges.CompositeEdges",),
)
class CompositeFaces:
    """The faces of the leaf cells of a 2D or 3D hierarchy, and the operators on them.

    A face is its normal axis, the lattice point of its lower corner and its
    span. A coarse face and the fine faces that tile it are different faces;
    the fine ones hang.

    Attributes:
        edges: The `CompositeEdges` of the hierarchy.
        nodes: Their `CompositeNodes`.
        num_faces: Number of faces.
        normal: The axis normal to every face, ``(n2,)``.
        lattice: Finest-lattice coordinates of the lower corner of every face,
            ``(n2, ndim)``.
        span: Side length of every face on the finest lattice.
        area: Physical measure of every face (a length in 2D).
        cell_faces: The faces of every leaf cell, ``(m, 2 ndim)``: for each axis
            the lower face, then the upper one.
        hanging: Whether a face hangs (lies in a coarser face).
        reduction: The constraint ``T2``: ``T2 @ f`` fills the hanging faces of
            ``f`` from the others (which it leaves unchanged).
        divergence: The discrete divergence ``D``, ``(m, n2)``: the net outward
            flux of every leaf cell.
    """

    # Construction -------------------------------------------------------------#
    def _keys(self, normal, lower, span) -> np.ndarray:
        """One integer per face: normal, lower corner (as a node key) and span."""
        ndim = self.nodes.ndim
        node_key = self.nodes._key(*(lower[..., axis] for axis in range(ndim)))
        return (node_key * ndim + normal) * self._max_span + span

    def _find(self, normal, lower, span) -> np.ndarray:
        """Face numbers of faces, -1 where there is none."""
        keys = self._keys(normal, lower, span)
        position = np.minimum(np.searchsorted(self._sorted, keys), self.num_faces - 1)
        return np.where(self._sorted[position] == keys, position, -1)

    def _find_hanging_faces(self) -> None:
        """Find the faces inside coarser cells' faces and their constraints."""
        nodes = self.nodes
        ndim = nodes.ndim
        ratio = nodes.hierarchy.refinement_ratio
        slaves, masters, weights, spans = [], [], [], []
        for coarse in np.unique(nodes.cell_span):
            coarse = int(coarse)
            cells = np.nonzero(nodes.cell_span == coarse)[0]
            fine = coarse // ratio
            while fine >= 1:
                count = coarse // fine
                share = (fine / coarse) ** (ndim - 1)
                for axis in range(ndim):
                    for side in (0, 1):
                        master = self.cell_faces[cells, 2 * axis + side]
                        grids = [
                            np.array([side * coarse])
                            if b == axis
                            else np.arange(count) * fine
                            for b in range(ndim)
                        ]
                        offsets = np.stack(
                            np.meshgrid(*grids, indexing="ij"), axis=-1
                        ).reshape(-1, ndim)
                        for offset in offsets:
                            face = self._find(
                                np.full(cells.size, axis),
                                nodes.cell_lower[cells] + offset,
                                fine,
                            )
                            found = face >= 0
                            slaves.append(face[found])
                            masters.append(master[found])
                            weights.append(np.full(int(found.sum()), share))
                            spans.append(np.full(int(found.sum()), coarse))
                fine //= ratio
        self.hanging = np.zeros(self.num_faces, dtype=bool)
        identity = sparse.csr_matrix(sparse.identity(self.num_faces))
        if not slaves or not sum(part.size for part in slaves):
            self.reduction = identity
            return
        slave = np.concatenate(slaves)
        master = np.concatenate(masters)
        weight = np.concatenate(weights)
        span = np.concatenate(spans)
        # one constraint per face, from the coarsest face it lies in
        order = np.lexsort((-span, slave))
        slave, master, weight = slave[order], master[order], weight[order]
        first = np.concatenate(([True], slave[1:] != slave[:-1]))
        slave, master, weight = slave[first], master[first], weight[first]
        self.hanging[slave] = True
        free = np.nonzero(~self.hanging)[0]
        reduction = sparse.csr_matrix(
            sparse.coo_matrix(
                (
                    np.concatenate((np.ones(free.size), weight)),
                    (np.concatenate((free, slave)), np.concatenate((free, master))),
                ),
                shape=(self.num_faces, self.num_faces),
            )
        )
        # a coarse face may itself lie in a still coarser one
        for _ in range(nodes._finest + 1):
            if not reduction[:, slave].nnz:
                break
            reduction = sparse.csr_matrix(reduction @ reduction)
            reduction.eliminate_zeros()
        else:
            raise RuntimeError("the hanging-face constraints do not close")
        self.reduction = reduction

    @sp.profile("patchwork.composite_faces", tags=PROFILE_TAGS)
    def __init__(self, edges: CompositeEdges) -> None:
        """Number the faces of the leaf cells next to ``edges`` and find the hanging ones."""
        nodes = edges.nodes
        self.edges = edges
        self.nodes = nodes
        ndim = nodes.ndim
        self._max_span = int(nodes.cell_span.max()) + 1
        normals, lowers, spans = [], [], []
        for axis in range(ndim):
            for side in (0, 1):
                normals.append(np.full(nodes.cell_span.size, axis))
                lower = nodes.cell_lower.copy()
                lower[:, axis] += side * nodes.cell_span
                lowers.append(lower)
                spans.append(nodes.cell_span)
        normal = np.stack(normals, axis=1)  # (m, 2 ndim)
        lower = np.stack(lowers, axis=1)  # (m, 2 ndim, ndim)
        span = np.stack(spans, axis=1)
        keys = self._keys(normal, lower, span)
        self._sorted, first, inverse = np.unique(
            keys.reshape(-1), return_index=True, return_inverse=True
        )
        self.num_faces = int(self._sorted.size)
        self.cell_faces = inverse.reshape(keys.shape)
        self.normal = normal.reshape(-1)[first]
        self.lattice = lower.reshape(-1, ndim)[first]
        for axis in range(ndim):
            if nodes._periodic[axis]:  # the seam's faces at the lower end
                self.lattice[:, axis] %= nodes._cells[axis]
        self.span = span.reshape(-1)[first]
        sides = self.span[:, None] * nodes._spacing[None, :]
        self.area = np.prod(
            np.where(np.arange(ndim)[None, :] == self.normal[:, None], 1.0, sides),
            axis=1,
        )
        cells = nodes.cell_span.size
        rows = np.repeat(np.arange(cells), 2 * ndim)
        signs = np.tile([-1.0, 1.0] * ndim, cells)
        self.divergence = sparse.csr_matrix(
            sparse.coo_matrix(
                (signs, (rows, self.cell_faces.reshape(-1))),
                shape=(cells, self.num_faces),
            )
        )
        # a periodic axis one cell wide: the cell's two faces are one face
        self.divergence.eliminate_zeros()
        self._find_hanging_faces()

    # Operators -------------------------------------------------------------------#
    @property
    def curl(self) -> sparse.csr_matrix:
        """The discrete curl ``C``, ``(n2, n1)``: the circulation of an edge field around every face (3D).

        Each face takes its leaf cell's four edges around it, counterclockwise
        seen from the side its normal points to. ``C @ G = 0`` for the gradient
        ``G`` of the edges.
        """
        nodes = self.nodes
        if nodes.ndim != 3:
            raise ValueError(
                "the curl of a 2D edge field is a value per leaf cell: use cell_curl"
            )
        bits = corner_bits(3)
        cells = nodes.cell_span.size
        rows, columns, entries = [], [], []
        for axis in range(3):
            b, c = _cyclic(axis)
            for side in (0, 1):
                face = self.cell_faces[:, 2 * axis + side]
                # (edge axis, its position bit on the other tangential axis, sign)
                for along, across, at, sign in (
                    (b, c, 0, 1.0),
                    (c, b, 1, 1.0),
                    (b, c, 1, -1.0),
                    (c, b, 0, -1.0),
                ):
                    starts = [start for start, _ in edge_corners(3, along)]
                    j = next(
                        k
                        for k, start in enumerate(starts)
                        if bits[start, axis] == side and bits[start, across] == at
                    )
                    column = self.edges.cell_edges[:, 4 * along + j]
                    rows.append(face)
                    columns.append(column)
                    entries.append(np.full(cells, sign))
        matrix = sparse.coo_matrix(
            (
                np.concatenate(entries),
                (np.concatenate(rows), np.concatenate(columns)),
            ),
            shape=(self.num_faces, self.edges.num_edges),
        )
        # every face appears in the cells on both sides of it: keep one copy
        matrix = sparse.csr_matrix(matrix)
        matrix.data = np.sign(matrix.data)
        matrix.eliminate_zeros()
        return matrix

    def cell_curl(self) -> sparse.csr_matrix:
        """The 2D curl ``(m, n1)``: the circulation of an edge field around every leaf cell.

        Counterclockwise: the lower x-edge, the right y-edge, minus the upper
        x-edge and the left y-edge. ``cell_curl() @ G = 0``.
        """
        nodes = self.nodes
        if nodes.ndim != 2:
            raise ValueError("cell_curl is the curl of a 2D edge field; in 3D use curl")
        cells = nodes.cell_span.size
        cell_edges = self.edges.cell_edges  # lower x, upper x, left y, right y
        rows = np.repeat(np.arange(cells), 4)
        columns = cell_edges[:, [0, 3, 1, 2]].reshape(-1)
        signs = np.tile([1.0, 1.0, -1.0, -1.0], cells)
        matrix = sparse.csr_matrix(
            sparse.coo_matrix(
                (signs, (rows, columns)), shape=(cells, self.edges.num_edges)
            )
        )
        matrix.eliminate_zeros()
        return matrix

    @method(
        equations={
            "energy": r"f^\top M_2 f = \int |\mathbf{B}|^2\, dV",
            "lumped": r"M_{gg} = \sum_{K \ni g} \frac{h_\perp}{2\, |g|}",
        },
        assumptions=(
            "lumped: the vertex quadrature along the normal, diagonal",
            "consistent: the exact integral of the linear normal profiles",
            r"for a field on hanging faces use $T_2^\top M_2 T_2$",
        ),
        components=("patchwork.spaces.mass.face_mass_blocks",),
    )
    def mass(
        self, lumped: bool = True, weight: np.ndarray | float | None = None
    ) -> sparse.csr_matrix:
        """The face mass matrix ``M2``: ``f^T M2 f`` is the integral of ``|B|^2``.

        Summed over the leaf cells, each with the face interpolant of its own
        faces. For a field on hanging faces, use ``T2^T M2 T2``. ``weight`` is
        a coefficient per leaf cell (a number or an array), e.g. a reluctivity.
        """
        size = self.nodes.cell_size
        shape = (self.num_faces, self.num_faces)
        mass = sparse.csr_matrix(shape)
        for axis in range(self.nodes.ndim):
            faces = self.cell_faces[:, [2 * axis, 2 * axis + 1]]
            mass = mass + scatter_blocks(
                _weighted(face_mass_blocks(size, axis, lumped), weight),
                faces,
                faces,
                shape,
            )
        return sparse.csr_matrix(mass)

    def evaluation(self, points: np.ndarray) -> tuple[sparse.csr_matrix, ...]:
        """The field of a face field at ``points`` ``(ndim, k)``: one ``(k, n2)`` matrix per axis.

        Each point takes the face interpolant of the leaf cell it is in. Apply
        them to face values with their hanging faces filled (``T2 @ f``). A
        point outside the domain gets empty rows.
        """
        nodes = self.nodes
        leaf, fraction, inside = nodes.point_cells(points)
        size = nodes.cell_size[leaf]
        count = leaf.size
        rows = np.repeat(np.arange(count), 2)
        matrices = []
        for axis in range(nodes.ndim):
            area = np.prod(np.delete(size, axis, axis=1), axis=1)
            across = fraction[axis]
            weight = np.column_stack((1.0 - across, across)) / area[:, None]
            weight[~inside] = 0.0
            columns = self.cell_faces[leaf][:, [2 * axis, 2 * axis + 1]]
            matrix = sparse.csr_matrix(
                sparse.coo_matrix(
                    (weight.reshape(-1), (rows, columns.reshape(-1))),
                    shape=(count, self.num_faces),
                )
            )
            matrix.eliminate_zeros()
            matrices.append(matrix)
        return tuple(matrices)
