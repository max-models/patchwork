r"""The mass matrices of the spaces, cell by cell: lumped or consistent.

On a tensor-product cell every lowest-order basis function is a product of 1D
profiles: linear (the hats ``1 - xi`` and ``xi``) along the axes on which it
lives at both ends, constant along an edge's own axis. Its mass block is then a
product of 1D blocks, one per axis,

$$
m_\mathrm{consistent} = \begin{pmatrix} 1/3 & 1/6 \\ 1/6 & 1/3 \end{pmatrix},
\qquad
m_\mathrm{lumped} = \begin{pmatrix} 1/2 & 0 \\ 0 & 1/2 \end{pmatrix},
$$

the exact integral of the products of the two hats, or the vertex (trapezoidal)
quadrature of it. The lumped blocks are diagonal: on the nodes they give the
control volumes, on the edges the conductances of the five-point (seven-point in
3D) stencil, so ``G^T M1 G`` with lumped ``M1`` is the finite-difference
Laplacian and with consistent ``M1`` the Q1 finite-element stiffness. The
spaces and the solvers take their blocks from here, so a finite-difference and a
finite-element version of an operator differ only in ``lumped``.
"""

from __future__ import annotations

import numpy as np
from methodcard import method

#: The 1D mass blocks of two linear hats on a unit interval.
_CONSISTENT = np.array([[1.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 1.0 / 3.0]])
_LUMPED = np.array([[0.5, 0.0], [0.0, 0.5]])


def _hats(lumped: bool, bits_a: np.ndarray, bits_b: np.ndarray) -> np.ndarray:
    """The 1D block's entries between the ends ``bits_a`` and ``bits_b`` (0 or 1)."""
    return (_LUMPED if lumped else _CONSISTENT)[bits_a, bits_b]


def corner_bits(ndim: int) -> np.ndarray:
    """The corners of a cell as bits, ``(2**ndim, ndim)``, the first axis fastest."""
    return np.array(
        [[(k >> axis) & 1 for axis in range(ndim)] for k in range(2**ndim)],
        dtype=np.int64,
    )


def edge_corners(ndim: int, axis: int) -> list[tuple[int, int]]:
    """The edges of a cell along ``axis``, as their (start, end) corners."""
    return [(k, k | (1 << axis)) for k in range(2**ndim) if not (k >> axis) & 1]


@method(
    equations={
        "nodes": r"M_{0,K} = |K| \bigotimes_{i=1}^{d} m",
        "lumped_nodes": r"M_{0,K} = \frac{|K|}{2^d}\,I",
    },
    assumptions=("a tensor-product (box) cell; Cartesian measure",),
)
def node_mass_blocks(size: np.ndarray, lumped: bool) -> np.ndarray:
    """The node mass block of every cell, ``(m, 2**d, 2**d)``, in corner order.

    ``size`` holds the cells' edge lengths, ``(m, d)``. ``u^T M u`` over a cell
    is the integral of the square of the (bi-, tri-)linear interpolant of its
    corner values ``u``; lumped, ``|K| / 2**d`` on the diagonal.
    """
    size = np.asarray(size, dtype=float)
    bits = corner_bits(size.shape[1])
    factor = np.ones((bits.shape[0], bits.shape[0]))
    for axis in range(size.shape[1]):
        factor = factor * _hats(lumped, bits[:, None, axis], bits[None, :, axis])
    return np.prod(size, axis=1)[:, None, None] * factor


@method(
    equations={
        "edges": r"M_{1,K}^{(a)} = \frac{\prod_{i \ne a} h_i}{h_a} \bigotimes_{i \ne a} m",
        "lumped_edges": r"M_{1,K}^{(a)} = \frac{\prod_{i \ne a} h_i}{2^{d-1} h_a}\,I",
    },
    assumptions=(
        "an edge value is the line integral of the field along it (Whitney edges)",
        "a tensor-product (box) cell; Cartesian measure",
    ),
)
def edge_mass_blocks(size: np.ndarray, axis: int, lumped: bool) -> np.ndarray:
    """The mass block of every cell's edges along ``axis``, ``(m, 2**(d-1), 2**(d-1))``.

    The edges are in the order of `edge_corners`. ``e^T M e`` over a cell is
    the integral of the square of the edge interpolant's component along
    ``axis``; lumped, ``prod(h_perp) / (2**(d-1) h_axis)`` on the diagonal, the
    conductance of the finite-difference stencil.
    """
    size = np.asarray(size, dtype=float)
    ndim = size.shape[1]
    starts = np.array([start for start, _ in edge_corners(ndim, axis)])
    bits = corner_bits(ndim)[starts]
    factor = np.ones((starts.size, starts.size))
    for other in range(ndim):
        if other != axis:
            factor = factor * _hats(lumped, bits[:, None, other], bits[None, :, other])
    scale = np.prod(np.delete(size, axis, axis=1), axis=1) / size[:, axis]
    return scale[:, None, None] * factor


@method(
    equations={
        "cross": r"M_{1,K}^{(a,b)} = \frac{|K|}{h_a h_b}\,\frac14 \bigotimes_{i \ne a, b} m \quad (a \ne b)",
    },
    assumptions=(
        "an edge value is the line integral of the field along it (Whitney edges)",
        r"the same for the lumped and the consistent mass along $a$ and $b$: the vertex quadrature of a product of a hat and a constant is exact",
        "a tensor-product (box) cell; Cartesian measure",
    ),
)
def edge_cross_blocks(
    size: np.ndarray, axis_a: int, axis_b: int, lumped: bool
) -> np.ndarray:
    """The coupling of every cell's edges along two different axes, ``(m, 2**(d-1), 2**(d-1))``.

    Rows are the edges along ``axis_a``, columns those along ``axis_b``, each
    in the order of `edge_corners`. ``e_a^T B e_b`` over a cell is the
    integral of the product of the edge interpolant's components along the two
    axes: what an off-diagonal coefficient ``K_ab`` multiplies in
    ``int E . K E``. The edge along ``a`` is a hat across ``b`` and constant
    along ``a``, the edge along ``b`` the reverse, so each of the two axes
    gives a factor ``1/2``; a third axis gives the 1D block ``m`` (lumped or
    consistent).
    """
    if axis_a == axis_b:
        raise ValueError("the cross block couples two different axes")
    size = np.asarray(size, dtype=float)
    ndim = size.shape[1]
    bits = corner_bits(ndim)
    bits_a = bits[[start for start, _ in edge_corners(ndim, axis_a)]]
    bits_b = bits[[start for start, _ in edge_corners(ndim, axis_b)]]
    factor = np.full((bits_a.shape[0], bits_b.shape[0]), 0.25)
    for other in range(ndim):
        if other not in (axis_a, axis_b):
            factor = factor * _hats(
                lumped, bits_a[:, None, other], bits_b[None, :, other]
            )
    scale = np.prod(size, axis=1) / (size[:, axis_a] * size[:, axis_b])
    return scale[:, None, None] * factor


@method(
    equations={
        "faces": r"M_{2,K}^{(a)} = \frac{h_a}{\prod_{i \ne a} h_i}\, m",
        "lumped_faces": r"M_{2,K}^{(a)} = \frac{h_a}{2 \prod_{i \ne a} h_i}\,I",
    },
    assumptions=(
        "a face value is the flux through it (Raviart–Thomas faces): the normal component varies linearly between the cell's two faces normal to the axis",
        "a tensor-product (box) cell; Cartesian measure",
    ),
)
def face_mass_blocks(size: np.ndarray, axis: int, lumped: bool) -> np.ndarray:
    """The mass block of every cell's two faces normal to ``axis``, ``(m, 2, 2)``.

    Lower face first. ``f^T M f`` over a cell is the integral of the square of
    the face interpolant's component along ``axis``; lumped,
    ``h_axis / (2 prod(h_perp))`` on the diagonal.
    """
    size = np.asarray(size, dtype=float)
    scale = size[:, axis] / np.prod(np.delete(size, axis, axis=1), axis=1)
    return scale[:, None, None] * (_LUMPED if lumped else _CONSISTENT)[None]


def scatter_blocks(
    blocks: np.ndarray, rows: np.ndarray, columns: np.ndarray, shape: tuple[int, int]
):
    """Sum per-cell blocks ``(m, a, b)`` into a sparse matrix.

    ``rows`` ``(m, a)`` and ``columns`` ``(m, b)`` are the global numbers of
    the blocks' rows and columns.
    """
    import scipy.sparse as sparse

    count, a, b = blocks.shape
    row = np.broadcast_to(rows[:, :, None], (count, a, b)).reshape(-1)
    column = np.broadcast_to(columns[:, None, :], (count, a, b)).reshape(-1)
    keep = blocks.reshape(-1) != 0.0
    return sparse.csr_matrix(
        sparse.coo_matrix(
            (blocks.reshape(-1)[keep], (row[keep], column[keep])), shape=shape
        )
    )


def _cell_gradient(ndim: int, axis: int) -> np.ndarray:
    """The differences of a cell's corners along its edges on ``axis``, ``(2**(d-1), 2**d)``."""
    pairs = edge_corners(ndim, axis)
    result = np.zeros((len(pairs), 2**ndim))
    for k, (start, end) in enumerate(pairs):
        result[k, start] = -1.0
        result[k, end] = 1.0
    return result


@method(
    equations={
        "stiffness": r"S_K = \sum_{a, b} D_a^\top \big(K_{ab}\, M_{1,K}^{(a,b)}\big) D_b, \qquad M_{1,K}^{(a,a)} = M_{1,K}^{(a)}",
        "edges": r"M_{1,K}^{(a,b)} \to \mathrm{diag}(\sqrt{\kappa_a})\, M_{1,K}^{(a,b)}\, \mathrm{diag}(\sqrt{\kappa_b}) \quad \text{(a coefficient per edge)}",
        "energy": r"u^\top S_K u = \int_K \nabla u \cdot \mathsf{K}\, \nabla u \, dA \quad \text{(consistent mass, constant } \mathsf{K}\text{)}",
    },
    assumptions=(
        r"$D_a$ the corner differences along the cell's edges on axis $a$ (the local discrete gradient)",
        r"a symmetric coefficient $\mathsf{K}$ per cell; with the lumped mass the vertex quadrature of $\int \nabla u \cdot \mathsf{K} \nabla u$, positive for a positive definite $\mathsf{K}$",
        r"a coefficient per edge $\kappa \geq 0$ scales the edge's rows and columns by $\sqrt{\kappa}$: on the lumped (diagonal) blocks the edge's conductance times $\kappa$",
    ),
    components=(
        "patchwork.spaces.mass.edge_mass_blocks",
        "patchwork.spaces.mass.edge_cross_blocks",
    ),
)
def stiffness_blocks(
    size: np.ndarray,
    lumped: bool,
    tensor: np.ndarray | None = None,
    edge_scale: np.ndarray | None = None,
) -> np.ndarray:
    """The nodal stiffness block ``G^T M1(K) G`` of every cell, ``(m, 2**d, 2**d)``.

    ``size`` holds the cells' edge lengths ``(m, d)``. ``tensor`` is a
    symmetric coefficient per cell ``(m, d, d)`` (none: the identity), which
    weights the edge mass blocks along each axis and couples the edges of
    different axes (`edge_cross_blocks`). ``edge_scale`` ``(m, d 2**(d-1))``
    holds the square roots of a coefficient per edge of each cell, the edges
    along the first axis first, each axis's in the order of `edge_corners`.
    With neither, the lumped blocks give the five-point (seven-point in 3D)
    stencil and the consistent ones the Q1 stiffness.
    """
    size = np.asarray(size, dtype=float)
    count, ndim = size.shape
    per_axis = 2 ** (ndim - 1)
    gradients = [_cell_gradient(ndim, axis) for axis in range(ndim)]
    result = np.zeros((count, 2**ndim, 2**ndim))
    for a in range(ndim):
        for b in range(ndim):
            if a == b:
                block = edge_mass_blocks(size, a, lumped)
            elif tensor is None:
                continue
            else:
                block = edge_cross_blocks(size, a, b, lumped)
            if tensor is not None:
                block = block * tensor[:, a, b, None, None]
            if edge_scale is not None:
                block = (
                    block
                    * edge_scale[:, a * per_axis : (a + 1) * per_axis, None]
                    * edge_scale[:, None, b * per_axis : (b + 1) * per_axis]
                )
            result += np.einsum("ki,mkl,lj->mij", gradients[a], block, gradients[b])
    return result
