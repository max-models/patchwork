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
