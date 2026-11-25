"""The NumPy version of the ``scatter_blocks`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``scatter_blocks_pyccel.py``, vectorized over the
markers with mpifield's NumPy ``spline_cell``.
"""

from __future__ import annotations

import itertools

import numpy as np
from mpifield.kernels.spline_cell import spline_cell


def scatter_blocks(
    rel: np.ndarray,
    count: np.ndarray,
    stride: np.ndarray,
    base: np.ndarray,
    values: np.ndarray,
    degree: int,
    buffer: np.ndarray,
) -> None:
    """Add the values of the markers to ``buffer``."""
    axes = []
    for a in range(rel.shape[0]):
        nodes, weights = spline_cell(rel[a], count[a], False, degree)
        axes.append([(k * stride[a], w) for k, w in zip(nodes, weights, strict=True)])
    for parts in itertools.product(*axes):
        where = base.copy()
        weight = values.copy()
        for at, w in parts:
            where += at
            weight = weight * w
        buffer += np.bincount(where, weights=weight, minlength=buffer.size)
