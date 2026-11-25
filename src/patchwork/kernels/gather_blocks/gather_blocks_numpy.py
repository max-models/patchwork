"""The NumPy version of the ``gather_blocks`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``gather_blocks_pyccel.py``, vectorized over the
markers with mpifield's NumPy ``spline_cell``.
"""

from __future__ import annotations

import itertools

import numpy as np
from mpifield.kernels.spline_cell import spline_cell


def gather_blocks(
    rel: np.ndarray,
    count: np.ndarray,
    stride: np.ndarray,
    base: np.ndarray,
    degree: int,
    buffer: np.ndarray,
    result: np.ndarray,
) -> None:
    """Write the gathered values of the markers to ``result``."""
    axes = []
    for a in range(rel.shape[0]):
        nodes, weights = spline_cell(rel[a], count[a], False, degree)
        axes.append([(k * stride[a], w) for k, w in zip(nodes, weights, strict=True)])
    total = np.zeros(result.shape)
    for parts in itertools.product(*axes):
        where = base.copy()
        weight = np.ones(result.shape)
        for at, w in parts:
            where += at
            weight = weight * w
        total += weight * buffer[where]
    result[...] = total
