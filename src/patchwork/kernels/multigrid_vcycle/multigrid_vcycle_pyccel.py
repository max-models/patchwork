"""One multigrid V-cycle with Gauss--Seidel smoothing, from zero.

The Pyccel version (CPU) of the ``multigrid_vcycle`` kernel; ``multigrid_vcycle_numpy.py``
next to this file is the NumPy version, with the same arguments. Loaded by
`patchwork.kernels.multigrid_vcycle`, compiled with Pyccel on first use; called by
`CompositeMultigrid.cycle` with ``smoother="gauss-seidel"``.

The ``levels`` operators, interpolations ``P`` and restrictions ``R = P^T`` are CSR
matrices concatenated level by level, the finest first: level ``l``'s row pointers are
``*_ptr[*_start[l] : *_start[l] + rows + 1]``, holding absolute positions in the
concatenated ``*_idx`` (columns within the level) and ``*_val``. Level ``l``'s vectors are
``[offsets[l], offsets[l + 1])`` of the work arrays ``bs``, ``xs`` and ``rs``.
``coarse`` maps the coarsest right-hand side to its solution (a dense inverse). Each
level is smoothed by ``sweeps`` forward Gauss--Seidel sweeps before the coarse
correction and as many backward sweeps after it (a symmetric cycle). ``x`` gets the
cycle applied to ``b``.
"""


def _gauss_seidel(
    ptr: "int[:]",
    idx: "int[:]",
    val: "float[:]",
    start: int,
    offset: int,
    rows: int,
    xs: "float[:]",
    bs: "float[:]",
    forward: bool,
) -> None:
    """One Gauss--Seidel sweep on a level, in place."""
    for step in range(rows):
        row = step if forward else rows - 1 - step
        total = bs[offset + row]
        diagonal = 0.0
        for k in range(ptr[start + row], ptr[start + row + 1]):
            column = idx[k]
            if column == row:
                diagonal += val[k]
            else:
                total -= val[k] * xs[offset + column]
        xs[offset + row] = total / diagonal


def multigrid_vcycle(
    levels: int,
    offsets: "int[:]",
    a_ptr: "int[:]",
    a_start: "int[:]",
    a_idx: "int[:]",
    a_val: "float[:]",
    p_ptr: "int[:]",
    p_start: "int[:]",
    p_idx: "int[:]",
    p_val: "float[:]",
    r_ptr: "int[:]",
    r_start: "int[:]",
    r_idx: "int[:]",
    r_val: "float[:]",
    coarse: "float[:,:]",
    sweeps: int,
    b: "float[:]",
    x: "float[:]",
    bs: "float[:]",
    xs: "float[:]",
    rs: "float[:]",
) -> None:
    """Apply one V-cycle to ``b`` into ``x`` (see the module)."""
    fine = offsets[1] - offsets[0]
    for i in range(fine):
        bs[i] = b[i]
    for level in range(levels - 1):
        offset = offsets[level]
        rows = offsets[level + 1] - offset
        start = a_start[level]
        for i in range(rows):
            xs[offset + i] = 0.0
        for _ in range(sweeps):
            _gauss_seidel(a_ptr, a_idx, a_val, start, offset, rows, xs, bs, True)
        for i in range(rows):
            total = bs[offset + i]
            for k in range(a_ptr[start + i], a_ptr[start + i + 1]):
                total -= a_val[k] * xs[offset + a_idx[k]]
            rs[offset + i] = total
        below = offsets[level + 1]
        coarse_rows = offsets[level + 2] - below
        rstart = r_start[level]
        for i in range(coarse_rows):
            total = 0.0
            for k in range(r_ptr[rstart + i], r_ptr[rstart + i + 1]):
                total += r_val[k] * rs[offset + r_idx[k]]
            bs[below + i] = total
    last = offsets[levels - 1]
    size = offsets[levels] - last
    for i in range(size):
        total = 0.0
        for j in range(size):
            total += coarse[i, j] * bs[last + j]
        xs[last + i] = total
    for level in range(levels - 2, -1, -1):
        offset = offsets[level]
        rows = offsets[level + 1] - offset
        below = offsets[level + 1]
        pstart = p_start[level]
        for i in range(rows):
            total = 0.0
            for k in range(p_ptr[pstart + i], p_ptr[pstart + i + 1]):
                total += p_val[k] * xs[below + p_idx[k]]
            xs[offset + i] += total
        start = a_start[level]
        for _ in range(sweeps):
            _gauss_seidel(a_ptr, a_idx, a_val, start, offset, rows, xs, bs, False)
    for i in range(fine):
        x[i] = xs[i]
