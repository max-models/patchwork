"""One multigrid V-cycle with multicolour Gauss--Seidel smoothing, from zero, with OpenMP.

The Pyccel version (CPU) of the ``multigrid_vcycle`` kernel; ``multigrid_vcycle_numpy.py``
next to this file is the NumPy version, with the same arguments. Loaded by
`patchwork.kernels.multigrid_vcycle`, compiled with Pyccel on first use; called by
`CompositeMultigrid.cycle` with ``smoother="gauss-seidel"``.

The ``levels`` operators, interpolations ``P`` and restrictions ``R = P^T`` are CSR
matrices concatenated level by level, the finest first: level ``l``'s row pointers are
``*_ptr[*_start[l] : *_start[l] + rows + 1]``, holding absolute positions in the
concatenated ``*_idx`` (columns within the level) and ``*_val``. Level ``l``'s vectors are
``[offsets[l], offsets[l + 1])`` of the work arrays ``bs``, ``xs`` and ``rs``.
``coarse`` maps the coarsest right-hand side to its solution (a dense inverse).

Gauss--Seidel goes colour by colour: the rows of a colour do not couple, so each colour
is updated in parallel over ``num_threads`` threads, and the result does not depend on
their number. Every level's unknowns are numbered colour by colour, so a colour is a
range of rows: level ``l``'s bounds are ``color_start[color_level[l] : color_level[l +
1]]`` (one more than its colours). Each level is smoothed by ``sweeps`` sweeps through the
colours in order before the coarse correction and as many in reverse order after it (a
symmetric cycle). ``x`` gets the cycle applied to ``b``.
"""


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
    color_start: "int[:]",
    color_level: "int[:]",
    sweeps: int,
    num_threads: int,
    b: "float[:]",
    x: "float[:]",
    bs: "float[:]",
    xs: "float[:]",
    rs: "float[:]",
) -> None:
    """Apply one V-cycle to ``b`` into ``x`` (see the module)."""
    fine = offsets[1] - offsets[0]
    # fmt: off
    #$ omp parallel for num_threads(num_threads)
    # fmt: on
    for i in range(fine):
        bs[i] = b[i]
    for level in range(levels - 1):
        offset = offsets[level]
        size = offsets[level + 1] - offset
        start = a_start[level]
        first = color_level[level]
        colors = color_level[level + 1] - first - 1
        # fmt: off
        #$ omp parallel for num_threads(num_threads)
        # fmt: on
        for i in range(size):
            xs[offset + i] = 0.0
        for _ in range(sweeps):
            for color in range(colors):
                low = color_start[first + color]
                high = color_start[first + color + 1]
                # fmt: off
                #$ omp parallel for num_threads(num_threads) private(total, diagonal, k, column)
                # fmt: on
                for row in range(low, high):
                    total = bs[offset + row]
                    diagonal = 0.0
                    for k in range(a_ptr[start + row], a_ptr[start + row + 1]):
                        column = a_idx[k]
                        if column == row:
                            diagonal += a_val[k]
                        else:
                            total -= a_val[k] * xs[offset + column]
                    xs[offset + row] = total / diagonal
        # fmt: off
        #$ omp parallel for num_threads(num_threads) private(total, k)
        # fmt: on
        for i in range(size):
            total = bs[offset + i]
            for k in range(a_ptr[start + i], a_ptr[start + i + 1]):
                total -= a_val[k] * xs[offset + a_idx[k]]
            rs[offset + i] = total
        below = offsets[level + 1]
        coarse_rows = offsets[level + 2] - below
        rstart = r_start[level]
        # fmt: off
        #$ omp parallel for num_threads(num_threads) private(total, k)
        # fmt: on
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
        size = offsets[level + 1] - offset
        below = offsets[level + 1]
        pstart = p_start[level]
        # fmt: off
        #$ omp parallel for num_threads(num_threads) private(total, k)
        # fmt: on
        for i in range(size):
            total = 0.0
            for k in range(p_ptr[pstart + i], p_ptr[pstart + i + 1]):
                total += p_val[k] * xs[below + p_idx[k]]
            xs[offset + i] += total
        start = a_start[level]
        first = color_level[level]
        colors = color_level[level + 1] - first - 1
        for _ in range(sweeps):
            for back in range(colors):
                color = colors - 1 - back
                low = color_start[first + color]
                high = color_start[first + color + 1]
                # fmt: off
                #$ omp parallel for num_threads(num_threads) private(total, diagonal, k, column)
                # fmt: on
                for row in range(low, high):
                    total = bs[offset + row]
                    diagonal = 0.0
                    for k in range(a_ptr[start + row], a_ptr[start + row + 1]):
                        column = a_idx[k]
                        if column == row:
                            diagonal += a_val[k]
                        else:
                            total -= a_val[k] * xs[offset + column]
                    xs[offset + row] = total / diagonal
    # fmt: off
    #$ omp parallel for num_threads(num_threads)
    # fmt: on
    for i in range(fine):
        x[i] = xs[i]
