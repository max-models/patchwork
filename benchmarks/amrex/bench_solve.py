"""patchwork side of the solver benchmarks against AMReX (B1-B3).

The same problems as ``cpp/bench_solve.cpp``: a Gaussian ``exp(-|x - 0.5|^2 / s^2)``,
``s = 0.1``, with zero Dirichlet faces, on a hierarchy whose levels each cover the
middle half of the one below.

- ``--mode nodal``: ``-lap(phi) = f`` on the nodes, `NodalPoisson`;
- ``--mode cell``: ``-div(a grad u) + u = f`` on the cells, `CellOperator`, with
  ``a = 1 + 0.5 sin(2 pi x) sin(2 pi y)`` (2D).

Prints one JSON line, with the keys of the AMReX driver.
"""

from __future__ import annotations

import argparse
import itertools
import json
import resource
import time

import numpy as np
from mpi4py import MPI

from patchwork import AMRHierarchy, CellOperator, CompositeCells, NodalPoisson

S = 0.1


def boxes(lo: int, hi: int, size: int, ndim: int) -> list[tuple]:
    """The box ``[lo, hi)^ndim`` cut into boxes of at most ``size`` per axis."""
    edges = list(range(lo, hi, size)) + [hi]
    pieces = list(zip(edges[:-1], edges[1:], strict=True))
    return [tuple(p) for p in itertools.product(pieces, repeat=ndim)]


def hierarchy(ndim: int, n: int, nlevels: int, placement: str = "all") -> AMRHierarchy:
    """Levels covering the middle half of the patch below, as in the AMReX driver.

    Under MPI with ``placement="balanced"`` every level is cut into patches of at most
    64 cells per axis, spread over the ranks (as AMReX's boxes).
    """
    split = placement != "all"
    mesh = AMRHierarchy(
        base_shape=(n,) * ndim,
        bounds=((0.0, 1.0),) * ndim,
        base_boxes=boxes(0, n, 64, ndim) if split else None,
        placement=placement,
    )
    lo, hi = 0, n
    for level in range(1, nlevels):
        quarter = (hi - lo) // 4
        lo, hi = 2 * (lo + quarter), 2 * (hi - quarter)
        for box in boxes(lo, hi, 64, ndim) if split else [((lo, hi),) * ndim]:
            mesh.add_patch(level=level, box=box)
    return mesh


def gaussian(x: np.ndarray, c: float = 0.5) -> np.ndarray:
    """The exact solution at points ``(ndim, n)``, centred at ``(c, ..., c)``."""
    return np.exp(-((x - c) ** 2).sum(axis=0) / S**2)


def laplacian(x: np.ndarray, c: float = 0.5) -> np.ndarray:
    """Laplacian of the Gaussian."""
    r2 = ((x - c) ** 2).sum(axis=0)
    return gaussian(x, c) * (4 * r2 / S**4 - 2 * x.shape[0] / S**2)


def centre(repeat: int, repeats: int) -> float:
    """A new source every solve (no warm start), the last one centred at 0.5."""
    return 0.5 + 0.04 * ((repeats - repeat) % 3)


def coefficient(x: np.ndarray) -> np.ndarray:
    """The diffusion coefficient of the cell case."""
    return 1.0 + 0.5 * np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1])


def cell_source(x: np.ndarray, c: float = 0.5) -> np.ndarray:
    """``-div(a grad u) + u`` for the Gaussian and `coefficient`."""
    u = gaussian(x, c)
    grad_u = -2 * (x - c) / S**2 * u
    grad_a = (
        0.5
        * 2
        * np.pi
        * np.array(
            [
                np.cos(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1]),
                np.sin(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]),
            ]
        )
    )
    return -((grad_a * grad_u).sum(axis=0) + coefficient(x) * laplacian(x, c)) + u


def peak_mb() -> float:
    """Peak resident memory of the process (Linux: kB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main() -> None:
    """Run one case and print its JSON line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("nodal", "cell"), default="nodal")
    parser.add_argument("--ndim", type=int, default=2)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--nlevels", type=int, default=3)
    parser.add_argument("--backend", default="lu")
    parser.add_argument("--mass", default="lumped")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--placement", default="all")
    parser.add_argument("--coarse-fine", default="symmetric")
    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    baseline = peak_mb()
    start = time.perf_counter()
    mesh = hierarchy(args.ndim, args.n, args.nlevels, args.placement)
    data = time.perf_counter() - start

    comm.Barrier()
    start = time.perf_counter()
    if args.mode == "nodal":
        solver = NodalPoisson(mesh, backend=args.backend, mass=args.mass)
        points = solver.nodes.coordinates.T
        sources = [
            -laplacian(points, centre(r, args.repeats)) for r in range(args.repeats + 1)
        ]
    else:
        cells = CompositeCells(mesh)
        points = cells.centers.T
        solver = CellOperator(
            mesh,
            diffusion=coefficient(points),
            reaction=1.0,
            cells=cells,
            coarse_fine=args.coarse_fine,
        )
        sources = [
            cell_source(points, centre(r, args.repeats))
            for r in range(args.repeats + 1)
        ]
    comm.Barrier()
    build = time.perf_counter() - start

    times = []
    for source in sources:
        start = time.perf_counter()
        result = solver.solve(source)
        comm.Barrier()
        times.append(time.perf_counter() - start)
    error = float(np.abs(np.real(result) - gaussian(points)).max())

    peak = comm.allreduce(peak_mb())
    if comm.rank:
        return
    print(
        json.dumps(
            {
                "code": "patchwork",
                "mode": args.mode,
                "backend": args.backend,
                "mass": args.mass,
                "coarse_fine": args.coarse_fine,
                "ndim": args.ndim,
                "n_cell": args.n,
                "nlevels": args.nlevels,
                "ranks": comm.size,
                "placement": args.placement,
                "points": int(points.shape[1]),
                "data_s": data,
                "build_s": build,
                "first_solve_s": times[0],
                "solve_s": float(np.median(times[1:])),
                "error": error,
                "peak_mb": peak,
                "baseline_mb": baseline * comm.size,
            }
        )
    )


if __name__ == "__main__":
    main()
