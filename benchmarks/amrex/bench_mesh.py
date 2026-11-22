"""patchwork side of the mesh benchmarks against AMReX (B4, B5).

The same cases as ``cpp/bench_mesh.cpp``:

- ``--mode halo`` (B4): the base of ``n`` cells per axis in boxes of at most 64, a fine
  level over the middle half in patches of ``tile`` fine cells per axis, a cell field
  with a halo of one; times (median) of `exchange_halos` and `restrict`, and of the
  first `exchange_halos` (which builds the schedule).
- ``--mode regrid`` (B5, 2D): the moving charge of ``examples/regrid_moving_charge.py``,
  cell-centered: per step `gradient_tags` and `regrid` (``max_patch_size=32``), which
  also moves the density.

Prints one JSON line.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
from mpi4py import MPI

from patchwork import AMRHierarchy, gradient_tags

WIDTH = 0.002


def density(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    """The moving Gaussian charge of the regrid example."""
    cx, cy = 0.25 + 0.5 * t, 0.3 + 0.4 * t
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / WIDTH) / (np.pi * WIDTH)


def boxes(lo: int, hi: int, size: int, ndim: int) -> list[tuple]:
    """The box ``[lo, hi)^ndim`` cut into boxes of at most ``size`` per axis."""
    edges = list(range(lo, hi, size)) + [hi]
    pieces = list(zip(edges[:-1], edges[1:], strict=True))
    return [tuple(p) for p in itertools.product(pieces, repeat=ndim)]


def halo_mode(ndim: int, n: int, tile: int, repeats: int, placement: str) -> dict:
    """B4."""
    mesh = AMRHierarchy(
        base_shape=(n,) * ndim,
        bounds=((0.0, 1.0),) * ndim,
        base_boxes=boxes(0, n, 64, ndim),
        placement=placement,
    )
    for box in boxes(n // 2, 3 * n // 2, tile, ndim):
        mesh.add_patch(level=1, box=box)
    mesh.allocate_field("u", halo=1)
    comm = mesh.comm
    for patch in mesh.local_patches:
        patch.field("u").local[...] = patch.level + 1.0
    comm.Barrier()
    start = time.perf_counter()
    mesh.exchange_halos("u")
    mesh.restrict("u")
    comm.Barrier()
    first = time.perf_counter() - start
    fill, restrict = [], []
    for _ in range(repeats):
        comm.Barrier()
        start = time.perf_counter()
        mesh.exchange_halos("u")
        comm.Barrier()
        middle = time.perf_counter()
        mesh.restrict("u")
        comm.Barrier()
        fill.append(middle - start)
        restrict.append(time.perf_counter() - middle)
    fine = mesh.patches_at_level(1)
    return {
        "mode": "halo",
        "ranks": comm.size,
        "ndim": ndim,
        "n_cell": n,
        "tile": tile,
        "patches": len(mesh.patches),
        "fine_patches": len(fine),
        "cells": int(sum(np.prod(p.shape) for p in mesh.patches)),
        "first_s": first,
        "fill_s": float(np.median(fill)),
        "restrict_s": float(np.median(restrict)),
    }


def regrid_mode(steps: int, threshold: float) -> dict:
    """B5."""
    mesh = AMRHierarchy(base_shape=(32, 32))
    mesh.allocate_field("rho")

    def deposit(mesh: AMRHierarchy, t: float) -> None:
        for patch in mesh.local_patches:
            x, y = patch.local_cell_centers(mesh, "rho")
            patch.field("rho").local[...] = density(x, y, t)

    def regrid(mesh: AMRHierarchy) -> AMRHierarchy:
        return mesh.regrid(gradient_tags(mesh, "rho", threshold), max_patch_size=32)

    deposit(mesh, 0.0)
    while True:  # deepen at t = 0 until the levels stop growing
        deeper = regrid(mesh)
        deposit(deeper, 0.0)
        if deeper.max_level == mesh.max_level:
            break
        mesh = deeper
    per_step = []
    for step in range(1, steps + 1):
        t = step / steps
        deposit(mesh, t)
        start = time.perf_counter()
        mesh = regrid(mesh)
        per_step.append(time.perf_counter() - start)
        deposit(mesh, t)
    return {
        "mode": "regrid",
        "steps": steps,
        "regrid_s": float(np.mean(per_step)),
        "levels": mesh.max_level + 1,
        "patches": len(mesh.patches),
        "cells": int(sum(np.prod(p.shape) for p in mesh.patches)),
    }


def main() -> None:
    """Run one case and print its JSON line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("halo", "regrid"), default="halo")
    parser.add_argument("--ndim", type=int, default=2)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--tile", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--placement", default="balanced")
    args = parser.parse_args()
    if args.mode == "halo":
        result = halo_mode(args.ndim, args.n, args.tile, args.repeats, args.placement)
    else:
        result = regrid_mode(args.steps, args.threshold)
    if MPI.COMM_WORLD.rank == 0:
        print(json.dumps({"code": "patchwork", **result}))


if __name__ == "__main__":
    main()
