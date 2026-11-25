"""Quadtree patches against clustered ones in a particle-in-cell cycle.

A beam of markers (a Gaussian of width ``--width``, ``--markers`` of them, on a uniform
background) moves across the unit square on a ``--base`` x ``--base`` base, and every step

- ``tag``: `marker_tags` (more than ``--per-cell`` markers per cell, ``--levels`` levels);
- ``regrid``: `regrid` with ``--method`` (``cluster`` or ``quadtree`` with ``--block-size``)
  and ``--placement`` (``balanced`` or ``morton``);
- ``migrate``, ``deposit``: the markers to their patches' ranks, cloud-in-cell deposit and
  `accumulate_halos`;
- ``solver``, ``solve``: a new `NodalPoisson` on the new mesh and its solve (serial only,
  or with ``--solve`` where the LU backend is enough);
- ``halos``: `exchange_halos` of the potential (or of the charge without a solve);
- ``gather``: `interpolate` at the markers.

Each part is a scope-profiler region (``quadtree_bench.<part>``); run with

    scope-profiler run -o prof.h5 benchmarks/quadtree/bench_quadtree.py --method quadtree
    mpiexec -n 4 scope-profiler run -o prof.h5 benchmarks/quadtree/bench_quadtree.py \\
        --placement morton --no-solve

and read the times with ``scope-profiler inspect prof.h5 --sort total``. Rank 0 also prints
one JSON line with the mesh: patches, refined cells, the share of patches a regrid keeps,
and the cells per rank.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import scope_profiler as sp
from maybempi import MPI

from patchwork import AMRHierarchy, NodalPoisson, marker_tags


def beam(rng, count: int, width: float, centre: tuple[float, float]):
    """Markers of a Gaussian beam on a uniform background (a fifth of them), in the unit square."""
    dense = count * 4 // 5
    x = np.concatenate(
        [rng.normal(centre[0], width, dense), rng.uniform(0, 1, count - dense)]
    )
    y = np.concatenate(
        [rng.normal(centre[1], width, dense), rng.uniform(0, 1, count - dense)]
    )
    inside = (x > 0) & (x < 1) & (y > 0) & (y < 1)
    return x[inside], y[inside]


def refined_cells(mesh) -> int:
    return sum(
        int(np.prod([hi - lo for lo, hi in patch.box]))
        for patch in mesh.patches
        if patch.level
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=["cluster", "quadtree"], default="quadtree")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-patch-size", type=int, default=None)
    parser.add_argument(
        "--placement", choices=["balanced", "morton"], default="balanced"
    )
    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--markers", type=int, default=200_000)
    parser.add_argument("--per-cell", type=float, default=20.0)
    parser.add_argument("--width", type=float, default=0.06)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--speed", type=float, default=0.01, help="beam shift per step")
    parser.add_argument("--no-solve", dest="solve", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    rng = np.random.default_rng(args.seed + rank)
    options = (
        {"method": "quadtree", "block_size": args.block_size}
        if args.method == "quadtree"
        else {"method": "cluster", "max_patch_size": args.max_patch_size}
    )
    mesh = AMRHierarchy(
        base_shape=(args.base, args.base),
        placement=args.placement,
        comm=comm,
    )
    mesh.allocate_field("rho", centering="node")
    if args.solve:
        mesh.allocate_field("phi", centering="node")

    def markers(step: int):
        x, y = beam(
            rng, args.markers // size, args.width, (0.25 + args.speed * step, 0.5)
        )
        return x, y, np.full(x.size, 1.0 / args.markers)

    # build the levels on the first beam, one level per regrid, untimed
    x, y, q = markers(0)
    for _ in range(args.levels):
        mesh = mesh.regrid(
            marker_tags(mesh, x, y, per_cell=args.per_cell, levels=args.levels),
            **options,
        )

    kept, patches, cells = [], [], []
    previous = {(p.level, p.box) for p in mesh.patches if p.level}
    for step in range(1, args.steps + 1):
        x, y, q = markers(step)
        with sp.region("quadtree_bench.tag"):
            tags = marker_tags(mesh, x, y, per_cell=args.per_cell, levels=args.levels)
        with sp.region("quadtree_bench.regrid"):
            mesh = mesh.regrid(tags, **options)
        boxes = {(p.level, p.box) for p in mesh.patches if p.level}
        kept.append(len(boxes & previous) / max(len(boxes), 1))
        previous = boxes
        patches.append(len(boxes))
        cells.append(refined_cells(mesh))
        with sp.region("quadtree_bench.migrate"):
            mx, my, mq = mesh.migrate(x, y, q, name="rho")
        with sp.region("quadtree_bench.deposit"):
            mesh.deposit(mx, my, mq, "rho")
            mesh.accumulate_halos("rho")
        field = "rho"
        if args.solve:
            with sp.region("quadtree_bench.solver"):
                solver = NodalPoisson(mesh)
            with sp.region("quadtree_bench.solve"):
                density = solver.nodes.from_patches("rho") / solver.nodes.volumes
                solver.solve(density, potential="phi")
            field = "phi"
        with sp.region("quadtree_bench.halos"):
            mesh.exchange_halos(field)
        with sp.region("quadtree_bench.gather"):
            mesh.interpolate(mx, my, field)

    load = mesh.cells_on_rank
    if rank == 0:
        print(
            json.dumps(
                {
                    "method": args.method,
                    "block_size": args.block_size
                    if args.method == "quadtree"
                    else None,
                    "placement": args.placement,
                    "ranks": size,
                    "patches": float(np.mean(patches)),
                    "refined_cells": float(np.mean(cells)),
                    "kept": float(np.mean(kept)),
                    "load_imbalance": max(load) / (sum(load) / len(load)),
                }
            )
        )


if __name__ == "__main__":
    main()
