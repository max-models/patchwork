"""One step of carafe's MHD on patches, for the scaling campaign (needs carafe).

The Orszag-Tang vortex on a periodic ``n x n`` base, regridded on the current density up to
``--levels`` refined levels (the tags of the initial state, so the mesh is the same on any
number of ranks), then timed: ``--steps`` steps of `carafe.MHD2D` (Heun, HLLD, constrained
transport of the potential on the nodes, refluxing), and one regrid. The phases of a step
come from scope-profiler's regions when the ``carafe`` and ``patchwork`` tags are on.

Run under ``mpiexec -n <ranks>``. Prints one JSON line (rank 0).
"""

from __future__ import annotations

import argparse
import json
import resource
import time

import numpy as np
from mpi4py import MPI

from patchwork import AMRHierarchy


def peak_mb() -> float:
    """Peak resident memory of the process (Linux: kB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main() -> None:
    """Build the case, time the steps and a regrid, print the JSON line."""
    from carafe.mhd import MHD2D

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--max-patch", type=int, default=32)
    parser.add_argument("--placement", default="balanced")
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    mesh = AMRHierarchy(
        (args.n, args.n),
        bounds=((0.0, 1.0), (0.0, 1.0)),
        periodic=(True, True),
        comm=comm,
        placement=args.placement,
    )
    model = MHD2D(mesh, resistivity=1e-4)
    k = 2.0 * np.pi
    state = {
        "density": 25.0 / (36.0 * np.pi),
        "velocity": (lambda x, y: -np.sin(k * y), lambda x, y: np.sin(k * x), 0.0),
        "pressure": 5.0 / (12.0 * np.pi),
        "vector_potential": lambda x, y: (
            (np.cos(2 * k * x) / (2 * k) + np.cos(k * y) / k) / np.sqrt(4 * np.pi)
        ),
    }
    model.set_state(**state)
    for _ in range(args.levels):
        model.regrid(
            model.current_tags(args.threshold, levels=args.levels),
            max_patch_size=args.max_patch,
        )
        model.set_state(**state)
    dt = model.stable_step()
    model.step(dt)  # builds the halo schedules
    times = []
    for _ in range(args.steps):
        comm.Barrier()
        start = time.perf_counter()
        model.step(dt)
        comm.Barrier()
        times.append(time.perf_counter() - start)
    comm.Barrier()
    start = time.perf_counter()
    model.regrid(
        model.current_tags(args.threshold, levels=args.levels),
        max_patch_size=args.max_patch,
    )
    comm.Barrier()
    regrid = time.perf_counter() - start
    from patchwork import CompositeCells

    leaf = CompositeCells(model.hierarchy).num_cells
    peak = comm.allreduce(peak_mb())
    if comm.rank == 0:
        print(
            json.dumps(
                {
                    "code": "patchwork",
                    "mode": "mhd",
                    "ranks": comm.size,
                    "n_cell": args.n,
                    "nlevels": args.levels + 1,
                    "placement": args.placement,
                    "patches": len(model.hierarchy.patches),
                    "leaf_cells": int(leaf),
                    "step_s": float(np.median(times)),
                    "regrid_s": regrid,
                    "peak_mb": peak,
                }
            )
        )


if __name__ == "__main__":
    main()
