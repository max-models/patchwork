"""Timings of the node-centered solve: the LU (SuperLU) on every rank against PETSc over the ranks.

Run under ``mpirun -n N python examples/distributed_scaling.py [--base 256] [--weak]``.
The hierarchy is a ``base x base`` mesh with a level-1 patch over its middle
half and a level-2 patch over a quarter of that, every patch on all ranks (the
default placement, each patch's rows split over the ranks). With ``--weak``
the base mesh grows with the ranks (``base * sqrt(N)`` per side), so the
unknowns per rank stay about the same.

Prints, on rank 0, the unknowns, the time to number the nodes, to assemble
(and factorize, for the LU), to solve, the Krylov iterations and the largest
difference of the two solutions. Times are the slowest rank's, the best of
``--repeat`` solves. The LU is run on every rank for itself, as
``NodalPoisson`` does by default (skip it with ``--no-lu``).
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from maybempi import MPI  # noqa: E402

from patchwork import AMRHierarchy, CompositeNodes, NodalPoisson  # noqa: E402
from patchwork.distributed import petsc  # noqa: E402

WORLD = MPI.COMM_WORLD
SIZE = WORLD.Get_size()
RANK = WORLD.Get_rank()


def hierarchy(base: int) -> AMRHierarchy:
    """A ``base``-cell square, refined twice around its middle."""
    mesh = AMRHierarchy(base_shape=(base, base), comm=WORLD)
    quarter = base // 4
    mesh.add_patch(level=1, box=((2 * quarter, 6 * quarter),) * 2)
    mesh.add_patch(level=2, box=((7 * quarter, 9 * quarter),) * 2)
    return mesh


def timed(work):
    """The result of ``work()`` and the slowest rank's time for it."""
    WORLD.Barrier()
    start = time.perf_counter()
    result = work()
    elapsed = time.perf_counter() - start
    return result, WORLD.allreduce(elapsed, op=MPI.MAX)


def source(x, y):
    """A charge cloud off the middle of the finest patch."""
    return np.exp(-((x - 0.45) ** 2 + (y - 0.55) ** 2) / 0.01)


def main() -> None:
    """Time both backends and print a table on rank 0."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=int, default=256)
    parser.add_argument("--weak", action="store_true")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--no-lu", action="store_true")
    args = parser.parse_args()
    base = args.base
    if args.weak:
        base = 8 * int(round(base * np.sqrt(SIZE) / 8))

    petsc()  # imported before timing anything
    mesh = hierarchy(base)
    nodes, numbering = timed(lambda: CompositeNodes(mesh))
    x, y = nodes.coordinates.T
    rhs = source(x, y)
    rows = []
    solutions = {}
    backends = ["petsc"] if args.no_lu else ["lu", "petsc"]
    for backend in backends:
        solver, setup = timed(
            lambda backend=backend: NodalPoisson(mesh, nodes=nodes, backend=backend)
        )
        best = np.inf
        for _ in range(args.repeat):
            if solver.system is not None:
                solver.system.reset()  # the same start for every repetition
            solutions[backend], elapsed = timed(lambda solver=solver: solver.solve(rhs))
            best = min(best, elapsed)
        iterations = solver.system.iterations if solver.system is not None else "-"
        held = WORLD.allreduce(solver.rows.size, op=MPI.MAX)
        rows.append((backend, solver.free.size, held, setup, best, iterations))

    difference = (
        np.abs(solutions["petsc"] - solutions["lu"]).max()
        if "lu" in solutions
        else float("nan")
    )
    if RANK == 0:
        print(
            f"ranks {SIZE}, base {base}, {nodes.num_nodes} nodes, "
            f"numbering {numbering:.3f} s, |petsc - lu| = {difference:.1e}"
        )
        print(
            f"{'backend':8} {'unknowns':>9} {'rows/rank':>9} "
            f"{'setup s':>8} {'solve s':>8} {'its':>4}"
        )
        for backend, unknowns, held, setup, solve, its in rows:
            print(
                f"{backend:8} {unknowns:9d} {held:9d} {setup:8.3f} {solve:8.3f} {its!s:>4}"
            )


if __name__ == "__main__":
    main()
