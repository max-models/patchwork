"""The curl-curl operator with AMS under ``mpirun -n N``: the rows split over the ranks.

The potential must be what the LU gives on one rank (``MPI.COMM_SELF``), which
every rank computes for itself. Prints ``MPI_JOB_OK`` on rank 0 if the check
passes; a failed check aborts the job.
"""

from __future__ import annotations

import numpy as np
from maybempi import MPI

from patchwork import AMRHierarchy, CurlCurl

WORLD = MPI.COMM_WORLD


def cube(comm) -> AMRHierarchy:
    mesh = AMRHierarchy(base_shape=(8, 8, 8), comm=comm)
    mesh.add_patch(level=1, box=((4, 12), (2, 10), (6, 14)))
    return mesh


def main() -> None:
    ams = CurlCurl(cube(WORLD), beta=1.0, reluctivity=2.0, solver="ams")
    lu = CurlCurl(cube(MPI.COMM_SELF), beta=1.0, reluctivity=2.0)
    edges = lu.edges
    middle = 0.5 * (
        edges.nodes.coordinates[edges.ends[:, 0]]
        + edges.nodes.coordinates[edges.ends[:, 1]]
    )
    current = np.where(
        edges.direction == 2, np.exp(-np.sum((middle - 0.5) ** 2, axis=1) / 0.02), 0.0
    )
    load = lu.load(current * edges.length)
    error = np.abs(ams.solve(load) - lu.solve(load)).max()
    if error > 1e-8:
        print(
            f"[rank {WORLD.Get_rank()}] FAILED: AMS differs by {error:.2e}", flush=True
        )
        WORLD.Abort(1)
    if WORLD.Get_rank() == 0:
        print("MPI_JOB_OK", flush=True)


if __name__ == "__main__":
    main()
