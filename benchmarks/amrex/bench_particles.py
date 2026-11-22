"""patchwork + mpiparticles side of the marker benchmark against AMReX (B6).

The case of ``cpp/bench_particles.cpp``: the hierarchy of ``bench_solve.py`` with every
level cut into patches of at most 64 cells per axis (``placement="balanced"`` under MPI),
markers from a Gaussian of width 0.15 around the centre, unit charges, held in an
mpiparticles `ParticleStore`. Times (median):

- migrate: a small random step, then `mpiparticles.migrate` with patchwork's `owner`
  as the decomposition, and `ParticleStore.compact`;
- deposit: `deposit` (cloud-in-cell on each marker's finest patch) and
  `accumulate_halos` (the halos and the shared nodes added to one copy);
- gather: `exchange_halos`, then `interpolate`;
- sort: `locate`, a key from the patch and the cell, `ParticleStore.sort_by_key`.

Run under ``mpiexec -n <ranks>`` for several ranks. Prints one JSON line (rank 0).
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import mpiparticles
import numpy as np
from mpi4py import MPI

from patchwork import AMRHierarchy


def boxes(lo: int, hi: int, size: int, ndim: int) -> list[tuple]:
    """The box ``[lo, hi)^ndim`` cut into boxes of at most ``size`` per axis."""
    edges = list(range(lo, hi, size)) + [hi]
    pieces = list(zip(edges[:-1], edges[1:], strict=True))
    return [tuple(p) for p in itertools.product(pieces, repeat=ndim)]


def hierarchy(ndim: int, n: int, nlevels: int, comm) -> AMRHierarchy:
    """The levels of ``bench_solve.py``, in patches of at most 64 cells."""
    mesh = AMRHierarchy(
        base_shape=(n,) * ndim,
        base_boxes=boxes(0, n, 64, ndim),
        placement="balanced",
        comm=comm,
    )
    lo, hi = 0, n
    for level in range(1, nlevels):
        quarter = (hi - lo) // 4
        lo, hi = 2 * (lo + quarter), 2 * (hi - quarter)
        for box in boxes(lo, hi, 64, ndim):
            mesh.add_patch(level=level, box=box)
    return mesh


class PatchOwners:
    """patchwork's `owner` as an mpiparticles decomposition."""

    def __init__(self, mesh: AMRHierarchy, name: str) -> None:
        """Own by the blocks of field ``name`` of ``mesh``."""
        self.mesh, self.name, self._comm = mesh, name, mesh.comm

    @property
    def comm(self):
        """The hierarchy's communicator."""
        return self._comm

    @property
    def rank(self) -> int:
        """This rank."""
        return self._comm.rank

    @property
    def size(self) -> int:
        """The number of ranks."""
        return self._comm.size

    def owners(self, positions: np.ndarray) -> np.ndarray:
        """The rank that holds each marker's cell (rank 0 for one outside)."""
        owner = self.mesh.owner(*positions, name=self.name)
        return np.where(owner < 0, 0, owner)

    def owned_mask(self, positions: np.ndarray) -> np.ndarray:
        """Which markers this rank holds."""
        return self.owners(positions) == self.rank


def main() -> None:
    """Run the case and print its JSON line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ndim", type=int, default=2)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--nlevels", type=int, default=3)
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size

    mesh = hierarchy(args.ndim, args.n, args.nlevels, comm)
    mesh.allocate_field("rho", centering="node", halo=1)
    rng = np.random.default_rng(1 + rank)
    local = args.count // size + (rank < args.count % size)
    positions = np.clip(rng.normal(0.5, 0.15, (args.ndim, local)), 1e-6, 1 - 1e-6)
    store = mpiparticles.ParticleStore(args.ndim, args.ndim, capacity=local)
    store.add(positions, np.zeros((args.ndim, local)), weights=1.0)
    owners = PatchOwners(mesh, "rho")
    owners.owners(positions)  # collective the first time
    mpiparticles.migrate(store, owners)
    finest = np.array(mesh.spacing(mesh.max_level))
    cells = np.round(1 / finest).astype(np.int64)

    timings = {key: [] for key in ("migrate", "deposit", "gather", "sort")}
    for repeat in range(args.repeats + 1):
        x = store.active_positions()
        x += 0.002 * (rng.random(x.shape) - 0.5)
        store.set_active_positions(np.clip(x, 1e-6, 1 - 1e-6))
        comm.Barrier()
        t0 = time.perf_counter()
        mpiparticles.migrate(store, owners)
        store.compact()  # drop the markers that left (sort_by_key keys every slot)
        comm.Barrier()
        t1 = time.perf_counter()
        x, w = store.active_positions(), store.active_weights()
        mesh.deposit(*x, w, name="rho")
        mesh.accumulate_halos("rho")
        comm.Barrier()
        t2 = time.perf_counter()
        mesh.exchange_halos("rho")
        mesh.interpolate(*x, name="rho")
        comm.Barrier()
        t3 = time.perf_counter()
        patch = mesh.locate(*x)
        index = np.minimum((x / finest[:, None]).astype(np.int64), cells[:, None] - 1)
        key = patch.astype(np.int64)
        for axis in range(args.ndim):
            key = key * cells[axis] + index[axis]
        store.sort_by_key(key)
        comm.Barrier()
        t4 = time.perf_counter()
        if repeat:
            for name, value in zip(
                timings, (t1 - t0, t2 - t1, t3 - t2, t4 - t3), strict=True
            ):
                timings[name].append(value)
    markers = comm.allreduce(store.num_alive)
    if rank == 0:
        print(
            json.dumps(
                {
                    "code": "patchwork",
                    "ndim": args.ndim,
                    "n_cell": args.n,
                    "nlevels": args.nlevels,
                    "ranks": size,
                    "markers": markers,
                    **{f"{k}_s": float(np.median(v)) for k, v in timings.items()},
                }
            )
        )


if __name__ == "__main__":
    main()
