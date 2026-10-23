"""The distributed node-centered solve: run under ``mpirun -n N`` (N >= 2).

Every rank assembles only the rows of the unknowns it owns and PETSc solves
them together; the potential must be what SuperLU gives for the same hierarchy
on one rank (``MPI.COMM_SELF``), which every rank computes for itself. Prints
``MPI_JOB_OK`` on rank 0 if all checks pass; a failed check aborts the job.
"""

from __future__ import annotations

import numpy as np
from maybempi import MPI

from patchwork import (
    AMRHierarchy,
    Dirichlet,
    EmbeddedDirichlet,
    Neumann,
    NodalPoisson,
)

WORLD = MPI.COMM_WORLD
SIZE = WORLD.Get_size()
RANK = WORLD.Get_rank()
FACES = ("x_lower", "x_upper", "y_lower", "y_upper")


def check(condition: bool, message: str) -> None:
    if not condition:
        print(f"[rank {RANK}] FAILED: {message}", flush=True)
        WORLD.Abort(1)


def source(x, y):
    return np.exp(-((x - 0.4) ** 2 + (y - 0.55) ** 2) / 0.02)


#: Fine patches and the ranks that hold them: one rank, a pair, all ranks.
PATCHES = (
    (1, ((16, 32), (16, 32)), SIZE - 1),
    (1, ((32, 48), (24, 48)), (0, 1)),
    (2, ((36, 60), (36, 60)), SIZE - 1),
    (2, ((68, 92), (52, 76)), None),
)


def rectangle(comm, placed: bool, base: int = 32, **kwargs) -> AMRHierarchy:
    mesh = AMRHierarchy(base_shape=(base, base), comm=comm, **kwargs)
    scale = base // 32
    for level, box, ranks in PATCHES:
        mesh.add_patch(
            level=level,
            box=tuple((lo * scale, hi * scale) for lo, hi in box),
            ranks=ranks if placed else None,
        )
    return mesh


def ell(comm, placed: bool) -> AMRHierarchy:
    """An L whose two base boxes are on different ranks."""
    mesh = AMRHierarchy(
        base_shape=(16, 16),
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
        base_boxes=[((0, 8), (0, 16)), ((8, 16), (8, 16))],
        base_ranks=[0, SIZE - 1] if placed else None,
        comm=comm,
    )
    mesh.add_patch(level=1, box=((12, 16), (12, 20)), ranks=1 if placed else None)
    mesh.add_patch(level=1, box=((16, 20), (16, 20)), ranks=0 if placed else None)
    return mesh


def compare(distributed: AMRHierarchy, serial: AMRHierarchy, what: str, **kwargs):
    """The PETSc solve on ``distributed`` against SuperLU on ``serial``."""
    solver = NodalPoisson(distributed, backend="petsc", **kwargs)
    reference = NodalPoisson(serial, **kwargs)
    nodes = solver.nodes
    check(
        np.array_equal(nodes.coordinates, reference.nodes.coordinates),
        f"{what}: the same nodes",
    )
    # every unknown is a row of exactly one rank
    rows = np.concatenate(WORLD.allgather(solver.rows))
    check(
        np.array_equal(np.sort(rows), solver.free), f"{what}: rows cover the unknowns"
    )
    check(
        solver.rows.size < solver.free.size or SIZE == 1,
        f"{what}: rank {RANK} holds all {solver.free.size} rows",
    )
    x, y = nodes.coordinates.T
    expected = reference.solve(source(x, y))
    found = solver.solve(source(x, y))
    error = np.abs(found - expected).max()
    check(
        error < 1e-8 * max(np.abs(expected).max(), 1.0),
        f"{what}: differs from SuperLU by {error:.2e}",
    )
    check(
        np.array_equal(WORLD.bcast(found, root=0), found),
        f"{what}: rank {RANK} has another solution than rank 0",
    )
    return solver


def main() -> None:
    check(SIZE >= 2, "the job needs at least two ranks")
    faces = {"x_lower": Neumann(0.0), "x_upper": Dirichlet(lambda x, y: 1.0 + y)}

    # 1. Patches on one rank, a pair and all ranks.
    compare(
        rectangle(WORLD, placed=True),
        rectangle(MPI.COMM_SELF, placed=False),
        "placed patches",
        faces=faces,
    )
    # 2. Every patch on every rank (the default): rows split along the first axis.
    compare(
        rectangle(WORLD, placed=False),
        rectangle(MPI.COMM_SELF, placed=False),
        "patches on all ranks",
        faces=faces,
    )
    # 3. An L over two ranks, with no flux through its inner edges.
    compare(
        ell(WORLD, placed=True),
        ell(MPI.COMM_SELF, placed=False),
        "an L",
        faces={"inner": Neumann(0.0), "y_upper": Dirichlet(2.0)},
    )
    # 4. A body across rank boundaries and a periodic axis (GMRES).
    body = EmbeddedDirichlet(
        lambda p: (p[0] - 0.5) ** 2 + (p[1] - 0.52) ** 2 < 0.1**2, -1.0
    )
    compare(
        rectangle(WORLD, placed=True, periodic=(False, True)),
        rectangle(MPI.COMM_SELF, placed=False, periodic=(False, True)),
        "a body and a periodic axis",
        embedded=[body],
    )

    # 5. A charge, the same on every rank, and the potential written to the patches.
    mesh = rectangle(WORLD, placed=True)
    solver = NodalPoisson(mesh, backend="petsc", faces=faces)
    reference = NodalPoisson(rectangle(MPI.COMM_SELF, placed=False), faces=faces)
    points = np.random.default_rng(7).uniform(0.0, 1.0, (2, 2000))
    charge = solver.nodes.deposit(points, np.full(2000, 1.0e-3))
    found = solver.solve(charge=charge, potential="phi")
    expected = reference.solve(charge=charge)
    check(np.abs(found - expected).max() < 1e-10, "charge solve")
    for patch in mesh.patches:
        gathered = mesh.gather(patch, "phi")
        check(
            np.array_equal(gathered, found[solver.nodes.patch_nodes(patch)]),
            f"the potential on patch {patch.box}",
        )

    # 6. A line, its patches on one rank, a pair and all ranks.
    def line(comm, placed: bool) -> AMRHierarchy:
        mesh = AMRHierarchy(base_shape=(32,), comm=comm)
        for level, box, ranks in (
            (1, ((16, 40),), SIZE - 1),
            (1, ((44, 60),), (0, 1)),
            (2, ((40, 72),), SIZE - 1),
            (2, ((92, 116),), None),
        ):
            mesh.add_patch(level=level, box=box, ranks=ranks if placed else None)
        return mesh

    def line_source(x):
        return np.exp(-((x - 0.4) ** 2) / 0.02)

    ends = {"x_lower": Neumann(0.5), "x_upper": Dirichlet(1.0)}
    solver = NodalPoisson(line(WORLD, placed=True), backend="petsc", faces=ends)
    reference = NodalPoisson(line(MPI.COMM_SELF, placed=False), faces=ends)
    rows = np.concatenate(WORLD.allgather(solver.rows))
    check(np.array_equal(np.sort(rows), solver.free), "a line: rows cover the unknowns")
    check(solver.rows.size < solver.free.size, f"a line: rank {RANK} holds every row")
    (x,) = solver.nodes.coordinates.T
    error = np.abs(solver.solve(line_source(x)) - reference.solve(line_source(x)))
    check(error.max() < 1e-8, f"a line: differs from SuperLU by {error.max():.2e}")

    # 7. A manufactured solution converges at second order, distributed.
    def exact(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y) + x * y

    errors = []
    for base in (32, 64, 128):
        solver = NodalPoisson(
            rectangle(WORLD, placed=True, base=base),
            faces={face: Dirichlet(exact) for face in FACES},
            backend="petsc",
        )
        x, y = solver.nodes.coordinates.T
        rhs = 2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
        errors.append(np.abs(solver.solve(rhs) - exact(x, y)).max())
    check(
        errors[0] / errors[1] > 3.3 and errors[1] / errors[2] > 3.3,
        f"second order: errors {errors}",
    )

    if RANK == 0:
        print("MPI_JOB_OK", flush=True)


if __name__ == "__main__":
    main()
