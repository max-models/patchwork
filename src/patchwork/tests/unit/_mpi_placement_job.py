"""Patches on their own communicators: run under ``mpirun -n N`` (N >= 2).

Every check compares with the same hierarchy built on one rank
(``MPI.COMM_SELF``), which every rank computes for itself. Prints
``MPI_JOB_OK`` on rank 0 if all checks pass; a failed check aborts the job.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from maybempi import MPI  # noqa: E402

from patchwork import (  # noqa: E402
    AMRHierarchy,
    Dirichlet,
    Neumann,
    NodalPoisson,
    plot_patches,
    solve_poisson,
)

WORLD = MPI.COMM_WORLD
SIZE = WORLD.Get_size()
RANK = WORLD.Get_rank()


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


def rectangle(comm, placed: bool) -> AMRHierarchy:
    mesh = AMRHierarchy(base_shape=(32, 32), comm=comm)
    for level, box, ranks in PATCHES:
        mesh.add_patch(level=level, box=box, ranks=ranks if placed else None)
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


def cell_solution(mesh: AMRHierarchy) -> list[np.ndarray]:
    mesh.allocate_field("rhs")
    for patch in mesh.local_patches:
        x, y = patch.local_cell_centers(mesh)
        patch.field("rhs").local[...] = source(x, y)
    solve_poisson(mesh, "rhs", potential="phi", boundary_value=0.5)
    return [mesh.gather(patch, "phi") for patch in mesh.patches]


def nodal_solution(mesh: AMRHierarchy, faces: dict) -> list[np.ndarray]:
    mesh.allocate_field("rho", centering="node")
    for patch in mesh.local_patches:
        x, y = patch.local_node_coordinates(mesh, "rho")
        patch.field("rho").local[...] = source(x, y)
    solver = NodalPoisson(mesh, faces=faces)
    solver.solve("rho", potential="phi")
    return [mesh.gather(patch, "phi") for patch in mesh.patches]


def same(found: list, expected: list, what: str) -> None:
    check(len(found) == len(expected), f"{what}: number of patches")
    for number, (a, b) in enumerate(zip(found, expected, strict=True)):
        check(a is not None and a.shape == b.shape, f"{what}: patch {number} shape")
        check(
            np.allclose(a, b, rtol=0.0, atol=1e-12),
            f"{what}: patch {number} differs by {np.abs(a - b).max():.2e}",
        )


def _copy(mesh: AMRHierarchy) -> AMRHierarchy:
    """The same patches on this rank alone."""
    copy = AMRHierarchy(
        base_shape=mesh.base_shape,
        base_boxes=[p.box for p in mesh.patches_at_level(0)],
        comm=MPI.COMM_SELF,
    )
    for patch in mesh.patches:
        if patch.level:
            copy.add_patch(level=patch.level, box=patch.box)
    return copy


def main() -> None:
    check(SIZE >= 2, "the job needs at least two ranks")

    # 1. Where the patches are: the same table on every rank, data on its ranks only.
    mesh = rectangle(WORLD, placed=True)
    mesh.allocate_field("density")
    expected_ranks = [
        tuple(range(SIZE)),
        (SIZE - 1,),
        (0, 1),
        (SIZE - 1,),
        tuple(range(SIZE)),
    ]
    for patch, ranks in zip(mesh.patches, expected_ranks, strict=True):
        ranks = tuple(sorted(set(ranks)))
        check(
            patch.ranks == ranks,
            f"patch {patch.box} on {patch.ranks}, expected {ranks}",
        )
        check(
            patch.is_local == (RANK in ranks),
            f"patch {patch.box}: is_local on rank {RANK}",
        )
        if patch.is_local:
            check(
                patch.comm.Get_size() == len(ranks), "size of the patch's communicator"
            )
            local = patch.field("density").local.size
        else:
            local = 0
            try:
                patch.field("density")
            except RuntimeError as error:
                check(
                    "holds no part" in str(error), "the error names the missing patch"
                )
            else:
                check(False, "a rank without the patch got its field")
        cells = WORLD.allreduce(local, op=MPI.SUM)
        check(
            cells == patch.shape[0] * patch.shape[1],
            f"patch {patch.box}: {cells} cells held",
        )
    check(
        [p.box for p in mesh.local_patches]
        == [p.box for p in mesh.patches if RANK in p.ranks],
        "local_patches",
    )
    total = sum(p.shape[0] * p.shape[1] for p in mesh.patches)
    check(abs(sum(mesh.cells_on_rank) - total) < 1e-9, "cells_on_rank adds up")

    # 2. A patch gathered to every rank, or to one.
    for patch in mesh.local_patches:
        (i0, i1), (j0, j1) = patch.field("density").layout.index_bounds
        i, j = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
        patch.field("density").local[...] = 1000.0 * patch.level + 100.0 * i + j
    for patch in mesh.patches:
        i, j = np.meshgrid(*(np.arange(n) for n in patch.shape), indexing="ij")
        expected = 1000.0 * patch.level + 100.0 * i + j
        everywhere = mesh.gather(patch, "density")
        check(
            np.array_equal(everywhere, expected), f"gather of {patch.box} on every rank"
        )
        for root in (0, SIZE - 1):
            at_root = mesh.gather(patch, "density", root=root)
            if RANK == root:
                check(
                    np.array_equal(at_root, expected),
                    f"gather of {patch.box} to {root}",
                )
            else:
                check(at_root is None, f"gather to {root} returned data on rank {RANK}")

    # 3. Both solvers give what one rank gives, wherever the patches are.
    same(
        cell_solution(rectangle(WORLD, placed=True)),
        cell_solution(rectangle(MPI.COMM_SELF, placed=False)),
        "cell-centered solve",
    )
    faces = {"x_lower": Neumann(0.0), "x_upper": Dirichlet(lambda x, y: 1.0 + y)}
    same(
        nodal_solution(rectangle(WORLD, placed=True), faces),
        nodal_solution(rectangle(MPI.COMM_SELF, placed=False), faces),
        "node-centered solve",
    )
    inner = {"inner": Neumann(0.0), "y_upper": Dirichlet(2.0)}
    same(
        nodal_solution(ell(WORLD, placed=True), inner),
        nodal_solution(ell(MPI.COMM_SELF, placed=False), inner),
        "node-centered solve on an L over two ranks",
    )
    same(
        cell_solution(ell(WORLD, placed=True)),
        cell_solution(ell(MPI.COMM_SELF, placed=False)),
        "cell-centered solve on an L over two ranks",
    )

    # 4. One rank per patch, chosen by the cells each rank already holds.
    balanced = AMRHierarchy(
        base_shape=(32, 32),
        base_boxes=[
            ((0, 16), (0, 16)),
            ((16, 32), (0, 16)),
            ((0, 16), (16, 32)),
            ((16, 32), (16, 32)),
        ],
        placement="balanced",
        comm=WORLD,
    )
    for level, box, _ in PATCHES:
        balanced.add_patch(level=level, box=box)
    check(
        all(len(p.ranks) == 1 for p in balanced.patches), "balanced: one rank per patch"
    )
    held = balanced.cells_on_rank
    largest = max(p.shape[0] * p.shape[1] for p in balanced.patches)
    check(max(held) - min(held) <= largest, f"balanced: cells per rank {held}")
    check(
        len({p.ranks for p in balanced.patches}) == min(SIZE, 8),
        "balanced: every rank used",
    )
    same(
        cell_solution(balanced),
        cell_solution(_copy(balanced)),
        "cell-centered solve, balanced placement",
    )

    # 5. The plot is drawn on one rank from patches that live elsewhere.
    plotted = rectangle(WORLD, placed=True)
    plotted.allocate_field("phi", centering="node")
    result = plot_patches(plotted, field="phi", root=0)
    check(
        (result is not None) == (RANK == 0),
        "plot_patches returns the figure on the root",
    )

    if RANK == 0:
        print("MPI_JOB_OK", flush=True)


if __name__ == "__main__":
    main()
