"""The distributed (PETSc) backend of the node-centered solver, on one rank and under ``mpirun``."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from patchwork import (
    AMRHierarchy,
    CompositeNodes,
    Dirichlet,
    EmbeddedDirichlet,
    Neumann,
    NodalPoisson,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("petsc4py") is None, reason="needs petsc4py"
)

JOB = Path(__file__).with_name("_mpi_distributed_job.py")
MPIRUN = shutil.which("mpirun")
FACES = ("x_lower", "x_upper", "y_lower", "y_upper")

LAYOUTS = {
    "uniform": [],
    "nested": [(1, ((8, 24), (8, 24))), (2, ((24, 40), (24, 40)))],
    "level jump": [(1, ((8, 24), (8, 24))), (2, ((16, 32), (16, 32)))],
    "two patches": [(1, ((4, 12), (4, 12))), (1, ((16, 28), (12, 30)))],
}


def _mesh(layout: str, base: int = 16, **kwargs) -> AMRHierarchy:
    mesh = AMRHierarchy(base_shape=(base, base), **kwargs)
    scale = base // 16
    for level, box in LAYOUTS[layout]:
        mesh.add_patch(
            level=level, box=tuple((lo * scale, hi * scale) for lo, hi in box)
        )
    return mesh


def _source(x, y):
    return np.exp(-((x - 0.4) ** 2 + (y - 0.55) ** 2) / 0.02)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_petsc_gives_what_superlu_gives(layout: str) -> None:
    mesh = _mesh(layout)
    faces = {"x_lower": Neumann(0.5), "x_upper": Dirichlet(lambda x, y: 1.0 + y)}
    nodes = CompositeNodes(mesh)
    direct = NodalPoisson(mesh, faces=faces, nodes=nodes)
    iterative = NodalPoisson(mesh, faces=faces, nodes=nodes, backend="petsc")
    x, y = nodes.coordinates.T

    expected = direct.solve(_source(x, y))
    found = iterative.solve(_source(x, y))

    assert np.abs(found - expected).max() < 1e-8 * np.abs(expected).max()
    assert not iterative.system.fell_back
    # one rank holds every row: the same system in the same order
    assert np.array_equal(iterative.rows, direct.free)
    assert abs(iterative.matrix - direct.matrix).max() < 1e-12


def test_bodies_charges_and_fields_match_superlu() -> None:
    mesh = _mesh("nested", periodic=(False, True))
    body = EmbeddedDirichlet(
        lambda p: (p[0] - 0.62) ** 2 + (p[1] - 0.64) ** 2 < 0.06**2, 2.0
    )
    direct = NodalPoisson(mesh, embedded=[body])
    iterative = NodalPoisson(mesh, embedded=[body], backend="petsc")
    rng = np.random.default_rng(3)
    points = rng.uniform(0.0, 1.0, (2, 500))
    charge = direct.nodes.deposit(points, np.full(500, 1.0e-2))

    expected = direct.solve(charge=charge)
    found = iterative.solve(charge=charge, potential="phi")

    assert np.abs(found - expected).max() < 1e-8 * np.abs(expected).max()
    assert iterative.system.ksp.getType() == "gmres"  # Shortley-Weller rows
    assert np.array_equal(iterative.nodes.from_patches("phi"), found)


def test_the_distributed_solution_converges_at_second_order() -> None:
    def exact(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y) + x * y

    def source(x, y):
        return 2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

    errors = []
    for base in (16, 32, 64):
        solver = NodalPoisson(
            _mesh("level jump", base),
            faces={face: Dirichlet(exact) for face in FACES},
            backend="petsc",
        )
        x, y = solver.nodes.coordinates.T
        errors.append(np.abs(solver.solve(source(x, y)) - exact(x, y)).max())

    assert errors[0] / errors[1] > 3.3 and errors[1] / errors[2] > 3.3
    assert errors[2] < 5e-4


def test_a_solve_that_does_not_converge_falls_back() -> None:
    mesh = _mesh("nested")
    direct = NodalPoisson(mesh)
    iterative = NodalPoisson(
        mesh, backend="petsc", petsc_options={"ksp_max_it": 1, "pc_type": "none"}
    )
    x, y = direct.nodes.coordinates.T

    found = iterative.solve(_source(x, y))

    assert iterative.system.fell_back
    assert np.abs(found - direct.solve(_source(x, y))).max() < 1e-7


def test_an_unknown_backend_is_refused() -> None:
    with pytest.raises(ValueError, match="backend"):
        NodalPoisson(_mesh("uniform"), backend="mumps")


@pytest.mark.skipif(
    MPIRUN is None or importlib.util.find_spec("mpi4py") is None,
    reason="needs mpirun and mpi4py",
)
@pytest.mark.parametrize("ranks", [2, 3, 4])
def test_the_distributed_solve_gives_what_one_rank_gives(ranks: int) -> None:
    """Rows on the ranks of the patches; see the job's docstring."""
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        env.update(OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
    result = subprocess.run(
        [MPIRUN, "-n", str(ranks), "--oversubscribe", sys.executable, str(JOB)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env=env,
    )

    assert "MPI_JOB_OK" in result.stdout, result.stdout[-2000:] + result.stderr[-2000:]


@pytest.mark.parametrize("periodic", [False, True])
def test_petsc_on_a_line_gives_what_superlu_gives(periodic: bool) -> None:
    mesh = AMRHierarchy(base_shape=(16,), periodic=(periodic,))
    mesh.add_patch(level=1, box=((8, 24),))
    mesh.add_patch(level=2, box=((24, 40),))
    faces = {} if periodic else {"x_lower": Neumann(0.5), "x_upper": Dirichlet(1.0)}
    body = EmbeddedDirichlet(lambda p: np.abs(p[0] - 0.71) < 0.03, 2.0)
    nodes = CompositeNodes(mesh)
    direct = NodalPoisson(mesh, faces=faces, embedded=[body], nodes=nodes)
    iterative = NodalPoisson(
        mesh, faces=faces, embedded=[body], nodes=nodes, backend="petsc"
    )
    (x,) = nodes.coordinates.T
    source = np.exp(-((x - 0.4) ** 2) / 0.02)

    expected = direct.solve(source)
    found = iterative.solve(source)

    assert np.abs(found - expected).max() < 1e-8 * np.abs(expected).max()
    assert np.array_equal(iterative.rows, direct.free)
