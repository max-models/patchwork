"""The geometric multigrid backend: the LU's answers, at a multigrid rate."""

from __future__ import annotations

import numpy as np
import pytest

from patchwork import (
    AMRHierarchy,
    Dirichlet,
    EmbeddedDirichlet,
    MultigridBackend,
    Neumann,
    NodalPoisson,
)
from patchwork.multigrid import MultigridSolver, prolongation_1d, prolongations

pytest.importorskip("petsc4py")


@pytest.mark.parametrize("num_nodes", [5, 6, 9, 10, 96, 97])
def test_the_interpolation_is_linear_and_keeps_both_ends(num_nodes: int) -> None:
    p = prolongation_1d(num_nodes, periodic=False).toarray()
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    coarse = np.flatnonzero(p.max(axis=1) == 1.0)
    np.testing.assert_allclose(p @ coarse.astype(float), np.arange(num_nodes))
    assert coarse[0] == 0
    assert coarse[-1] == num_nodes - 1


def test_the_periodic_interpolation_wraps_around() -> None:
    p = prolongation_1d(8, periodic=True).toarray()
    assert p.shape == (8, 4)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    np.testing.assert_allclose(p[7], [0.5, 0.0, 0.0, 0.5])


def test_short_and_odd_periodic_axes_are_not_coarsened() -> None:
    assert prolongation_1d(3, periodic=False) is None
    assert prolongation_1d(9, periodic=True) is None
    hierarchy = prolongations((97, 33), (False, True))
    assert hierarchy[0].shape == (97 * 33, 49 * 33)
    assert prolongations((20, 20), (False, False)) == []  # small: one level


def _disc(points: np.ndarray) -> np.ndarray:
    return (points[0] - 0.45) ** 2 + (points[1] - 0.6) ** 2 < 0.2**2


CASES = {
    "1d dirichlet": (
        lambda: AMRHierarchy(base_shape=(4000,), bounds=((0.0, 1.0),)),
        {"x_lower": Dirichlet(1.0), "x_upper": Dirichlet(lambda x: 3.0 * x)},
        [],
    ),
    "2d dirichlet, odd cells": (
        lambda: AMRHierarchy(base_shape=(97, 70), bounds=((0.0, 1.0), (0.0, 1.3))),
        {"x_lower": Dirichlet(1.0), "x_upper": Dirichlet(lambda x, y: y)},
        [],
    ),
    "2d periodic y, neumann x": (
        lambda: AMRHierarchy(
            base_shape=(80, 64), bounds=((0.0, 1.0), (0.0, 1.3)), periodic=(False, True)
        ),
        {"x_lower": Neumann(0.5), "x_upper": Dirichlet(-1.0)},
        [],
    ),
    "2d with a body": (
        lambda: AMRHierarchy(base_shape=(96, 96), bounds=((0.0, 1.0), (0.0, 1.2))),
        {},
        [EmbeddedDirichlet(_disc, 1.0)],
    ),
    "2d cylindrical with a body": (
        lambda: AMRHierarchy(
            base_shape=(96, 64),
            bounds=((0.0, 1.0), (0.0, 1.2)),
            geometry="cylindrical",
        ),
        {"x_upper": Dirichlet(2.0)},
        [EmbeddedDirichlet(_disc, 1.0)],
    ),
}


@pytest.mark.parametrize("case", CASES)
def test_multigrid_gives_the_lu_solution_at_a_multigrid_rate(case: str) -> None:
    mesh, faces, embedded = CASES[case]
    mesh = mesh()
    direct = NodalPoisson(mesh, faces=faces, embedded=embedded)
    multigrid = NodalPoisson(
        mesh, faces=faces, embedded=embedded, nodes=direct.nodes, backend="multigrid"
    )
    source = np.random.default_rng(1).standard_normal(direct.nodes.num_nodes)

    expected = direct.solve(source)
    found = multigrid.solve(source)

    solver = multigrid.solver
    assert isinstance(solver, MultigridSolver)
    assert solver.num_levels > 1
    assert solver.iterations < 25  # a multigrid rate, not a Krylov one
    assert np.abs(found - expected).max() < 1e-8 * np.abs(expected).max()
    # the previous solution is the next initial guess: a second solve is cheap
    multigrid.solve(source)
    assert solver.iterations <= 1
    solver.reset()
    multigrid.solve(source)
    assert solver.iterations > 1


def test_a_small_mesh_is_one_level_solved_directly() -> None:
    mesh = AMRHierarchy(base_shape=(20, 20))
    faces = {"x_lower": Dirichlet(0.0)}
    multigrid = NodalPoisson(mesh, faces=faces, backend="multigrid")
    direct = NodalPoisson(mesh, faces=faces, nodes=multigrid.nodes)
    source = np.ones(direct.nodes.num_nodes)
    assert multigrid.solver.num_levels == 1
    np.testing.assert_allclose(multigrid.solve(source), direct.solve(source), atol=1e-9)


def _refined() -> AMRHierarchy:
    mesh = AMRHierarchy(base_shape=(8, 8))
    mesh.add_patch(level=1, box=((4, 12), (4, 12)))
    return mesh


@pytest.mark.parametrize(
    ("mesh", "faces", "reason"),
    [
        (_refined, {"x_lower": Dirichlet(0.0)}, "uniform mesh"),
        (
            lambda: AMRHierarchy(
                base_shape=(8, 8), base_boxes=[((0, 4), (0, 8)), ((4, 8), (0, 4))]
            ),
            {"x_lower": Dirichlet(0.0)},
            "fills its box",
        ),
        (lambda: AMRHierarchy(base_shape=(8, 8)), {}, "singular"),
    ],
)
def test_multigrid_refuses_what_it_cannot_solve(mesh, faces, reason) -> None:
    neumann = ("x_lower", "x_upper", "y_lower", "y_upper")
    faces = {name: Neumann(0.0) for name in neumann} | faces
    with pytest.raises(ValueError, match=reason):
        NodalPoisson(mesh(), faces=faces, backend="multigrid")


def test_options_reach_the_krylov_solver() -> None:
    mesh = AMRHierarchy(base_shape=(64, 64))
    backend = MultigridBackend(rtol=1e-6, options={"ksp_type": "gmres"})
    poisson = NodalPoisson(mesh, faces={"x_lower": Dirichlet(0.0)}, backend=backend)
    assert poisson.solver.ksp.getType() == "gmres"
    assert "rtol=1e-06" in repr(backend)
