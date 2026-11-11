"""Tests for the geometric multigrid over the patches (`MultigridBackend`)."""

from __future__ import annotations

import numpy as np
import pytest

from patchwork import (
    AMRHierarchy,
    CompositeNodes,
    Dirichlet,
    EmbeddedDirichlet,
    MultigridBackend,
    Neumann,
    NodalPoisson,
)
from patchwork.solvers.multigrid import (
    _unknowns,
    coarser_hierarchies,
    prolongation,
)


def _nested(n: int, levels: int, **kwargs) -> AMRHierarchy:
    """``n`` x ``n`` base cells and ``levels`` patches, each over the middle of the last."""
    mesh = AMRHierarchy(base_shape=(n, n), **kwargs)
    lower, upper = n // 4, 3 * n // 4
    for level in range(1, levels + 1):
        width = upper - lower
        lower, upper = 2 * lower + width // 2, 2 * lower + 3 * width // 2
        lower, upper = lower - lower % 2, upper - upper % 2
        mesh.add_patch(level=level, box=((lower, upper), (lower, upper + 2)))
    return mesh


def _tensor(solver: NodalPoisson) -> np.ndarray:
    """A smooth, anisotropic, positive definite coefficient per leaf cell."""
    x, y = solver.nodes.cell_centers.T
    tensor = np.zeros((x.size, 2, 2))
    tensor[:, 0, 0] = 1.0 + 10.0 * x * y
    tensor[:, 1, 1] = 1.0 + 5.0 * np.sin(3.0 * x) ** 2
    tensor[:, 0, 1] = tensor[:, 1, 0] = 0.3 * np.cos(y)
    return tensor


def _source(solver: NodalPoisson) -> np.ndarray:
    x, y = solver.nodes.coordinates.T
    return np.cos(3.0 * x) * np.sin(2.0 * np.pi * y)


def _iterations(n: int, levels: int, periodic: bool = False) -> tuple[int, float]:
    mesh = _nested(n, levels, periodic=(periodic, periodic))
    solver = NodalPoisson(mesh, backend="multigrid")
    solver.set_coefficient(_tensor(solver))
    potential = solver.solve(_source(solver))
    error = 0.0
    if not periodic:
        direct = NodalPoisson(mesh, nodes=solver.nodes, coefficient=_tensor(solver))
        exact = direct.solve(_source(solver))
        error = float(abs(potential - exact).max() / abs(exact).max())
    return solver.solver.iterations, error


@pytest.mark.parametrize("periodic", [False, True])
def test_iterations_do_not_grow_with_levels_or_cells(periodic: bool) -> None:
    counts = {}
    for n in (32, 64):
        for levels in (0, 1, 2, 3):
            counts[n, levels], error = _iterations(n, levels, periodic)
            assert error < 1e-9
    assert max(counts.values()) <= 13
    assert max(counts.values()) - min(counts.values()) <= 3


def test_the_spaces_are_nested_and_the_interpolation_exact_for_linears() -> None:
    mesh = _nested(16, 2)
    hierarchies = coarser_hierarchies(mesh, unknowns=50)
    # two levels dropped, then the base level halved twice (16 -> 8 -> 4)
    assert [h.max_level for h in hierarchies] == [1, 0, 0, 0]
    assert [h.base_shape for h in hierarchies[1:]] == [(16, 16), (8, 8), (4, 4)]
    # Neumann faces: every node is an unknown (a Dirichlet face's nodes are not,
    # and the coarse correction is zero there)
    faces = {
        face: Neumann(0.0) for face in ("x_lower", "x_upper", "y_lower", "y_upper")
    }
    solver = NodalPoisson(mesh, faces=faces, backend="multigrid")
    fine, fine_unknowns = solver.nodes, solver.free
    for hierarchy in hierarchies:
        coarse = CompositeNodes(hierarchy)
        coarse_unknowns = _unknowns(coarse, solver.system)
        matrix = prolongation(coarse, coarse_unknowns, fine, fine_unknowns)
        x, y = coarse.coordinates[coarse_unknowns].T
        fx, fy = fine.coordinates[fine_unknowns].T
        assert np.allclose(matrix @ (2.0 * x - y + 1.0), 2.0 * fx - fy + 1.0)
        fine, fine_unknowns = coarse, coarse_unknowns


def test_conditions_layouts_and_masses_match_the_lu() -> None:
    body = EmbeddedDirichlet(
        lambda points: (points[0] - 0.3) ** 2 + (points[1] - 0.6) ** 2 < 0.02, 1.0
    )
    cases = [
        ({"faces": {"x_lower": Neumann(1.0), "y_upper": Dirichlet(0.5)}}, (32, 32), {}),
        ({"mass": "consistent"}, (32, 32), {}),
        ({"embedded": (body,)}, (32, 32), {}),
        ({}, (32, 32), {"refinement_ratio": 4}),
        ({}, (32, 32), {"geometry": "cylindrical"}),
        ({}, (32, 32), {"base_boxes": [((0, 32), (0, 16)), ((0, 16), (16, 32))]}),
        ({}, (30, 34), {}),
    ]
    for options, shape, mesh_options in cases:
        mesh = AMRHierarchy(base_shape=shape, **mesh_options)
        ratio = mesh.refinement_ratio
        mesh.add_patch(level=1, box=((4 * ratio, 12 * ratio), (2 * ratio, 10 * ratio)))
        direct = NodalPoisson(mesh, **options)
        source = _source(direct)
        exact = direct.solve(source)
        multigrid = NodalPoisson(
            mesh, nodes=direct.nodes, backend="multigrid", **options
        )
        assert np.allclose(multigrid.solve(source), exact, atol=1e-9 * abs(exact).max())
        assert multigrid.solver.iterations <= 20, (options, mesh_options)


def test_a_line_and_a_box() -> None:
    line = AMRHierarchy(base_shape=(64,))
    line.add_patch(level=1, box=((40, 80),))
    box = AMRHierarchy(base_shape=(16, 16, 16))
    box.add_patch(level=1, box=((8, 24),) * 3)
    for mesh in (line, box):
        direct = NodalPoisson(mesh, coefficient=2.0)
        source = np.sin(np.pi * direct.nodes.coordinates[:, 0])
        exact = direct.solve(source)
        multigrid = NodalPoisson(
            mesh,
            nodes=direct.nodes,
            coefficient=2.0,
            backend=MultigridBackend(coarsest_unknowns=20),
        )
        assert np.allclose(multigrid.solve(source), exact, atol=1e-9 * abs(exact).max())
        assert multigrid.solver.num_levels > 2
        assert multigrid.solver.iterations <= 12


def test_a_singular_operator_gives_the_solution_of_zero_mean() -> None:
    mesh = _nested(32, 1, periodic=(True, True))
    multigrid = NodalPoisson(mesh, backend="multigrid")
    source = _source(multigrid)
    potential = multigrid.solve(source)
    volumes = multigrid.nodes.volumes
    assert abs(volumes @ potential) < 1e-12
    right = multigrid.load @ source
    residual = multigrid.matrix @ potential[multigrid.free] - (right - right.mean())
    assert np.linalg.norm(residual) < 1e-9 * np.linalg.norm(right)
    # a uniform periodic mesh: the FFT's answer
    uniform = AMRHierarchy(base_shape=(32, 32), periodic=(True, True))
    fft = NodalPoisson(uniform, backend="fft")
    multigrid = NodalPoisson(uniform, nodes=fft.nodes, backend="multigrid")
    source = _source(fft)
    assert np.allclose(multigrid.solve(source), fft.solve(source), atol=1e-10)


def test_set_coefficient_keeps_the_interpolations() -> None:
    mesh = _nested(32, 2)
    solver = NodalPoisson(mesh, backend=MultigridBackend(rtol=1e-12))
    built = solver.solver
    interpolations = built.interpolations
    solver.set_coefficient(_tensor(solver))
    assert solver.solver is built and solver.solver.interpolations is interpolations
    direct = NodalPoisson(mesh, nodes=solver.nodes, coefficient=_tensor(solver))
    source = _source(solver)
    assert np.allclose(solver.solve(source), direct.solve(source), atol=1e-11)


def test_one_cycle_is_a_preconditioner() -> None:
    import scipy.sparse.linalg

    mesh = _nested(32, 2)
    solver = NodalPoisson(mesh, backend="multigrid", coefficient=3.0)
    cycle = solver.solver
    b = solver.load @ _source(solver)
    # one V-cycle reduces the residual by a factor of about ten
    x = cycle.cycle(b)
    assert np.linalg.norm(b - solver.matrix @ x) < 0.2 * np.linalg.norm(b)
    x, info = scipy.sparse.linalg.cg(
        solver.matrix, b, rtol=1e-10, M=cycle.as_linear_operator()
    )
    assert info == 0
    assert np.allclose(x, cycle.solve(b), atol=1e-8 * abs(x).max())


def test_no_convergence_and_zero_right_hand_sides() -> None:
    mesh = _nested(32, 1)
    solver = NodalPoisson(mesh, backend=MultigridBackend(max_iterations=2))
    with pytest.raises(RuntimeError, match="did not converge"):
        solver.solve(_source(solver))
    assert not solver.solver.solve(np.zeros(solver.free.size)).any()
    body = EmbeddedDirichlet(lambda points: points[0] > 0.8, 1.0)
    bodies = NodalPoisson(
        mesh, embedded=(body,), backend=MultigridBackend(max_iterations=1)
    )
    assert not bodies.solver.solve(np.zeros(bodies.free.size)).any()
    with pytest.raises(RuntimeError, match="did not converge"):
        bodies.solve(_source(bodies))
    small = NodalPoisson(AMRHierarchy(base_shape=(8, 8)), backend="multigrid")
    assert small.solver.num_levels == 1
    assert "rtol=1e-10" in repr(MultigridBackend())
