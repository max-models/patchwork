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


@pytest.mark.parametrize("smoother", ["chebyshev", "gauss-seidel"])
def test_a_singular_operator_gives_the_solution_of_zero_mean(smoother: str) -> None:
    mesh = _nested(32, 1, periodic=(True, True))
    multigrid = NodalPoisson(mesh, backend=MultigridBackend(smoother=smoother))
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
    multigrid = NodalPoisson(
        uniform, nodes=fft.nodes, backend=MultigridBackend(smoother=smoother)
    )
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


def _box_3d(shape: tuple[int, int, int], **kwargs) -> NodalPoisson:
    """A 3D box: Dirichlet on the x faces, Neumann on the others, a body in the middle."""
    mesh = AMRHierarchy(base_shape=shape)
    faces = {"x_lower": Dirichlet(0.0), "x_upper": Dirichlet(1.0)}
    for face in ("y_lower", "y_upper", "z_lower", "z_upper"):
        faces[face] = Neumann(0.0)
    body = EmbeddedDirichlet(
        lambda p: (np.abs(p[0] - 0.5) < 0.06) & (p[1] ** 2 + p[2] ** 2 > 0.35**2), 0.3
    )
    return NodalPoisson(mesh, faces=faces, embedded=(body,), **kwargs)


def test_linear_interpolation_on_an_odd_lattice() -> None:
    from patchwork.solvers.multigrid import _linear_1d

    for n, coarse in ((2, [0, 1]), (3, [0, 2]), (6, [0, 2, 4, 5]), (7, [0, 2, 4, 6])):
        matrix = _linear_1d(n)
        assert matrix.shape == (n, len(coarse))
        line = 0.3 + 2.0 * np.arange(n)
        assert np.allclose(matrix @ line[coarse], line)  # exact for linears


@pytest.mark.parametrize("shape", [(24, 12, 12), (25, 13, 11)])
def test_odd_base_levels_coarsen_on_their_lattice(shape) -> None:
    """A base level `_halved` cannot coarsen goes on with linear interpolation on its nodes."""
    solver = _box_3d(shape, backend=MultigridBackend(rtol=1e-10, coarsest_unknowns=100))
    multigrid = solver.solver
    assert multigrid.num_levels >= 3
    direct = _box_3d(shape)
    source = np.ones(solver.nodes.num_nodes)
    assert np.allclose(solver.solve(source), direct.solve(source), atol=1e-8)
    assert multigrid.iterations < 30


def test_gauss_seidel_cycles_compiled_and_uncompiled_agree() -> None:
    import cunumpy as xp

    solver = _box_3d(
        (25, 13, 11),
        backend=MultigridBackend(
            smoother="gauss-seidel", smoothing_steps=1, coarsest_unknowns=100
        ),
    )
    multigrid = solver.solver
    assert multigrid._compiled is not None
    b = np.random.default_rng(0).standard_normal(multigrid.matrix.shape[0])
    arguments, multigrid._compiled = multigrid._compiled, None
    uncompiled = multigrid.cycle(b)
    multigrid._compiled = arguments
    results = []
    for implementation in ("numpy", "python", "pyccel"):
        try:
            with xp.kernels.use_host_kernel_implementation(implementation):
                results.append(multigrid.cycle(b))
        except Exception:  # noqa: BLE001 -- no Pyccel, or no compiler
            assert implementation == "pyccel"
    for result in results:
        assert np.allclose(
            result, uncompiled, rtol=0, atol=1e-12 * abs(uncompiled).max()
        )
    direct = _box_3d((25, 13, 11))
    source = np.ones(solver.nodes.num_nodes)
    assert np.allclose(solver.solve(source), direct.solve(source), atol=1e-8)
    with pytest.raises(ValueError, match="smoother"):
        MultigridBackend(smoother="jacobi").setup(solver.system)


def test_a_new_diagonal_is_taken_without_the_galerkin_product() -> None:
    """`update_diagonal`, `update_operator` and `set_reaction` give the levels of a full update."""
    import scipy.sparse as sparse

    from patchwork.solvers.multigrid import _galerkin_of_diagonal

    backend = MultigridBackend(
        smoother="gauss-seidel", smoothing_steps=1, coarsest_unknowns=100
    )
    solver = _box_3d((25, 13, 11), backend=backend)
    multigrid = solver.solver
    b = np.random.default_rng(1).standard_normal(multigrid.matrix.shape[0])
    reaction = np.random.default_rng(2).uniform(0.0, 50.0, solver.nodes.num_nodes)
    solver.set_reaction(reaction)  # through update(system): only the diagonal changed
    assert multigrid._diagonal_map is not None
    fast = multigrid.cycle(b)
    jacobian = multigrid.matrix.copy()
    multigrid._reference = None
    solver.set_reaction(reaction)  # a full update
    assert multigrid._diagonal_map is None
    full = multigrid.cycle(b)
    assert np.allclose(fast, full, rtol=0, atol=1e-12 * abs(full).max())
    solver.set_reaction(None)
    multigrid.update_diagonal(jacobian.diagonal())
    assert np.allclose(multigrid.cycle(b), full, rtol=0, atol=1e-12 * abs(full).max())
    multigrid.update_operator(jacobian)
    assert np.allclose(multigrid.cycle(b), full, rtol=0, atol=1e-12 * abs(full).max())
    # the map itself: P^T diag(d) P on the pattern
    p = multigrid.interpolations[0]
    d = np.random.default_rng(3).uniform(size=p.shape[0])
    pattern, mapping = _galerkin_of_diagonal(p, sparse.csr_matrix(p.T @ p))
    product = sparse.csr_matrix(
        (mapping @ d, pattern.indices, pattern.indptr), shape=pattern.shape
    )
    assert abs(product - p.T @ sparse.diags(d) @ p).max() < 1e-14
    multigrid._reference = None
    with pytest.raises(ValueError, match="full update"):
        multigrid.update_diagonal(jacobian.diagonal())


def test_a_singular_3d_box_with_the_compiled_cycle() -> None:
    """Neumann on every face: the compiled cycle's coarsest solve pins the constant."""
    mesh = AMRHierarchy(base_shape=(17, 9, 9))
    faces = {
        face: Neumann(0.0)
        for face in ("x_lower", "x_upper", "y_lower", "y_upper", "z_lower", "z_upper")
    }
    backend = MultigridBackend(
        smoother="gauss-seidel", smoothing_steps=1, coarsest_unknowns=50
    )
    solver = NodalPoisson(mesh, faces=faces, backend=backend)
    assert solver.solver._compiled is not None and solver.solver.singular
    x, y, z = solver.nodes.coordinates.T
    source = np.cos(np.pi * x) * np.cos(np.pi * y)
    potential = solver.solve(source)
    assert abs(solver.nodes.volumes @ potential) < 1e-12
    right = solver.load @ source
    residual = solver.matrix @ potential[solver.free] - (right - right.mean())
    assert np.linalg.norm(residual) < 1e-8 * np.linalg.norm(right)
    solver.set_coefficient(2.0)  # not only the diagonal: a full update
    assert solver.solver._diagonal_map is None


def _independent(matrix, groups) -> bool:
    """Whether the colours cover every row once and no two coupled rows share one."""
    import scipy.sparse as sparse

    size = matrix.shape[0]
    colour = np.full(size, -1)
    for number, group in enumerate(groups):
        colour[group] = number
    coo = sparse.coo_matrix(matrix)
    off = coo.row != coo.col
    return (
        sum(g.size for g in groups) == size
        and (colour >= 0).all()
        and not np.any(colour[coo.row[off]] == colour[coo.col[off]])
    )


def test_colours_are_parities_on_lattices_and_independent_sets_elsewhere() -> None:
    from patchwork.solvers.multigrid import _colour_groups

    backend = MultigridBackend(
        smoother="gauss-seidel", smoothing_steps=1, coarsest_unknowns=50
    )
    box = _box_3d((24, 13, 11), backend=backend).solver
    assert len(box._groups[0]) <= 2  # red-black on the seven-point stencil
    assert all(len(groups) <= 8 for groups in box._groups[1:])  # 27 points: parities
    for level, groups in zip(box.levels, box._groups, strict=True):
        assert _independent(level.matrix, groups)
    # refined patches: no lattice parity works on the composite level, the general colours do
    nested = NodalPoisson(
        _nested(32, 2), backend=MultigridBackend(smoother="gauss-seidel")
    )
    for level, groups in zip(nested.solver.levels, nested.solver._groups, strict=True):
        assert _independent(level.matrix, groups)
    matrix = nested.solver.levels[0].matrix
    assert _independent(matrix, _colour_groups(matrix))  # without coordinates
    assert _colour_groups(matrix[:0, :0]) == []


def test_the_compiled_cycle_does_not_depend_on_the_threads() -> None:
    backend = MultigridBackend(
        smoother="gauss-seidel", smoothing_steps=1, coarsest_unknowns=50
    )
    multigrid = _box_3d((25, 13, 11), backend=backend).solver
    b = np.random.default_rng(4).standard_normal(multigrid.matrix.shape[0])
    results = []
    for threads in (1, 3):
        multigrid.num_threads = threads
        results.append(multigrid.cycle(b))
    assert np.array_equal(results[0], results[1])
    system = _box_3d((8, 8, 8)).system
    assert MultigridBackend(num_threads=2).setup(system).num_threads == 2


def test_the_interpolation_has_no_round_off_weights() -> None:
    from patchwork.solvers.multigrid import ROUND_OFF_WEIGHT

    solver = NodalPoisson(AMRHierarchy(base_shape=(32, 16, 16)), backend="multigrid")
    for interpolation in solver.solver.interpolations:
        assert np.abs(interpolation.data).min() >= ROUND_OFF_WEIGHT


@pytest.mark.parametrize("levels", [0, 2])
def test_a_reaction_is_taken_without_the_finest_galerkin_products(levels: int) -> None:
    """`update_reaction` gives the operator and the cycle of `set_reaction`, refined too."""
    backend = MultigridBackend(smoother="gauss-seidel", smoothing_steps=1)
    solver = NodalPoisson(_nested(32, levels), backend=backend)
    multigrid = solver.solver
    reaction = np.random.default_rng(6).uniform(0.0, 40.0, solver.nodes.num_nodes)
    b = np.random.default_rng(7).standard_normal(multigrid.matrix.shape[0])
    multigrid.update_reaction(solver.volumes * reaction)
    taken, fast = multigrid.matrix.copy(), multigrid.cycle(b)
    solver.set_reaction(reaction)  # assembled and reduced by NodalPoisson
    assert abs(taken - solver.matrix).max() < 1e-12 * abs(solver.matrix).max()
    multigrid._reference = None
    solver.set_reaction(reaction)  # a full update
    full = multigrid.cycle(b)
    assert np.allclose(fast, full, rtol=0, atol=1e-11 * abs(full).max())
    solver.set_reaction(None)
    without = NodalPoisson(_nested(32, levels), backend=backend).solver
    without._stiffness = None
    with pytest.raises(ValueError, match="without a reaction"):
        without.update_reaction(np.zeros(solver.nodes.num_nodes))
