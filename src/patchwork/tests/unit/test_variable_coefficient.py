"""Tests for the nodal operator with a variable, symmetric tensor coefficient."""

from __future__ import annotations

import numpy as np
import pytest
import sympy

from patchwork import (
    AMRHierarchy,
    CompositeEdges,
    EmbeddedDirichlet,
    Neumann,
    NodalPoisson,
)
from patchwork.spaces.mass import edge_cross_blocks, stiffness_blocks


def _mesh(n: int = 8, **kwargs) -> AMRHierarchy:
    mesh = AMRHierarchy(base_shape=(n, n), **kwargs)
    mesh.add_patch(level=1, box=((n // 2, 3 * n // 2), (n // 2, 5 * n // 4)))
    mesh.add_patch(level=2, box=((5 * n // 4, 9 * n // 4), (5 * n // 4, 2 * n)))
    return mesh


def _centers(solver: NodalPoisson) -> np.ndarray:
    return solver.nodes.cell_centers.T


@pytest.mark.parametrize("mass", ["lumped", "consistent"])
def test_unit_coefficients_give_the_laplacian(mass: str) -> None:
    plain = NodalPoisson(_mesh(), mass=mass)
    edges = CompositeEdges(plain.nodes)
    for coefficient in (
        {"coefficient": 1.0},
        {"coefficient": np.ones((plain.nodes.cell_size.shape[0], 2))},
        {"edge_coefficient": np.ones(edges.num_edges)},
    ):
        weighted = NodalPoisson(
            plain.nodes.hierarchy,
            mass=mass,
            nodes=plain.nodes,
            edges=edges,
            **coefficient,
        )
        assert abs(weighted.matrix - plain.matrix).max() < 1e-13


@pytest.mark.parametrize("mass", ["lumped", "consistent"])
def test_a_scalar_coefficient_is_the_weighted_edge_mass(mass: str) -> None:
    solver = NodalPoisson(_mesh(), mass=mass)
    nodes = solver.nodes
    edges = CompositeEdges(nodes)
    weight = 1.0 + np.random.default_rng(1).random(nodes.cell_size.shape[0])
    solver.set_coefficient(weight)
    reduction, gradient = nodes.reduction, edges.gradient
    lumped = mass == "lumped"
    full = reduction.T @ gradient.T @ edges.mass(lumped, weight) @ gradient @ reduction
    assert abs(solver.matrix - full[solver.free][:, solver.free]).max() < 1e-13


def sparse_diagonal(values: np.ndarray):
    import scipy.sparse as sparse

    return sparse.diags(values)


def test_an_edge_coefficient_scales_the_edge_conductances() -> None:
    solver = NodalPoisson(_mesh())
    edges = CompositeEdges(solver.nodes)
    kappa = 1.0 + np.random.default_rng(2).random(edges.num_edges)
    solver.set_coefficient(edge_coefficient=kappa)
    nodes, gradient = solver.nodes, edges.gradient
    lumped = sparse_diagonal(edges.mass(lumped=True).diagonal() * kappa)
    full = nodes.reduction.T @ gradient.T @ lumped @ gradient @ nodes.reduction
    assert abs(solver.matrix - full[solver.free][:, solver.free]).max() < 1e-13


@pytest.mark.parametrize("lumped", [True, False])
def test_the_stiffness_is_the_energy_of_a_linear_potential(lumped: bool) -> None:
    # grad u . K grad u over the cell, for a linear u and a constant tensor
    rng = np.random.default_rng(3)
    for ndim in (2, 3):
        size = 0.5 + rng.random((4, ndim))
        root = rng.standard_normal((4, ndim, ndim))
        tensor = root @ np.swapaxes(root, 1, 2) + np.eye(ndim)
        slope = rng.standard_normal(ndim)
        corners = np.array(
            [[(k >> axis) & 1 for axis in range(ndim)] for k in range(2**ndim)]
        )
        values = (corners[None] * size[:, None, :]) @ slope  # (4, corners)
        blocks = stiffness_blocks(size, lumped, tensor)
        energy = np.einsum("mi,mij,mj->m", values, blocks, values)
        exact = np.prod(size, axis=1) * np.einsum("i,mij,j->m", slope, tensor, slope)
        assert np.allclose(energy, exact, rtol=1e-12)


def test_the_cross_block_is_the_integral_of_the_product() -> None:
    # 2D: int E_x E_y over the cell = (e_lower + e_upper)(e_left + e_right) / 4
    size = np.array([[0.7, 1.3]])
    block = edge_cross_blocks(size, 0, 1, lumped=True)[0]
    assert np.allclose(block, 0.25)
    assert np.allclose(edge_cross_blocks(size, 0, 1, lumped=False)[0], 0.25)
    with pytest.raises(ValueError, match="different axes"):
        edge_cross_blocks(size, 1, 1, lumped=True)


def test_a_positive_tensor_gives_a_symmetric_positive_operator() -> None:
    solver = NodalPoisson(_mesh())
    x, y = _centers(solver)
    tensor = np.zeros((x.size, 2, 2))
    tensor[:, 0, 0] = 2.0 + x
    tensor[:, 1, 1] = 1.0 + y
    tensor[:, 0, 1] = tensor[:, 1, 0] = 0.9 * np.sqrt(tensor[:, 0, 0] * tensor[:, 1, 1])
    solver.set_coefficient(tensor)
    matrix = solver.matrix.toarray()
    assert np.allclose(matrix, matrix.T)
    assert np.linalg.eigvalsh(matrix).min() > 0.0


def _manufactured(n: int, mass: str) -> float:
    """The error of a manufactured solution of -div(K grad u) = f with a full tensor."""
    mesh = AMRHierarchy(base_shape=(n, n))
    mesh.add_patch(level=1, box=((n // 2, 3 * n // 2),) * 2)
    mesh.add_patch(level=2, box=((3 * n // 2, 5 * n // 2),) * 2)
    solver = NodalPoisson(mesh, mass=mass)
    X, Y = sympy.symbols("x y")
    exact = sympy.sin(sympy.pi * X) * sympy.sin(sympy.pi * Y)
    tensor = [[1 + X * Y, X / 3], [X / 3, 2 + sympy.sin(X)]]
    gradient = [sympy.diff(exact, X), sympy.diff(exact, Y)]
    flux = [sum(tensor[i][j] * gradient[j] for j in range(2)) for i in range(2)]
    source = -(sympy.diff(flux[0], X) + sympy.diff(flux[1], Y))
    x, y = _centers(solver)
    values = np.empty((x.size, 2, 2))
    for i in range(2):
        for j in range(2):
            values[:, i, j] = sympy.lambdify((X, Y), tensor[i][j])(x, y)
    solver.set_coefficient(values)
    px, py = solver.nodes.coordinates.T
    difference = solver.solve(sympy.lambdify((X, Y), source)(px, py)) - sympy.lambdify(
        (X, Y), exact
    )(px, py)
    return float(np.sqrt(np.sum(difference**2 * solver.nodes.volumes)))


@pytest.mark.parametrize("mass", ["lumped", "consistent"])
def test_a_manufactured_solution_with_a_tensor_converges_at_second_order(
    mass: str,
) -> None:
    errors = [_manufactured(n, mass) for n in (8, 16, 32)]
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert orders.min() > 1.9


def test_a_neumann_value_is_the_flux_of_the_coefficient() -> None:
    # u = x on [0, 1]^2, K = 3: -div(K grad u) = 0, K du/dn = -3 on x_lower
    mesh = _mesh()
    faces = {"x_lower": Neumann(-3.0), "y_lower": Neumann(0.0), "y_upper": Neumann(0.0)}
    from patchwork import Dirichlet

    faces["x_upper"] = Dirichlet(1.0)
    solver = NodalPoisson(mesh, faces=faces, coefficient=3.0)
    potential = solver.solve(np.zeros(solver.nodes.num_nodes))
    assert np.allclose(potential, solver.nodes.coordinates[:, 0], atol=1e-12)


def test_a_coefficient_in_1d_and_3d() -> None:
    line = AMRHierarchy(base_shape=(16,))
    line.add_patch(level=1, box=((8, 20),))
    solver = NodalPoisson(line, coefficient=2.0)
    x = solver.nodes.coordinates[:, 0]
    # -(2 u')' = 2 f is -u'' = f
    assert np.allclose(
        solver.solve(2.0 * np.pi**2 * np.sin(np.pi * x)),
        NodalPoisson(line).solve(np.pi**2 * np.sin(np.pi * x)),
    )
    edges_1d = NodalPoisson(
        line, edge_coefficient=np.full(solver.nodes.cell_size.shape[0], 2.0)
    )
    assert abs(edges_1d.matrix - solver.matrix).max() < 1e-12
    box = AMRHierarchy(base_shape=(4, 4, 4))
    box.add_patch(level=1, box=((2, 6),) * 3)
    plain = NodalPoisson(box)
    weighted = NodalPoisson(
        box, coefficient=np.full((plain.nodes.cell_size.shape[0], 3), 1.0)
    )
    assert abs(plain.matrix - weighted.matrix).max() < 1e-13


def test_set_coefficient_keeps_the_conditions_and_resets() -> None:
    from patchwork import Dirichlet

    mesh = _mesh()
    solver = NodalPoisson(mesh, faces={"x_upper": Dirichlet(2.0)})
    before = solver.solve(np.ones(solver.nodes.num_nodes))
    solver.set_coefficient(4.0)
    # -div(4 grad u) = 4 is -div(grad u) = 1, with the same face value
    scaled = solver.solve(4.0 * np.ones(solver.nodes.num_nodes))
    assert np.allclose(scaled, before)
    solver.set_coefficient()
    assert np.allclose(solver.solve(np.ones(solver.nodes.num_nodes)), before)


def test_bad_coefficients_are_refused() -> None:
    mesh = _mesh()
    solver = NodalPoisson(mesh)
    count = solver.nodes.cell_size.shape[0]
    with pytest.raises(ValueError, match="per leaf cell"):
        solver.set_coefficient(np.ones(count + 1))
    tensor = np.zeros((count, 2, 2))
    tensor[:, 0, 1] = 1.0
    with pytest.raises(ValueError, match="symmetric"):
        solver.set_coefficient(tensor)
    with pytest.raises(ValueError, match="edges"):
        solver.set_coefficient(edge_coefficient=np.ones(3))
    edges = CompositeEdges(solver.nodes)
    with pytest.raises(ValueError, match="negative"):
        solver.set_coefficient(edge_coefficient=-np.ones(edges.num_edges))
    with pytest.raises(ValueError, match="CompositeEdges of these nodes"):
        NodalPoisson(mesh, edge_coefficient=np.ones(edges.num_edges), edges=edges)
    body = EmbeddedDirichlet(lambda points: points[0] > 0.9, 1.0)
    with pytest.raises(ValueError, match="no embedded bodies"):
        NodalPoisson(mesh, embedded=(body,), coefficient=2.0)


def test_cell_centers_are_the_means_of_the_corners() -> None:
    nodes = NodalPoisson(_mesh(bounds=((-1.0, 2.0), (0.5, 1.5)))).nodes
    corners = nodes.coordinates[nodes.cell_corners].mean(axis=1)
    assert np.allclose(nodes.cell_centers, corners)
