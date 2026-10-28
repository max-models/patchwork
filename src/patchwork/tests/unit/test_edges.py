"""Tests for the lowest-order edge space of a 2D hierarchy."""

from __future__ import annotations

import numpy as np
import pytest

from patchwork import (
    AMRHierarchy,
    CompositeEdges,
    CompositeNodes,
    Dirichlet,
    Neumann,
    NodalPoisson,
)

#: Patch layouts on a 16 x 16 base mesh, as in the nodal tests: none, nested, a
#: level-2 patch that touches its parent's edge, and two separate patches.
LAYOUTS = {
    "uniform": [],
    "nested": [(1, ((8, 24), (8, 24))), (2, ((24, 40), (24, 40)))],
    "level jump": [(1, ((8, 24), (8, 24))), (2, ((16, 32), (16, 32)))],
    "two patches": [(1, ((4, 12), (4, 12))), (1, ((16, 28), (12, 30)))],
}


def _edges(layout: str, **kwargs) -> CompositeEdges:
    mesh = AMRHierarchy(base_shape=(16, 16), bounds=((0.0, 2.0), (0.0, 1.0)), **kwargs)
    for level, box in LAYOUTS[layout]:
        mesh.add_patch(level=level, box=box)
    return CompositeEdges(CompositeNodes(mesh))


def _points(count: int = 2000, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.vstack((2.0 * rng.random(count), rng.random(count)))


def _potential(nodes: CompositeNodes, seed: int = 5) -> np.ndarray:
    """A random potential with its hanging nodes filled: a field of the composite space."""
    return nodes.reduction @ np.random.default_rng(seed).normal(size=nodes.num_nodes)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_gradient_of_a_linear_potential_is_exact(layout: str) -> None:
    edges = _edges(layout)
    x, y = edges.nodes.coordinates.T
    e = edges.gradient @ (0.5 + 3.0 * x - 2.0 * y)
    along = np.where(edges.direction == 0, 3.0, -2.0)
    np.testing.assert_allclose(e, along * edges.length, atol=1e-13)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("periodic", [(False, False), (False, True), (True, True)])
def test_the_constraints_commute_with_the_gradient(layout, periodic) -> None:
    """``G T = T1 G T``: the gradient of a composite potential is a composite edge field."""
    edges = _edges(layout, periodic=periodic)
    nodes = edges.nodes
    expected = (edges.gradient @ nodes.reduction).toarray()
    constrained = (edges.reduction @ edges.gradient @ nodes.reduction).toarray()
    np.testing.assert_allclose(constrained, expected, atol=1e-13)
    assert edges.hanging.any() == (layout != "uniform")


def test_a_half_edge_carries_half_the_coarse_edge() -> None:
    edges = _edges("nested")
    hanging = np.flatnonzero(edges.hanging)
    rows = edges.reduction[hanging]
    # each hanging edge takes its share of the coarse edge it lies in
    np.testing.assert_allclose(
        rows.sum(axis=1).A1, edges.span[hanging] / edges.span[rows.indices], atol=0
    )
    assert np.all(np.diff(rows.indptr) == 1)
    assert not edges.hanging[rows.indices].any()


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("periodic", [(False, False), (False, True)])
def test_the_lumped_mass_gives_the_nodal_poisson_operator(layout, periodic) -> None:
    """``T^T G^T M1 G T`` on the free nodes is ``NodalPoisson``'s matrix."""
    edges = _edges(layout, periodic=periodic)
    nodes = edges.nodes
    faces = {"x_lower": Dirichlet(0.0), "x_upper": Neumann(0.0)}
    if not periodic[1]:
        faces |= {"y_lower": Neumann(0.0), "y_upper": Dirichlet(1.0)}
    poisson = NodalPoisson(nodes.hierarchy, faces=faces, nodes=nodes)
    gradient = edges.gradient @ nodes.reduction
    stiffness = (gradient.T @ edges.mass(lumped=True) @ gradient).tocsr()
    free = poisson.free
    difference = stiffness[free][:, free] - poisson.matrix
    assert abs(difference).max() < 1e-12 * abs(poisson.matrix).max()


@pytest.mark.parametrize("lumped", [True, False])
def test_the_mass_is_the_energy_of_a_uniform_field(lumped: bool) -> None:
    edges = _edges("nested")
    field = np.array([0.7, -1.3])
    e = edges.length * field[edges.direction]  # the line integrals of a uniform field
    mass = edges.mass(lumped=lumped)
    assert e @ mass @ e == pytest.approx(2.0 * float(field @ field), rel=1e-13)
    # symmetric and positive definite on the free edges
    free = np.flatnonzero(~edges.hanging)
    reduced = (edges.reduction.T @ mass @ edges.reduction)[free][:, free].toarray()
    np.testing.assert_allclose(reduced, reduced.T, atol=1e-14)
    assert np.linalg.eigvalsh(reduced).min() > 0.0


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_gather_of_the_gradient_is_the_gradient_of_the_bilinear_potential(
    layout: str,
) -> None:
    edges = _edges(layout, periodic=(False, True))
    nodes = edges.nodes
    phi = _potential(nodes)
    e = edges.gradient @ phi
    points = _points()
    field = np.array([matrix @ e for matrix in edges.evaluation(points)])
    # the bilinear interpolant is linear along each axis inside a cell
    step = 1e-7
    for axis in range(2):
        shift = np.zeros((2, 1))
        shift[axis] = step
        upper = nodes.evaluation(points + shift) @ phi
        lower = nodes.evaluation(points - shift) @ phi
        same = (
            nodes.point_cells(points + shift)[0] == nodes.point_cells(points - shift)[0]
        )
        np.testing.assert_allclose(
            field[axis][same],
            ((upper - lower) / (2 * step))[same],
            rtol=1e-6,
            atol=1e-6,
        )


def test_a_point_outside_the_domain_has_no_field() -> None:
    edges = _edges("nested")
    e = edges.gradient @ _potential(edges.nodes)
    points = np.array([[-0.1, 2.5, 1.0], [0.5, 0.5, 1.5]])
    for matrix in edges.evaluation(points):
        np.testing.assert_array_equal(matrix @ e, 0.0)


def test_the_deposit_is_the_transposed_gather() -> None:
    edges = _edges("level jump", periodic=(False, True))
    points = _points(300)
    vectors = np.random.default_rng(7).normal(size=(2, 300))
    e = edges.reduction @ np.random.default_rng(8).normal(size=edges.num_edges)
    gathered = np.array([matrix @ e for matrix in edges.evaluation(points)])
    assert edges.deposit(points, vectors) @ e == pytest.approx(
        float(np.sum(vectors * gathered)), rel=1e-12
    )


def test_a_uniform_mesh_has_two_edges_per_node_and_no_hanging_edges() -> None:
    edges = _edges("uniform", periodic=(True, True))
    assert edges.num_edges == 2 * edges.nodes.num_nodes
    assert not edges.hanging.any()
    np.testing.assert_array_equal(edges.cell_edges.shape, (256, 4))


def test_the_edge_space_is_two_dimensional() -> None:
    nodes = CompositeNodes(AMRHierarchy(base_shape=(8,)))
    with pytest.raises(NotImplementedError, match="two-dimensional"):
        CompositeEdges(nodes)
