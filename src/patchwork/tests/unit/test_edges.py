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


def _continuity_error(edges: CompositeEdges, start, end, charges) -> float:
    """``(G T)^T j`` against the change of the composite nodal charge, relative."""
    nodes = edges.nodes
    current = edges.deposit_paths(start, end, charges)
    divergence = (edges.gradient @ nodes.reduction).T @ current
    change = nodes.reduction.T @ (
        nodes.deposit(end, charges) - nodes.deposit(start, charges)
    )
    return float(np.abs(divergence - change).max() / np.abs(charges).max())


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ((0.03, 0.03), (0.05, 0.06)),  # inside a coarse cell
        ((0.03, 0.03), (0.2, 0.05)),  # across faces of coarse cells
        ((0.45, 0.4), (0.55, 0.62)),  # across the edge of the level-1 patch
        ((0.4, 0.4), (1.3, 0.85)),  # through both patches and out
        ((1.0, 0.97), (1.1, 1.08)),  # across the periodic seam
        ((0.6, 0.5), (0.6, 0.7)),  # along a lattice line
        ((0.5, 0.5), (0.75, 0.75)),  # through lattice corners
    ],
)
def test_the_path_deposit_conserves_the_charge(start, end) -> None:
    edges = _edges("nested", periodic=(False, True))
    start, end = np.array(start)[:, None], np.array(end)[:, None]
    assert _continuity_error(edges, start, end, np.array([1.5])) < 1e-12


@pytest.mark.parametrize("layout", LAYOUTS)
def test_many_random_paths_conserve_the_charge(layout: str) -> None:
    edges = _edges(layout, periodic=(False, True))
    rng = np.random.default_rng(4)
    start = _points(500, seed=9) * np.array([[0.9], [1.0]]) + np.array([[0.1], [0.0]])
    end = start + rng.normal(scale=0.15, size=start.shape)
    end[0] = np.clip(end[0], 0.0, 2.0)  # bounded in x, periodic in y
    charges = rng.normal(size=500)
    assert _continuity_error(edges, start, end, charges) < 1e-12


def test_the_work_of_a_gradient_field_is_the_potential_difference() -> None:
    edges = _edges("level jump", periodic=(False, True))
    nodes = edges.nodes
    phi = _potential(nodes)
    start = np.array([[0.3, 0.9, 1.6], [0.2, 0.95, 0.4]])
    end = np.array([[1.4, 1.1, 0.7], [0.65, 1.3, 0.55]])
    charges = np.array([1.0, -2.0, 0.5])
    work = edges.deposit_paths(start, end, charges) @ (edges.gradient @ phi)
    expected = charges @ (nodes.evaluation(end) @ phi - nodes.evaluation(start) @ phi)
    assert work == pytest.approx(expected, rel=1e-12)


def test_a_uniform_field_does_the_work_of_its_displacement() -> None:
    edges = _edges("two patches")
    field = np.array([2.0, -0.5])
    e = edges.length * field[edges.direction]
    start = np.array([[0.1, 1.9], [0.1, 0.8]])
    end = np.array([[1.7, 0.2], [0.9, 0.15]])
    charges = np.array([0.7, 1.1])
    work = edges.deposit_paths(start, end, charges) @ e
    assert work == pytest.approx(float(charges @ (field @ (end - start))), rel=1e-12)


def test_a_path_that_stays_put_deposits_nothing_and_shapes_are_checked() -> None:
    edges = _edges("nested")
    point = np.array([[0.4], [0.3]])
    np.testing.assert_array_equal(edges.deposit_paths(point, point, np.ones(1)), 0.0)
    with pytest.raises(ValueError, match="paths"):
        edges.deposit_paths(point, np.zeros((2, 2)), np.ones(1))
