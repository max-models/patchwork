"""Lumped and consistent mass: the spaces' matrices and NodalPoisson's two schemes."""

import numpy as np
import pytest

from patchwork import AMRHierarchy, CompositeEdges, CompositeNodes, NodalPoisson
from patchwork.spaces.boundaries import EmbeddedDirichlet
from patchwork.spaces.mass import edge_mass_blocks, node_mass_blocks


def _nested(n: int = 8) -> AMRHierarchy:
    """A unit square with a refined patch and a twice-refined one inside it."""
    mesh = AMRHierarchy(base_shape=(n, n))
    mesh.add_patch(level=1, box=((n // 2, 3 * n // 2),) * 2)
    mesh.add_patch(level=2, box=((3 * n // 2, 5 * n // 2),) * 2)
    return mesh


# Blocks ------------------------------------------------------------------------#
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("lumped", [True, False])
def test_node_blocks_integrate_a_constant(ndim: int, lumped: bool) -> None:
    size = np.array([[0.5, 0.25, 2.0][:ndim]])
    block = node_mass_blocks(size, lumped)[0]
    assert block.shape == (2**ndim, 2**ndim)
    assert np.isclose(block.sum(), np.prod(size))
    assert np.allclose(block, block.T)


def test_edge_blocks_in_3d() -> None:
    """Lumped: h_perp / (4 h) on the diagonal; consistent: 1/9, 1/18, 1/36 times that scale."""
    size = np.array([[1.0, 2.0, 3.0]])
    lumped = edge_mass_blocks(size, 0, lumped=True)[0]
    assert np.allclose(lumped, np.eye(4) * 2.0 * 3.0 / 4.0)
    consistent = edge_mass_blocks(size, 0, lumped=False)[0]
    scale = 2.0 * 3.0
    assert np.isclose(consistent[0, 0], scale / 9.0)
    assert np.isclose(consistent[0, 1], scale / 18.0)
    assert np.isclose(consistent[0, 3], scale / 36.0)
    assert np.isclose(consistent.sum(), scale)  # a uniform field along x


# Node mass ---------------------------------------------------------------------#
def test_lumped_node_mass_is_the_control_volumes() -> None:
    nodes = CompositeNodes(_nested())
    assert np.allclose(nodes.mass().diagonal(), nodes.volumes)
    assert nodes.mass().nnz == nodes.num_nodes


def test_consistent_node_mass_integrates_bilinear_fields() -> None:
    """``u^T M0 u`` is exact for a bilinear field, hanging nodes included."""
    mesh = AMRHierarchy(base_shape=(6, 4), bounds=((0.0, 3.0), (-1.0, 1.0)))
    mesh.add_patch(level=1, box=((4, 8), (2, 6)))
    nodes = CompositeNodes(mesh)
    x, y = nodes.coordinates.T
    mass = nodes.mass(lumped=False)
    one = np.ones(nodes.num_nodes)
    assert np.isclose(one @ mass @ one, 6.0)
    u = 1.0 + x * y
    # the integral of (1 + x y)^2 over [0, 3] x [-1, 1]
    assert np.isclose(u @ mass @ u, 6.0 + 9.0 * (2.0 / 3.0))
    assert np.allclose((mass - mass.T).data, 0.0)


def test_consistent_node_mass_in_1d() -> None:
    nodes = CompositeNodes(AMRHierarchy(base_shape=(5,)))
    (x,) = nodes.coordinates.T
    mass = nodes.mass(lumped=False)
    assert np.isclose(x @ mass @ x, 1.0 / 3.0)


# One assembly: G^T M1 G ---------------------------------------------------------#
@pytest.mark.parametrize("mass", ["lumped", "consistent"])
def test_nodal_operator_is_the_edge_energy(mass: str) -> None:
    """``NodalPoisson``'s matrix is ``(T^T G^T M1 G T)`` on the unknowns, for both masses."""
    solver = NodalPoisson(_nested(), mass=mass)
    nodes = solver.nodes
    edges = CompositeEdges(nodes)
    full = nodes.reduction.T @ (
        edges.gradient.T
        @ edges.mass(lumped=mass == "lumped")
        @ edges.gradient
        @ nodes.reduction
    )
    unknowns = solver.system.unknowns
    expected = full.tocsr()[unknowns][:, unknowns]
    assert abs(solver.matrix - expected).max() < 1e-12


def test_consistent_stencil_is_the_q1_stiffness() -> None:
    """Square cells: 8/3 on the diagonal, -1/3 to all eight neighbours."""
    solver = NodalPoisson(AMRHierarchy(base_shape=(4, 4)), mass="consistent")
    lattice = solver.nodes.lattice[solver.system.unknowns]
    centre = int(np.flatnonzero((lattice == (2, 2)).all(axis=1))[0])
    row = solver.matrix[centre].toarray().ravel()
    assert np.isclose(row[centre], 8.0 / 3.0)
    neighbours = np.flatnonzero(np.abs(lattice - (2, 2)).max(axis=1) == 1)
    assert neighbours.size == 8
    assert np.allclose(row[neighbours], -1.0 / 3.0)
    assert np.isclose(row.sum(), 0.0)


def test_consistent_matches_lumped_stiffness_in_1d() -> None:
    mesh = AMRHierarchy(base_shape=(8,))
    mesh.add_patch(level=1, box=((4, 12),))
    lumped = NodalPoisson(mesh)
    consistent = NodalPoisson(mesh, mass="consistent")
    assert abs(lumped.matrix - consistent.matrix).max() < 1e-12
    assert abs(lumped.load - consistent.load).max() > 0.0


def test_consistent_mass_on_a_periodic_axis_stays_symmetric() -> None:
    mesh = AMRHierarchy(base_shape=(8, 8), periodic=(True, False))
    mesh.add_patch(level=1, box=((0, 6), (4, 10)))
    solver = NodalPoisson(mesh, mass="consistent")
    assert abs(solver.matrix - solver.matrix.T).max() < 1e-12


# Accuracy -----------------------------------------------------------------------#
def _error(n: int, mass: str) -> float:
    solver = NodalPoisson(_nested(n), mass=mass)
    x, y = solver.nodes.coordinates.T
    exact = np.sin(np.pi * x) * np.sin(np.pi * y)
    difference = solver.solve(2.0 * np.pi**2 * exact) - exact
    return float(np.sqrt(np.sum(difference**2 * solver.nodes.volumes)))


@pytest.mark.parametrize("mass", ["lumped", "consistent"])
def test_second_order_on_nested_patches(mass: str) -> None:
    errors = [_error(n, mass) for n in (16, 32)]
    order = np.log2(errors[0] / errors[1])
    assert order > 1.8, (mass, errors)


def test_consistent_solves_a_bilinear_potential_exactly() -> None:
    """The Q1 solution is exact for a harmonic bilinear potential, hanging nodes or not."""
    from patchwork import Dirichlet

    def phi(x, y):
        return 1.0 + 2.0 * x - y + 3.0 * x * y

    faces = {
        face: Dirichlet(phi) for face in ("x_lower", "x_upper", "y_lower", "y_upper")
    }
    solver = NodalPoisson(_nested(), faces=faces, mass="consistent")
    x, y = solver.nodes.coordinates.T
    assert np.allclose(solver.solve(np.zeros_like(x)), phi(x, y), atol=1e-12)


# Arguments ----------------------------------------------------------------------#
def test_unknown_mass_is_rejected() -> None:
    with pytest.raises(ValueError, match="mass must be one of"):
        NodalPoisson(AMRHierarchy(base_shape=(4, 4)), mass="exact")


def test_consistent_mass_takes_no_bodies() -> None:
    body = EmbeddedDirichlet(lambda points: points[0] > 0.7, 1.0)
    with pytest.raises(ValueError, match="lumped"):
        NodalPoisson(
            AMRHierarchy(base_shape=(8, 8)), embedded=[body], mass="consistent"
        )


def test_auto_takes_the_lu_for_the_consistent_scheme() -> None:
    """The FFT diagonalizes the five-point stencil only: ``"auto"`` takes the LU."""
    from patchwork import FFTBackend, LUBackend

    mesh = AMRHierarchy(base_shape=(16, 16))
    lumped = NodalPoisson(mesh, backend="auto")
    consistent = NodalPoisson(mesh, backend="auto", mass="consistent")
    assert isinstance(lumped.backend.choose(lumped.system), FFTBackend)
    assert isinstance(consistent.backend.choose(consistent.system), LUBackend)
    with pytest.raises(ValueError, match="lumped"):
        NodalPoisson(mesh, backend="fft", mass="consistent")
    x, y = consistent.nodes.coordinates.T
    exact = np.sin(np.pi * x) * np.sin(np.pi * y)
    error = consistent.solve(2.0 * np.pi**2 * exact) - exact
    assert np.abs(error).max() < 1e-2
