"""The 3D edge space and the face space: the exact sequence on refined patches."""

import numpy as np
import pytest

from patchwork import AMRHierarchy, CompositeEdges, CompositeFaces, CompositeNodes

BOUNDS = ((0.0, 1.0), (0.0, 2.0), (0.0, 1.5))


def spaces(periodic=None):
    mesh = AMRHierarchy(base_shape=(8, 8, 8), bounds=BOUNDS, periodic=periodic)
    mesh.add_patch(level=1, box=((4, 12), (2, 10), (6, 14)))
    mesh.add_patch(level=2, box=((12, 18), (8, 16), (14, 22)))
    nodes = CompositeNodes(mesh)
    edges = CompositeEdges(nodes)
    return nodes, edges, CompositeFaces(edges)


def trilinear(x, y, z):
    return 1.0 + x - 2.0 * y + 3.0 * z + x * y - y * z + 2.0 * x * z + 5.0 * x * y * z


def trilinear_gradient(x, y, z):
    return (
        1.0 + y + 2.0 * z + 5.0 * y * z,
        -2.0 + x - z + 5.0 * x * z,
        3.0 - y + 2.0 * x + 5.0 * x * y,
    )


def random_points(count: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.vstack([rng.random(count) * (hi - lo) + lo for lo, hi in BOUNDS])


# 3D edges -----------------------------------------------------------------------#
@pytest.mark.parametrize("periodic", [None, (False, True, False)])
def test_3d_edge_constraints_commute_with_the_gradient(periodic) -> None:
    nodes, edges, _ = spaces(periodic)
    assert edges.hanging.sum() > 0
    gradient, reduction = edges.gradient, nodes.reduction
    assert (
        abs(gradient @ reduction - edges.reduction @ gradient @ reduction).max() < 1e-14
    )


def test_hanging_edges_inside_coarse_faces_take_two_masters() -> None:
    _, edges, _ = spaces()
    rows = edges.reduction[np.flatnonzero(edges.hanging)]
    masters = np.diff(rows.indptr)
    assert set(masters) == {1, 2}  # on a coarse edge, inside a coarse face


def test_3d_whitney_gather_of_a_gradient_is_the_gradient() -> None:
    nodes, edges, _ = spaces()
    x, y, z = nodes.coordinates.T
    e = edges.gradient @ nodes.reduction @ trilinear(x, y, z)
    points = random_points(500)
    for matrix, expected in zip(
        edges.evaluation(points), trilinear_gradient(*points), strict=True
    ):
        assert np.abs(matrix @ e - expected).max() < 1e-12


@pytest.mark.parametrize("lumped", [True, False])
def test_3d_edge_mass_is_the_energy_of_a_uniform_field(lumped: bool) -> None:
    _, edges, _ = spaces()
    for axis in range(3):
        e = np.where(edges.direction == axis, edges.length, 0.0)
        e = edges.reduction @ e
        assert np.isclose(e @ edges.mass(lumped) @ e, 3.0)


def test_3d_path_deposit_conserves_the_charge_across_patch_faces() -> None:
    nodes, edges, _ = spaces()
    rng = np.random.default_rng(1)
    start = random_points(400, 2)
    end = start + rng.normal(scale=0.15, size=start.shape)
    for axis, (lo, hi) in enumerate(BOUNDS):
        end[axis] = np.clip(end[axis], lo, hi)
    charges = rng.normal(size=400)
    current = edges.deposit_paths(start, end, charges)
    divergence = (edges.gradient @ nodes.reduction).T @ current
    change = nodes.reduction.T @ (
        nodes.deposit(end, charges) - nodes.deposit(start, charges)
    )
    assert np.abs(divergence - change).max() < 1e-12
    # the work of a gradient field is the change of its potential
    x, y, z = nodes.coordinates.T
    e = edges.gradient @ nodes.reduction @ trilinear(x, y, z)
    work = np.sum(charges * (trilinear(*end) - trilinear(*start)))
    assert np.isclose(current @ e, work)


def test_3d_deposit_is_the_transposed_evaluation() -> None:
    _, edges, _ = spaces()
    points = random_points(100, 3)
    vectors = np.random.default_rng(4).normal(size=(3, 100))
    e = np.random.default_rng(5).normal(size=edges.num_edges)
    gathered = [matrix @ e for matrix in edges.evaluation(points)]
    assert np.isclose(edges.deposit(points, vectors) @ e, np.sum(vectors * gathered))


# Faces ---------------------------------------------------------------------------#
@pytest.mark.parametrize("periodic", [None, (True, False, True)])
def test_the_sequence_is_exact(periodic) -> None:
    _, edges, faces = spaces(periodic)
    curl, divergence = faces.curl, faces.divergence
    assert faces.hanging.sum() > 0
    assert abs(curl @ edges.gradient).max() == 0.0
    assert abs(divergence @ curl).max() == 0.0
    commuted = curl @ edges.reduction - faces.reduction @ curl @ edges.reduction
    assert abs(commuted).max() < 1e-14
    # the net flux out of every leaf cell of a curl is zero, patch faces included
    assert abs(divergence @ faces.reduction @ curl @ edges.reduction).max() < 1e-14


def test_the_curl_of_a_linear_potential_gives_its_uniform_field() -> None:
    nodes, edges, faces = spaces()
    middle = 0.5 * (
        nodes.coordinates[edges.ends[:, 0]] + nodes.coordinates[edges.ends[:, 1]]
    )
    x, y, z = middle.T
    # A = (-y / 2 + 0.3 z, x / 2 - 0.2 z, 0.4 y): curl A = (0.6, 0.3, 1)
    potential = np.stack((-y / 2 + 0.3 * z, x / 2 - 0.2 * z, 0.4 * y), axis=1)
    e = potential[np.arange(edges.num_edges), edges.direction] * edges.length
    b = faces.reduction @ faces.curl @ e
    points = random_points(300, 6)
    for matrix, expected in zip(faces.evaluation(points), (0.6, 0.3, 1.0), strict=True):
        assert np.abs(matrix @ b - expected).max() < 1e-12


def test_the_divergence_is_the_net_outward_flux() -> None:
    nodes, _, faces = spaces()
    # the field (x, 0, 0): its flux through x-faces is x times their area
    middle = (faces.lattice + 0.5 * faces.span[:, None]) * nodes._spacing
    flux = np.where(faces.normal == 0, middle[:, 0] * faces.area, 0.0)
    assert np.allclose(faces.divergence @ flux, np.prod(nodes.cell_size, axis=1))


@pytest.mark.parametrize("lumped", [True, False])
def test_face_mass_is_the_energy_of_a_uniform_field(lumped: bool) -> None:
    _, _, faces = spaces()
    for axis in range(3):
        f = faces.reduction @ np.where(faces.normal == axis, faces.area, 0.0)
        assert np.isclose(f @ faces.mass(lumped) @ f, 3.0)
    mass = faces.mass(lumped=False)
    assert abs(mass - mass.T).max() == 0.0


def test_faces_in_2d_are_the_rotated_edges() -> None:
    mesh = AMRHierarchy(base_shape=(8, 8))
    mesh.add_patch(level=1, box=((4, 12), (4, 12)))
    nodes = CompositeNodes(mesh)
    edges = CompositeEdges(nodes)
    faces = CompositeFaces(edges)
    assert faces.num_faces == edges.num_edges
    assert faces.hanging.sum() == edges.hanging.sum()
    assert abs(faces.cell_curl() @ edges.gradient).max() == 0.0
    with pytest.raises(ValueError, match="cell_curl"):
        _ = faces.curl
    _, _, faces_3d = spaces()
    with pytest.raises(ValueError, match="in 3D use curl"):
        faces_3d.cell_curl()
