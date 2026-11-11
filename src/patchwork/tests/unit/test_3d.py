"""Three-dimensional hierarchies: patches, halos, regridding, nodes and the nodal Poisson."""

import numpy as np
import pytest

from patchwork import (
    AMRHierarchy,
    CellOperator,
    CompositeCells,
    CompositeNodes,
    Dirichlet,
    EmbeddedDirichlet,
    Neumann,
    NodalPoisson,
    cluster,
    function_tags,
)
from patchwork.spaces.boundaries import faces_of

BOUNDS = ((0.0, 1.0), (0.0, 2.0), (0.0, 1.5))


def nested(n: int = 8, bounds=BOUNDS, periodic=None) -> AMRHierarchy:
    """A box with a refined patch and a twice-refined one inside it."""
    mesh = AMRHierarchy(base_shape=(n, n, n), bounds=bounds, periodic=periodic)
    mesh.add_patch(
        level=1,
        box=((n // 2, 3 * n // 2), (n // 4, 5 * n // 4), (3 * n // 4, 7 * n // 4)),
    )
    mesh.add_patch(
        level=2,
        box=(
            (5 * n // 4 * 2 // 2 + n // 4, 9 * n // 4),
            (n, 2 * n),
            (7 * n // 4, 11 * n // 4),
        ),
    )
    return mesh


def trilinear(x, y, z):
    return 1.0 + x - 2.0 * y + 3.0 * z + x * y - y * z + 2.0 * x * z + 5.0 * x * y * z


def linear(x, y, z):
    return 1.0 + 2.0 * x - 3.0 * y + 0.5 * z


# The hierarchy -------------------------------------------------------------------#
def test_a_3d_hierarchy_has_three_axes_and_six_faces() -> None:
    mesh = nested()
    assert mesh.ndim == 3
    assert mesh.max_level == 2
    assert set(faces_of(3)) == {f"{a}_{s}" for a in "xyz" for s in ("lower", "upper")}


def test_locate_in_3d_finds_the_finest_patch() -> None:
    mesh = nested()
    rng = np.random.default_rng(0)
    x, y, z = (rng.random(2000) * (hi - lo) + lo for lo, hi in BOUNDS)
    found = mesh.locate(x, y, z)
    expected = np.zeros(x.size, dtype=np.int64)
    for number, patch in enumerate(mesh.patches):
        spacing = mesh.spacing(patch.level)
        inside = np.ones(x.size, dtype=bool)
        for axis, values in enumerate((x, y, z)):
            index = np.floor(values / spacing[axis])
            inside &= (index >= patch.box[axis][0]) & (index < patch.box[axis][1])
        expected[inside] = number  # patches are listed coarse to fine
    assert np.array_equal(found, expected)
    assert mesh.locate(np.array([2.0]), np.array([0.5]), np.array([0.5]))[0] == -1


def test_locate_wraps_a_periodic_z_axis() -> None:
    mesh = AMRHierarchy(base_shape=(4, 4, 4), periodic=(False, False, True))
    inside = mesh.locate(np.array([0.5]), np.array([0.5]), np.array([0.25]))
    wrapped = mesh.locate(np.array([0.5]), np.array([0.5]), np.array([1.25]))
    assert inside[0] == wrapped[0] == 0


def test_cluster_covers_3d_tags_with_disjoint_boxes() -> None:
    rng = np.random.default_rng(3)
    tags = np.zeros((16, 12, 10), dtype=bool)
    i, j, k = np.ogrid[:16, :12, :10]
    tags |= (i - 4) ** 2 + (j - 4) ** 2 + (k - 3) ** 2 < 9
    tags |= (i - 12) ** 2 + (j - 8) ** 2 + (k - 7) ** 2 < 6
    tags |= rng.random(tags.shape) < 0.005
    boxes = cluster(tags, efficiency=0.6, max_size=6)
    count = np.zeros(tags.shape, dtype=int)
    for box in boxes:
        assert len(box) == 3
        assert all(hi - lo <= 6 for lo, hi in box)
        count[tuple(slice(lo, hi) for lo, hi in box)] += 1
    assert count.max() == 1
    assert (count[tags] == 1).all()


# Fields -----------------------------------------------------------------------------#
@pytest.mark.parametrize("centering", ["node", "cell"])
def test_halos_of_a_linear_field_are_exact_in_3d(centering: str) -> None:
    """Every halo point inside the domain is filled, from a neighbour or from below."""
    sentinel = 1.0e9
    mesh = nested()
    mesh.allocate_field("u", centering=centering, halo=1)
    shift = 0.0 if centering == "node" else 0.5
    for patch in mesh.local_patches:
        storage = patch.field("u").local_with_halos
        storage[...] = sentinel
        points = (
            patch.local_node_coordinates(mesh, "u")
            if centering == "node"
            else patch.local_cell_centers(mesh, "u")
        )
        storage[1:-1, 1:-1, 1:-1] = linear(*points)
    mesh.exchange_halos("u")
    filled = 0
    for patch in mesh.local_patches:
        storage = np.asarray(patch.field("u").local_with_halos)
        spacing = mesh.spacing(patch.level)
        # this rank's block of the patch with its halo (a patch may be split)
        layout = patch.field("u").layout
        axes = [
            BOUNDS[axis][0] + (lo + np.arange(b0 - h, b1 + h) + shift) * spacing[axis]
            for axis, ((lo, _), (b0, b1), h) in enumerate(
                zip(patch.box, layout.index_bounds, layout.halo, strict=True)
            )
        ]
        x, y, z = np.meshgrid(*axes, indexing="ij")
        inside = np.ones(x.shape, dtype=bool)
        for values, (lo, hi) in zip((x, y, z), BOUNDS, strict=True):
            inside &= (values > lo - 1e-12) & (values < hi + 1e-12)
        assert storage.shape == x.shape
        assert np.abs(storage[inside] - linear(x, y, z)[inside]).max() < 1e-12
        ghost = np.ones(x.shape, dtype=bool)
        ghost[1:-1, 1:-1, 1:-1] = False
        filled += int((ghost & inside).sum())
    assert mesh.comm.allreduce(filled) > 0


def test_regrid_in_3d_follows_the_tags_and_moves_fields_exactly() -> None:
    mesh = AMRHierarchy(base_shape=(8, 8, 8))
    mesh.allocate_field("u", centering="node", halo=1)
    mesh.allocate_field("rho", halo=1, conservative=True)
    for patch in mesh.local_patches:
        patch.field("u").local[...] = linear(*patch.local_node_coordinates(mesh, "u"))
        patch.field("rho").local[...] = linear(*patch.local_cell_centers(mesh, "rho"))
    total = mesh.integral("rho")

    def ball(x, y, z):
        return (x - 0.5) ** 2 + (y - 0.4) ** 2 + (z - 0.6) ** 2 < 0.2**2

    tags = function_tags(mesh, ball, levels=2)
    new = mesh.regrid(tags, max_patch_size=8)
    assert new.max_level == 2
    assert all(len(p.box) == 3 for p in new.patches)
    assert new.integral("rho") == pytest.approx(total, rel=1e-13)
    for level, level_tags in tags.items():
        covered = np.zeros(new.level_shape(level + 1), dtype=bool)
        for patch in new.patches_at_level(level + 1):
            covered[tuple(slice(lo, hi) for lo, hi in patch.box)] = True
        r = new.refinement_ratio
        n = level_tags.shape
        refined = covered.reshape(n[0], r, n[1], r, n[2], r).any(axis=(1, 3, 5))
        assert refined[level_tags].all()
    for patch in new.local_patches:
        x, y, z = patch.local_node_coordinates(new, "u")
        assert np.abs(patch.field("u").local - linear(x, y, z)).max() < 1e-12


# Nodes --------------------------------------------------------------------------------#
def test_hanging_nodes_on_coarse_edges_and_faces_interpolate_trilinear_fields() -> None:
    nodes = CompositeNodes(nested())
    x, y, z = nodes.coordinates.T
    u = trilinear(x, y, z)
    assert nodes.hanging.sum() > 0
    assert np.abs(nodes.reduction @ u - u).max() < 1e-12
    # hanging nodes in the middle of a coarse face take four masters
    rows = nodes.reduction[np.flatnonzero(nodes.hanging)]
    assert (np.diff(rows.indptr) == 4).any()
    assert np.isclose(nodes.volumes.sum(), 3.0)


def test_3d_deposit_keeps_the_charge_and_is_the_transposed_evaluation() -> None:
    nodes = CompositeNodes(nested())
    rng = np.random.default_rng(2)
    points = np.vstack([rng.random(300) * (hi - lo) + lo for lo, hi in BOUNDS])
    charges = rng.normal(size=300)
    deposited = nodes.deposit(points, charges)
    assert np.isclose(deposited.sum(), charges.sum())
    assert np.allclose(deposited, nodes.evaluation(points).T @ charges)
    x, y, z = nodes.coordinates.T
    u = nodes.reduction @ trilinear(x, y, z)
    assert np.allclose(nodes.evaluation(points) @ u, trilinear(*points))


def test_boundary_pieces_in_3d_are_quads_that_tile_the_faces() -> None:
    nodes = CompositeNodes(nested())
    for face, (axis, _) in faces_of(3).items():
        corners, measure, midpoint = nodes.face_pieces(face)
        assert corners.shape[1] == 4
        others = [a for a in range(3) if a != axis]
        area = np.prod([BOUNDS[a][1] - BOUNDS[a][0] for a in others])
        assert np.isclose(measure.sum(), area)
        assert np.isclose(nodes.face_measure(face).sum(), area)
    with pytest.raises(ValueError, match="face_pieces"):
        nodes.face_edges("x_lower")


# The nodal Poisson ---------------------------------------------------------------------#
@pytest.mark.parametrize("mass", ["lumped", "consistent"])
def test_3d_poisson_converges_at_second_order(mass: str) -> None:
    def error(n: int) -> float:
        mesh = AMRHierarchy(base_shape=(n, n, n))
        mesh.add_patch(level=1, box=((n // 2, 3 * n // 2),) * 3)
        solver = NodalPoisson(mesh, mass=mass)
        x, y, z = solver.nodes.coordinates.T
        exact = np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
        difference = solver.solve(3.0 * np.pi**2 * exact) - exact
        return float(np.sqrt(np.sum(difference**2 * solver.nodes.volumes)))

    coarse, fine = error(8), error(16)
    assert np.log2(coarse / fine) > 1.9


def test_consistent_3d_poisson_is_exact_for_trilinear_potentials() -> None:
    faces = {face: Dirichlet(trilinear) for face in faces_of(3)}
    solver = NodalPoisson(nested(), faces=faces, mass="consistent")
    x, y, z = solver.nodes.coordinates.T
    assert np.abs(solver.solve(np.zeros_like(x)) - trilinear(x, y, z)).max() < 1e-11


def test_neumann_faces_in_3d() -> None:
    """``phi = z^2 / 2 - z`` on [0, 1]^3: zero flux at z = 1, one at z = 0, f = -1."""
    mesh = AMRHierarchy(base_shape=(8, 8, 8))
    mesh.add_patch(level=1, box=((4, 12),) * 3)
    faces = {
        "z_lower": Neumann(1.0),  # outward normal derivative -d/dz = 1
        "z_upper": Neumann(0.0),
        "x_lower": Dirichlet(lambda x, y, z: z**2 / 2 - z),
        "x_upper": Dirichlet(lambda x, y, z: z**2 / 2 - z),
        "y_lower": Dirichlet(lambda x, y, z: z**2 / 2 - z),
        "y_upper": Dirichlet(lambda x, y, z: z**2 / 2 - z),
    }
    solver = NodalPoisson(mesh, faces=faces)
    x, y, z = solver.nodes.coordinates.T
    phi = solver.solve(-np.ones_like(x))
    assert np.abs(phi - (z**2 / 2 - z)).max() < 2e-3


def test_a_body_in_3d_holds_its_potential_and_converges() -> None:
    def ball(points):
        return np.sum((points - 0.5) ** 2, axis=0) < 0.2**2

    def error(n: int) -> float:
        body = EmbeddedDirichlet(ball, 1.0)
        solver = NodalPoisson(AMRHierarchy(base_shape=(n, n, n)), embedded=[body])
        phi = solver.solve(np.zeros(solver.nodes.num_nodes))
        x, y, z = solver.nodes.coordinates.T
        r = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2)
        assert np.allclose(phi[r < 0.2 - 1e-9], 1.0)
        # far from the walls the potential falls like 1/r; compare a shell
        shell = (r > 0.25) & (r < 0.3)
        return float(np.std(phi[shell] * r[shell]))

    assert error(16) < 0.05


def test_cells_of_a_3d_hierarchy() -> None:
    mesh = nested()
    cells = CompositeCells(mesh)
    assert np.isclose(cells.volumes.sum(), 3.0)
    with pytest.raises(NotImplementedError, match="1D and 2D"):
        CellOperator(mesh)
