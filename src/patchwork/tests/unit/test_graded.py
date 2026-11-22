"""Graded axes: the hierarchy's map, the node space and `NodalPoisson` on it."""

from __future__ import annotations

import numpy as np
import pytest

from patchwork import (
    AMRHierarchy,
    CompositeNodes,
    Dirichlet,
    EmbeddedDirichlet,
    NodalPoisson,
)

GROUNDED = {
    face: Dirichlet(0.0) for face in ("x_lower", "x_upper", "y_lower", "y_upper")
}


def graded(n: int, strength: float = 0.6, lower: float = 0.0, upper: float = 1.0):
    """``n + 1`` nodes of a smooth map of ``[lower, upper]``: cells from 0.4 to 1.6 h."""
    s = np.linspace(0.0, 1.0, n + 1)
    x = s - strength * np.sin(2.0 * np.pi * s) / (2.0 * np.pi)
    return lower + (upper - lower) * x


def wall_graded(n: int, lower: float = 0.0, upper: float = 1.0):
    """``n + 1`` nodes crowded at the lower end: the cells grow from 1/3 to 5/3 of a mean one."""
    s = np.linspace(0.0, 1.0, n + 1)
    return lower + (upper - lower) * (s + 2.0 * s**2) / 3.0


def _orders(errors) -> np.ndarray:
    return np.log2(np.array(errors[:-1]) / np.array(errors[1:]))


# The hierarchy ---------------------------------------------------------------------- #


def test_the_bounds_of_a_graded_axis_are_its_ends() -> None:
    mesh = AMRHierarchy(
        (4, 2), (None, (0.0, 2.0)), axis_nodes=([0, 0.1, 0.3, 0.6, 1.0], None)
    )
    assert mesh.bounds == ((0.0, 1.0), (0.0, 2.0)) and mesh.graded
    assert not AMRHierarchy((4, 2)).graded
    with pytest.raises(ValueError, match="differ from the ends"):
        AMRHierarchy(
            (4, 2), ((0.0, 2.0), (0.0, 1.0)), axis_nodes=(np.linspace(0, 1, 5), None)
        )
    with pytest.raises(ValueError, match="increase strictly"):
        AMRHierarchy((4,), axis_nodes=([0, 0.2, 0.2, 0.5, 1.0],))
    with pytest.raises(ValueError, match="5 positions"):
        AMRHierarchy((4,), axis_nodes=([0, 0.5, 1.0],))
    with pytest.raises(ValueError, match="one entry per axis"):
        AMRHierarchy((4, 4), axis_nodes=(None,))


def test_finer_levels_split_the_base_cells_evenly() -> None:
    nodes = [0.0, 0.1, 0.3, 0.6, 1.0]
    mesh = AMRHierarchy((4, 1), axis_nodes=(nodes, None))
    np.testing.assert_allclose(
        mesh.node_positions(1, 0), [0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0]
    )
    np.testing.assert_allclose(mesh.cell_sizes(0, 0, np.arange(4)), np.diff(nodes))
    x = np.array([0.0, 0.05, 0.25, 0.99, 1.0])
    np.testing.assert_allclose(mesh.to_physical(0, mesh.to_logical(0, x)), x)
    # the uniform axis is the identity
    y = np.array([0.3])
    assert mesh.to_logical(1, y) is y and mesh.to_physical(1, y) is y


def test_points_are_located_in_their_physical_cells() -> None:
    mesh = AMRHierarchy((4, 1), axis_nodes=([0, 0.1, 0.3, 0.6, 1.0], None))
    fine = mesh.add_patch(level=1, box=((2, 4), (0, 2)))  # base cell 1: 0.1 < x < 0.3
    found = mesh.locate(np.array([0.05, 0.15, 0.29, 0.31, 0.95]), np.full(5, 0.5))
    number = mesh.patches.index(fine)
    np.testing.assert_array_equal(found == number, [False, True, True, False, False])
    assert fine.physical_bounds(mesh) == ((0.1, 0.3), (0.0, 1.0))
    np.testing.assert_allclose(fine.node_coordinates(mesh)[0], [0.1, 0.2, 0.3])


@pytest.mark.parametrize("geometry", ["cartesian", "cylindrical"])
def test_cells_and_faces_have_their_physical_measure(geometry) -> None:
    r_nodes = wall_graded(6, 0.5, 2.0)
    z_nodes = graded(5)
    mesh = AMRHierarchy((6, 5), axis_nodes=(r_nodes, z_nodes), geometry=geometry)
    i, j = np.ix_(np.arange(6), np.arange(5))
    volumes = mesh.cell_volumes(0, i, j)
    if geometry == "cartesian":
        assert volumes.sum() == pytest.approx(1.5)
    else:
        assert volumes.sum() == pytest.approx(np.pi * (2.0**2 - 0.5**2))
    # the faces close every cell: a constant flux along each axis has no divergence,
    # and the radial flux r (div = 2) in r-z
    radial = mesh.face_areas(0, 0, np.arange(7)[:, None], np.arange(5))
    axial = mesh.face_areas(0, 1, np.arange(6)[:, None], np.arange(6))  # 6 faces in z
    np.testing.assert_allclose(np.diff(axial, axis=1), 0.0, atol=1e-15)
    if geometry == "cylindrical":
        flux = radial * r_nodes[:, None]
        np.testing.assert_allclose(np.diff(flux, axis=0) / volumes, 2.0)
    # fine faces over a coarse one add up to it
    fine = mesh.face_areas(1, 1, np.arange(12)[:, None], np.arange(11))  # 11 faces in z
    np.testing.assert_allclose(fine[::2, ::2] + fine[1::2, ::2], axial)


def test_the_integral_and_a_regrid_keep_the_grading() -> None:
    mesh = AMRHierarchy((8, 8), axis_nodes=(wall_graded(8), graded(8)))
    mesh.allocate_field("u", conservative=True)
    for patch in mesh:
        x, y = patch.local_cell_centers(mesh, "u")
        patch.field("u").local[...] = 1.0 + x
    tags = {0: np.zeros((8, 8), dtype=bool)}
    tags[0][:2, :] = True
    new = mesh.regrid(tags)
    np.testing.assert_array_equal(new.axis_nodes[0], mesh.axis_nodes[0])
    assert new.max_level == 1
    # the integral of 1 + x over the unit square, from the midpoint rule
    exact = 1.5
    assert mesh.integral("u") == pytest.approx(exact, rel=1e-2)
    assert new.integral("u") == pytest.approx(mesh.integral("u"), rel=1e-13)


# The node space and NodalPoisson ------------------------------------------------------- #


def test_the_nodes_are_at_their_physical_positions() -> None:
    mesh = AMRHierarchy((4, 4), axis_nodes=(graded(4), None))
    mesh.add_patch(level=1, box=((2, 6), (2, 6)))
    nodes = CompositeNodes(mesh)
    xs = np.unique(nodes.coordinates[:, 0])
    np.testing.assert_allclose(
        xs, np.union1d(graded(4), mesh.node_positions(1, 0)[2:7])
    )
    assert nodes.volumes.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(
        nodes.cell_centers, nodes.coordinates[nodes.cell_corners].mean(axis=1)
    )


def test_deposit_and_evaluation_are_transposes() -> None:
    mesh = AMRHierarchy((8, 8), axis_nodes=(wall_graded(8), graded(8)))
    mesh.add_patch(level=1, box=((0, 6), (4, 12)))
    nodes = CompositeNodes(mesh)
    points = np.random.default_rng(1).uniform(0.0, 1.0, (2, 400))
    charges = np.random.default_rng(2).uniform(-1.0, 1.0, 400)
    deposited = nodes.deposit(points, charges)
    np.testing.assert_allclose(
        deposited, nodes.evaluation(points).T @ charges, atol=1e-13
    )
    # the evaluation reproduces a bilinear function exactly
    x, y = nodes.coordinates.T
    values = 1.0 + 2.0 * x - 3.0 * y + x * y
    px, py = points
    np.testing.assert_allclose(
        nodes.evaluation(points) @ values,
        1.0 + 2.0 * px - 3.0 * py + px * py,
        atol=1e-12,
    )


def test_the_patch_deposit_matches_the_composite_one() -> None:
    mesh = AMRHierarchy((8, 8), axis_nodes=(wall_graded(8), graded(8)))
    mesh.add_patch(level=1, box=((0, 6), (4, 12)))
    mesh.allocate_field("rho", centering="node", halo=1)
    points = np.random.default_rng(3).uniform(0.0, 1.0, (2, 300))
    charges = np.random.default_rng(4).uniform(0.0, 1.0, 300)
    mesh.deposit(*points, charges, "rho")
    mesh.accumulate_halos("rho")
    nodes = CompositeNodes(mesh)
    np.testing.assert_allclose(
        nodes.from_patches("rho"), nodes.deposit(points, charges), atol=1e-13
    )
    mesh.allocate_field("phi", centering="node", halo=1)
    x, y = nodes.coordinates.T
    nodes.to_patches(2.0 * x + y, "phi")
    mesh.exchange_halos("phi")
    np.testing.assert_allclose(
        mesh.interpolate(*points, "phi"), 2 * points[0] + points[1]
    )


def test_equally_spaced_nodes_give_the_uniform_solve() -> None:
    plain = AMRHierarchy((16, 16))
    listed = AMRHierarchy(
        (16, 16), axis_nodes=(np.linspace(0, 1, 17), np.linspace(0, 1, 17))
    )
    for mesh in (plain, listed):
        mesh.add_patch(level=1, box=((8, 24), (8, 24)))
    body = EmbeddedDirichlet(
        lambda p: (p[0] - 0.4) ** 2 + (p[1] - 0.55) ** 2 < 0.15**2, value=1.0
    )
    solutions = []
    for mesh in (plain, listed):
        solver = NodalPoisson(mesh, faces=GROUNDED, embedded=(body,))
        x, y = solver.nodes.coordinates.T
        solutions.append(solver.solve(np.sin(3 * x) * y))
    np.testing.assert_allclose(solutions[1], solutions[0], atol=1e-12)


def _manufactured(solver: NodalPoisson) -> float:
    """phi = sin(pi x) sin(pi y) exp(x): the volume-weighted L2 error."""
    x, y = solver.nodes.coordinates.T
    exact = np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(x)
    laplacian = (
        ((1.0 - 2.0 * np.pi**2) * np.sin(np.pi * x) + 2.0 * np.pi * np.cos(np.pi * x))
        * np.sin(np.pi * y)
        * np.exp(x)
    )
    difference = solver.solve(-laplacian) - exact
    return float(np.sqrt(np.sum(difference**2 * solver.volumes)))


@pytest.mark.parametrize("refine", [False, True])
def test_the_graded_laplacian_converges_at_second_order(refine) -> None:
    errors = []
    for n in (16, 32, 64):
        mesh = AMRHierarchy((n, n), axis_nodes=(wall_graded(n), graded(n)))
        if refine:
            mesh.add_patch(level=1, box=((n // 2, 3 * n // 2), (n // 2, 3 * n // 2)))
        errors.append(_manufactured(NodalPoisson(mesh, faces=GROUNDED)))
    assert np.all(_orders(errors) > 1.85), errors


def test_the_graded_axisymmetric_laplacian_converges_at_second_order() -> None:
    errors = []
    for n in (16, 32, 64):
        mesh = AMRHierarchy(
            (n, n), axis_nodes=(graded(n, 0.5), wall_graded(n)), geometry="cylindrical"
        )
        mesh.add_patch(level=1, box=((n // 2, 3 * n // 2), (0, n)))
        solver = NodalPoisson(
            mesh,
            metric="cylindrical",
            faces={face: GROUNDED[face] for face in ("x_upper", "y_lower", "y_upper")},
        )
        r, z = solver.nodes.coordinates.T
        exact = (1.0 - r**2) * np.sin(np.pi * z)
        source = (4.0 + np.pi**2 * (1.0 - r**2)) * np.sin(np.pi * z)
        difference = solver.solve(source) - exact
        errors.append(float(np.sqrt(np.sum(difference**2 * solver.volumes))))
    assert np.all(_orders(errors) > 1.85), errors


def test_a_body_on_a_graded_mesh_converges_at_second_order() -> None:
    """A grounded box around a disc at 1: the potential of a coaxial line, ln(R/r)/ln(R/a)."""
    a, outer = 0.15, 0.45
    body = EmbeddedDirichlet(
        lambda p: (p[0] - 0.5) ** 2 + (p[1] - 0.5) ** 2 < a**2, value=1.0
    )
    errors = []
    for n in (16, 32, 64):
        mesh = AMRHierarchy((n, n), axis_nodes=(graded(n, 0.5), graded(n, -0.5)))

        def exact(px, py):
            r = np.hypot(px - 0.5, py - 0.5)
            return np.log(outer / np.maximum(r, a)) / np.log(outer / a)

        faces = {face: Dirichlet(exact) for face in GROUNDED}
        solver = NodalPoisson(mesh, faces=faces, embedded=(body,))
        px, py = solver.nodes.coordinates.T
        free = np.hypot(px - 0.5, py - 0.5) > a
        error = np.abs(solver.solve(np.zeros(px.size)) - exact(px, py))[free]
        errors.append(float(error.max()))
    assert np.all(_orders(errors) > 1.7), errors


def test_the_fft_backend_refuses_a_graded_axis() -> None:
    mesh = AMRHierarchy((8, 8), axis_nodes=(graded(8), None))
    with pytest.raises(ValueError, match="graded"):
        NodalPoisson(mesh, faces=GROUNDED, backend="fft")


@pytest.mark.parametrize("backend", ["lu", "multigrid"])
def test_the_backends_agree_on_a_graded_mesh(backend) -> None:
    mesh = AMRHierarchy((32, 32), axis_nodes=(wall_graded(32), graded(32)))
    mesh.add_patch(level=1, box=((0, 16), (16, 48)))
    reference = NodalPoisson(mesh, faces=GROUNDED)
    solver = NodalPoisson(mesh, faces=GROUNDED, backend=backend)
    x, y = solver.nodes.coordinates.T
    source = np.exp(-((x - 0.2) ** 2 + (y - 0.6) ** 2) / 0.01)
    np.testing.assert_allclose(solver.solve(source), reference.solve(source), atol=1e-8)


def test_a_graded_line_and_a_graded_box_converge_at_second_order() -> None:
    for ndim in (1, 3):
        errors = []
        for n in (32, 64, 128) if ndim == 1 else (8, 16, 32):
            mesh = AMRHierarchy(
                (n,) * ndim, axis_nodes=(wall_graded(n),) + (graded(n),) * (ndim - 1)
            )
            faces = {
                f"{a}_{side}": Dirichlet(0.0)
                for a in "xyz"[:ndim]
                for side in ("lower", "upper")
            }
            solver = NodalPoisson(mesh, faces=faces)
            coordinates = solver.nodes.coordinates.T
            exact = np.prod([np.sin(np.pi * c) for c in coordinates], axis=0)
            difference = solver.solve(ndim * np.pi**2 * exact) - exact
            errors.append(float(np.sqrt(np.sum(difference**2 * solver.volumes))))
        assert np.all(_orders(errors) > 1.85), (ndim, errors)


# The cells and CellOperator ------------------------------------------------------------ #


def test_the_leaf_cells_are_physical_boxes() -> None:
    from patchwork import CompositeCells

    mesh = AMRHierarchy((8, 8), axis_nodes=(wall_graded(8), graded(8)))
    mesh.add_patch(level=1, box=((0, 6), (4, 12)))
    cells = CompositeCells(mesh)
    assert cells.volumes.sum() == pytest.approx(1.0)
    points = np.random.default_rng(5).uniform(0.0, 1.0, (2, 500))
    found = cells.locate(points)
    low = cells.centers[found] - 0.5 * cells.sizes[found]
    high = cells.centers[found] + 0.5 * cells.sizes[found]
    assert np.all((points.T >= low - 1e-15) & (points.T <= high + 1e-15))
    # the faces between cells and on the boundary close every cell
    total = 0.0
    for axis in (0, 1):
        faces = cells.boundary(axis)
        area, _, _ = cells.face_metric(
            axis, faces["plane"], faces["start"], faces["count"]
        )
        total += area.sum()
    assert total == pytest.approx(4.0)


@pytest.mark.parametrize("coarse_fine", ["symmetric", "conservative"])
def test_the_graded_cell_operator_converges_at_second_order(coarse_fine) -> None:
    from patchwork import CellOperator

    def exact(x, y):
        return np.sin(np.pi * x) * np.cos(1.3 * y) + 0.2 * x * y

    def source(x, y):
        return (np.pi**2 + 1.69) * np.sin(np.pi * x) * np.cos(1.3 * y)

    errors = []
    for n in (16, 32, 64):
        mesh = AMRHierarchy((n, n), axis_nodes=(wall_graded(n), graded(n)))
        mesh.add_patch(level=1, box=((n // 2, 3 * n // 2), (n // 2, 3 * n // 2)))
        solver = CellOperator(
            mesh,
            faces={face: Dirichlet(exact) for face in GROUNDED},
            coarse_fine=coarse_fine,
        )
        u = solver.solve(source)
        x, y = solver.cells.centers.T
        error = u - exact(x, y)
        errors.append(float(np.sqrt(np.sum(error**2 * solver.cells.volumes))))
    assert np.all(_orders(errors) > 1.8), errors


def test_the_graded_azimuthal_operator_keeps_its_null_field() -> None:
    """A_phi = B r / 2 is the vector potential of a uniform B: a null field, exactly.

    With the conservative coarse-fine faces, whose ghost is exact for a linear
    field (the symmetric ones are not, on a uniform mesh either).
    """
    from patchwork import CellOperator, Neumann

    mesh = AMRHierarchy(
        (12, 10),
        axis_nodes=(wall_graded(12, 0.0, 1.0), graded(10)),
        geometry="cylindrical",
    )
    mesh.add_patch(level=1, box=((4, 14), (6, 14)))
    solver = CellOperator(
        mesh,
        component="azimuthal",
        coarse_fine="conservative",
        faces={
            "x_upper": Dirichlet(lambda r, z: 0.5 * r),
            "y_lower": Neumann(0.0),
            "y_upper": Neumann(0.0),
        },
    )
    u = solver.solve(0.0)
    np.testing.assert_allclose(u, 0.5 * solver.cells.centers[:, 0], atol=1e-12)


# The edges ----------------------------------------------------------------------------- #


def _graded_edges(periodic=(False, False)):
    from patchwork import CompositeEdges

    mesh = AMRHierarchy(
        (16, 16), axis_nodes=(wall_graded(16, 0.0, 2.0), graded(16)), periodic=periodic
    )
    mesh.add_patch(level=1, box=((4, 16), (8, 24)))
    mesh.add_patch(level=2, box=((12, 24), (24, 40)))
    return CompositeEdges(CompositeNodes(mesh))


@pytest.mark.parametrize("periodic", [(False, False), (False, True), (True, True)])
def test_the_path_deposit_conserves_the_charge_on_a_graded_mesh(periodic) -> None:
    edges = _graded_edges(periodic)
    nodes = edges.nodes
    rng = np.random.default_rng(7)
    start = np.vstack((rng.uniform(0.0, 2.0, 600), rng.uniform(0.0, 1.0, 600)))
    end = start + rng.normal(scale=0.2, size=start.shape)
    for axis, wraps in enumerate(periodic):
        if not wraps:
            end[axis] = np.clip(end[axis], 0.0, (2.0, 1.0)[axis])
    charges = rng.normal(size=600)
    current = edges.deposit_paths(start, end, charges)
    divergence = (edges.gradient @ nodes.reduction).T @ current
    change = nodes.reduction.T @ (
        nodes.deposit(end, charges) - nodes.deposit(start, charges)
    )
    assert np.abs(divergence - change).max() < 1e-12
    # the compiled deposit is the NumPy pieces'
    np.testing.assert_allclose(
        current, edges._deposit_pieces(start, end, charges), atol=1e-12
    )


def test_a_uniform_field_on_a_graded_mesh() -> None:
    edges = _graded_edges()
    # the edges along x at y = 0 span the domain's length 2
    nodes = edges.nodes
    bottom = (edges.direction == 0) & (nodes.coordinates[edges.ends[:, 0], 1] == 0.0)
    assert edges.length[bottom].sum() == pytest.approx(2.0)
    field = np.array([2.0, -0.5])
    e = edges.length * field[edges.direction]
    start = np.array([[0.1, 1.9, 0.05], [0.1, 0.8, 0.95]])
    end = np.array([[1.7, 0.2, 0.06], [0.9, 0.15, 0.02]])
    charges = np.array([0.7, 1.1, -0.4])
    work = edges.deposit_paths(start, end, charges) @ e
    assert work == pytest.approx(float(charges @ (field @ (end - start))), rel=1e-12)
    points = np.random.default_rng(8).uniform(0.0, 1.0, (2, 200)) * np.array(
        [[2.0], [1.0]]
    )
    gathered = edges.gather(points, e)
    np.testing.assert_allclose(gathered, field[:, None] * np.ones((1, 200)), atol=1e-12)
    for matrix, value in zip(edges.evaluation(points), field, strict=True):
        np.testing.assert_allclose(matrix @ e, value, atol=1e-12)


def test_the_face_areas_are_physical() -> None:
    from patchwork import CompositeFaces

    edges = _graded_edges()
    faces = CompositeFaces(edges)
    flux = np.where(faces.normal == 0, faces.area, 0.0)  # a uniform flux along x
    np.testing.assert_allclose(faces.divergence @ flux, 0.0, atol=1e-13)


# graded_nodes --------------------------------------------------------------------------- #


def test_graded_nodes_follow_the_spacing() -> None:
    from patchwork import graded_nodes

    # 0.1 mm at the meniscus, 1 mm at the end of a 20 mm gap
    nodes = graded_nodes(0.0, 20e-3, [(0.0, 0.1e-3), (20e-3, 1.0e-3)])
    sizes = np.diff(nodes)
    assert nodes[0] == 0.0 and nodes[-1] == 20e-3
    assert sizes[0] == pytest.approx(0.1e-3, rel=0.05)
    assert sizes[-1] == pytest.approx(1.0e-3, rel=0.05)
    assert np.max(sizes[1:] / sizes[:-1]) < 1.1  # neighbours within 10 %
    # the cell count is the integral of 1 / dx: 20 / 0.9 ln 10 cells
    assert sizes.size == round(20.0 / 0.9 * np.log(10.0))
    uniform = graded_nodes(0.0, 1.0, lambda x: 0.1)
    np.testing.assert_allclose(uniform, np.linspace(0.0, 1.0, 11))
    assert graded_nodes(0.0, 1.0, [(0.5, 0.1)], cells=4).size == 5
    with pytest.raises(ValueError, match="positive"):
        graded_nodes(0.0, 1.0, [(0.0, 0.1), (1.0, -0.1)])
    with pytest.raises(ValueError, match="pairs"):
        graded_nodes(0.0, 1.0, [0.1, 0.2])


def test_graded_nodes_pass_through_given_positions() -> None:
    from patchwork import graded_nodes

    nodes = graded_nodes(
        0.0, 0.04, [(0.0, 0.5e-3), (0.04, 0.25e-3)], through=(0.016, 0.033, 0.05)
    )
    for x in (0.016, 0.033):
        assert np.min(np.abs(nodes - x)) == 0.0
    assert nodes[0] == 0.0 and nodes[-1] == 0.04 and np.all(np.diff(nodes) > 0)
    sizes = np.diff(nodes)
    assert sizes[0] == pytest.approx(0.5e-3, rel=0.1) and sizes[-1] == pytest.approx(
        0.25e-3, rel=0.1
    )
    with pytest.raises(ValueError, match="combined"):
        graded_nodes(0.0, 1.0, [(0.0, 0.1)], cells=4, through=(0.5,))
