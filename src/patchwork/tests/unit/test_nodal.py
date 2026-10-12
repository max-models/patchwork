"""Tests for the node-centered composite Poisson solver."""

from __future__ import annotations

import numpy as np
import pytest

from patchwork import (
    AMRHierarchy,
    CompositeNodes,
    Dirichlet,
    EmbeddedDirichlet,
    Neumann,
    NodalPoisson,
    solve_nodal_poisson,
)

FACES = ("x_lower", "x_upper", "y_lower", "y_upper")

#: Patch layouts on a 16 x 16 base mesh: none, nested, and a level-2 patch that
#: touches the edge of its parent (cells four times finer next to base cells).
LAYOUTS = {
    "uniform": [],
    "nested": [(1, ((8, 24), (8, 24))), (2, ((24, 40), (24, 40)))],
    "level jump": [(1, ((8, 24), (8, 24))), (2, ((16, 32), (16, 32)))],
    "two patches": [(1, ((4, 12), (4, 12))), (1, ((16, 28), (12, 30)))],
}


def _mesh(layout: str, base: int = 16, **kwargs) -> AMRHierarchy:
    mesh = AMRHierarchy(base_shape=(base, base), **kwargs)
    scale = base // 16
    for level, box in LAYOUTS[layout]:
        mesh.add_patch(
            level=level, box=tuple((lo * scale, hi * scale) for lo, hi in box)
        )
    return mesh


def _error(solver: NodalPoisson, exact, source) -> float:
    x, y = solver.nodes.coordinates.T
    return float(np.abs(solver.solve(source(x, y)) - exact(x, y)).max())


def test_node_fields_have_one_value_per_cell_corner() -> None:
    mesh = _mesh("nested")
    mesh.allocate_field("phi", centering="node")
    mesh.allocate_field("rho")

    for patch in mesh:
        assert patch.field("phi").shape == patch.node_shape
        assert patch.field("rho").shape == patch.shape
        assert patch.node_shape == (patch.shape[0] + 1, patch.shape[1] + 1)
    assert mesh.centering("phi") == "node" and mesh.centering("rho") == "cell"
    later = mesh.add_patch(level=3, box=((48, 64), (48, 64)))
    assert later.field("phi").shape == later.node_shape


def test_patches_share_the_nodes_they_have_in_common() -> None:
    mesh = _mesh("nested")
    nodes = CompositeNodes(mesh)
    base, middle, fine = mesh.patches

    # every other node of a fine patch is a node of its parent
    assert np.array_equal(
        nodes.patch_nodes(middle)[::2, ::2], nodes.patch_nodes(base)[4:13, 4:13]
    )
    assert np.array_equal(
        nodes.patch_nodes(fine)[::2, ::2], nodes.patch_nodes(middle)[4:13, 4:13]
    )
    # 17^2 base nodes, plus the new nodes of each finer patch
    assert nodes.num_nodes == 17**2 + (17**2 - 9**2) * 2
    # the midpoints of the coarse edges on both patch boundaries hang
    assert nodes.hanging.sum() == 2 * 4 * 8


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_linear_potential_is_reproduced_exactly(layout: str) -> None:
    """No charge and a linear potential on the faces: linear everywhere, hanging nodes too."""

    def exact(x, y):
        return 1.0 + 2.0 * x - 3.0 * y

    solver = NodalPoisson(
        _mesh(layout), faces={face: Dirichlet(exact) for face in FACES}
    )

    assert _error(solver, exact, lambda x, y: 0.0 * x) < 1e-12


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_operator_is_symmetric_and_conserves_charge(layout: str) -> None:
    solver = NodalPoisson(_mesh(layout))
    matrix = solver.matrix

    assert abs(matrix - matrix.T).max() < 1e-12
    # a unit source everywhere loads the unknowns with the area they stand for
    load = solver.load @ np.ones(solver.nodes.num_nodes)
    on_faces = np.zeros(solver.nodes.num_nodes)
    for face in FACES:
        on_faces[solver.nodes.face_nodes(face)] = 1.0
    boundary = solver.nodes.volumes[on_faces > 0].sum()
    assert load.sum() == pytest.approx(1.0 - boundary)


def test_a_uniform_mesh_has_the_five_point_stencil() -> None:
    mesh = AMRHierarchy(base_shape=(4, 4), bounds=((0.0, 2.0), (0.0, 1.0)))
    solver = NodalPoisson(mesh)
    hx, hy = mesh.spacing(0)
    centre = int(np.argmin(np.abs(solver.nodes.coordinates - (1.0, 0.5)).sum(axis=1)))
    row = solver.matrix.toarray()[list(solver.free).index(centre)]

    # control volume hx * hy times (2 / hx^2 + 2 / hy^2), and the four links
    assert row.max() == pytest.approx(2.0 * hy / hx + 2.0 * hx / hy)
    assert sorted(np.round(row[row < 0], 12)) == pytest.approx(
        sorted([-hy / hx] * 2 + [-hx / hy] * 2)
    )


@pytest.mark.parametrize("layout", ["uniform", "nested", "level jump"])
def test_the_solution_converges_at_second_order(layout: str) -> None:
    def exact(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def source(x, y):
        return 2.0 * np.pi**2 * exact(x, y)

    errors = [
        _error(NodalPoisson(_mesh(layout, base)), exact, source)
        for base in (16, 32, 64)
    ]

    assert errors[0] / errors[1] > 3.3 and errors[1] / errors[2] > 3.3
    assert errors[2] < 5e-4


def test_neumann_faces_and_a_periodic_axis_converge_at_second_order() -> None:
    def exact(x, y):
        return np.cos(2.0 * np.pi * y) * np.exp(x) + x

    def source(x, y):
        return (4.0 * np.pi**2 - 1.0) * np.cos(2.0 * np.pi * y) * np.exp(x)

    faces = {
        # the outward derivative on the lower x face is -d/dx
        "x_lower": Neumann(lambda x, y: -(np.cos(2.0 * np.pi * y) + 1.0)),
        "x_upper": Dirichlet(exact),
    }
    errors = []
    for base in (16, 32, 64):
        mesh = AMRHierarchy(base_shape=(base, base), periodic=(False, True))
        # a patch on the Neumann face and one along the whole periodic axis
        mesh.add_patch(level=1, box=((0, base // 2), (base // 2, 3 * base // 2)))
        mesh.add_patch(level=1, box=((base, 3 * base // 2), (0, 2 * base)))
        errors.append(_error(NodalPoisson(mesh, faces=faces), exact, source))

    assert errors[0] / errors[1] > 3.5 and errors[1] / errors[2] > 3.5


def _disc(radius: float):
    def exact(x, y):
        r = np.maximum(np.hypot(x - 0.5, y - 0.5), 1e-12)
        return np.where(r < radius, 1.0, 1.0 + 0.5 * np.log(r / radius))

    body = EmbeddedDirichlet(
        inside=lambda points: np.hypot(points[0] - 0.5, points[1] - 0.5) < radius,
        value=1.0,
    )
    return exact, body


def test_an_embedded_body_converges_at_second_order() -> None:
    exact, body = _disc(0.15)
    faces = {face: Dirichlet(exact) for face in FACES}
    errors = []
    for base in (16, 32, 64):
        solver = NodalPoisson(
            AMRHierarchy(base_shape=(base, base)), faces=faces, embedded=[body]
        )
        x, y = solver.nodes.coordinates.T
        potential = solver.solve(np.zeros(solver.nodes.num_nodes))
        errors.append(np.abs(potential - exact(x, y)).max())

    assert errors[0] / errors[1] > 3.0 and errors[1] / errors[2] > 3.5
    # nodes inside the body hold its potential
    inside = np.hypot(x - 0.5, y - 0.5) < 0.15
    assert np.all(potential[inside] == 1.0)


def test_patches_resolve_a_body_the_base_mesh_cannot() -> None:
    """A disc of half a cell's radius: nested patches around it cut the error tenfold."""
    exact, body = _disc(0.03)
    faces = {face: Dirichlet(exact) for face in FACES}

    def error(patches) -> tuple[float, int]:
        mesh = AMRHierarchy(base_shape=(32, 32))
        for level, box in patches:
            mesh.add_patch(level=level, box=box)
        solver = NodalPoisson(mesh, faces=faces, embedded=[body])
        x, y = solver.nodes.coordinates.T
        potential = solver.solve(np.zeros(solver.nodes.num_nodes))
        base = np.all(solver.nodes.lattice % solver.nodes.cell_span.max() == 0, axis=1)
        return float(np.abs(potential - exact(x, y))[base].max()), solver.free.size

    coarse, coarse_unknowns = error([])
    refined, refined_unknowns = error(
        [(1, ((24, 40), (24, 40))), (2, ((56, 72), (56, 72)))]
    )

    assert refined < 0.1 * coarse
    assert refined_unknowns < 1.4 * coarse_unknowns


def test_conditions_change_without_factorizing_again() -> None:
    _, body = _disc(0.15)
    mesh = _mesh("nested")
    solver = NodalPoisson(
        mesh,
        faces={"x_lower": Dirichlet(0.0), "y_upper": Neumann(0.0)},
        embedded=[body],
    )
    factorization = solver._factorization
    source = np.ones(solver.nodes.num_nodes)

    solver.set_conditions(
        faces={"x_lower": Dirichlet(lambda x, y: 5.0 * y), "y_upper": Neumann(2.0)},
        body_values=[-3.0],
    )

    fresh = NodalPoisson(
        mesh,
        faces={"x_lower": Dirichlet(lambda x, y: 5.0 * y), "y_upper": Neumann(2.0)},
        embedded=[EmbeddedDirichlet(body.inside, -3.0)],
    )
    assert solver._factorization is factorization
    assert solver.solve(source) == pytest.approx(fresh.solve(source), abs=1e-12)
    with pytest.raises(ValueError, match="cannot change its kind"):
        solver.set_conditions(faces={"x_lower": Neumann(0.0)})


def test_a_surface_through_a_periodic_face_is_found_on_the_short_edge() -> None:
    """A slab around the periodic face: the same potential as around the middle."""

    def solve(centre: float) -> np.ndarray:
        def inside(points):
            distance = np.abs((points[1] - centre + 0.5) % 1.0 - 0.5)
            return distance < 0.11

        mesh = AMRHierarchy(base_shape=(8, 16), periodic=(False, True))
        solver = NodalPoisson(
            mesh,
            faces={"x_lower": Dirichlet(0.0), "x_upper": Neumann(0.0)},
            embedded=[EmbeddedDirichlet(inside, value=2.0)],
        )
        potential = solver.solve(np.ones(solver.nodes.num_nodes))
        return potential[solver.nodes.patch_nodes(mesh.patches[0])][:, :16]

    at_face, in_middle = solve(0.0), solve(0.5)

    assert np.roll(at_face, 8, axis=1) == pytest.approx(in_middle, abs=1e-10)


def test_fields_go_to_the_patches_and_back() -> None:
    mesh = _mesh("nested")
    mesh.allocate_field("rhs", centering="node")
    for patch in mesh:
        x, y = patch.node_coordinates(mesh)
        patch.field("rhs").local[...] = (
            2.0 * np.pi**2 * np.outer(np.sin(np.pi * x), np.sin(np.pi * y))
        )

    solver = solve_nodal_poisson(mesh, "rhs", potential="phi")

    assert mesh.centering("phi") == "node"
    for patch in mesh:
        x, y = patch.node_coordinates(mesh)
        exact = np.outer(np.sin(np.pi * x), np.sin(np.pi * y))
        assert np.abs(patch.field("phi").local - exact).max() < 5e-3
    nodes = solver.nodes
    assert nodes.from_patches("phi") == pytest.approx(solver.solve("rhs"))


def test_interpolation_between_levels_is_exact_for_linear_functions() -> None:
    mesh = AMRHierarchy(base_shape=(8, 6), periodic=(False, True))
    mesh.add_patch(level=1, box=((4, 12), (0, 12)))
    mesh.add_patch(level=2, box=((12, 20), (8, 16)))
    nodes = CompositeNodes(mesh)
    x = nodes.coordinates[:, 0]
    values = 1.0 + 2.0 * x

    base = 1.0 + 2.0 * np.arange(9)[:, None] / 8 * np.ones((1, 6))
    assert nodes.level_shape(0) == (9, 6) and nodes.level_shape(2) == (33, 24)
    assert nodes.from_level(0) @ base.ravel() == pytest.approx(values)
    fine = (nodes.to_level(2) @ values).reshape(33, 24)
    assert fine == pytest.approx(
        1.0 + 2.0 * np.arange(33)[:, None] / 32 * np.ones((1, 24))
    )
    points = np.array([[0.1, 0.33, 0.61, 0.999], [0.0, 0.4, 0.5, 0.97]])
    assert nodes.evaluation(points) @ values == pytest.approx(1.0 + 2.0 * points[0])


def test_patch_fields_are_sampled_from_the_finest_patch_around_a_node() -> None:
    mesh = AMRHierarchy(base_shape=(8, 6), periodic=(False, True))
    mesh.add_patch(level=1, box=((4, 12), (0, 12)))  # along the whole periodic axis
    mesh.add_patch(level=2, box=((12, 20), (8, 16)))
    nodes = CompositeNodes(mesh)

    def function(x, y):
        return np.sin(2.0 * np.pi * y) * x

    arrays = []
    for patch in mesh.patches:
        x, y = patch.node_coordinates(mesh)
        values = function(*np.meshgrid(x, y, indexing="ij"))
        arrays.append(np.stack([values, 2.0 * values], axis=-1))
    x, y = np.meshgrid(np.arange(33) / 32, np.arange(24) / 24, indexing="ij")
    in_fine = (x >= 12 / 32) & (x <= 20 / 32) & (y >= 8 / 24) & (y <= 16 / 24)
    in_middle = (x >= 0.25) & (x <= 0.75) & ~in_fine
    on_middle_nodes = (np.arange(33)[:, None] % 2 == 0) & (
        np.arange(24)[None, :] % 2 == 0
    )

    sampled = nodes.sample_patches(arrays)

    assert sampled.shape == (33, 24, 2)
    assert sampled[..., 1] == pytest.approx(2.0 * sampled[..., 0])
    assert sampled[in_fine, 0] == pytest.approx(function(x, y)[in_fine])
    exact_here = in_middle & on_middle_nodes
    assert sampled[exact_here, 0] == pytest.approx(function(x, y)[exact_here])
    # between the nodes of a coarser patch: its bilinear interpolant
    assert sampled[0, 1, 0] == pytest.approx(0.0)
    assert sampled[2, 2, 0] == pytest.approx(
        0.5 * (function(0.0, 0.0) + function(0.125, 1 / 6)) * 0.5
        + 0.5 * (function(0.125, 0.0) + function(0.0, 1 / 6)) * 0.5
    )
    # a coarser lattice takes the finer patches at its own nodes
    coarser = nodes.sample_patches(arrays, level=1)
    assert coarser.shape == (17, 12, 2)
    assert coarser[..., 0][in_fine[::2, ::2]] == pytest.approx(
        function(x, y)[::2, ::2][in_fine[::2, ::2]]
    )


def test_a_potential_known_only_up_to_a_constant_is_refused() -> None:
    mesh = AMRHierarchy(base_shape=(4, 4), periodic=(True, True))

    with pytest.raises(ValueError, match="singular"):
        NodalPoisson(mesh)
    with pytest.raises(ValueError, match="periodic axis"):
        NodalPoisson(mesh, faces={"x_lower": Dirichlet(0.0)})
    with pytest.raises(ValueError, match="unknown faces"):
        NodalPoisson(AMRHierarchy(base_shape=(4, 4)), faces={"left": Dirichlet(0.0)})
