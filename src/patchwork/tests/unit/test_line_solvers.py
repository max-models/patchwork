"""The composite solvers on a 1D hierarchy: NodalPoisson and CellOperator on a line.

The 1D counterparts of ``test_nodal.py``, ``test_cells.py`` and
``test_cells_bodies_curl.py``, and two of picnic's 1D field solvers, written out
here (picnic is not a dependency): ``DirichletPoisson1D``, the tridiagonal
finite-difference solve on a uniform node grid, and ``SheetField1D``, the exact
field of charged sheets between walls.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import epsilon_0
from scipy.linalg import solve_banded

from patchwork import (
    AMRHierarchy,
    CellOperator,
    CompositeNodes,
    Dirichlet,
    EmbeddedDirichlet,
    EmbeddedFloating,
    Neumann,
    NodalPoisson,
    cell_body,
    solve_poisson,
)

WALLS = ("x_lower", "x_upper")

#: Patch layouts on a base of 16 cells: none, nested, a level-2 patch at the
#: edge of its parent (cells four times finer next to base cells), two patches.
LAYOUTS = {
    "uniform": [],
    "nested": [(1, ((8, 24),)), (2, ((24, 40),))],
    "level jump": [(1, ((8, 24),)), (2, ((16, 32),))],
    "two patches": [(1, ((4, 12),)), (1, ((16, 28),))],
}


def line(layout: str, base: int = 16, **kwargs) -> AMRHierarchy:
    mesh = AMRHierarchy(base_shape=(base,), **kwargs)
    scale = base // 16
    for level, ((lo, hi),) in LAYOUTS[layout]:
        mesh.add_patch(level=level, box=((lo * scale, hi * scale),))
    return mesh


def orders(errors) -> np.ndarray:
    errors = np.asarray(errors)
    return np.log2(errors[:-1] / errors[1:])


# NodalPoisson ----------------------------------------------------------------------- #


def test_nodes_of_a_line() -> None:
    mesh = line("nested")
    nodes = CompositeNodes(mesh)
    base, middle, fine = mesh.patches

    # 17 base nodes, and the new ones of each patch; none hangs
    assert nodes.num_nodes == 17 + 8 + 8
    assert not nodes.hanging.any()
    assert np.array_equal(nodes.patch_nodes(middle)[::2], nodes.patch_nodes(base)[4:13])
    assert nodes.cell_corners.shape == (nodes.cell_span.size, 2)
    # the control volumes add up to the length of the line
    assert nodes.volumes.sum() == pytest.approx(1.0)
    first, second, length, midpoint = nodes.face_edges("x_upper")
    assert first.tolist() == second.tolist() == [nodes.num_nodes - 1]
    assert length.tolist() == [1.0] and midpoint.tolist() == [[1.0]]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_quadratic_potential_is_exact_across_coarse_fine_nodes(layout) -> None:
    """In 1D the three-point stencil is exact for quadratics, with unequal arms too."""

    def exact(x):
        return 1.0 + 2.0 * x - 3.0 * x**2

    solver = NodalPoisson(
        line(layout), faces={face: Dirichlet(exact) for face in WALLS}
    )
    (x,) = solver.nodes.coordinates.T

    np.testing.assert_allclose(solver.solve(np.full(x.size, 6.0)), exact(x), atol=1e-12)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_operator_is_symmetric_and_conserves_charge(layout) -> None:
    solver = NodalPoisson(line(layout))
    matrix = solver.matrix

    assert abs(matrix - matrix.T).max() < 1e-12
    load = solver.load @ np.ones(solver.nodes.num_nodes)
    walls = solver.nodes.volumes[[0, -1]].sum()
    assert load.sum() == pytest.approx(1.0 - walls)


def test_a_uniform_line_has_the_three_point_stencil() -> None:
    mesh = AMRHierarchy(base_shape=(8,), bounds=((0.0, 2.0),))
    solver = NodalPoisson(mesh)
    (h,) = mesh.spacing(0)
    tridiagonal = (
        np.diag(np.full(7, 2.0)) - np.diag(np.ones(6), 1) - np.diag(np.ones(6), -1)
    )
    np.testing.assert_allclose(solver.matrix.toarray() * h, tridiagonal, atol=1e-12)


@pytest.mark.parametrize("layout", ["uniform", "nested", "level jump"])
def test_the_nodal_solution_converges_at_second_order(layout) -> None:
    def exact(x):
        return np.sin(3.0 * x) + np.exp(x)

    def source(x):
        return 9.0 * np.sin(3.0 * x) - np.exp(x)

    errors = []
    for base in (16, 32, 64):
        solver = NodalPoisson(
            line(layout, base), faces={face: Dirichlet(exact) for face in WALLS}
        )
        (x,) = solver.nodes.coordinates.T
        errors.append(np.abs(solver.solve(source(x)) - exact(x)).max())

    assert np.all(orders(errors) > 1.9), errors


def test_a_neumann_face_converges_at_second_order() -> None:
    def exact(x):
        return np.cos(2.0 * x) + x

    errors = []
    for base in (16, 32, 64):
        solver = NodalPoisson(
            line("level jump", base),
            # the outward derivative at x = 0 is -du/dx = -1
            faces={"x_lower": Neumann(-1.0), "x_upper": Dirichlet(exact)},
        )
        (x,) = solver.nodes.coordinates.T
        errors.append(np.abs(solver.solve(4.0 * np.cos(2.0 * x)) - exact(x)).max())

    assert np.all(orders(errors) > 1.9), errors


def test_a_body_on_a_periodic_line_is_exact_for_a_quadratic_potential() -> None:
    """The Shortley-Weller arms reach the surfaces, which are not nodes."""
    a, b = 0.437, 0.6013  # the body, inside the level-2 patch; zero on its surfaces
    mesh = line("nested", periodic=(True,))

    def inside(points):
        return (points[0] > a) & (points[0] < b)

    solver = NodalPoisson(mesh, embedded=[EmbeddedDirichlet(inside, 0.0)])
    (x,) = solver.nodes.coordinates.T
    phi = solver.solve(np.ones(x.size))

    # -phi'' = 1 between b and a + 1 (through the periodic face), phi = 0 at both
    distance = np.where(x >= b, x - b, x + 1.0 - b)
    exact = np.where(inside(x[None, :]), 0.0, distance * (a + 1.0 - b - distance) / 2)
    np.testing.assert_allclose(phi, exact, atol=1e-12)


def test_a_periodic_line_needs_an_anchor_and_takes_no_face() -> None:
    with pytest.raises(ValueError, match="singular"):
        NodalPoisson(line("uniform", periodic=(True,)))
    with pytest.raises(ValueError, match="periodic"):
        NodalPoisson(line("uniform", periodic=(True,)), faces={"x_lower": Neumann()})
    with pytest.raises(ValueError, match="unknown faces"):
        NodalPoisson(line("uniform"), faces={"y_lower": Dirichlet()})


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_deposit_keeps_the_charge_and_is_the_transposed_evaluation(layout) -> None:
    nodes = CompositeNodes(line(layout))
    rng = np.random.default_rng(3)
    points = rng.random((1, 300))
    charges = rng.normal(size=300)

    charge = nodes.deposit(points, charges)

    assert charge.sum() == pytest.approx(charges.sum(), rel=1e-12)
    np.testing.assert_allclose(charge, nodes.evaluation(points).T @ charges)
    # a linear field is gathered exactly
    (x,) = nodes.coordinates.T
    np.testing.assert_allclose(
        nodes.evaluation(points) @ (2.0 * x - 1.0), 2.0 * points[0] - 1.0
    )
    assert nodes.deposit(np.array([[-0.5, 1.5]]), np.ones(2)).sum() == 0.0


def test_fields_go_to_the_patches_and_back_and_are_sampled() -> None:
    mesh = line("level jump")
    solver = NodalPoisson(mesh, faces={"x_upper": Dirichlet(1.0)})
    (x,) = solver.nodes.coordinates.T
    phi = solver.solve(np.zeros(x.size), potential="phi")

    np.testing.assert_allclose(solver.nodes.from_patches("phi"), phi)
    np.testing.assert_allclose(phi, x, atol=1e-12)
    sampled = solver.nodes.to_level(0) @ phi
    np.testing.assert_allclose(sampled, np.linspace(0.0, 1.0, 17), atol=1e-12)
    arrays = [np.asarray(patch.field("phi").local) for patch in mesh.patches]
    np.testing.assert_allclose(
        solver.nodes.sample_patches(arrays), np.linspace(0.0, 1.0, 65), atol=1e-12
    )


def test_a_line_of_separate_pieces_has_inner_faces() -> None:
    """Base boxes that do not touch: the ends inside the interval are ``"inner"``."""
    mesh = AMRHierarchy(base_shape=(16,), base_boxes=[((0, 6),), ((10, 16),)])
    mesh.add_patch(level=1, box=((2, 10),))
    faces = {
        "x_lower": Dirichlet(0.0),
        "x_upper": Dirichlet(1.0),
        "inner": Dirichlet(lambda x: x),
    }
    solver = NodalPoisson(mesh, faces=faces)
    (x,) = solver.nodes.coordinates.T
    np.testing.assert_allclose(x[solver.nodes.face_nodes("inner")], [0.375, 0.625])
    np.testing.assert_allclose(solver.solve(np.zeros(x.size)), x, atol=1e-12)
    operator = CellOperator(mesh, faces=faces)
    (centres,) = operator.cells.centers.T
    np.testing.assert_allclose(operator.solve(0.0), centres, atol=1e-12)
    assert mesh.locate(np.array([0.3, 0.45, 0.7])).tolist() == [2, -1, 1]


# picnic's 1D field solvers -------------------------------------------------------- #


def dirichlet_poisson_1d(rho, spacing, phi_lower, phi_upper):
    """picnic's ``DirichletPoisson1D.solve``: -phi'' = rho / eps0 on a node grid."""
    interior = rho.size - 2
    banded = np.zeros((3, interior))
    banded[0, 1:] = -1.0
    banded[1, :] = 2.0
    banded[2, :-1] = -1.0
    rhs = rho[1:-1] * spacing**2 / epsilon_0
    rhs[0] += phi_lower
    rhs[-1] += phi_upper
    phi = np.empty(rho.size)
    phi[0], phi[-1] = phi_lower, phi_upper
    phi[1:-1] = solve_banded((1, 1), banded, rhs)
    return phi


def test_the_same_potential_as_picnic_s_dirichlet_poisson_1d() -> None:
    length, cells = 0.067, 128
    mesh = AMRHierarchy(base_shape=(cells,), bounds=((0.0, length),))
    solver = NodalPoisson(
        mesh, faces={"x_lower": Dirichlet(450.0), "x_upper": Dirichlet(0.0)}
    )
    (x,) = solver.nodes.coordinates.T
    rho = 1.0e-6 * (1.0 + np.sin(40.0 * x)) * np.exp(-x / 0.02)

    for voltage in (450.0, -450.0):  # an RF electrode: new values, no new LU
        solver.set_conditions(faces={"x_lower": Dirichlet(voltage)})
        expected = dirichlet_poisson_1d(rho, length / cells, voltage, 0.0)
        np.testing.assert_allclose(
            solver.solve(rho / epsilon_0), expected, rtol=0, atol=1e-10 * 450.0
        )


def sheet_potential(positions, sigma, points, length, left="dirichlet"):
    """picnic's ``SheetField1D.potential`` with eps = 1: sheets between two walls.

    The right wall is grounded; the left one too, or field-free (``"neumann"``).
    """
    if left == "neumann":
        field_0 = 0.0
        phi_0 = np.sum(sigma * (length - positions))
    else:
        field_0 = -np.sum(sigma * (length - positions)) / length
        phi_0 = 0.0
    below = positions[None, :] < points[:, None]
    return (
        phi_0
        - field_0 * points
        - (below * sigma * (points[:, None] - positions)).sum(1)
    )


@pytest.mark.parametrize("left", ["dirichlet", "neumann"])
@pytest.mark.parametrize("layout", ["uniform", "level jump", "two patches"])
def test_deposited_sheets_have_picnic_s_exact_sheet_potential(left, layout) -> None:
    """A linear deposit and the three-point operator give the exact potential at the nodes.

    In 1D the composite operator is that of linear finite elements and the
    cloud-in-cell deposit is their load of a point charge, whose potential is
    piecewise linear: the nodes are exact, on refined lines too.
    """
    length = 2.0
    mesh = line(layout, bounds=((0.0, length),))
    faces = {"x_upper": Dirichlet(0.0)}
    faces["x_lower"] = Neumann(0.0) if left == "neumann" else Dirichlet(0.0)
    solver = NodalPoisson(mesh, faces=faces)
    rng = np.random.default_rng(7)
    positions = rng.uniform(0.0, length, 200)
    sigma = rng.normal(size=200)
    (x,) = solver.nodes.coordinates.T

    phi = solver.solve(charge=solver.nodes.deposit(positions[None, :], sigma))

    expected = sheet_potential(positions, sigma, x, length, left)
    np.testing.assert_allclose(
        phi, expected, rtol=0, atol=1e-11 * np.abs(expected).max()
    )


# CellOperator ----------------------------------------------------------------------- #


def refined(n: int, **kwargs) -> AMRHierarchy:
    """A line of n cells with a level-1 patch over part of it and a level-2 one in that."""
    mesh = AMRHierarchy(base_shape=(n,), **kwargs)
    mesh.add_patch(level=1, box=((n // 2, n + n // 2),))
    mesh.add_patch(level=2, box=((3 * n // 2, 5 * n // 2),))
    return mesh


def exact(x):
    return np.sin(np.pi * x) + 0.2 * x


def diffusion(x):
    return 1.0 + 0.5 * np.sin(2.0 * x)


def reaction(x):
    return 2.0 + x


def source(x):
    derivative = np.pi * np.cos(np.pi * x) + 0.2
    second = -(np.pi**2) * np.sin(np.pi * x)
    return -(np.cos(2.0 * x) * derivative + diffusion(x) * second) + reaction(
        x
    ) * exact(x)


def test_leaf_cells_tile_the_line() -> None:
    operator = CellOperator(refined(8), factorize=False)
    cells = operator.cells
    assert cells.volumes.sum() == pytest.approx(1.0)
    assert cells.centers.shape == (cells.num_cells, 1)
    np.testing.assert_array_equal(
        cells.locate(cells.centers.T), np.arange(cells.num_cells)
    )
    assert cells.locate(np.array([[-0.1, 1.1]])).tolist() == [-1, -1]


@pytest.mark.parametrize("coarse_fine", ["symmetric", "conservative"])
def test_a_manufactured_solution_converges_at_second_order(coarse_fine) -> None:
    errors = []
    for n in (16, 32, 64):
        operator = CellOperator(
            refined(n),
            diffusion=diffusion,
            reaction=reaction,
            faces={face: Dirichlet(exact) for face in WALLS},
            coarse_fine=coarse_fine,
        )
        (x,) = operator.cells.centers.T
        errors.append(np.abs(operator.solve(source) - exact(x)).max())

    assert np.all(orders(errors) > 1.9), errors


def test_a_patch_over_the_whole_line_is_the_uniform_fine_grid() -> None:
    coarse = AMRHierarchy(base_shape=(16,))
    coarse.add_patch(level=1, box=((0, 32),))
    fine = AMRHierarchy(base_shape=(32,))
    settings = {
        "diffusion": diffusion,
        "reaction": reaction,
        "faces": {"x_lower": Dirichlet(1.0), "x_upper": Neumann(0.5)},
    }

    np.testing.assert_allclose(
        CellOperator(coarse, **settings).solve(source),
        CellOperator(fine, **settings).solve(source),
        atol=1e-12,
    )


@pytest.mark.parametrize("coarse_fine", ["symmetric", "conservative"])
def test_a_complex_skin_layer_matches_its_closed_form(coarse_fine) -> None:
    """-u'' + i k^2 u = 0 from u = 1 at the wall: the decaying skin layer."""
    k = 30.0
    gamma = np.sqrt(1j) * k

    def layer(x):
        return np.sinh(gamma * (1.0 - x)) / np.sinh(gamma)

    errors = []
    for n in (32, 64, 128):
        mesh = AMRHierarchy(base_shape=(n,))
        mesh.add_patch(level=1, box=((0, n // 2),))
        mesh.add_patch(level=2, box=((0, n // 2),))
        operator = CellOperator(
            mesh,
            reaction=1j * k**2,
            faces={"x_lower": Dirichlet(1.0), "x_upper": Dirichlet(0.0)},
            coarse_fine=coarse_fine,
        )
        (x,) = operator.cells.centers.T
        errors.append(np.abs(operator.solve(0.0) - layer(x)).max())

    assert np.all(orders(errors) > 1.9), errors


def test_the_operator_is_symmetric_and_conservative() -> None:
    operator = CellOperator(refined(16), diffusion=diffusion, reaction=1j)
    matrix = operator.matrix
    assert abs(matrix - matrix.T).max() < 1e-12
    # the conservative variant: every face flux leaves one cell and enters the other
    conservative = CellOperator(
        refined(16), diffusion=diffusion, coarse_fine="conservative", factorize=False
    )
    walls = np.zeros(conservative.cells.num_cells)
    for cells, strength in zip(
        conservative._wall_cells, conservative._wall_strength, strict=True
    ):
        np.add.at(walls, cells, strength)
    column_sums = np.asarray(conservative.stiffness.sum(axis=0)).ravel()
    np.testing.assert_allclose(column_sums, walls, atol=1e-10)


def test_a_periodic_line_and_a_neumann_face() -> None:
    errors = []
    for n in (16, 32, 64):
        operator = CellOperator(
            refined(n, periodic=(True,)), reaction=1.0, coarse_fine="conservative"
        )
        (x,) = operator.cells.centers.T
        u = operator.solve(lambda x: (4.0 * np.pi**2 + 1.0) * np.cos(2.0 * np.pi * x))
        errors.append(np.abs(u - np.cos(2.0 * np.pi * x)).max())
    assert np.all(orders(errors) > 1.9), errors

    # an outward gradient of 2 at the upper face and u = 1 at the lower: u = 1 + 2x
    operator = CellOperator(
        refined(16), faces={"x_lower": Dirichlet(1.0), "x_upper": Neumann(2.0)}
    )
    (x,) = operator.cells.centers.T
    np.testing.assert_allclose(operator.solve(0.0), 1.0 + 2.0 * x, atol=1e-12)


@pytest.mark.parametrize("coarse_fine", ["symmetric", "conservative"])
def test_the_gradient_of_a_linear_field_is_exact(coarse_fine) -> None:
    def linear(x):
        return 2.0 * x - 1.0

    operator = CellOperator(
        refined(16),
        faces={face: Dirichlet(linear) for face in WALLS},
        coarse_fine=coarse_fine,
        factorize=False,
    )
    (x,) = operator.cells.centers.T
    np.testing.assert_allclose(operator.gradient(linear(x)), 2.0, atol=1e-10)
    with pytest.raises(ValueError, match="2D"):
        operator.curl(linear(x))


@pytest.mark.parametrize("coarse_fine", ["symmetric", "conservative"])
def test_the_gradient_is_second_order_across_coarse_fine_faces(coarse_fine) -> None:
    errors = []
    for n in (16, 32, 64):
        operator = CellOperator(
            refined(n),
            faces={face: Dirichlet(exact) for face in WALLS},
            coarse_fine=coarse_fine,
            factorize=False,
        )
        (x,) = operator.cells.centers.T
        gradient = operator.gradient(exact(x))[:, 0]
        errors.append(np.abs(gradient - (np.pi * np.cos(np.pi * x) + 0.2)).max())
    assert np.all(orders(errors) > 1.8), errors


@pytest.mark.parametrize("refine", [False, True])
def test_a_floating_slab_between_two_walls(refine) -> None:
    """No net flux into the slab: the same slope on both sides, the slab at the middle value."""
    mesh = AMRHierarchy(base_shape=(20,))
    if refine:
        mesh.add_patch(level=1, box=((12, 28),))

    def inside(points):
        return (points[0] > 0.4) & (points[0] < 0.6)

    operator = CellOperator(
        mesh,
        faces={"x_lower": Dirichlet(0.0), "x_upper": Dirichlet(1.0)},
        embedded=[EmbeddedFloating(inside)],
    )
    (x,) = operator.cells.centers.T
    u = operator.solve(0.0)
    expected = np.where(x < 0.5, 1.25 * x, 1.0 - 1.25 * (1.0 - x))
    expected[inside(x[None, :])] = 0.5
    np.testing.assert_allclose(u, expected, atol=1e-12)


def test_a_body_of_cells_holds_its_value() -> None:
    mesh = refined(16)
    cells = CellOperator(mesh, factorize=False).cells
    (x,) = cells.centers.T
    body = cell_body(cells, x > 0.75, 2.0)
    operator = CellOperator(mesh, faces={"x_lower": Dirichlet(0.0)}, embedded=[body])
    u = operator.solve(0.0)
    np.testing.assert_allclose(u[x > 0.75], 2.0)
    # linear from 0 at the wall to 2 at the body's surface, x = 0.75
    np.testing.assert_allclose(u[x < 0.75], 2.0 * x[x < 0.75] / 0.75, atol=1e-12)


def test_solutions_go_to_the_patches_and_back() -> None:
    mesh = refined(16)
    operator = CellOperator(mesh, faces={face: Dirichlet(exact) for face in WALLS})
    u = operator.solve(lambda x: np.pi**2 * np.sin(np.pi * x), potential="u")
    np.testing.assert_allclose(operator.cells.from_patches("u"), u)
    # the cells under a patch hold the mean of its cells
    assert mesh.integral("u") == pytest.approx(float(u @ operator.cells.volumes))


def test_the_legacy_cell_solve_and_the_conditions_are_checked() -> None:
    mesh = line("uniform")
    mesh.allocate_field("rho")
    with pytest.raises(NotImplementedError, match="CellOperator"):
        solve_poisson(mesh, "rho")
    with pytest.raises(ValueError, match="unknown faces"):
        CellOperator(mesh, faces={"y_upper": Dirichlet()})
    with pytest.raises(ValueError, match="singular"):
        CellOperator(line("uniform", periodic=(True,)))


def test_a_line_is_plotted() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from patchwork import plot_patches

    mesh = line("level jump")
    figure, axes = plot_patches(mesh)
    assert len(axes.lines) == len(mesh.patches)
    matplotlib.pyplot.close(figure)
    mesh.allocate_field("phi", centering="node")
    figure, axes = plot_patches(mesh, field="phi")
    assert len(axes.lines) == len(mesh.patches)
    matplotlib.pyplot.close(figure)
