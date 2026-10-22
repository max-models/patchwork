"""Tests for the cell-centered composite operator (Cartesian)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import mu_0

from patchwork import (
    AMRHierarchy,
    CellOperator,
    CompositeCells,
    Dirichlet,
    EmbeddedDirichlet,
    Neumann,
)

FACES = ("x_lower", "x_upper", "y_lower", "y_upper")


def _refined(n: int, **kwargs) -> AMRHierarchy:
    """An n x n unit square with a level-1 patch over part of it and a level-2 one."""
    mesh = AMRHierarchy(base_shape=(n, n), **kwargs)
    mesh.add_patch(level=1, box=((n // 2, n + n // 2), (n // 2, n + n // 2)))
    mesh.add_patch(level=2, box=((3 * n // 2, 5 * n // 2), (3 * n // 2, 5 * n // 2)))
    return mesh


# a manufactured solution with variable coefficients
def _exact(x, y):
    return np.sin(np.pi * x) * np.cos(1.3 * y) + 0.2 * x * y


def _diffusion(x, y):
    return 1.0 + 0.5 * np.sin(2.0 * x + y)


def _reaction(x, y):
    return 2.0 + x


def _source(x, y):
    ux = np.pi * np.cos(np.pi * x) * np.cos(1.3 * y) + 0.2 * y
    uy = -1.3 * np.sin(np.pi * x) * np.sin(1.3 * y) + 0.2 * x
    laplacian = -(np.pi**2 + 1.69) * np.sin(np.pi * x) * np.cos(1.3 * y)
    ax, ay = np.cos(2.0 * x + y), 0.5 * np.cos(2.0 * x + y)
    divergence = ax * ux + ay * uy + _diffusion(x, y) * laplacian
    return -divergence + _reaction(x, y) * _exact(x, y)


def _manufactured_errors(n: int, coarse_fine: str) -> tuple[float, float]:
    mesh = _refined(n)
    solver = CellOperator(
        mesh,
        diffusion=_diffusion,
        reaction=_reaction,
        faces={face: Dirichlet(_exact) for face in FACES},
        coarse_fine=coarse_fine,
    )
    u = solver.solve(_source)
    x, y = solver.cells.centers.T
    error = np.abs(u - _exact(x, y))
    return float(error.max()), float(np.sqrt(np.sum(error**2 * solver.cells.volumes)))


def test_leaf_cells_tile_the_domain() -> None:
    mesh = _refined(8)
    cells = CompositeCells(mesh)
    assert cells.volumes.sum() == pytest.approx(1.0)
    assert (cells.owner >= 0).all()
    # every lattice cell's owner contains it
    lattice = np.indices(cells.owner.shape).reshape(2, -1).T
    owner = cells.owner.reshape(-1)
    assert np.all(lattice >= cells.lattice[owner])
    assert np.all(lattice < cells.lattice[owner] + cells.span[owner, None])
    # the faces of the cells and of the domain close every cell's surface
    perimeter = np.zeros(cells.num_cells)
    for axis in (0, 1):
        faces = cells.faces(axis)
        area, _, _ = cells.face_metric(
            axis, faces["plane"], faces["start"], faces["count"]
        )
        np.add.at(perimeter, faces["low"], area)
        np.add.at(perimeter, faces["high"], area)
        boundary = cells.boundary(axis)
        area, _, _ = cells.face_metric(
            axis, boundary["plane"], boundary["start"], boundary["count"]
        )
        np.add.at(perimeter, boundary["cell"], area)
    np.testing.assert_allclose(perimeter, 2.0 * cells.sizes.sum(axis=1))


@pytest.mark.parametrize("coarse_fine", ["symmetric", "conservative"])
def test_a_manufactured_solution_converges_at_second_order(coarse_fine: str) -> None:
    errors = np.array([_manufactured_errors(n, coarse_fine) for n in (16, 32, 64)])
    max_order = np.log2(errors[:-1, 0] / errors[1:, 0])
    mean_order = np.log2(errors[:-1, 1] / errors[1:, 1])
    assert mean_order.min() > 1.9  # L2, both treatments
    if coarse_fine == "conservative":
        assert max_order.min() > 1.75  # second order next to the coarse-fine edges too
    else:
        # symmetric: first order next to a coarse-fine edge (see the module docstring)
        assert max_order.min() > 0.9


def test_a_patch_over_the_whole_domain_is_the_uniform_fine_grid() -> None:
    covered = AMRHierarchy(base_shape=(8, 8))
    covered.add_patch(level=1, box=((0, 16), (0, 16)))
    uniform = AMRHierarchy(base_shape=(16, 16))
    results = []
    for mesh in (covered, uniform):
        solver = CellOperator(
            mesh,
            diffusion=_diffusion,
            reaction=_reaction,
            faces={face: Dirichlet(_exact) for face in FACES},
        )
        x, y = solver.cells.centers.T
        order = np.lexsort((y, x))
        results.append(solver.solve(_source)[order])
    np.testing.assert_allclose(results[0], results[1], rtol=0.0, atol=1e-13)


def test_refinement_over_a_sharp_feature_is_as_good_as_the_fine_grid() -> None:
    # a narrow bump at the centre, resolved by the patches: the refined mesh is
    # about as accurate as one uniform grid at its finest spacing
    width = 0.05

    def bump(x, y):
        return np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / width**2)

    def source(x, y):  # -laplacian u + u
        r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2
        return -bump(x, y) * (4.0 * r2 / width**4 - 4.0 / width**2) + bump(x, y)

    def error(mesh: AMRHierarchy) -> float:
        solver = CellOperator(mesh, reaction=1.0, coarse_fine="conservative")
        x, y = solver.cells.centers.T
        return float(np.abs(solver.solve(source) - bump(x, y)).max())

    errors = {
        "base": error(AMRHierarchy(base_shape=(16, 16))),
        "refined": error(_refined(16)),
        "fine": error(AMRHierarchy(base_shape=(64, 64))),
    }
    assert errors["refined"] < 0.1 * errors["base"]
    assert errors["refined"] < 2.0 * errors["fine"]


@pytest.mark.parametrize("coarse_fine", ["symmetric", "conservative"])
def test_a_complex_skin_layer_matches_its_closed_form(coarse_fine: str) -> None:
    # -u'' + i omega mu0 sigma u = 0 in a conductor driven at x = 0: exp(-k x)
    omega, sigma, length = 2.0 * np.pi * 1.0e6, 1.0e4, 0.05
    k = np.sqrt(1j * omega * mu_0 * sigma)  # skin depth 5 mm
    errors = []
    for n in (16, 32, 64):
        mesh = AMRHierarchy(
            base_shape=(n, 2),
            bounds=((0.0, length), (0.0, 2.0 * length / n)),
            periodic=(False, True),
        )
        mesh.add_patch(level=1, box=((0, n), (0, 4)))  # the first half, 2x finer
        mesh.add_patch(level=2, box=((0, n), (0, 8)))  # the first quarter, 4x finer
        exact = lambda x, y: np.exp(-k * x) + 0.0 * y  # noqa: E731
        solver = CellOperator(
            mesh,
            reaction=1j * omega * mu_0 * sigma,
            faces={"x_lower": Dirichlet(exact), "x_upper": Dirichlet(exact)},
            coarse_fine=coarse_fine,
        )
        u = solver.solve(0.0)
        x, y = solver.cells.centers.T
        errors.append(np.abs(u - exact(x, y)).max())
    errors = np.array(errors)
    assert errors[-1] < 2e-3
    assert np.log2(errors[:-1] / errors[1:]).min() > 1.8


def test_the_operator_is_complex_symmetric() -> None:
    mesh = _refined(8)
    solver = CellOperator(
        mesh, diffusion=_diffusion, reaction=lambda x, y: 1.0 + 2j * x
    )
    matrix = solver.matrix
    assert np.iscomplexobj(matrix.data)
    assert abs(matrix - matrix.T).max() < 1e-12  # symmetric
    assert abs(matrix - matrix.conj().T).max() > 1e-3  # not Hermitian
    conservative = CellOperator(mesh, diffusion=_diffusion, coarse_fine="conservative")
    assert abs(conservative.matrix - conservative.matrix.T).max() > 1e-3


def test_a_body_holds_its_value_and_bends_the_field_around_it() -> None:
    # between a cylinder at 1 (radius 0.1) and one at 0 (radius 0.4): ln(r / R2) / ln(R1 / R2)
    r1, r2 = 0.1, 0.4
    errors = []
    for n in (32, 64):
        mesh = AMRHierarchy(base_shape=(n, n), bounds=((-0.5, 0.5), (-0.5, 0.5)))
        mesh.add_patch(level=1, box=((n // 2, 3 * n // 2), (n // 2, 3 * n // 2)))

        def radius(points):
            return np.hypot(points[0], points[1])

        inner = EmbeddedDirichlet(lambda p: radius(p) < r1, value=1.0)
        outer = EmbeddedDirichlet(lambda p: radius(p) > r2, value=0.0)
        solver = CellOperator(mesh, embedded=[outer, inner])
        u = solver.solve(0.0)
        x, y = solver.cells.centers.T
        r = np.hypot(x, y)
        assert np.all(u[r < r1] == 1.0) and np.all(u[r > r2] == 0.0)
        between = (r > r1) & (r < r2)
        exact = np.log(r / r2) / np.log(r1 / r2)
        errors.append(np.abs(u[between] - exact[between]).max())
        # a new body value without assembling again: the field is linear in it
        solver.set_conditions(body_values=[0.0, 2.0])
        np.testing.assert_allclose(solver.solve(0.0)[between], 2.0 * u[between])
    assert errors[-1] < 1e-2
    assert errors[0] / errors[1] > 1.8


def test_a_periodic_axis_and_neumann_faces() -> None:
    # u = cos(2 pi y) * (1 + x^2), periodic in y, du/dn given at x = 0 and x = 1
    def exact(x, y):
        return np.cos(2 * np.pi * y) * (1.0 + x**2)

    def source(x, y):
        return -np.cos(2 * np.pi * y) * (2.0 - 4 * np.pi**2 * (1.0 + x**2)) + exact(
            x, y
        )

    errors = []
    for n in (16, 32, 64):
        mesh = AMRHierarchy(base_shape=(n, n), periodic=(False, True))
        mesh.add_patch(level=1, box=((0, n), (0, 2 * n)))  # across the periodic seam
        solver = CellOperator(
            mesh,
            reaction=1.0,
            faces={
                "x_lower": Neumann(0.0),
                "x_upper": Neumann(lambda x, y: 2.0 * np.cos(2 * np.pi * y)),
            },
            coarse_fine="conservative",
        )
        u = solver.solve(source)
        x, y = solver.cells.centers.T
        errors.append(np.abs(u - exact(x, y)).max())
    errors = np.array(errors)
    assert np.log2(errors[:-1] / errors[1:]).min() > 1.8


def test_factorize_once_solve_many_and_change_the_reaction() -> None:
    mesh = _refined(8)
    solver = CellOperator(mesh, diffusion=_diffusion, reaction=1.0)
    first, second = solver.solve(_source), solver.solve(lambda x, y: x)
    both = solver.solve(lambda x, y: _source(x, y) + x)
    np.testing.assert_allclose(both, first + second, atol=1e-12)
    solver.set_reaction(_reaction)
    fresh = CellOperator(mesh, diffusion=_diffusion, reaction=_reaction)
    np.testing.assert_allclose(solver.solve(_source), fresh.solve(_source), atol=1e-12)
    # the residual of a solution is zero
    u = solver.solve(_source)
    residual = (
        solver.apply(u)
        - (_source(*solver.cells.centers.T) * solver.cells.volumes)[solver.free]
        - solver.constant
    )
    assert np.abs(residual).max() < 1e-10


def test_solutions_go_to_the_patches_and_back() -> None:
    mesh = _refined(8)
    solver = CellOperator(mesh, reaction=1.0 + 1.0j)
    u = solver.solve(_source, potential="u")
    assert np.iscomplexobj(u)
    np.testing.assert_allclose(solver.cells.from_patches("u"), u)
    # covered cells hold the mean of their fine cells: base cell (2, 2) is the
    # level-1 patch's first 2 x 2 cells
    coarse = mesh.gather(mesh.patches_at_level(0)[0], "u")
    fine = mesh.gather(mesh.patches_at_level(1)[0], "u")
    assert coarse[2, 2] == pytest.approx(fine[:2, :2].mean())
    again = solver.solve("u")  # a cell field as the source
    assert again.shape == u.shape


def test_inner_edges_of_a_domain_that_does_not_fill_its_rectangle() -> None:
    # an L: the upper right quarter is missing; its edges take the "inner" value
    mesh = AMRHierarchy(
        base_shape=(16, 16), base_boxes=[((0, 16), (0, 8)), ((0, 8), (8, 16))]
    )
    solver = CellOperator(
        mesh,
        faces={face: Dirichlet(1.0) for face in FACES} | {"inner": Dirichlet(1.0)},
    )
    np.testing.assert_allclose(solver.solve(0.0), 1.0)


def test_conditions_are_checked() -> None:
    with pytest.raises(ValueError, match="singular"):
        CellOperator(
            AMRHierarchy(base_shape=(4, 4)), faces={f: Neumann(0.0) for f in FACES}
        )
    with pytest.raises(ValueError, match="periodic"):
        CellOperator(
            AMRHierarchy(base_shape=(4, 4), periodic=(True, False)),
            faces={"x_lower": Dirichlet(0.0)},
        )
    with pytest.raises(ValueError, match="azimuthal"):
        CellOperator(AMRHierarchy(base_shape=(4, 4)), component="azimuthal")
    with pytest.raises(ValueError, match="negative"):
        CellOperator(AMRHierarchy(base_shape=(4, 4)), diffusion=-1.0)
    with pytest.raises(ValueError, match="singular"):
        CellOperator(AMRHierarchy(base_shape=(4, 4)), diffusion=0.0)
    with pytest.raises(ValueError, match="coarse_fine"):
        CellOperator(AMRHierarchy(base_shape=(4, 4)), coarse_fine="other")
