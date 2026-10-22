"""Floating bodies, bodies of cells, and the gradient and curl of CellOperator.

Closed forms: between two walls a floating slab takes the value that makes the
fluxes on its two sides balance (a piecewise linear solution, which the scheme
reproduces exactly; with a finer patch the conservative coarse–fine variant does, the
symmetric one only to first order next to the patch, by design); a linear field has an exact gradient; the uniform axial field
``A_phi = B r / 2`` has the exact curl ``(0, B)``, also next to the axis and across
coarse–fine faces.
"""

from __future__ import annotations

import numpy as np
import pytest

from patchwork import (
    AMRHierarchy,
    CellOperator,
    Dirichlet,
    EmbeddedDirichlet,
    EmbeddedFloating,
    Neumann,
    cell_body,
)

WALLS = {
    "x_lower": Dirichlet(0.0),
    "x_upper": Dirichlet(1.0),
    "y_lower": Neumann(0.0),
    "y_upper": Neumann(0.0),
}


def slab_mesh(refined: bool) -> AMRHierarchy:
    mesh = AMRHierarchy(base_shape=(40, 4), bounds=((0.0, 1.0), (0.0, 0.1)))
    if refined:
        # a finer patch across the slab's left surface
        mesh.add_patch(1, ((24, 40), (0, 8)))
    return mesh


def in_slab(points):
    return (points[0] > 0.4) & (points[0] < 0.6)


@pytest.mark.parametrize("refined", [False, True])
@pytest.mark.parametrize("source", [0.0, 0.1])
def test_a_floating_slab_between_two_walls(refined, source):
    """``s1 - s2 = Q / width``, ``U = 0.4 s1``, ``1 = U + 0.4 s2``."""
    operator = CellOperator(
        slab_mesh(refined),
        faces=WALLS,
        embedded=[EmbeddedFloating(in_slab, source)],
        coarse_fine="conservative",
    )
    u = operator.solve(0.0)
    s1 = (2.5 + source / 0.1 * 1.0) / 2.0
    value = 0.4 * s1
    s2 = (1.0 - value) / 0.4
    x = operator.cells.centers[:, 0]
    expected = np.where(
        x < 0.4, s1 * x, np.where(x > 0.6, value + s2 * (x - 0.6), value)
    )
    np.testing.assert_allclose(u, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(operator.floating_values(u), [value], atol=1e-12)
    # the system's residual, the floating body's row (its net flux) included
    unknowns = np.concatenate([u[operator.free], operator.floating_values(u)])
    load = np.concatenate([operator.constant, [source]])
    np.testing.assert_allclose(operator.matrix @ unknowns, load, atol=1e-12)


def test_the_operator_with_floating_bodies_is_symmetric():
    mesh = AMRHierarchy(base_shape=(24, 24))
    mesh.add_patch(1, ((12, 36), (12, 36)))
    ring = EmbeddedFloating(
        lambda p: np.abs(np.hypot(p[0] - 0.5, p[1] - 0.5) - 0.3) < 0.03
    )
    operator = CellOperator(mesh, reaction=lambda x, y: 2.0 + 1j * x, embedded=[ring])
    matrix = operator.matrix
    assert abs(matrix - matrix.T).max() < 1e-12
    # an isolated floating ring takes no net flux: the solution's flux into it is 0
    u = operator.solve(lambda x, y: np.exp(-((x - 0.2) ** 2) / 0.01))
    coupling = operator.floating_coupling
    own = operator.matrix[-1, -1]
    flux = coupling[:, 0].T @ u[operator.free] + own * operator.floating_values(u)[0]
    assert abs(flux) < 1e-12 * np.abs(u).max()


def test_a_floating_shield_against_the_uniform_fine_grid():
    """A patch over the whole domain is the uniform grid at the fine spacing."""

    def build(refined):
        mesh = AMRHierarchy(base_shape=(16, 16) if refined else (32, 32))
        if refined:
            mesh.add_patch(1, ((0, 32), (0, 32)))
        bodies = [
            EmbeddedDirichlet(lambda p: np.hypot(p[0] - 0.2, p[1] - 0.5) < 0.08, 1.0),
            EmbeddedDirichlet(lambda p: np.hypot(p[0] - 0.8, p[1] - 0.5) < 0.08, -1.0),
            EmbeddedFloating(lambda p: np.abs(p[0] - 0.5) < 0.04),
        ]
        return CellOperator(mesh, embedded=bodies).solve(0.0)

    np.testing.assert_allclose(build(True), build(False), rtol=0, atol=1e-12)


def test_a_floating_body_needs_a_free_neighbour():
    with pytest.raises(ValueError, match="touch no free cell"):
        CellOperator(
            AMRHierarchy(base_shape=(4, 4)),
            embedded=[EmbeddedFloating(lambda p: np.ones(p.shape[1], dtype=bool))],
        )


@pytest.mark.parametrize("refined", [False, True])
@pytest.mark.parametrize("floating", [False, True])
def test_a_body_of_cells_has_its_surface_on_their_faces(refined, floating):
    mesh = slab_mesh(refined)
    operator = CellOperator(mesh, faces=WALLS, factorize=False)
    cells = operator.cells
    mask = (cells.centers[:, 0] > 0.6) & (cells.centers[:, 0] < 0.7)
    body = cell_body(cells, mask, 0.0 if floating else 0.75, floating=floating)
    solution = CellOperator(
        mesh, faces=WALLS, embedded=[body], coarse_fine="conservative"
    ).solve(0.0)
    x = cells.centers[:, 0]
    # surfaces at x = 0.6 and 0.7, the cells' faces; floating: equal fluxes on both
    # sides, U = 0.6 s and 1 = U + 0.3 s
    value = 0.6 / 0.9 if floating else 0.75
    left, right = value / 0.6, (1.0 - value) / 0.3
    expected = np.where(
        x < 0.6, left * x, np.where(x > 0.7, value + right * (x - 0.7), value)
    )
    np.testing.assert_allclose(solution, expected, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="leaf cells"):
        cell_body(cells, mask[:-1])


def test_locating_points_in_the_leaf_cells():
    mesh = slab_mesh(True)
    cells = CellOperator(mesh, faces=WALLS, factorize=False).cells
    found = cells.locate(cells.centers.T)
    np.testing.assert_array_equal(found, np.arange(cells.num_cells))
    assert cells.locate(np.array([[-0.1, 0.5], [0.05, 0.05]]))[0] == -1


# Gradient and curl ---------------------------------------------------------------- #


def refined_mesh(n=16, **settings):
    mesh = AMRHierarchy(base_shape=(n, n), **settings)
    mesh.add_patch(1, ((n // 2, 3 * n // 2), (n // 2, 3 * n // 2)))
    return mesh


@pytest.mark.parametrize("variant", ["symmetric", "conservative"])
def test_the_gradient_of_a_linear_field_is_exact(variant):
    def linear(x, y):
        return 2.0 * x - 3.0 * y + 1.0

    faces = {face: Dirichlet(linear) for face in WALLS}
    operator = CellOperator(
        refined_mesh(), faces=faces, coarse_fine=variant, factorize=False
    )
    centres = operator.cells.centers
    u = linear(centres[:, 0], centres[:, 1])
    shape = (u.size, 2)
    np.testing.assert_allclose(
        operator.gradient(u), np.broadcast_to([2.0, -3.0], shape), atol=1e-10
    )
    np.testing.assert_allclose(
        operator.curl(u), np.broadcast_to([-3.0, -2.0], shape), atol=1e-10
    )


@pytest.mark.parametrize("variant", ["symmetric", "conservative"])
def test_the_gradient_is_second_order_across_coarse_fine_edges(variant):
    def exact(x, y):
        return np.sin(np.pi * x) * np.cos(np.pi * y)

    errors = []
    for n in (16, 32, 64):
        faces = {face: Dirichlet(exact) for face in WALLS}
        operator = CellOperator(
            refined_mesh(n), faces=faces, coarse_fine=variant, factorize=False
        )
        x, y = operator.cells.centers.T
        error = operator.gradient(exact(x, y)) - np.stack(
            [
                np.pi * np.cos(np.pi * x) * np.cos(np.pi * y),
                -np.pi * np.sin(np.pi * x) * np.sin(np.pi * y),
            ],
            axis=1,
        )
        errors.append(np.abs(error).max())
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(orders > 1.8), orders


def test_the_uniform_axial_field_is_exact_in_r_z():
    mesh = AMRHierarchy(base_shape=(16, 16), geometry="cylindrical")
    mesh.add_patch(1, ((0, 16), (8, 24)))
    mesh.add_patch(2, ((8, 24), (24, 40)))
    b = 3.0
    operator = CellOperator(
        mesh,
        component="azimuthal",
        faces={
            "x_upper": Dirichlet(lambda r, z: b * r / 2),
            "y_lower": Neumann(0.0),
            "y_upper": Neumann(0.0),
        },
        factorize=False,
    )
    r = operator.cells.centers[:, 0]
    np.testing.assert_allclose(
        operator.curl(b * r / 2), np.broadcast_to([0.0, b], (r.size, 2)), atol=1e-12
    )
    with pytest.raises(ValueError, match="curl, not a gradient"):
        operator.gradient(r)


def test_the_curl_of_a_solved_loop_field_is_the_operator_s():
    """The curl of an azimuthal solution: ``B_z`` is the same from the faces of a cell."""
    mesh = AMRHierarchy(
        base_shape=(32, 32), bounds=((0.0, 1.0), (-0.5, 0.5)), geometry="cylindrical"
    )
    mesh.add_patch(1, ((8, 32), (24, 40)))
    operator = CellOperator(
        mesh,
        component="azimuthal",
        faces={
            "y_lower": Neumann(0.0),
            "y_upper": Neumann(0.0),
            "x_upper": Neumann(0.0),
        },
        reaction=0.0,
        embedded=[EmbeddedDirichlet(lambda p: p[0] > 0.9, 0.0)],
    )
    r, z = operator.cells.centers.T
    current = np.where((np.abs(r - 0.3) < 0.03) & (np.abs(z) < 0.03), 1.0, 0.0)
    a = operator.solve(current)
    b = operator.curl(a)
    assert np.all(np.isfinite(b))
    # B_z of a current loop is largest inside the loop near its plane
    centre = np.argmin(np.hypot(r - 0.05, z))
    assert b[centre, 1] > 0.0 and abs(b[centre, 0]) < 0.1 * b[centre, 1]


def test_face_gradients_are_located_half_way():
    mesh = AMRHierarchy(base_shape=(8, 8))
    faces = {face: Dirichlet(lambda x, y: x) for face in WALLS}
    operator = CellOperator(mesh, faces=faces, factorize=False)
    x = operator.cells.centers[:, 0]
    estimates = operator.face_gradients(x, 0)
    np.testing.assert_allclose(estimates["value"], 1.0, atol=1e-12)
    # interior faces at half a cell, walls at a quarter
    assert set(np.round(np.abs(estimates["offset"]) * 8, 6)) == {0.25, 0.5}


@pytest.mark.parametrize("variant", ["symmetric", "conservative"])
def test_a_field_uniform_along_no_flux_walls_stays_one_dimensional(variant):
    """Across a patch, coarse values interpolate along the face to the mirror image
    beyond a no-flux wall, so a field that does not vary along it is exact."""
    operator = CellOperator(slab_mesh(True), faces=WALLS, coarse_fine=variant)
    u = operator.solve(0.0)
    np.testing.assert_allclose(u, operator.cells.centers[:, 0], rtol=0, atol=1e-12)


def test_an_azimuthal_component_mirrors_oddly_across_the_axis():
    """A coarse–fine face along the axis: ``A_phi = B r / 2`` stays a null field."""
    mesh = AMRHierarchy(base_shape=(16, 16), geometry="cylindrical")
    mesh.add_patch(1, ((0, 32), (0, 16)))  # the lower half, from the axis out
    b = 2.0
    operator = CellOperator(
        mesh,
        component="azimuthal",
        faces={
            "x_upper": Dirichlet(lambda r, z: b * r / 2),
            "y_lower": Neumann(0.0),
            "y_upper": Neumann(0.0),
        },
    )
    u = operator.solve(0.0)
    np.testing.assert_allclose(u, b * operator.cells.centers[:, 0] / 2, atol=1e-12)
