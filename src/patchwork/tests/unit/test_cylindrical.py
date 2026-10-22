"""Tests for the axisymmetric (r–z) metric: cell volumes, integrals, restriction,
conservative regridding and the curl-curl of an azimuthal component."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import special
from scipy.constants import mu_0

from patchwork import AMRHierarchy, CellOperator, CompositeCells, Neumann


def _add_patch(mesh: AMRHierarchy, level: int, r: tuple, z: tuple) -> None:
    """A patch of ``level`` over at least the physical box ``r x z``."""
    dr, dz = mesh.spacing(level)
    ratio = mesh.refinement_ratio
    (r_lower, z_lower) = mesh.lower_bound

    def index(value: float, lower: float, step: float, up: bool) -> int:
        cells = (value - lower) / step / ratio
        return int(np.ceil(cells) if up else np.floor(cells)) * ratio

    mesh.add_patch(
        level,
        (
            (index(r[0], r_lower, dr, False), index(r[1], r_lower, dr, True)),
            (index(z[0], z_lower, dz, False), index(z[1], z_lower, dz, True)),
        ),
    )


def test_a_cylindrical_hierarchy_is_checked() -> None:
    with pytest.raises(ValueError, match="r >= 0"):
        AMRHierarchy((4, 4), ((-1.0, 1.0), (0.0, 1.0)), geometry="cylindrical")
    with pytest.raises(ValueError, match="periodic"):
        AMRHierarchy((4, 4), periodic=(True, False), geometry="cylindrical")
    with pytest.raises(ValueError, match="geometry"):
        AMRHierarchy((4, 4), geometry="spherical")
    mesh = AMRHierarchy((4, 4), ((0.0, 2.0), (0.0, 1.0)), geometry="cylindrical")
    assert mesh.on_axis and mesh.cylindrical
    np.testing.assert_allclose(mesh.cell_volumes(0, np.arange(4)).sum() * 4, 4 * np.pi)


def test_leaf_cells_are_rings() -> None:
    mesh = AMRHierarchy((8, 8), ((0.0, 1.0), (0.0, 2.0)), geometry="cylindrical")
    _add_patch(mesh, 1, (0.0, 0.5), (0.5, 1.5))
    cells = CompositeCells(mesh)
    assert cells.volumes.sum() == pytest.approx(2.0 * np.pi)  # pi r^2 h


def test_the_integral_weights_cells_by_their_rings() -> None:
    mesh = AMRHierarchy((8, 8), ((0.0, 1.0), (-1.0, 1.0)), geometry="cylindrical")
    _add_patch(mesh, 1, (0.25, 0.75), (-0.5, 0.5))
    _add_patch(mesh, 2, (0.375, 0.625), (-0.25, 0.25))
    errors = {}
    for name, function, exact in (
        ("one", lambda r, z: 1.0 + 0.0 * r, 2.0 * np.pi),  # pi R^2 H
        ("z", lambda r, z: z + 1.0, 2.0 * np.pi),
        ("r", lambda r, z: r, 2.0 * 2.0 * np.pi / 3.0),  # 2 pi H R^3 / 3
    ):
        mesh.allocate_field(name)
        for patch in mesh.local_patches:
            r, z = patch.local_cell_centers(mesh, name)
            patch.field(name).local[...] = function(r, z)
        errors[name] = abs(mesh.integral(name) - exact) / exact
    assert errors["one"] < 1e-14 and errors["z"] < 1e-14
    # the midpoint rule in the radius: dr^2 / (4 R^2) on the base grid, 1 / 256
    assert errors["r"] < 1.0 / 256.0


def test_restriction_and_regridding_conserve_in_the_metric() -> None:
    mesh = AMRHierarchy((16, 16), ((0.0, 1.0), (0.0, 1.0)), geometry="cylindrical")
    _add_patch(mesh, 1, (0.0, 0.5), (0.25, 0.75))
    mesh.allocate_field("density", conservative=True)
    for patch in mesh.local_patches:
        r, z = patch.local_cell_centers(mesh, "density")
        patch.field("density").local[...] = 1.0 + 3.0 * r + np.sin(4 * z)
    before = mesh.integral("density")
    # the restricted coarse cells carry the same amount as their fine cells
    mesh.restrict("density")
    coarse_only = AMRHierarchy(
        (16, 16), ((0.0, 1.0), (0.0, 1.0)), geometry="cylindrical"
    )
    coarse_only.allocate_field("density")
    coarse_only.patches[0].field("density").local[...] = (
        mesh.patches[0].field("density").local
    )
    assert coarse_only.integral("density") == pytest.approx(before, rel=1e-13)
    # a regrid that refines elsewhere keeps the integral to round-off, with slopes in r
    tags = {0: np.zeros((16, 16), dtype=bool)}
    tags[0][2:12, 1:6] = True
    new = mesh.regrid(tags)
    assert new.integral("density") == pytest.approx(before, rel=1e-13)


def test_a_current_loop_has_its_closed_form_field() -> None:
    # A_phi of a loop of radius a: mu0 I / (pi k) sqrt(a / r) ((1 - k^2/2) K - E)
    a, tube, current = 0.1, 0.004, 1.0
    mesh = AMRHierarchy((32, 64), ((0.0, 1.6), (-1.6, 1.6)), geometry="cylindrical")
    for level, (r, z) in enumerate(
        [
            ((0.0, 0.6), (-0.4, 0.4)),
            ((0.0, 0.32), (-0.2, 0.2)),
            ((0.0, 0.2), (-0.12, 0.12)),
            ((0.04, 0.16), (-0.06, 0.06)),
            ((0.07, 0.13), (-0.03, 0.03)),
        ],
        start=1,
    ):
        _add_patch(mesh, level, r, z)
    solver = CellOperator(mesh, component="azimuthal")
    cells = solver.cells
    r, z = cells.centers.T
    inside = (r - a) ** 2 + z**2 < tube**2
    in_plane = np.prod(cells.sizes, axis=1)
    density = np.where(inside, current / in_plane[inside].sum(), 0.0)
    potential = solver.solve(integrated=mu_0 * density * cells.volumes)
    m = 4 * a * r / ((a + r) ** 2 + z**2)
    exact = (
        mu_0
        * current
        / (np.pi * np.sqrt(m))
        * np.sqrt(a / r)
        * ((1 - m / 2) * special.ellipk(m) - special.ellipe(m))
    )
    distance = np.hypot(r - a, z)
    near = (distance > 3 * tube) & (distance < 0.05)
    error = np.abs(potential[near] - exact[near]).max() / np.abs(exact[near]).max()
    assert error < 3e-3


@pytest.mark.parametrize("coarse_fine", ["symmetric", "conservative"])
def test_a_conducting_column_in_a_solenoid_has_its_bessel_solution(
    coarse_fine: str,
) -> None:
    # B_z = mu0 K I0(k r) / I0(k R) in the column, mu0 K between it and the coil
    omega, sigma = 2 * np.pi * 1.0e6, 1.0e4
    radius, coil, thickness, sheet = 0.1, 0.15, 0.01, 1000.0
    k = np.sqrt(1j * omega * mu_0 * sigma)
    field = mu_0 * sheet
    errors = []
    for n in (20, 40, 80):
        mesh = AMRHierarchy(
            (n, 2),
            ((0.0, 0.2), (0.0, 0.4 / n)),
            periodic=(False, True),
            geometry="cylindrical",
        )
        _add_patch(mesh, 1, (0.06, 0.12), (0.0, 0.4 / n))
        _add_patch(mesh, 2, (0.08, 0.11), (0.0, 0.4 / n))
        solver = CellOperator(
            mesh,
            component="azimuthal",
            reaction=lambda r, z: np.where(r < radius, 1j * omega * mu_0 * sigma, 0.0),
            faces={"x_upper": Neumann(0.0)},  # an infinite solenoid: no field outside
            coarse_fine=coarse_fine,
        )
        cells = solver.cells
        r, _ = cells.centers.T
        density = np.where((r > coil) & (r < coil + thickness), sheet / thickness, 0.0)
        potential = solver.solve(integrated=mu_0 * density * cells.volumes)

        def bessel(x):  # I1(k x) / I0(k R), scaled to stay finite
            return (
                special.ive(1, k * x)
                / special.ive(0, k * radius)
                * np.exp((k * x).real - (k * radius).real)
            )

        at_edge = field * bessel(radius) / k
        exact = np.where(
            r < radius,
            field * bessel(r) / k,
            (radius * at_edge + field * (r**2 - radius**2) / 2) / r,
        )
        inside = r < coil
        errors.append(
            np.abs(potential[inside] - exact[inside]).max()
            / np.abs(exact[inside]).max()
        )
    errors = np.array(errors)
    assert errors[-1] < 5e-4
    assert np.log2(errors[1:-1] / errors[2:]).min() > 1.9


def test_a_uniform_axial_field_is_a_null_field_next_to_the_axis() -> None:
    # A_phi = B r / 2 has no curl-curl: with that value on the outer face and
    # the axis condition, the solution is exact to round-off on any mesh
    mesh = AMRHierarchy(
        (8, 4), ((0.0, 1.0), (0.0, 0.5)), periodic=(False, True), geometry="cylindrical"
    )
    _add_patch(mesh, 1, (0.0, 0.5), (0.0, 0.5))
    from patchwork import Dirichlet

    solver = CellOperator(
        mesh, component="azimuthal", faces={"x_upper": Dirichlet(lambda r, z: 0.5 * r)}
    )
    potential = solver.solve(0.0)
    r, _ = solver.cells.centers.T
    assert np.abs(potential - 0.5 * r).max() < 1e-12


def test_a_scalar_in_r_z_has_no_flux_through_the_axis() -> None:
    # -laplacian u = 4 (u = 1 - r^2) with u = 0 at r = 1: second order in r
    errors = []
    for n in (16, 32, 64):
        mesh = AMRHierarchy(
            (n, 2),
            ((0.0, 1.0), (0.0, 2.0 / n)),
            periodic=(False, True),
            geometry="cylindrical",
        )
        _add_patch(mesh, 1, (0.0, 0.5), (0.0, 2.0 / n))
        solver = CellOperator(mesh, coarse_fine="conservative")
        u = solver.solve(4.0)
        r, _ = solver.cells.centers.T
        errors.append(np.abs(u - (1.0 - r**2)).max())
    errors = np.array(errors)
    assert np.log2(errors[:-1] / errors[1:]).min() > 1.8
    with pytest.raises(ValueError, match="axis"):
        CellOperator(mesh, faces={"x_lower": Neumann(0.0)})
