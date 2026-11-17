"""`NodalPoisson` with the cylindrical metric and with a reaction term."""

from __future__ import annotations

import numpy as np
import pytest

from patchwork import AMRHierarchy, Dirichlet, EmbeddedDirichlet, Neumann, NodalPoisson

GROUNDED = {
    "x_upper": Dirichlet(0.0),
    "y_lower": Dirichlet(0.0),
    "y_upper": Dirichlet(0.0),
}


def _rz(n: int, refine: bool = False, **kwargs) -> NodalPoisson:
    mesh = AMRHierarchy(
        base_shape=(n, n), bounds=((0.0, 1.0), (0.0, 1.0)), geometry="cylindrical"
    )
    if refine:
        mesh.add_patch(level=1, box=((n // 2, 3 * n // 2), (n // 2, 3 * n // 2)))
    return NodalPoisson(mesh, metric="cylindrical", faces=GROUNDED, **kwargs)


def _error(solver: NodalPoisson, c: float = 0.0) -> float:
    """phi = (1 - r^2) sin(pi z) in the volume-weighted L2 norm."""
    r, z = solver.nodes.coordinates.T
    exact = (1.0 - r**2) * np.sin(np.pi * z)
    source = (4.0 + np.pi**2 * (1.0 - r**2) + c * (1.0 - r**2)) * np.sin(np.pi * z)
    difference = solver.solve(source) - exact
    return float(np.sqrt(np.sum(difference**2 * solver.volumes)))


def _orders(errors: list[float]) -> np.ndarray:
    return np.log2(np.array(errors[:-1]) / np.array(errors[1:]))


@pytest.mark.parametrize("refine", [False, True])
def test_the_axisymmetric_laplacian_converges_at_second_order(refine) -> None:
    errors = [_error(_rz(n, refine)) for n in (16, 32, 64)]
    assert np.all(_orders(errors) > 1.9)


def test_the_control_volumes_are_the_rings() -> None:
    solver = _rz(8)
    assert solver.volumes.sum() == pytest.approx(np.pi)  # the cylinder r < 1, 0 < z < 1
    r, z = solver.nodes.coordinates.T
    inner = (r > 0) & (r < 1) & (z > 0) & (z < 1)
    np.testing.assert_allclose(solver.volumes[inner], 2 * np.pi * r[inner] / 64)
    on_axis = (r == 0) & (z > 0) & (z < 1)
    h = 1 / 8  # the ring 0 < r < h/2: pi (h/2)^2 h
    np.testing.assert_allclose(solver.volumes[on_axis], np.pi * (h / 2) ** 2 * h)


def test_the_stencil_is_the_finite_volumes_of_the_rings() -> None:
    """Interior row r_j = j h: (1 -/+ 1/2j) radially, 1 axially, divided by the volume."""
    solver = _rz(8)
    lattice = solver.nodes.lattice
    row = int(np.flatnonzero((lattice[solver.free] == (3, 4)).all(axis=1))[0])
    entries = solver.matrix[row].toarray().ravel() / solver.volumes[solver.free[row]]
    neighbours = {
        (dr, dz): entries[np.flatnonzero((lattice[solver.free] == (3 + dr, 4 + dz)).all(axis=1))[0]]
        for dr, dz in ((-1, 0), (1, 0), (0, -1), (0, 1))
    }  # fmt: skip
    h2 = (1 / 8) ** 2
    assert neighbours[(-1, 0)] * h2 == pytest.approx(-(1 - 1 / 6))
    assert neighbours[(1, 0)] * h2 == pytest.approx(-(1 + 1 / 6))
    assert neighbours[(0, -1)] * h2 == pytest.approx(-1.0)
    assert entries[row] * h2 == pytest.approx(4.0)
    axis = int(np.flatnonzero((lattice[solver.free] == (0, 4)).all(axis=1))[0])
    up = int(np.flatnonzero((lattice[solver.free] == (1, 4)).all(axis=1))[0])
    a = solver.matrix[axis].toarray().ravel() / solver.volumes[solver.free[axis]] * h2
    assert a[up] == pytest.approx(-4.0)  # 4 (phi_1 - phi_0) / h^2 on the axis
    assert a[axis] == pytest.approx(4.0 + 2.0)


def test_a_coaxial_line_with_an_embedded_conductor() -> None:
    """phi = V ln(b/r) / ln(b/a) between a body r < a and the face r = b."""
    a, b = 0.23, 1.0
    errors = []
    for n in (40, 80, 160):
        mesh = AMRHierarchy(
            base_shape=(n, n // 2),
            bounds=((0.0, b), (0.0, 0.5)),
            geometry="cylindrical",
        )
        solver = NodalPoisson(
            mesh,
            metric="cylindrical",
            embedded=(EmbeddedDirichlet(lambda p: p[0] < a, 1.0),),
            faces={
                "x_upper": Dirichlet(0.0),
                "y_lower": Neumann(),
                "y_upper": Neumann(),
            },
        )
        r = solver.nodes.coordinates[:, 0]
        exact = np.log(b / np.maximum(r, a)) / np.log(b / a)
        errors.append(np.abs(solver.solve(np.zeros(r.size)) - exact).max())
    assert errors[-1] < 5e-5
    assert errors[-1] < errors[0] / 3


def test_a_neumann_face_carries_its_ring_area() -> None:
    """A radial field E_r = -q / (2 pi r) from a line charge: the flux through r = 1 balances it."""
    mesh = AMRHierarchy(
        base_shape=(32, 4), bounds=((0.0, 1.0), (0.0, 0.25)), geometry="cylindrical"
    )
    solver = NodalPoisson(
        mesh,
        metric="cylindrical",
        faces={"x_upper": Neumann(-1.0), "y_lower": Neumann(), "y_upper": Neumann()},
        backend="lu",
        reaction=1e-12,  # fixes the constant
    )
    r = solver.nodes.coordinates[:, 0]
    source = np.where(r < 0.5, 1.0, 0.0)
    total_source = (solver.volumes * source).sum()
    flux = 2 * np.pi * 1.0 * 0.25  # -dphi/dn = 1 over the ring r = 1
    # the right-hand side sums to the source plus the boundary flux
    rhs = solver.load @ source + solver.constant
    assert rhs.sum() == pytest.approx(total_source - flux, rel=1e-12)


@pytest.mark.parametrize("cylindrical", [False, True])
def test_a_reaction_term_converges_and_can_be_removed(cylindrical) -> None:
    c = 7.0
    if cylindrical:
        errors = [_error(_rz(n, reaction=c), c) for n in (16, 32, 64)]
        solver = _rz(8, reaction=c)
    else:

        def plain(n: int, **kwargs) -> NodalPoisson:
            return NodalPoisson(AMRHierarchy(base_shape=(n, n)), **kwargs)

        def error(solver: NodalPoisson) -> float:
            x, y = solver.nodes.coordinates.T
            exact = np.sin(np.pi * x) * np.sin(np.pi * y)
            difference = solver.solve((2 * np.pi**2 + c) * exact) - exact
            return float(np.sqrt(np.sum(difference**2 * solver.volumes)))

        errors = [error(plain(n, reaction=c)) for n in (16, 32, 64)]
        solver = plain(8, reaction=c)
    assert np.all(_orders(errors) > 1.9)
    reacting = solver.matrix.copy()
    np.testing.assert_allclose(
        (reacting - solver.stiffness).diagonal(), c * solver.volumes[solver.unknowns]
    )
    solver.set_reaction(None)
    assert abs(solver.matrix - solver.stiffness).max() == 0.0
    solver.set_reaction(c)
    assert abs(solver.matrix - reacting).max() < 1e-12


def test_a_reaction_rules_out_the_fft() -> None:
    mesh = AMRHierarchy(base_shape=(8, 8))
    assert (
        NodalPoisson(mesh, backend="auto")
        .backend.choose(NodalPoisson(mesh).system)
        .name
        == "fft"
    )
    with pytest.raises(ValueError, match="reaction"):
        NodalPoisson(mesh, backend="fft", reaction=1.0)
    solver = NodalPoisson(mesh, backend="auto", reaction=1.0)
    assert solver.backend.choose(solver.system).name == "lu"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"metric": "polar"}, "metric"),
        ({"metric": "cylindrical", "faces": {"x_lower": Dirichlet(0.0)}}, "axis"),
        ({"metric": "cylindrical", "mass": "consistent"}, "lumped"),
        ({"metric": "cylindrical", "coefficient": 2.0}, "coefficient"),
        ({"reaction": np.ones(3)}, "reaction"),
    ],
)
def test_bad_settings_are_rejected(kwargs, match) -> None:
    mesh = AMRHierarchy(base_shape=(4, 4), geometry="cylindrical")
    with pytest.raises(ValueError, match=match):
        NodalPoisson(mesh, **kwargs)
    with pytest.raises(ValueError, match="r-z"):
        NodalPoisson(AMRHierarchy(base_shape=(4, 4)), metric="cylindrical")
