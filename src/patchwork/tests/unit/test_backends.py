"""The solver backends: the same operator, solved with an LU, the FFT or PETSc."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse.linalg

from patchwork import (
    AMRHierarchy,
    AutoBackend,
    Backend,
    Dirichlet,
    EmbeddedDirichlet,
    FFTBackend,
    LUBackend,
    Neumann,
    NodalPoisson,
    PetscBackend,
)
from patchwork.backends import FFTSolver, TridiagonalLU, backend_of


def _mesh(shape, periodic=None) -> AMRHierarchy:
    """A uniform mesh on a box with unequal sides and spacings."""
    bounds = tuple((0.0, 1.0 + 0.3 * axis) for axis in range(len(shape)))
    return AMRHierarchy(base_shape=shape, bounds=bounds, periodic=periodic)


CASES = {
    "1d dirichlet": (
        (17,),
        (False,),
        {"x_lower": Dirichlet(1.0), "x_upper": Dirichlet(lambda x: 3.0 * x)},
    ),
    "2d dirichlet": (
        (20, 13),
        (False, False),
        {"x_lower": Dirichlet(1.0), "x_upper": Dirichlet(lambda x, y: y)},
    ),
    "2d dirichlet x, periodic y": (
        (20, 13),
        (False, True),
        {
            "x_lower": Dirichlet(-2.0),
            "x_upper": Dirichlet(lambda x, y: np.sin(2.0 * np.pi * y / 1.3)),
        },
    ),
    "2d periodic x, dirichlet y": (
        (16, 9),
        (True, False),
        {"y_lower": Dirichlet(0.2), "y_upper": Dirichlet(lambda x, y: x)},
    ),
    "2d dirichlet x, neumann y": (
        (20, 13),
        (False, False),
        {"y_lower": Neumann(1.0), "y_upper": Neumann(lambda x, y: x * x)},
    ),
    "2d neumann x, dirichlet y": (
        (11, 12),
        (False, False),
        {"x_lower": Neumann(0.5), "x_upper": Neumann(0.0)},
    ),
}


@pytest.mark.parametrize("case", CASES)
def test_the_fft_gives_the_lu_solution_to_round_off(case: str) -> None:
    shape, periodic, faces = CASES[case]
    mesh = _mesh(shape, periodic)
    direct = NodalPoisson(mesh, faces=faces)
    fast = NodalPoisson(mesh, faces=faces, nodes=direct.nodes, backend="fft")
    source = np.random.default_rng(0).standard_normal(direct.nodes.num_nodes)

    expected = direct.solve(source)
    found = fast.solve(source)

    assert isinstance(fast.solver, FFTSolver)
    assert np.abs(found - expected).max() < 1e-13 * np.abs(expected).max()
    charge = direct.nodes.volumes * source
    assert (
        np.abs(fast.solve(charge=charge) - expected).max()
        < 1e-13 * np.abs(expected).max()
    )


@pytest.mark.parametrize(
    ("shape", "periodic", "neumann"),
    [
        ((16,), (True,), False),
        ((12, 10), (True, True), False),
        ((15,), (False,), True),
        ((12, 10), (False, False), True),
        ((12, 10), (True, False), True),
    ],
)
def test_the_fft_solves_a_singular_operator_for_the_zero_mean_solution(
    shape, periodic, neumann
) -> None:
    mesh = _mesh(shape, periodic)
    faces = (
        {face: Neumann(0.0) for face in ("x_lower", "x_upper", "y_lower", "y_upper")}
        if neumann
        else {}
    )
    faces = {
        face: condition
        for face, condition in faces.items()
        if face[0] in "xy"[: len(shape)] and not periodic["xy".index(face[0])]
    }
    with pytest.raises(ValueError, match="singular"):
        NodalPoisson(mesh, faces=faces)
    solver = NodalPoisson(mesh, faces=faces, backend="fft")
    volumes = solver.nodes.volumes
    source = np.random.default_rng(1).standard_normal(volumes.size)
    source -= volumes @ source / volumes.sum()  # the solvable part

    potential = solver.solve(source)

    residual = solver.matrix @ potential - solver.load @ source
    assert np.abs(residual).max() < 1e-12 * np.abs(solver.load @ source).max()
    assert abs(volumes @ potential) < 1e-12 * np.abs(potential).max()


def test_the_spectral_fft_inverts_the_exact_laplacian_of_a_mode() -> None:
    mesh = _mesh((32, 16), (True, True))
    solver = NodalPoisson(mesh, backend=FFTBackend(spectral=True))
    x, y = solver.nodes.coordinates.T
    kx, ky = 2.0 * np.pi * 3, 2.0 * np.pi * 2 / 1.3
    mode = np.sin(kx * x) * np.cos(ky * y)

    potential = solver.solve((kx**2 + ky**2) * mode)

    assert np.abs(potential - mode).max() < 1e-13


def test_the_fft_runs_in_threads() -> None:
    shape, periodic, faces = CASES["2d dirichlet x, periodic y"]
    mesh = _mesh(shape, periodic)
    one = NodalPoisson(mesh, faces=faces, backend="fft")
    two = NodalPoisson(mesh, faces=faces, backend=FFTBackend(workers=2))
    source = np.random.default_rng(2).standard_normal(one.nodes.num_nodes)
    assert np.abs(one.solve(source) - two.solve(source)).max() < 1e-14


def _disc(points: np.ndarray) -> np.ndarray:
    return (points[0] - 0.5) ** 2 + (points[1] - 0.5) ** 2 < 0.04


def _refined() -> AMRHierarchy:
    mesh = _mesh((8, 8))
    mesh.add_patch(level=1, box=((4, 12), (4, 12)))
    return mesh


@pytest.mark.parametrize(
    ("mesh", "faces", "embedded", "reason"),
    [
        (lambda: _mesh((8, 8)), {}, [EmbeddedDirichlet(_disc, 1.0)], "bodies"),
        (
            lambda: _refined(),
            {},
            [],
            "uniform mesh",
        ),
        (
            lambda: AMRHierarchy(
                base_shape=(8, 8), base_boxes=[((0, 4), (0, 8)), ((4, 8), (0, 4))]
            ),
            {},
            [],
            "fills its box",
        ),
        (lambda: _mesh((8, 8)), {"x_lower": Neumann(0.0)}, [], "same kind"),
        (
            lambda: AMRHierarchy(
                base_shape=(8, 8),
                bounds=((0.0, 1.0), (0.0, 1.0)),
                geometry="cylindrical",
            ),
            {},
            [],
            "Cartesian",
        ),
    ],
)
def test_the_fft_refuses_what_it_cannot_solve(mesh, faces, embedded, reason) -> None:
    with pytest.raises(ValueError, match=reason):
        NodalPoisson(mesh(), faces=faces, embedded=embedded, backend="fft")


def test_spectral_needs_every_axis_periodic() -> None:
    with pytest.raises(ValueError, match="spectral"):
        NodalPoisson(_mesh((8, 8), (False, True)), backend=FFTBackend(spectral=True))


def test_auto_takes_the_fft_where_it_applies_and_the_lu_elsewhere() -> None:
    shape, periodic, faces = CASES["2d dirichlet x, periodic y"]
    uniform = NodalPoisson(_mesh(shape, periodic), faces=faces, backend="auto")
    refined = NodalPoisson(_refined(), backend="auto")
    singular = _mesh((8,), (True,))

    assert isinstance(uniform.solver, FFTSolver)
    assert not isinstance(refined.solver, FFTSolver)
    assert isinstance(NodalPoisson(singular, backend="auto").solver, FFTSolver)
    # a singular operator the FFT cannot take falls to the LU, which refuses it
    neumann = {face: Neumann(0.0) for face in ("x_lower", "x_upper")}
    refined_line = AMRHierarchy(base_shape=(8,))
    refined_line.add_patch(level=1, box=((4, 8),))
    with pytest.raises(ValueError, match="singular"):
        NodalPoisson(refined_line, faces=neumann, backend="auto")


def test_backends_by_name_and_by_instance() -> None:
    assert isinstance(backend_of("lu"), LUBackend)
    assert isinstance(backend_of("fft"), FFTBackend)
    assert isinstance(backend_of("auto"), AutoBackend)
    assert isinstance(backend_of("petsc"), PetscBackend)
    custom = FFTBackend(workers=1)
    assert backend_of(custom) is custom
    assert "workers=1" in repr(custom)
    assert "rtol=1e-08" in repr(PetscBackend(rtol=1e-8))
    assert repr(LUBackend()) == "LUBackend()"
    with pytest.raises(ValueError, match="backend"):
        backend_of("mumps")
    with pytest.raises(NotImplementedError):
        Backend().setup(NodalPoisson(_mesh((4,))).system)


def test_a_mesh_without_unknowns_needs_no_backend() -> None:
    solver = NodalPoisson(_mesh((1,)), backend="fft")
    assert solver.solver is None
    assert solver.solve(np.ones(2)) == pytest.approx([0.0, 0.0])


@pytest.mark.parametrize("body", [False, True])
def test_a_line_is_factorized_as_a_tridiagonal_matrix(body: bool) -> None:
    embedded = (
        [EmbeddedDirichlet(lambda points: np.abs(points[0] - 0.61) < 0.13, 2.0)]
        if body
        else []
    )
    faces = {"x_lower": Dirichlet(1.0), "x_upper": Neumann(0.5)}
    solver = NodalPoisson(_mesh((40,)), faces=faces, embedded=embedded)
    rhs = np.random.default_rng(3).standard_normal(solver.free.size)

    assert isinstance(solver.solver, TridiagonalLU)
    # Shortley-Weller rows next to a body are not symmetric: the pivoted LU
    assert solver.solver.symmetric is not body
    expected = scipy.sparse.linalg.spsolve(solver.matrix.tocsc(), rhs)
    assert (
        np.abs(solver.solver.solve(rhs) - expected).max()
        < 1e-12 * np.abs(expected).max()
    )
    assert isinstance(NodalPoisson(_mesh((40,)), backend="auto").solver, TridiagonalLU)


def test_a_periodic_line_with_a_body_is_not_tridiagonal() -> None:
    body = EmbeddedDirichlet(lambda points: np.abs(points[0] - 0.5) < 0.1, 1.0)
    solver = NodalPoisson(_mesh((16,), (True,)), embedded=[body], backend="auto")
    assert not isinstance(solver.solver, TridiagonalLU | FFTSolver)
