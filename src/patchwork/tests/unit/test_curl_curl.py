"""The curl-curl operator on the edges: accuracy, the null space, LU against AMS."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from patchwork import AMRHierarchy, CurlCurl
from patchwork.solvers.curl_curl import has_ams

needs_ams = pytest.mark.skipif(not has_ams(), reason="needs petsc4py with hypre")
JOB = Path(__file__).with_name("_mpi_curl_curl_job.py")


def cube(n: int, periodic=None) -> AMRHierarchy:
    mesh = AMRHierarchy(base_shape=(n, n, n), periodic=periodic)
    mesh.add_patch(level=1, box=((n // 2, 3 * n // 2),) * 3)
    return mesh


def manufactured(solver: CurlCurl) -> tuple[np.ndarray, np.ndarray]:
    """``A = (sin(pi y) sin(pi z), 0, 0)``: tangential zero on the box; its load."""
    edges = solver.edges
    coordinates = edges.nodes.coordinates
    middle = 0.5 * (coordinates[edges.ends[:, 0]] + coordinates[edges.ends[:, 1]])
    shape = np.sin(np.pi * middle[:, 1])
    if edges.nodes.ndim == 3:
        shape = shape * np.sin(np.pi * middle[:, 2])
    exact = np.where(edges.direction == 0, shape * edges.length, 0.0)
    curl_curl = (edges.nodes.ndim - 1) * np.pi**2
    return exact, solver.load((curl_curl + 1.0) * exact)


def error(solver: CurlCurl, potential: np.ndarray, exact: np.ndarray) -> float:
    difference = solver.edges.reduction @ (potential - exact)
    return float(np.sqrt(abs(difference @ solver.edges.mass() @ difference)))


@pytest.mark.parametrize("mass", ["lumped", "consistent"])
def test_curl_curl_converges_on_refined_patches(mass: str) -> None:
    errors = []
    for n in (8, 16):
        solver = CurlCurl(cube(n), beta=1.0, mass=mass)
        exact, load = manufactured(solver)
        errors.append(error(solver, solver.solve(load), exact))
    assert np.log2(errors[0] / errors[1]) > 1.6, errors


def test_gradients_are_the_null_space_of_the_curl_term() -> None:
    solver = CurlCurl(cube(8), beta=1.0)
    edges, nodes = solver.edges, solver.nodes
    gradient = edges.gradient @ nodes.reduction
    curl_term = solver.curl.T @ solver.faces.mass() @ solver.curl
    assert abs(curl_term @ gradient).max() < 1e-12


def test_electric_faces_fix_the_tangential_edges_and_magnetic_ones_do_not() -> None:
    mesh = cube(4)
    electric = CurlCurl(mesh, beta=1.0)
    magnetic = CurlCurl(
        mesh, beta=1.0, faces={"x_lower": "magnetic"}, edges=electric.edges
    )
    edges = electric.edges
    on_x_lower = (edges.lattice[:, 0] == 0) & (edges.direction != 0)
    assert electric.fixed[on_x_lower].all()
    # inside the face: not also on one of the other (electric) faces
    top = edges.nodes._cells
    end = edges.lattice + edges.span[:, None] * (
        np.arange(3)[None, :] == edges.direction[:, None]
    )
    inside = on_x_lower.copy()
    for axis in (1, 2):
        inside &= (edges.lattice[:, axis] > 0) & (end[:, axis] < top[axis])
    assert inside.any()
    assert not magnetic.fixed[inside].any()
    assert magnetic.free.size > electric.free.size


def test_a_complex_beta_is_solved_by_the_lu() -> None:
    solver = CurlCurl(cube(4), beta=1.0 - 2.0j)
    exact, load = manufactured(solver)
    potential = solver.solve(load.astype(complex))
    assert potential.dtype == complex
    assert np.iscomplexobj(potential) and abs(potential.imag).max() > 0


def test_without_beta_the_lu_needs_a_gauge() -> None:
    with pytest.raises(ValueError, match="gauge"):
        CurlCurl(cube(4))
    gauged = CurlCurl(cube(4), gauge=1e-6)
    assert gauged.matrix.shape[0] == gauged.free.size


def test_the_arguments_are_checked() -> None:
    mesh = cube(4)
    with pytest.raises(ValueError, match="condition"):
        CurlCurl(mesh, beta=1.0, faces={"x_lower": "open"})
    with pytest.raises(ValueError, match="unknown faces"):
        CurlCurl(mesh, beta=1.0, faces={"w_lower": "electric"})
    with pytest.raises(ValueError, match="periodic"):
        CurlCurl(
            cube(4, periodic=(True, False, False)),
            beta=1.0,
            faces={"x_lower": "magnetic"},
        )
    with pytest.raises(ValueError, match="mass"):
        CurlCurl(mesh, beta=1.0, mass="exact")
    with pytest.raises(ValueError, match="solver"):
        CurlCurl(mesh, beta=1.0, solver="cg")
    with pytest.raises(ValueError, match="reluctivity"):
        CurlCurl(mesh, beta=1.0, reluctivity=-1.0)
    with pytest.raises(ValueError, match="leaf cells"):
        CurlCurl(mesh, beta=np.ones(3))
    with pytest.raises(ValueError, match="2D or 3D"):
        CurlCurl(AMRHierarchy(base_shape=(4,)), beta=1.0)


def test_coefficients_can_vary_over_the_cells() -> None:
    def reluctivity(x, y, z):
        return np.where(x < 0.5, 1.0, 4.0)

    solver = CurlCurl(cube(8), reluctivity=reluctivity, beta=lambda x, y, z: 1.0 + y)
    exact, load = manufactured(solver)
    assert np.isfinite(solver.solve(load)).all()


def test_2d_curl_curl_converges() -> None:
    errors = []
    for n in (8, 16):
        mesh = AMRHierarchy(base_shape=(n, n))
        mesh.add_patch(level=1, box=((n // 2, 3 * n // 2),) * 2)
        solver = CurlCurl(mesh, beta=1.0)
        exact, load = manufactured(solver)
        errors.append(error(solver, solver.solve(load), exact))
    assert np.log2(errors[0] / errors[1]) > 1.6, errors


# Material hooks ----------------------------------------------------------------------#
def test_face_and_edge_masses_reproduce_the_cell_coefficients() -> None:
    reference = CurlCurl(cube(4), reluctivity=2.0, beta=3.0)
    edges, faces = reference.edges, reference.faces
    given = CurlCurl(
        cube(4),
        face_mass=faces.mass(weight=2.0).diagonal(),
        edge_mass=edges.mass(weight=3.0),
        edges=edges,
    )
    assert abs(given.matrix - reference.matrix).max() < 1e-12
    exact, load = manufactured(reference)
    assert np.abs(given.solve(load) - reference.solve(load)).max() < 1e-12


def test_fixed_edges_are_held_at_zero() -> None:
    plain = CurlCurl(cube(4), beta=1.0)
    edges = plain.edges
    centre = edges.nodes.coordinates[edges.ends].mean(axis=1)
    inside = np.all(np.abs(centre - 0.5) < 0.2, axis=1) & ~edges.hanging
    held = CurlCurl(cube(4), beta=1.0, fixed=inside, edges=edges)
    assert held.free.size == plain.free.size - np.count_nonzero(inside)
    exact, load = manufactured(held)
    assert not held.solve(load)[inside].any()


def test_a_complex_face_mass_takes_the_lu() -> None:
    plain = CurlCurl(cube(4), beta=1.0)
    lossy = plain.faces.mass().diagonal() * (1.0 + 0.5j)
    solver = CurlCurl(cube(4), beta=1.0, face_mass=lossy, edges=plain.edges)
    exact, load = manufactured(solver)
    assert abs(solver.solve(load.astype(complex)).imag).max() > 0
    with pytest.raises(ValueError, match="real"):
        CurlCurl(cube(4), beta=1.0, face_mass=lossy, solver="ams", edges=plain.edges)


def test_an_edge_mass_alone_makes_the_operator_regular() -> None:
    plain = CurlCurl(cube(4), beta=1.0)
    solver = CurlCurl(cube(4), edge_mass=plain.edges.mass(), edges=plain.edges)
    assert abs(solver.matrix - plain.matrix).max() < 1e-12


def test_the_material_hooks_are_checked() -> None:
    with pytest.raises(ValueError, match="face_mass must be"):
        CurlCurl(cube(4), beta=1.0, face_mass=np.ones(3))
    with pytest.raises(ValueError, match="diagonal"):
        CurlCurl(cube(4), beta=1.0, edge_mass=np.ones((3, 3)))
    with pytest.raises(ValueError, match="fixed"):
        CurlCurl(cube(4), beta=1.0, fixed=np.ones(3, dtype=bool))


# AMS -------------------------------------------------------------------------------#
@needs_ams
@pytest.mark.parametrize("mass", ["lumped", "consistent"])
def test_ams_gives_what_the_lu_gives(mass: str) -> None:
    lu = CurlCurl(cube(8), beta=1.0, mass=mass, reluctivity=lambda x, y, z: 1.0 + x)
    ams = CurlCurl(
        cube(8),
        beta=1.0,
        mass=mass,
        reluctivity=lambda x, y, z: 1.0 + x,
        solver="ams",
        edges=lu.edges,
    )
    exact, load = manufactured(lu)
    assert np.abs(ams.solve(load) - lu.solve(load)).max() < 1e-8
    assert 0 < ams.iterations < 40


@needs_ams
def test_ams_solves_the_singular_magnetostatic_problem() -> None:
    """Without beta, for a divergence-free current: the field of the LU with a tiny gauge."""
    ams = CurlCurl(cube(8), solver="ams")
    lu = CurlCurl(cube(8), gauge=1e-9, edges=ams.edges)
    faces = ams.faces
    middle = (
        faces.lattice
        + 0.5 * faces.span[:, None] * (np.arange(3)[None, :] != faces.normal[:, None])
    ) * ams.nodes._spacing
    bump = np.exp(-np.sum((middle - 0.5) ** 2, axis=1) / 0.02)
    t = faces.reduction @ np.where(faces.normal == 2, bump * faces.area, 0.0)
    load = ams.curl.T @ faces.mass() @ t  # the curl of a field: divergence-free
    gradient = ams.edges.gradient @ ams.nodes.reduction
    assert abs(gradient.T @ load).max() < 1e-14
    field = ams.flux(ams.solve(load))
    reference = lu.flux(lu.solve(load))
    assert np.abs(field - reference).max() < 1e-6 * np.abs(reference).max()


@needs_ams
def test_auto_takes_ams_for_real_and_the_lu_for_complex_coefficients() -> None:
    assert CurlCurl(cube(4), beta=1.0, solver="auto").solver == "ams"
    assert CurlCurl(cube(4), beta=1.0j, solver="auto").solver == "lu"
    with pytest.raises(ValueError, match="real"):
        CurlCurl(cube(4), beta=1.0j, solver="ams")


@needs_ams
@pytest.mark.skipif(shutil.which("mpirun") is None, reason="needs mpirun")
@pytest.mark.parametrize("ranks", [2, 3])
def test_ams_under_mpi_gives_what_one_rank_gives(ranks: int) -> None:
    env = os.environ.copy()
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        env.update(OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
    result = subprocess.run(
        [
            shutil.which("mpirun"),
            "-n",
            str(ranks),
            "--oversubscribe",
            sys.executable,
            str(JOB),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-2000:]
    assert "MPI_JOB_OK" in result.stdout
