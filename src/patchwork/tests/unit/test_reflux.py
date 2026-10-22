"""Refluxing: conservative explicit steps of a cell field across coarse–fine faces.

The file runs serially and under ``mpirun -n N python -m pytest`` (see
``test_mpi.py``), with the patches spread over the ranks there are.
"""

from __future__ import annotations

import numpy as np
import pytest
from maybempi import MPI

from patchwork import AMRHierarchy, CompositeCells, FluxRegister

WORLD = MPI.COMM_WORLD
SIZE = WORLD.Get_size()


def _storage(patch, name):
    """The rank's storage of a patch field, its halo width and its block."""
    array = patch.field(name)
    return np.asarray(array.local_with_halos), array.layout.halo[0], array.layout


def upwind(name, velocity):
    """First-order upwind fluxes ``v u`` of a constant velocity (needs a halo of 1)."""
    vx, vy = velocity

    def fluxes(patch):
        u, h, layout = _storage(patch, name)
        nb, nc = layout.local_shape
        rows = u[h - 1 : h + nb + 1, h : h + nc]
        cols = u[h : h + nb, h - 1 : h + nc + 1]
        fx = vx * (rows[:-1] if vx > 0 else rows[1:])
        fy = vy * (cols[:, :-1] if vy > 0 else cols[:, 1:])
        return fx, fy

    return fluxes


def fromm(name, velocity):
    """Second-order upwind-biased fluxes (Fromm), for a positive velocity (halo 2)."""
    vx, vy = velocity

    def fluxes(patch):
        u, h, layout = _storage(patch, name)
        nb, nc = layout.local_shape
        rows = u[h - 2 : h + nb + 1, h : h + nc]
        cols = u[h : h + nb, h - 2 : h + nc + 1]
        fx = vx * (rows[1:-1] + 0.25 * (rows[2:] - rows[:-2]))
        fy = vy * (cols[:, 1:-1] + 0.25 * (cols[:, 2:] - cols[:, :-2]))
        return fx, fy

    return fluxes


def diffusive(mesh, name, a=1.0):
    """Fluxes ``-a grad u`` from the two cells of a face; none through walls."""

    def fluxes(patch):
        u, h, layout = _storage(patch, name)
        nb, nc = layout.local_shape
        (b0, b1), (c0, c1) = layout.index_bounds
        dx, dy = mesh.spacing(patch.level)
        rows = u[h - 1 : h + nb + 1, h : h + nc]
        cols = u[h : h + nb, h - 1 : h + nc + 1]
        fx = -a * (rows[1:] - rows[:-1]) / dx
        fy = -a * (cols[:, 1:] - cols[:, :-1]) / dy
        nx, ny = mesh.level_shape(patch.level)
        i = patch.box[0][0] + np.arange(b0, b1 + 1)
        j = patch.box[1][0] + np.arange(c0, c1 + 1)
        if not mesh.periodic[0]:
            fx[(i == 0) | (i == nx)] = 0.0
        if not mesh.periodic[1]:
            fy[:, (j == 0) | (j == ny)] = 0.0
        return fx, fy

    return fluxes


def fill(mesh, name, function):
    for patch in mesh.local_patches:
        x, y = patch.local_cell_centers(mesh, name)
        patch.field(name).local[...] = function(x, y)
    mesh.restrict(name)


def periodic_mesh(n, split=0, halo=1, levels=1):
    """A periodic unit square of ``n x n`` cells, refined in the middle."""
    mesh = AMRHierarchy((n, n), periodic=(True, True), comm=WORLD)
    mesh.add_patch(1, ((n // 2, 3 * n // 2), (n // 2, 3 * n // 2)), ranks=SIZE - 1)
    if levels > 1:
        mesh.add_patch(
            2, ((3 * n // 2, 5 * n // 2), (3 * n // 2, 5 * n // 2)), ranks=None
        )
    mesh.allocate_field("u", halo=halo, split=split)
    return mesh


def blob(x, y):
    return 1.0 + np.exp(-((x - 0.3) ** 2 + (y - 0.35) ** 2) / 0.01)


def run(mesh, fluxes, dt, steps, *, reflux=True):
    """Explicit Euler steps of all levels with one time step."""
    register = FluxRegister(mesh)
    for _ in range(steps):
        mesh.exchange_halos("u")
        register.update("u", fluxes, dt)
        if reflux:
            register.reflux("u")
        else:
            mesh.restrict("u")


def test_the_register_finds_the_coarse_faces_on_the_edge_of_finer_levels() -> None:
    mesh = periodic_mesh(8, levels=2)
    register = FluxRegister(mesh)
    # level 1 over 4 x 4 base cells, level 2 over 4 x 4 level-1 cells: 16 faces each
    assert register.num_faces == 32
    one_level = AMRHierarchy((8, 8), comm=WORLD)
    assert FluxRegister(one_level).num_faces == 0


@pytest.mark.parametrize("split", [0, None])
def test_advection_across_a_patch_is_conservative_with_refluxing(split) -> None:
    errors = {}
    for reflux in (True, False):
        mesh = periodic_mesh(16, split=split, levels=2)
        fill(mesh, "u", blob)
        before = mesh.integral("u")
        h = mesh.spacing(2)[0]
        run(mesh, upwind("u", (1.0, 0.5)), 0.4 * h / 1.5, 40, reflux=reflux)
        errors[reflux] = abs(mesh.integral("u") - before) / before
    assert errors[True] < 1e-14
    assert errors[False] > 1e-4  # the coarse and fine fluxes disagree


def test_diffusion_with_walls_is_conservative_with_refluxing() -> None:
    errors = {}
    for reflux in (True, False):
        mesh = AMRHierarchy((16, 12), ((0.0, 1.0), (0.0, 0.75)), comm=WORLD)
        mesh.add_patch(1, ((0, 12), (6, 18)), ranks=0)  # along the lower x wall
        mesh.add_patch(2, ((4, 16), (16, 28)), ranks=SIZE - 1)
        mesh.allocate_field("u", halo=1)
        fill(mesh, "u", blob)
        before = mesh.integral("u")
        h = mesh.spacing(2)[0]
        run(mesh, diffusive(mesh, "u", 0.1), 0.2 * h**2 / 0.1, 30, reflux=reflux)
        errors[reflux] = abs(mesh.integral("u") - before) / before
    assert errors[True] < 1e-14
    assert errors[False] > 1e-5


def test_the_cylindrical_metric_is_conservative_with_refluxing() -> None:
    errors = {}
    for reflux in (True, False):
        mesh = AMRHierarchy(
            (12, 12), ((0.0, 1.0), (-0.5, 0.5)), geometry="cylindrical", comm=WORLD
        )
        mesh.add_patch(1, ((0, 10), (6, 18)), ranks=SIZE - 1)  # on the axis
        mesh.add_patch(1, ((14, 22), (2, 10)), ranks=0)
        mesh.allocate_field("u", halo=1)
        fill(mesh, "u", lambda r, z: 1.0 + np.exp(-((r - 0.4) ** 2 + z**2) / 0.02))
        before = mesh.integral("u")
        h = mesh.spacing(1)[0]
        run(mesh, diffusive(mesh, "u"), 0.2 * h**2, 30, reflux=reflux)
        errors[reflux] = abs(mesh.integral("u") - before) / before
    assert errors[True] < 1e-14
    assert errors[False] > 1e-5


def test_the_areas_of_the_fine_faces_add_up_to_the_coarse_face() -> None:
    mesh = AMRHierarchy((4, 4), ((0.5, 1.5), (0.0, 1.0)), geometry="cylindrical")
    for axis in (0, 1):
        coarse = mesh.face_areas(0, axis, np.arange(4))
        fine = mesh.face_areas(1, axis, np.arange(8))
        if axis == 0:  # radial faces: two fine ones (along z) over a coarse one
            np.testing.assert_allclose(2 * fine[::2], coarse)
        else:  # axial faces: two fine rings make a coarse one
            np.testing.assert_allclose(fine[::2] + fine[1::2], coarse)
    # the faces close the cells: the divergence of a uniform radial flux r
    volumes = mesh.cell_volumes(0, np.arange(4))
    radial = mesh.face_areas(0, 0, np.arange(5))
    radius = 0.5 + 0.25 * np.arange(5)
    np.testing.assert_allclose((np.diff(radial * radius)) / volumes, 2.0)


def heun(mesh, register, fluxes, dt):
    """One step of Heun's method (SSP-RK2), the stages recorded with ``dt / 2``."""
    start = [patch.field("u").local.copy() for patch in mesh.local_patches]
    for _ in range(2):
        mesh.exchange_halos("u")
        register.update("u", fluxes, dt, scale=0.5 * dt)
        mesh.restrict("u")
    for patch, u0 in zip(mesh.local_patches, start, strict=True):
        u = patch.field("u").local
        u[...] = 0.5 * (u0 + u)
    register.reflux("u")


def smooth(t):
    def exact(x, y):
        return 1.0 + 0.5 * np.sin(2 * np.pi * (x - t)) * np.sin(2 * np.pi * (y - t))

    return exact


def test_a_smooth_advection_converges_at_second_order() -> None:
    errors = []
    for n in (16, 32, 64):
        mesh = periodic_mesh(n, halo=2)
        fill(mesh, "u", smooth(0.0))
        before = mesh.integral("u")
        register = FluxRegister(mesh)
        h, t_end = mesh.spacing(1)[0], 0.25
        steps = int(np.ceil(t_end / (0.4 * h)))
        for _ in range(steps):
            heun(mesh, register, fromm("u", (1.0, 1.0)), t_end / steps)
        assert abs(mesh.integral("u") - before) < 1e-14 * before
        cells = CompositeCells(mesh)
        exact = smooth(t_end)(*cells.centers.T)
        error = np.abs(cells.from_patches("u") - exact)
        errors.append(np.sum(error * cells.volumes) / np.sum(cells.volumes))
    orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert np.all(orders > 1.9), orders


def test_diffusion_converges_across_refined_patches() -> None:
    errors = []
    for n in (8, 16, 32):
        mesh = periodic_mesh(n, levels=2)
        fill(mesh, "u", lambda x, y: np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y))
        h, t_end = mesh.spacing(2)[0], 0.01
        steps = int(np.ceil(t_end / (0.2 * h**2)))
        run(mesh, diffusive(mesh, "u"), t_end / steps, steps)
        cells = CompositeCells(mesh)
        x, y = cells.centers.T
        exact = np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)
        exact *= np.exp(-8 * np.pi**2 * t_end)
        error = np.abs(cells.from_patches("u") - exact)
        errors.append(np.sum(error * cells.volumes))
    orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert np.all(orders > 1.8), orders


def subcycle(mesh, register, fluxes, dt, level=0):
    """One step of ``level`` and ``ratio`` steps of each finer level per step."""
    ratio = mesh.refinement_ratio
    mine = [p for p in mesh.local_patches if p.level == level]
    mesh.exchange_halos("u", level=level, below="u_between")
    for patch in mine:
        patch.field("u_before").local[...] = patch.field("u").local
    register.update("u", fluxes, dt, level=level)
    if level == mesh.max_level:
        return
    for step in range(ratio):
        theta = step / ratio  # the level in between, for the halos of the finer one
        for patch in mine:
            before, after = patch.field("u_before").local, patch.field("u").local
            patch.field("u_between").local[...] = (1 - theta) * before + theta * after
        subcycle(mesh, register, fluxes, dt / ratio, level + 1)
    register.reflux("u", level=level)


@pytest.mark.parametrize("fluxes", ["upwind", "diffusive"])
def test_subcycling_is_conservative_and_agrees_with_one_step(fluxes) -> None:
    results = []
    for subcycling in (True, False):
        mesh = periodic_mesh(16, levels=2)
        for name in ("u_before", "u_between"):
            mesh.allocate_field(name, halo=1)
        fill(mesh, "u", blob)
        before = mesh.integral("u")
        if fluxes == "upwind":
            flux, dt = upwind("u", (1.0, 0.5)), 0.4 * mesh.spacing(2)[0] / 1.5
            steps = 30
        else:
            flux, dt = diffusive(mesh, "u", 0.1), 0.2 * mesh.spacing(2)[0] ** 2 / 0.1
            steps = 20
        register = FluxRegister(mesh)
        if subcycling:  # the base takes four times the finest step
            for _ in range(steps // 4 + 1):
                subcycle(mesh, register, flux, 4 * dt)
        else:
            run(mesh, flux, dt, 4 * (steps // 4 + 1))
        assert abs(mesh.integral("u") - before) < 1e-14 * before
        cells = CompositeCells(mesh)
        results.append(cells.from_patches("u"))
    # the same solution, to the error of the coarser time steps of coarse levels
    assert np.max(np.abs(results[0] - results[1])) < 0.05


def test_restrict_and_halos_by_level() -> None:
    mesh = periodic_mesh(8, levels=2)
    mesh.allocate_field("v", halo=1)
    for patch in mesh.local_patches:
        patch.field("u").local[...] = patch.level
        patch.field("v").local[...] = 10.0
    mesh.restrict("u", level=1)  # level 2 onto level 1, not level 1 onto level 0
    for patch in mesh.local_patches:
        u = patch.field("u").local
        if patch.level == 1:
            assert np.any(u == 2.0) and np.any(u == 1.0)
        if patch.level == 0:
            assert np.all(u == 0.0)
    mesh.exchange_halos("u", level=2, below="v")
    for patch in mesh.local_patches:
        u = np.asarray(patch.field("u").local_with_halos)
        first = patch.field("u").layout.index_bounds[0][0] == 0
        if patch.level == 2 and first:  # its lower halo: from v of level 1
            assert np.all(u[0, 1:-1] == 10.0)
        if patch.level == 1:  # not filled
            assert np.all(u[0, 1:-1] == 0.0)
    with pytest.raises(ValueError, match="centered"):
        mesh.allocate_field("w", centering="node")
        mesh.exchange_halos("u", level=1, below="w")


def test_fluxes_of_the_wrong_shape_are_refused() -> None:
    mesh = periodic_mesh(8)
    register = FluxRegister(mesh)
    with pytest.raises(ValueError, match="shapes"):
        register.update("u", lambda patch: (np.zeros((1, 1)), np.zeros((1, 1))), 0.1)
