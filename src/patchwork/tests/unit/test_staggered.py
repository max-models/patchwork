"""Face- and edge-centered patch fields: shapes, halos, restriction, regridding.

The checks run serially and under ``mpirun -n N python -m pytest``, with the
patches on the ranks there are. The last check is the one constrained transport
relies on: with a vector potential on the edges, restricted, the flux of its curl
through a coarse face is the sum of the fine fluxes through it.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from maybempi import MPI

from patchwork import AMRHierarchy
from patchwork.mesh.hierarchy import CENTERINGS, staggering

WORLD = MPI.COMM_WORLD
SIZE = WORLD.Get_size()
SENTINEL = 1.0e9
FACES = ("x-face", "y-face", "z-face")
EDGES = ("x-edge", "y-edge", "z-edge")


def mesh_3d(periodic=(False, False, False)) -> AMRHierarchy:
    """A 6 x 4 x 4 base with two level-1 patches and a level-2 patch."""
    mesh = AMRHierarchy(
        base_shape=(6, 4, 4),
        bounds=((0.0, 1.5), (0.0, 1.0), (0.0, 1.0)),
        periodic=periodic,
        comm=WORLD,
    )
    mesh.add_patch(level=1, box=((2, 8), (0, 8), (2, 6)), ranks=0)
    mesh.add_patch(level=1, box=((8, 12), (2, 6), (0, 4)), ranks=SIZE - 1)
    mesh.add_patch(level=2, box=((6, 14), (4, 12), (6, 10)), ranks=SIZE // 2)
    return mesh


def linear(x, y, z):
    return 1.0 + 2.0 * x - 0.5 * y + 0.75 * z


def test_the_staggering_of_every_centering() -> None:
    assert staggering("cell", 3) == (False, False, False)
    assert staggering("node", 2) == (True, True)
    assert staggering("x-face", 3) == (True, False, False)
    assert staggering("z-face", 3) == (False, False, True)
    assert staggering("y-edge", 3) == (True, False, True)
    # in 2D the x-edges are the y-faces and the z-edges the nodes; in 1D the
    # x-faces are the nodes and the x-edges the cells
    assert staggering("x-edge", 2) == staggering("y-face", 2) == (False, True)
    assert staggering("z-edge", 2) == (True, True)
    assert staggering("x-face", 1) == (True,)
    assert staggering("x-edge", 1) == (False,)
    with pytest.raises(ValueError, match="has no z-faces"):
        staggering("z-face", 2)
    with pytest.raises(ValueError, match="has no y-edges"):
        staggering("y-edge", 1)
    with pytest.raises(ValueError, match="centering must be one of"):
        staggering("corner", 2)


def test_face_and_edge_fields_have_one_more_point_along_their_node_axes() -> None:
    mesh = mesh_3d()
    for name in CENTERINGS:
        mesh.allocate_field(name, centering=name, split=None)
    for patch in mesh.local_patches:
        nx, ny, nz = patch.shape
        assert patch.field("x-face").shape == (nx + 1, ny, nz)
        assert patch.field("z-face").shape == (nx, ny, nz + 1)
        assert patch.field("x-edge").shape == (nx, ny + 1, nz + 1)
        assert patch.field("y-edge").shape == (nx + 1, ny, nz + 1)
        assert patch.field("node").shape == (nx + 1, ny + 1, nz + 1)
        x, y, z = patch.local_coordinates(mesh, "y-edge")
        dx, dy, dz = mesh.spacing(patch.level)
        (i0, _), (j0, _), (k0, _) = patch.box
        assert x[0, 0, 0] == pytest.approx(i0 * dx)
        assert y[0, 0, 0] == pytest.approx((j0 + 0.5) * dy)
        assert z[0, 0, 0] == pytest.approx(k0 * dz)
        assert x.shape == patch.field("y-edge").local.shape
    for name in CENTERINGS:
        assert mesh.centering(name) == name


def test_staggered_fields_are_cartesian_only() -> None:
    mesh = AMRHierarchy(
        base_shape=(4, 4), bounds=((0.0, 1.0), (0.0, 1.0)), geometry="cylindrical"
    )
    with pytest.raises(NotImplementedError, match="Cartesian only"):
        mesh.allocate_field("b", centering="x-face")


def test_gradient_tags_refuse_staggered_fields() -> None:
    from patchwork import gradient_tags

    mesh = mesh_3d()
    mesh.allocate_field("b", centering="x-face")
    with pytest.raises(NotImplementedError, match="cell- or node-centered"):
        gradient_tags(mesh, "b", 0.1)


def fill(mesh, name, function, halo):
    """``function`` at the own points of every patch, the sentinel in the halos."""
    for patch in mesh.local_patches:
        storage = patch.field(name).local_with_halos
        storage[...] = SENTINEL
        inner = (slice(halo, -halo),) * 3
        storage[inner] = function(*patch.local_coordinates(mesh, name))


@pytest.mark.parametrize("split", [None, 0], ids=["whole", "split"])
@pytest.mark.parametrize("centering", FACES + EDGES)
def test_halos_of_a_linear_field_are_exact(centering, split) -> None:
    mesh = mesh_3d()
    mesh.allocate_field("u", halo=1, centering=centering, split=split)
    fill(mesh, "u", linear, 1)

    mesh.exchange_halos("u")

    nodal = staggering(centering, 3)
    filled = 0
    for patch in mesh.local_patches:
        array = patch.field("u")
        values = np.asarray(array.local_with_halos)
        h = mesh.spacing(patch.level)
        axes = [
            (lo + np.arange(b0 - 1, b1 + 1) + 0.5 * (not on)) * dx
            for (lo, _), (b0, b1), on, dx in zip(
                patch.box, array.layout.index_bounds, nodal, h, strict=True
            )
        ]
        expected = linear(*np.meshgrid(*axes, indexing="ij"))
        there = values != SENTINEL
        np.testing.assert_allclose(values[there], expected[there], rtol=0, atol=1e-12)
        filled += int(there.sum()) - int(np.prod(array.local.shape))
    assert WORLD.allreduce(filled, op=MPI.SUM) > 0


@pytest.mark.parametrize("centering", FACES + EDGES)
def test_restriction_gives_the_mean_over_the_cell_axes(centering) -> None:
    mesh = mesh_3d()
    mesh.allocate_field("u", halo=1, centering=centering, split=None)
    rng = np.random.default_rng(3)
    # a random field on level 2, and a linear one with an offset below: the
    # restriction must give level 1 the fine mean, level 0 the level-1 one
    for patch in mesh.local_patches:
        coordinates = patch.local_coordinates(mesh, "u")
        values = linear(*coordinates) + (100.0 if patch.level < 2 else 0.0)
        if patch.level == 2:
            values = values + rng.normal(size=values.shape)
        patch.field("u").local[...] = values
    fine = {
        patch.box: np.array(patch.field("u").local)
        for patch in mesh.local_patches
        if patch.level == 2
    }

    mesh.restrict("u")

    nodal = staggering(centering, 3)
    r = mesh.refinement_ratio
    for patch in mesh.local_patches:
        if patch.level != 1:
            continue
        values = np.asarray(patch.field("u").local)
        for box, fine_values in fine.items():
            # the level-1 points of the level-2 patch
            under = tuple(
                (lo // r, hi // r + on) for (lo, hi), on in zip(box, nodal, strict=True)
            )
            part = tuple(
                (max(a, lo), min(b, hi + on))
                for (a, b), (lo, hi), on in zip(under, patch.box, nodal, strict=True)
            )
            if any(a >= b for a, b in part):
                continue
            mean = fine_values[
                tuple(slice(None, None, r) if on else slice(None) for on in nodal)
            ]
            shape = [
                part
                for size, on in zip(mean.shape, nodal, strict=True)
                for part in ((size, 1) if on else (size // r, r))
            ]
            mean = mean.reshape(shape).mean(axis=(1, 3, 5))
            got = values[
                tuple(
                    slice(a - lo, b - lo)
                    for (a, b), (lo, _) in zip(part, patch.box, strict=True)
                )
            ]
            want = mean[
                tuple(
                    slice(a - u, b - u)
                    for (a, b), (u, _) in zip(part, under, strict=True)
                )
            ]
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def curl_flux(mesh, patch, axis):
    """The flux of ``curl A`` through this patch's faces normal to ``axis``.

    The circulation of the edge values ``A_b`` (times the edge lengths) around
    every face: Stokes' theorem on the cell faces.
    """
    b, c = (axis + 1) % 3, (axis + 2) % 3
    h = mesh.spacing(patch.level)
    ab = np.asarray(patch.field("xyz"[b] + "-edge").local)  # node along a and c
    ac = np.asarray(patch.field("xyz"[c] + "-edge").local)  # node along a and b

    def take(array, along, start):
        index = [slice(None)] * 3
        index[along] = slice(start, array.shape[along] - 1 + start)
        return array[tuple(index)]

    # around the face: +A_b at c, +A_c at b+1, -A_b at c+1, -A_c at b
    return (take(ab, c, 0) - take(ab, c, 1)) * h[b] + (
        take(ac, b, 1) - take(ac, b, 0)
    ) * h[c]


def test_the_curl_of_a_restricted_edge_potential_keeps_the_fine_fluxes() -> None:
    mesh = mesh_3d(periodic=(True, True, True))
    for name in EDGES:
        mesh.allocate_field(name, halo=1, centering=name, split=None)
    # rough values, but a periodic function of the position: patches that touch
    # (across the periodic faces too) share their edges, with the same values
    for patch in mesh.local_patches:
        for axis, name in enumerate(EDGES):
            x, y, z = patch.local_coordinates(mesh, name)
            phase = 2.0 * np.pi * (5.0 * x / 1.5 + 7.0 * y + 3.0 * z)
            patch.field(name).local[...] = np.sin(phase + axis) + np.cos(3.0 * phase)

    for name in EDGES:
        mesh.restrict(name)

    patches = {patch.box: patch for patch in mesh.local_patches}
    r = mesh.refinement_ratio
    checked = 0
    for fine in mesh.local_patches:
        if fine.level == 0:
            continue
        for axis in range(3):
            fine_flux = curl_flux(mesh, fine, axis)
            # sum the fine fluxes over the r x r fine faces of each coarse face
            shape = [
                part
                for a, size in enumerate(fine_flux.shape)
                for part in ((size, 1) if a == axis else (size // r, r))
            ]
            summed = fine_flux[
                tuple(
                    slice(None, None, r) if a == axis else slice(None) for a in range(3)
                )
            ]
            shape = [
                part
                for a, size in enumerate(summed.shape)
                for part in ((size, 1) if a == axis else (size // r, r))
            ]
            summed = summed.reshape(shape).sum(axis=(1, 3, 5))
            under = tuple(
                (lo // r, hi // r + (a == axis)) for a, (lo, hi) in enumerate(fine.box)
            )
            for coarse in mesh.patches_at_level(fine.level - 1):
                if coarse.box not in patches:
                    continue
                coarse_flux = curl_flux(mesh, patches[coarse.box], axis)
                own = tuple(
                    (lo, hi + (a == axis)) for a, (lo, hi) in enumerate(coarse.box)
                )
                part = tuple(
                    (max(u0, o0), min(u1, o1))
                    for (u0, u1), (o0, o1) in zip(under, own, strict=True)
                )
                if any(a >= b for a, b in part):
                    continue
                got = coarse_flux[
                    tuple(
                        slice(a - o, b - o)
                        for (a, b), (o, _) in zip(part, own, strict=True)
                    )
                ]
                want = summed[
                    tuple(
                        slice(a - u, b - u)
                        for (a, b), (u, _) in zip(part, under, strict=True)
                    )
                ]
                np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
                checked += got.size
    if SIZE == 1:
        assert checked > 0


def test_the_divergence_of_the_curl_is_zero_in_every_cell() -> None:
    mesh = mesh_3d()
    for name in EDGES:
        mesh.allocate_field(name, halo=1, centering=name, split=None)
    rng = np.random.default_rng(11)
    for patch in mesh.local_patches:
        for name in EDGES:
            patch.field(name).local[...] = rng.normal(size=patch.field(name).shape)
    for patch in mesh.local_patches:
        div = sum(np.diff(curl_flux(mesh, patch, axis), axis=axis) for axis in range(3))
        assert np.abs(div).max() < 1e-12


@pytest.mark.parametrize("centering", ("y-face", "z-edge"))
def test_regridding_moves_a_linear_field_exactly(centering) -> None:
    mesh = mesh_3d()
    mesh.allocate_field("u", halo=1, centering=centering, split=None)
    fill(mesh, "u", linear, 1)
    tags = {level: np.zeros(mesh.level_shape(level), dtype=bool) for level in (0, 1)}
    tags[0][1:4, 1:3, 1:3] = True
    tags[1][8:11, 2:5, 3:6] = True

    new = mesh.regrid(tags, fill_halos=True)

    assert new.max_level == 2
    for patch in new.local_patches:
        values = np.asarray(patch.field("u").local)
        expected = linear(*patch.local_coordinates(new, "u"))
        np.testing.assert_allclose(values, expected, rtol=0, atol=1e-12)


def test_halo_schedules_are_kept_by_staggering() -> None:
    """Faces of two axes in one hierarchy keep schedules of their own."""
    mesh = mesh_3d()
    for name in FACES:
        mesh.allocate_field(name, halo=1, centering=name, split=None)
        fill(mesh, name, linear, 1)
    for name in FACES:
        mesh.exchange_halos(name)
    for patch in mesh.local_patches:
        for name in FACES:
            values = np.asarray(patch.field(name).local_with_halos)
            there = values != SENTINEL
            nodal = staggering(name, 3)
            axes = [
                (np.arange(lo - 1, hi + 1 + on) + 0.5 * (not on)) * dx
                for (lo, hi), on, dx in zip(
                    patch.box, nodal, mesh.spacing(patch.level), strict=True
                )
            ]
            expected = linear(*np.meshgrid(*axes, indexing="ij"))
            np.testing.assert_allclose(
                values[there], expected[there], rtol=0, atol=1e-12
            )


@pytest.mark.parametrize("ndim", [1, 2])
def test_faces_in_fewer_dimensions_are_cells_and_nodes(ndim) -> None:
    """A 1D x-face field behaves as a node field, a 2D z-edge one too."""
    mesh = AMRHierarchy(base_shape=(8,) * ndim, comm=WORLD)
    mesh.add_patch(level=1, box=((4, 12),) * ndim)
    name = "x-face" if ndim == 1 else "z-edge"
    for centering in (name, "node"):
        mesh.allocate_field(centering, halo=1, centering=centering, split=None)
        for patch in mesh.local_patches:
            x = patch.local_coordinates(mesh, centering)
            patch.field(centering).local[...] = np.sin(3.0 * sum(x)) + patch.level
        mesh.restrict(centering)
        mesh.exchange_halos(centering)
    for patch in mesh.local_patches:
        np.testing.assert_array_equal(
            np.asarray(patch.field(name).local_with_halos),
            np.asarray(patch.field("node").local_with_halos),
        )


def test_every_centering_is_listed_once() -> None:
    assert len(set(CENTERINGS)) == len(CENTERINGS) == 8
    for centering, ndim in itertools.product(CENTERINGS, (3,)):
        assert len(staggering(centering, ndim)) == ndim
