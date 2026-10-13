"""Halo exchange between patches and restriction to coarse patches.

The checks look at every point of every rank's storage, so the file runs
serially and under ``mpirun -n N python -m pytest`` (see ``test_mpi.py``), with
the patches spread over the ranks there are.
"""

from __future__ import annotations

import numpy as np
import pytest
from maybempi import MPI

from patchwork import AMRHierarchy

WORLD = MPI.COMM_WORLD
SIZE = WORLD.Get_size()
RANK = WORLD.Get_rank()
SENTINEL = 1.0e9
HALO = 2

#: Level, box and ranks of the fine patches, on a 16 x 12 base periodic in y.
PATCHES = (
    (1, ((4, 14), (0, 24)), tuple(sorted({0, SIZE - 1}))),  # along the periodic axis
    (1, ((14, 24), (6, 18)), 1 % SIZE),  # touches the one before
    (1, ((0, 4), (8, 16)), SIZE - 1),  # at the lower x wall
    (1, ((26, 32), (18, 24)), 0),  # at the upper x wall and the periodic face
    (2, ((12, 24), (8, 20)), None),  # every rank
    (2, ((24, 32), (16, 24)), (SIZE - 1,)),  # over two level-1 patches
)


def hierarchy(centering: str) -> AMRHierarchy:
    mesh = AMRHierarchy(base_shape=(16, 12), periodic=(False, True), comm=WORLD)
    for level, box, ranks in PATCHES:
        mesh.add_patch(level=level, box=box, ranks=ranks)
    mesh.allocate_field("u", halo=HALO, centering=centering)
    return mesh


def linear(x, y):
    return 1.0 + 2.0 * x


def periodic(x, y):
    return 1.0 + 2.0 * x + 0.3 * np.cos(2.0 * np.pi * y)


def own_points(mesh: AMRHierarchy, patch, nodal: bool):
    extra = 1 if nodal else 0
    return tuple((lo, hi + extra) for lo, hi in patch.box)


def contains(mesh, level, i, j, nodal):
    """Whether level points (i, j), j taken modulo the period, are points of a patch."""
    n_y = mesh.level_shape(level)[1]
    found = np.zeros(i.shape, dtype=bool)
    for patch in mesh.patches_at_level(level):
        (x0, x1), (y0, y1) = own_points(mesh, patch, nodal)
        for jj in (j % n_y, j % n_y + n_y):
            found |= (i >= x0) & (i < x1) & (jj >= y0) & (jj < y1)
    return found


def storage_points(mesh, patch, nodal):
    """Level indices of every point of this rank's storage of field ``u``."""
    layout = patch.field("u").layout
    axes = [
        lo + np.arange(b0 - h, b1 + h)
        for (lo, _), (b0, b1), h in zip(
            patch.box, layout.index_bounds, layout.halo, strict=True
        )
    ]
    return np.meshgrid(*axes, indexing="ij")


def coordinates(mesh, level, i, j, nodal):
    dx, dy = mesh.spacing(level)
    shift = 0.0 if nodal else 0.5
    return (i + shift) * dx, (j + shift) * dy


def fill(mesh, function, nodal, interior_only=True):
    """``function`` on the patches' own points; the halos hold the sentinel."""
    for patch in mesh.local_patches:
        i, j = storage_points(mesh, patch, nodal)
        storage = patch.field("u").local_with_halos
        storage[...] = SENTINEL
        values = function(*coordinates(mesh, patch.level, i, j, nodal))
        storage[HALO:-HALO, HALO:-HALO] = values[HALO:-HALO, HALO:-HALO]


def complete_stencil(mesh, level, i, j, nodal):
    """Whether every coarse point with a weight is a point of the level below."""
    ratio = mesh.refinement_ratio
    ok = np.ones(i.shape, dtype=bool)
    stencil = []
    for values in (i, j):
        position = values / ratio if nodal else (values + 0.5) / ratio - 0.5
        lower = np.floor(position).astype(int)
        stencil.append((lower, position - lower))
    (ci, wi), (cj, wj) = stencil
    n_x = mesh.level_shape(level - 1)[0]
    for a in (0, 1):
        for b in (0, 1):
            weight = (wi if a else 1 - wi) * (wj if b else 1 - wj)
            pi, pj = ci + a, cj + b
            inside = (pi >= 0) & (pi < n_x + (1 if nodal else 0))
            there = contains(mesh, level - 1, np.clip(pi, 0, None), pj, nodal) & inside
            ok &= (weight == 0.0) | there
    return ok


@pytest.mark.parametrize("centering", ["cell", "node"])
@pytest.mark.parametrize("function", [linear, periodic], ids=["linear", "periodic"])
def test_every_halo_point_gets_its_value(centering, function) -> None:
    nodal = centering == "node"
    mesh = hierarchy(centering)
    fill(mesh, function, nodal)

    mesh.exchange_halos("u")

    counts = {"copied": 0, "interpolated": 0, "outside": 0}
    for patch in mesh.local_patches:
        level = patch.level
        i, j = storage_points(mesh, patch, nodal)
        storage = np.asarray(patch.field("u").local_with_halos)
        expected = function(*coordinates(mesh, level, i, j, nodal))
        (x0, x1), (y0, y1) = own_points(mesh, patch, nodal)
        ghost = ~((i >= x0) & (i < x1) & (j >= y0) & (j < y1))
        n_x = mesh.level_shape(level)[0]
        outside = (i < 0) | (i >= n_x + (1 if nodal else 0))
        copied = ghost & ~outside & contains(mesh, level, i, j, nodal)
        rest = ghost & ~outside & ~copied

        np.testing.assert_array_equal(storage[~ghost], expected[~ghost])
        np.testing.assert_allclose(
            storage[copied], expected[copied], rtol=0, atol=1e-12
        )
        assert np.all(storage[ghost & outside] == SENTINEL)
        counts["copied"] += int(copied.sum())
        counts["outside"] += int((ghost & outside).sum())
        if level == 0:
            continue
        exact = rest & complete_stencil(mesh, level, i, j, nodal)
        tolerance = 1e-12 if function is linear else 0.02
        np.testing.assert_allclose(
            storage[exact], expected[exact], rtol=0, atol=tolerance
        )
        # a partial stencil: a mean of values of the function
        partial = rest & ~exact
        assert np.all((storage[partial] > 0.0) & (storage[partial] < 4.0))
        counts["interpolated"] += int(rest.sum())
    totals = {key: WORLD.allreduce(value, op=MPI.SUM) for key, value in counts.items()}
    assert all(totals.values()), totals  # every kind of halo point was seen


@pytest.mark.parametrize("centering", ["cell", "node"])
def test_restriction_goes_down_level_by_level(centering) -> None:
    nodal = centering == "node"
    mesh = hierarchy(centering)
    by_level = {
        0: lambda x, y: np.full_like(x, SENTINEL),
        1: lambda x, y: linear(x, y) + 100.0,
    }
    for patch in mesh.local_patches:
        i, j = storage_points(mesh, patch, nodal)
        function = by_level.get(patch.level, linear)
        patch.field("u").local_with_halos[...] = function(
            *coordinates(mesh, patch.level, i, j, nodal)
        )

    mesh.restrict("u")

    ratio = mesh.refinement_ratio
    for patch in mesh.local_patches:
        if patch.level == 2:
            continue
        i, j = storage_points(mesh, patch, nodal)
        inner = (slice(HALO, -HALO), slice(HALO, -HALO))
        i, j = i[inner], j[inner]
        values = np.asarray(patch.field("u").local_with_halos)[inner]
        expected = by_level[patch.level](*coordinates(mesh, patch.level, i, j, nodal))
        # under level-2 patches: the fine values, through level 1 for level 0
        for finer in range(patch.level + 1, 3):
            scale = ratio ** (finer - patch.level)
            if nodal:
                fi, fj = i * scale, j * scale
                under = contains(mesh, finer, fi, fj, nodal)
                value = linear if finer == 2 else by_level[1]
                fine_value = value(*coordinates(mesh, finer, fi, fj, nodal))
            else:
                children = [
                    (i * scale + a, j * scale + b)
                    for a in range(scale)
                    for b in range(scale)
                ]
                under = np.ones(i.shape, dtype=bool)
                for ci, cj in children:
                    under &= contains(mesh, finer, ci, cj, nodal)
                fine_value = linear(*coordinates(mesh, patch.level, i, j, nodal))
                if finer == 1:
                    # a level-0 cell under level 1: the mean of its level-1 cells,
                    # each the mean of level 2 (exact) or level 1's own value
                    level1 = np.zeros(i.shape)
                    for ci, cj in children:
                        own = linear(*coordinates(mesh, 1, ci, cj, nodal))
                        deeper = np.ones(i.shape, dtype=bool)
                        for a in range(ratio):
                            for b in range(ratio):
                                deeper &= contains(
                                    mesh, 2, ci * ratio + a, cj * ratio + b, nodal
                                )
                        level1 += np.where(deeper, own, own + 100.0)
                    fine_value = level1 / len(children)
            expected = np.where(under, fine_value, expected)
            if nodal and finer == 1:
                # a level-0 node under a level-2 node gets the level-2 value
                deep = contains(mesh, 2, i * ratio**2, j * ratio**2, nodal) & under
                expected = np.where(
                    deep, linear(*coordinates(mesh, 0, i, j, nodal)), expected
                )
        np.testing.assert_allclose(values, expected, rtol=0, atol=1e-9)


def test_schedules_are_kept_until_a_patch_is_added() -> None:
    mesh = hierarchy("cell")
    mesh.exchange_halos("u")
    mesh.restrict("u")
    kept = dict(mesh._schedules)

    mesh.exchange_halos("u")
    assert mesh._schedules == kept
    mesh.add_patch(level=1, box=((26, 32), (0, 6)), ranks=0)
    assert not mesh._schedules
    mesh.exchange_halos("u")
    assert ("halo", "u") in mesh._schedules
    with pytest.raises(KeyError, match="no field 'v'"):
        mesh.exchange_halos("v")


def test_a_field_without_halo_needs_no_exchange() -> None:
    mesh = AMRHierarchy(base_shape=(8, 8), comm=WORLD)
    mesh.add_patch(level=1, box=((4, 12), (4, 12)), ranks=0)
    mesh.allocate_field("u", halo=0)
    for patch in mesh.local_patches:
        patch.field("u").local[...] = 2.0
    mesh.exchange_halos("u")
    mesh.restrict("u")
    for patch in mesh.local_patches:
        assert np.all(np.asarray(patch.field("u").local) == 2.0)
