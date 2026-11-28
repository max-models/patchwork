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


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_the_box_index_finds_every_overlapping_box(ndim: int) -> None:
    from patchwork.mesh.halos import _BoxIndex, _overlap

    rng = np.random.default_rng(ndim)
    lower = rng.integers(0, 60, (80, ndim))
    boxes = [
        tuple((int(lo), int(lo + n)) for lo, n in zip(row, size, strict=True))
        for row, size in zip(lower, rng.integers(1, 12, (80, ndim)), strict=True)
    ]
    index = _BoxIndex(boxes)
    for _ in range(200):
        start = rng.integers(-10, 70, ndim)
        query = tuple(
            (int(s), int(s + n))
            for s, n in zip(start, rng.integers(1, 15, ndim), strict=True)
        )
        expected = [k for k, box in enumerate(boxes) if _overlap(box, query)]
        found = index.candidates(query)
        assert set(expected) <= set(found)
        assert found == sorted(found)
    assert index.candidates(((5, 5),) * ndim) == []


def test_the_halo_schedule_of_many_patches_matches_their_neighbours() -> None:
    # 256 tiles of 4 x 4 fine cells: each halo filled from its neighbours
    mesh = AMRHierarchy(base_shape=(32, 32), periodic=(True, False))
    for i in range(0, 64, 4):
        for j in range(16, 48, 2):
            mesh.add_patch(level=1, box=((i, i + 4), (j, j + 2)))
    mesh.allocate_field("u", halo=1)
    for patch in mesh.local_patches:
        x, y = patch.local_cell_centers(mesh, "u")
        patch.field("u").local[...] = x + 2 * y
    mesh.exchange_halos("u")
    for patch in mesh.local_patches:
        field = patch.field("u")
        x, y = patch.local_cell_centers(mesh, "u")
        h = mesh.spacing(1)
        with_halo = np.asarray(field.local_with_halos)
        # the halo row above (same level or interpolated, both exact for a
        # linear function away from the periodic seam)
        inner = slice(1, -1)
        expected = x + 2 * (y + h[1])
        if patch.box[0][0] > 0 and patch.box[0][1] < 64:
            np.testing.assert_allclose(
                with_halo[inner, -1], expected[:, -1], atol=1e-12
            )


def _quadratic(*x):
    """A full quadratic polynomial in 1, 2 or 3 variables."""
    value = 1.0 + 0.5 * x[0] - 0.8 * x[0] ** 2
    for axis, xa in enumerate(x[1:], start=1):
        value = value + (axis - 2.5) * xa + 0.3 * axis * xa**2 - 0.7 * x[0] * xa
    if len(x) == 3:
        value = value + 1.1 * x[1] * x[2]
    return value


def _fill_levels(mesh, function, nodal, halo, offset=None):
    """``function`` at the own points of every patch (plus ``offset[level]``), halos at the sentinel.

    Returns the exact values on each local patch's storage, by patch.
    """
    exact = {}
    shift = 0.0 if nodal else 0.5
    for patch in mesh.local_patches:
        h = mesh.spacing(patch.level)
        lower = mesh.lower_bound
        axes = [
            x0 + (np.arange(b0 - halo, b1 + halo + (1 if nodal else 0)) + shift) * dx
            for (b0, b1), dx, x0 in zip(patch.box, h, lower, strict=True)
        ]
        values = function(*np.meshgrid(*axes, indexing="ij"))
        storage = patch.field("u").local_with_halos
        storage[...] = SENTINEL
        inner = (slice(halo, -halo),) * mesh.ndim
        storage[inner] = values[inner] + (offset or {}).get(patch.level, 0.0)
        exact[id(patch)] = values
    return exact


@pytest.mark.parametrize("centering", ["node", "cell"])
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_quadratic_halos_reproduce_quadratic_polynomials(centering, ndim) -> None:
    nodal = centering == "node"
    base = (8, 6, 4)[:ndim]
    mesh = AMRHierarchy(base_shape=base, comm=WORLD)
    # one patch at the lower walls, one inside
    mesh.add_patch(level=1, box=tuple((0, 6) for _ in range(ndim)))
    mesh.add_patch(level=1, box=((10, 14),) + tuple((2, 6) for _ in range(ndim - 1)))
    mesh.allocate_field("u", halo=HALO, centering=centering, split=None)
    exact = _fill_levels(mesh, _quadratic, nodal, HALO)

    mesh.exchange_halos("u", interpolation="quadratic")

    for patch in mesh.local_patches:
        values = np.asarray(patch.field("u").local_with_halos)
        inside = values != SENTINEL
        # the halo points in the domain are filled, and exactly
        own = np.prod([hi - lo + (1 if nodal else 0) for lo, hi in patch.box])
        assert patch.level == 0 or inside.sum() > own
        np.testing.assert_allclose(
            values[inside], exact[id(patch)][inside], rtol=0, atol=1e-12
        )
    # the bilinear halos are not exact for it
    mesh.exchange_halos("u")
    errors = [
        np.abs(np.asarray(p.field("u").local_with_halos) - exact[id(p)]).max()
        for p in mesh.local_patches
        if p.level == 1
    ]
    assert WORLD.allreduce(max(errors, default=0.0), op=MPI.MAX) > 1e-4


def test_quadratic_halos_keep_the_shared_nodes() -> None:
    mesh = AMRHierarchy(base_shape=(8, 8), comm=WORLD)
    mesh.add_patch(level=1, box=((4, 12), (4, 12)))
    mesh.allocate_field("u", halo=HALO, centering="node", split=None)
    # the fine level disagrees with the coarse one by 5: its own nodes, those it
    # shares with the coarse level included, must keep their values
    _fill_levels(mesh, _quadratic, True, HALO, offset={1: 5.0})
    before = {
        id(p): np.array(p.field("u").local_with_halos) for p in mesh.local_patches
    }

    mesh.exchange_halos("u", interpolation="quadratic")

    for patch in mesh.local_patches:
        values = np.asarray(patch.field("u").local_with_halos)
        own = (slice(HALO, -HALO),) * 2
        np.testing.assert_array_equal(values[own], before[id(patch)][own])
        if patch.level == 1:
            # halo nodes on coarse nodes take the coarse value (no offset)
            halo = np.ones(values.shape, dtype=bool)
            halo[own] = False
            assert np.all(np.abs(values[halo] - before[id(patch)][halo]) > 1.0)


def _laplacian_error(n: int, interpolation: str, periodic: bool) -> float:
    """Max error of the 5-point Laplacian at the edge nodes of a fine node patch."""

    def function(x, y):
        return np.sin(2.0 * np.pi * x + 0.3) * np.cos(2.0 * np.pi * y - 0.2)

    mesh = AMRHierarchy(base_shape=(n, n), periodic=(periodic,) * 2, comm=WORLD)
    # inside, or at the upper corner of a periodic domain (halos across the seam)
    box = ((n, 2 * n),) * 2 if periodic else ((n // 2, 3 * n // 2),) * 2
    mesh.add_patch(level=1, box=box)
    mesh.allocate_field("u", halo=1, centering="node", split=None)
    exact = _fill_levels(mesh, function, True, 1)
    mesh.exchange_halos("u", interpolation=interpolation)
    error = 0.0
    for patch in mesh.local_patches:
        if patch.level != 1:
            continue
        u = np.asarray(patch.field("u").local_with_halos)
        h = mesh.spacing(1)[0]
        lap = (
            u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]
        ) / h**2
        expected = -8.0 * np.pi**2 * exact[id(patch)][1:-1, 1:-1]
        edge = np.ones(lap.shape, dtype=bool)
        edge[1:-1, 1:-1] = False  # the nodes on the coarse-fine edge
        error = max(error, float(np.abs(lap - expected)[edge].max()))
    return WORLD.allreduce(error, op=MPI.MAX)


@pytest.mark.parametrize("periodic", [False, True], ids=["bounded", "periodic"])
def test_the_laplacian_at_a_coarse_fine_edge_converges_with_quadratic_halos(
    periodic,
) -> None:
    sizes = (8, 16, 32)
    bilinear = [_laplacian_error(n, "bilinear", periodic) for n in sizes]
    quadratic = [_laplacian_error(n, "quadratic", periodic) for n in sizes]
    # bilinear: O(1), does not go down; quadratic: first order or better
    assert bilinear[-1] > 0.5 * bilinear[0]
    assert bilinear[-1] > 1.0
    orders = np.log2(np.array(quadratic[:-1]) / np.array(quadratic[1:]))
    assert np.all(orders > 0.9), (quadratic, orders)
    assert quadratic[-1] < 0.1 * bilinear[-1]


def test_an_unknown_interpolation_is_refused() -> None:
    mesh = hierarchy("node")
    with pytest.raises(ValueError, match="interpolation must be one of"):
        mesh.exchange_halos("u", interpolation="cubic")
