"""Tagging, clustering and regridding.

Runs serially and under ``mpirun -n N python -m pytest`` (see ``test_mpi.py``):
the regridded patches are spread over the ranks there are.
"""

from __future__ import annotations

import numpy as np
import pytest
from maybempi import MPI

from patchwork import AMRHierarchy, cluster, function_tags, gradient_tags
from patchwork.regrid import _domain_cells, _shrink

WORLD = MPI.COMM_WORLD
SIZE = WORLD.Get_size()


def covered_cells(shape, boxes):
    """How many boxes cover every cell."""
    count = np.zeros(shape, dtype=int)
    for (i0, i1), (j0, j1) in boxes:
        count[i0:i1, j0:j1] += 1
    return count


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("max_size", [None, 5])
def test_clusters_cover_the_tags_without_overlapping(seed, max_size) -> None:
    rng = np.random.default_rng(seed)
    tags = np.zeros((40, 30), dtype=bool)
    for _ in range(4):  # a few blobs, and scattered cells
        ci, cj, r = rng.integers(0, 40), rng.integers(0, 30), rng.integers(2, 7)
        i, j = np.ogrid[:40, :30]
        tags |= (i - ci) ** 2 + (j - cj) ** 2 < r**2
    tags |= rng.random(tags.shape) < 0.01
    allowed = np.ones(tags.shape, dtype=bool)
    allowed[:, 25:] = seed % 2 == 0  # sometimes a strip where no box may go

    boxes = cluster(tags, efficiency=0.6, max_size=max_size, allowed=allowed)

    count = covered_cells(tags.shape, boxes)
    assert count.max() == 1  # disjoint
    assert np.all(count[tags & allowed] == 1)  # every allowed tag is covered
    assert not np.any(count[~allowed])
    for (i0, i1), (j0, j1) in boxes:
        if max_size is not None:
            assert max(i1 - i0, j1 - j0) <= max_size
        else:
            assert tags[i0:i1, j0:j1].mean() >= 0.6


def test_clusters_separate_at_gaps_and_keep_a_filled_box_whole() -> None:
    tags = np.zeros((20, 20), dtype=bool)
    tags[2:6, 3:9] = True
    tags[12:18, 12:14] = True
    assert cluster(tags) == [((2, 6), (3, 9)), ((12, 18), (12, 14))]
    assert cluster(tags, max_size=3) == [
        ((2, 4), (3, 6)),
        ((2, 4), (6, 9)),
        ((4, 6), (3, 6)),
        ((4, 6), (6, 9)),
        ((12, 15), (12, 14)),
        ((15, 18), (12, 14)),
    ]
    assert cluster(np.zeros((4, 4), dtype=bool)) == []


def test_cluster_checks_its_arguments() -> None:
    tags = np.ones((4, 4), dtype=bool)
    with pytest.raises(ValueError, match="1D or 2D"):
        cluster(np.ones((2, 2, 2), dtype=bool))
    with pytest.raises(ValueError, match="efficiency"):
        cluster(tags, efficiency=0.0)
    with pytest.raises(ValueError, match="max_size"):
        cluster(tags, max_size=0)
    with pytest.raises(ValueError, match="shape of tags"):
        cluster(tags, allowed=np.ones((3, 3), dtype=bool))


def assert_properly_nested(mesh: AMRHierarchy, nesting: int) -> None:
    """Every patch inside the level below, ``nesting`` cells from its inner edges."""
    ratio = mesh.refinement_ratio
    for level in range(1, mesh.max_level + 1):
        below = covered_cells(
            mesh.level_shape(level - 1),
            [p.box for p in mesh.patches_at_level(level - 1)],
        ).astype(bool)
        domain = _domain_cells(mesh, level - 1)
        allowed = _shrink(below | ~domain, nesting, mesh.periodic) & below
        for patch in mesh.patches_at_level(level):
            (i0, i1), (j0, j1) = patch.box
            assert all(v % ratio == 0 for v in (i0, i1, j0, j1))
            assert allowed[
                i0 // ratio : i1 // ratio, j0 // ratio : j1 // ratio
            ].all(), (
                level,
                patch.box,
            )


def linear(x, y):
    return 1.0 + 2.0 * x - 3.0 * y


def blob(x, y):
    return np.exp(-((x - 0.35) ** 2 + (y - 0.6) ** 2) / 0.004)


def fill(mesh, function):
    for patch in mesh.local_patches:
        x, y = patch.local_node_coordinates(mesh, "u")
        patch.field("u").local[...] = function(x, y)
        for name in ("c", "rho"):
            x, y = patch.local_cell_centers(mesh, name)
            patch.field(name).local[...] = function(x, y)


def largest_error(mesh, function, cells=("c", "rho")):
    error = 0.0
    for patch in mesh.local_patches:
        x, y = patch.local_node_coordinates(mesh, "u")
        error = max(error, float(np.abs(patch.field("u").local - function(x, y)).max()))
        for name in cells:
            x, y = patch.local_cell_centers(mesh, name)
            values = patch.field(name).local
            error = max(error, float(np.abs(values - function(x, y)).max()))
    return WORLD.allreduce(error, op=MPI.MAX)


def hierarchy(**options) -> AMRHierarchy:
    mesh = AMRHierarchy(base_shape=(32, 24), comm=WORLD, **options)
    mesh.allocate_field("u", centering="node", halo=1)
    mesh.allocate_field("c", halo=1)
    mesh.allocate_field("rho", halo=1, conservative=True)
    return mesh


@pytest.mark.parametrize("placement", ["all", "balanced"])
def test_regridding_follows_the_tags_and_moves_the_fields_exactly(placement) -> None:
    mesh = hierarchy(placement=placement)
    fill(mesh, linear)

    for centre in (0.3, 0.5, 0.7):  # a disc that moves, three levels around it
        tags = function_tags(
            mesh, lambda x, y, c=centre: np.hypot(x - c, y - 0.5) < 0.12, levels=3
        )
        mesh = mesh.regrid(tags, max_patch_size=16)

        assert mesh.max_level == 3
        assert_properly_nested(mesh, nesting=1)
        assert all(
            max(hi - lo for lo, hi in p.box) <= 16 for p in mesh.patches if p.level
        )
        for level, level_tags in tags.items():
            ratio = mesh.refinement_ratio
            fine = covered_cells(
                mesh.level_shape(level + 1),
                [p.box for p in mesh.patches_at_level(level + 1)],
            )
            n, m = level_tags.shape
            refined = fine.reshape(n, ratio, m, ratio).any(axis=(1, 3))
            assert refined[level_tags].all()
        # copied from old patches or interpolated from the new level below:
        # both exact for a linear function
        assert largest_error(mesh, linear) < 1e-12
    if placement == "balanced" and SIZE > 1:
        holders = {p.ranks for p in mesh.patches if p.level}
        assert all(len(ranks) == 1 for ranks in holders)
        assert len(holders) == SIZE


def test_values_of_old_patches_are_copied_not_interpolated() -> None:
    mesh = hierarchy()
    mesh.add_patch(level=1, box=((16, 40), (16, 32)))
    fill(mesh, blob)
    before = {p.box: mesh.gather(p, "u").copy() for p in mesh.patches_at_level(1)}

    # a level-1 patch that overlaps the old one in part
    tags = {0: np.zeros((32, 24), dtype=bool)}
    tags[0][12:22, 9:15] = True
    new = mesh.regrid(tags, buffer=0, efficiency=1.0)

    (old_box,) = before
    (patch,) = new.patches_at_level(1)
    assert patch.box == ((24, 44), (18, 30))
    values = new.gather(patch, "u")
    (i0, _), (j0, _) = patch.box
    (o0, o1), (p0, p1) = old_box
    # nodes 24..40 x 18..30 were the old patch's nodes
    rows, columns = o1 - i0 + 1, values.shape[1]
    np.testing.assert_array_equal(
        values[:rows, :],
        before[old_box][i0 - o0 :, j0 - p0 : j0 - p0 + columns],
    )
    # the nodes beyond it are the bilinear interpolation of the base nodes
    base = new.gather(new.patches[0], "u")
    i = np.arange(i0 + rows, patch.box[0][1] + 1) / 2
    j = np.arange(j0, patch.box[1][1] + 1) / 2
    li, lj = np.floor(i).astype(int), np.floor(j).astype(int)
    wi, wj = (i - li)[:, None], (j - lj)[None, :]
    li, lj = li[:, None], lj[None, :]
    expected = (
        (1 - wi) * (1 - wj) * base[li, lj]
        + wi * (1 - wj) * base[np.minimum(li + 1, 32), lj]
        + (1 - wi) * wj * base[li, np.minimum(lj + 1, 24)]
        + wi * wj * base[np.minimum(li + 1, 32), np.minimum(lj + 1, 24)]
    )
    np.testing.assert_allclose(values[rows:], expected, rtol=0, atol=1e-15)


def test_gradient_tags_find_a_step_on_every_rank() -> None:
    mesh = hierarchy(placement="balanced")
    mesh = mesh.regrid(
        function_tags(mesh, lambda x, y: (x > 0.25) & (x < 0.75), levels=1)
    )

    def step(x, y):
        return np.where(x + 0.0 * y < 0.52, 0.0, 1.0)

    fill(mesh, step)
    tags = gradient_tags(mesh, "c", 0.5)
    from patchwork.regrid import combine_tags

    combined = combine_tags(mesh, tags)
    # level-1 cells of 1/64: the jump is between cells 32 and 33, both tagged
    columns = np.flatnonzero(combined[1].any(axis=1))
    assert list(columns) == [32, 33]
    assert combined[1][32:34].all()  # level 1 spans the whole y range
    assert not combined[0].any() or set(np.flatnonzero(combined[0].any(axis=1))) <= {
        15,
        16,
        17,
    }
    nodal = combine_tags(mesh, gradient_tags(mesh, "u", 0.5))
    assert set(np.flatnonzero(nodal[1].any(axis=1))) == {33}


def test_an_l_shaped_domain_is_refined_up_to_its_inner_corner() -> None:
    mesh = AMRHierarchy(
        base_shape=(16, 16),
        base_boxes=[((0, 8), (0, 16)), ((8, 16), (8, 16))],
        comm=WORLD,
        placement="balanced",
    )
    mesh.allocate_field("u", centering="node")
    corner = function_tags(
        mesh, lambda x, y: np.hypot(x - 0.5, y - 0.5) < 0.15, levels=2
    )
    mesh = mesh.regrid(corner)

    assert mesh.max_level == 2
    assert_properly_nested(mesh, nesting=1)
    # patches reach the corner from both arms, never into the missing quadrant
    domain = _domain_cells(mesh, 2)
    for patch in mesh.patches_at_level(2):
        (i0, i1), (j0, j1) = patch.box
        assert domain[i0:i1, j0:j1].all()
    near = covered_cells((64, 64), [p.box for p in mesh.patches_at_level(2)])
    assert near[31, 31] and near[31, 32] and near[32, 32]


def test_a_periodic_axis_is_refined_across_its_seam() -> None:
    mesh = hierarchy(periodic=(False, True))
    fill(mesh, lambda x, y: 1.0 + 2.0 * x)
    tags = function_tags(mesh, lambda x, y: np.minimum(y, 1.0 - y) < 0.1, levels=1)
    mesh = mesh.regrid(tags)

    boxes = [p.box for p in mesh.patches_at_level(1)]
    assert any(j0 == 0 for _, (j0, _) in boxes)
    assert any(j1 == 48 for _, (_, j1) in boxes)
    # the patches reach the x walls: the conservative "rho" is flat in the
    # coarse cells there, not linear
    assert largest_error(mesh, lambda x, y: 1.0 + 2.0 * x + 0.0 * y, ("c",)) < 1e-12


def test_levels_without_tags_are_dropped_and_communicators_are_shared() -> None:
    mesh = hierarchy()
    mesh = mesh.regrid(function_tags(mesh, lambda x, y: x < 0.3, levels=2))
    assert mesh.max_level == 2

    coarser = mesh.regrid({0: np.zeros((32, 24), dtype=bool)})

    assert coarser.max_level == 0
    assert coarser._communicators is mesh._communicators
    assert coarser.field_names == mesh.field_names


def test_regrid_checks_its_tags() -> None:
    mesh = hierarchy()
    with pytest.raises(ValueError, match="shape"):
        mesh.regrid({0: np.zeros((4, 4), dtype=bool)})
    with pytest.raises(ValueError, match="per level"):
        mesh.regrid({-1: np.zeros((32, 24), dtype=bool)})
    with pytest.raises(ValueError, match="max_patch_size"):
        mesh.regrid({0: np.ones((32, 24), dtype=bool)}, max_patch_size=1)
    with pytest.raises(KeyError, match="no field"):
        gradient_tags(mesh, "v", 1.0)


@pytest.mark.parametrize("seed", [*range(8), 44])  # 44 needs the clipping to nest
def test_random_tags_on_three_levels_give_properly_nested_patches(seed) -> None:
    rng = np.random.default_rng(seed)
    mesh = AMRHierarchy(base_shape=(24, 24), comm=WORLD)
    tags = {}
    for level in range(3):
        n = 24 * 2**level
        level_tags = np.zeros((n, n), dtype=bool)
        for _ in range(3):
            ci, cj = rng.integers(0, n, 2)
            r = int(rng.integers(1, 4 * 2**level))
            i, j = np.ogrid[:n, :n]
            level_tags |= (i - ci) ** 2 + (j - cj) ** 2 < r * r
        tags[level] = level_tags

    new = mesh.regrid(tags, efficiency=float(rng.choice([0.3, 0.5, 0.8])))

    assert_properly_nested(new, nesting=1)


@pytest.mark.parametrize("periodic", [(False, False), (False, True)])
def test_a_conservative_field_keeps_its_integral_through_regrids(periodic) -> None:
    mesh = hierarchy(periodic=periodic, placement="balanced")

    previous = None
    for centre in (0.3, 0.45, 0.6, 0.6):  # the last regrid keeps the patches
        # each level evolved on its own, as in a simulation: the coarse cells
        # under a patch no longer hold the means of its cells
        # a narrow pulse on a wide one, curved at the patches' edges too
        fill(
            mesh,
            lambda x, y, c=centre: (
                blob(x - c + 0.35, y)
                + 0.5 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.05)
            ),
        )
        total = mesh.integral("rho")
        tags = function_tags(
            mesh, lambda x, y, c=centre: np.hypot(x - c, y - 0.6) < 0.15, levels=3
        )

        mesh = mesh.regrid(tags, max_patch_size=16)

        assert mesh.integral("rho") == pytest.approx(total, rel=1e-13, abs=0)
        if centre != previous:
            # the bilinear field "c" held the same values, but its new cells
            # do not keep the integral
            assert abs(mesh.integral("c") - total) > 1e-6
        previous = centre
    total = mesh.integral("rho")
    mesh = mesh.regrid({0: np.zeros((32, 24), dtype=bool)})  # every patch dropped
    assert mesh.integral("rho") == pytest.approx(total, rel=1e-13, abs=0)


def test_conservative_filling_makes_no_new_extremes() -> None:
    mesh = hierarchy()

    def step(x, y):
        return np.where(x + 0.5 * y < 0.6, 1.0, 3.0)

    fill(mesh, step)
    mesh = mesh.regrid(function_tags(mesh, lambda x, y: np.abs(x - 0.45) < 0.2, 2))

    for patch in mesh.local_patches:
        values = np.asarray(patch.field("rho").local)
        assert values.min() >= 1.0 - 1e-12
        assert values.max() <= 3.0 + 1e-12


def test_the_integral_of_cell_and_node_fields() -> None:
    mesh = AMRHierarchy(
        base_shape=(16, 16),
        base_boxes=[((0, 8), (0, 16)), ((8, 16), (8, 16))],  # an L of area 3/4
        comm=WORLD,
        placement="balanced",
    )
    mesh.add_patch(level=1, box=((4, 20), (16, 28)))  # over both base boxes
    mesh.add_patch(level=2, box=((12, 28), (36, 52)))
    mesh.allocate_field("u", centering="node")
    mesh.allocate_field("c")
    for patch in mesh.local_patches:
        x, y = patch.local_node_coordinates(mesh, "u")
        patch.field("u").local[...] = linear(x, y)
        patch.field("c").local[...] = 2.0

    assert mesh.integral("c") == pytest.approx(2.0 * 0.75, rel=1e-14)
    # the trapezoidal rule is exact for a linear field: 1 + 2x - 3y over the
    # unit square without the quadrant [0.5, 1] x [0, 0.5]
    square = 1.0 + 1.0 - 1.5
    quadrant = 0.25 * (1.0 + 2.0 * 0.75 - 3.0 * 0.25)
    assert mesh.integral("u") == pytest.approx(square - quadrant, rel=1e-13)

    # values under a finer patch do not count
    for patch in mesh.local_patches:
        if patch.level < 2:
            under = patch.field("c")
            (b0, _), (c0, _) = under.layout.index_bounds
            (i0, _), (j0, _) = patch.box
            i, j = np.meshgrid(
                i0 + b0 + np.arange(under.local.shape[0]),
                j0 + c0 + np.arange(under.local.shape[1]),
                indexing="ij",
            )
            finer = covered_cells(
                mesh.level_shape(patch.level + 1),
                [p.box for p in mesh.patches_at_level(patch.level + 1)],
            )
            n, m = finer.shape
            ratio = mesh.refinement_ratio
            refined = finer.reshape(n // ratio, ratio, m // ratio, ratio).any(
                axis=(1, 3)
            )
            np.asarray(under.local)[refined[i, j].astype(bool)] = 100.0
    assert mesh.integral("c") == pytest.approx(2.0 * 0.75, rel=1e-14)


def test_only_cell_centered_fields_can_be_conservative() -> None:
    mesh = AMRHierarchy(base_shape=(8, 8), comm=WORLD)
    with pytest.raises(ValueError, match="cell-centered"):
        mesh.allocate_field("u", centering="node", conservative=True)


# Smaller items: halos after a regrid, forbidden cells, patch sizes and placement ---------- #


def middle_tags(mesh):
    """Tags of level 0 over a rectangle in the middle of a 16 x 16 unit square."""
    return function_tags(
        mesh, lambda x, y: (x > 0.25) & (x < 0.6) & (y > 0.3) & (y < 0.7), levels=1
    )


def test_regrid_fills_the_halos_when_asked() -> None:
    mesh = AMRHierarchy(base_shape=(16, 16), comm=WORLD)
    mesh.allocate_field("u", halo=2)
    for patch in mesh.local_patches:
        x, y = patch.local_cell_centers(mesh)
        patch.field("u").local[...] = 1.0 + 2.0 * x
    tags = middle_tags(mesh)

    plain = mesh.regrid(tags)
    filled = mesh.regrid(tags, fill_halos=True)
    reference = mesh.regrid(tags)
    reference.exchange_halos("u")

    differ = 0
    for a, b, c in zip(
        plain.local_patches, filled.local_patches, reference.local_patches, strict=True
    ):
        assert np.array_equal(
            b.field("u").local_with_halos, c.field("u").local_with_halos
        )
        differ += int(
            not np.array_equal(
                a.field("u").local_with_halos, b.field("u").local_with_halos
            )
        )
    assert differ > 0
    assert plain.max_level == filled.max_level == 1


def test_forbidden_cells_are_never_covered_and_the_default_is_unchanged() -> None:
    mesh = AMRHierarchy(base_shape=(16, 16), comm=WORLD)
    tags = middle_tags(mesh)
    free = mesh.regrid(tags)
    assert [p.box for p in mesh.regrid(tags, forbid=None).patches] == [
        p.box for p in free.patches
    ]

    # no patch over the cells with x > 0.4: the level-1 cells i >= 13 of 32
    mask = np.zeros(mesh.level_shape(1), dtype=bool)
    mask[13:, :] = True
    by_mask = mesh.regrid(tags, forbid={1: mask})
    by_function = mesh.regrid(tags, forbid=lambda x, y: x > 0.4)

    assert max(p.box[0][1] for p in free.patches_at_level(1)) > 13
    for new in (by_mask, by_function):
        cover = np.zeros(new.level_shape(1), dtype=int)
        for patch in new.patches_at_level(1):
            (i0, i1), (j0, j1) = patch.box
            cover[i0:i1, j0:j1] += 1
        assert cover.max() == 1
        assert not cover[13:, :].any()
        assert cover[:13, :].any()  # the tags left of it are still refined
    assert [p.box for p in by_mask.patches] == [p.box for p in by_function.patches]


def test_forbidden_cells_keep_patches_off_a_periodic_seam() -> None:
    mesh = AMRHierarchy(base_shape=(16, 16), periodic=(False, True), comm=WORLD)
    tags = function_tags(mesh, lambda x, y: (x > 0.3) & (x < 0.6), levels=1)

    free = mesh.regrid(tags)
    away = mesh.regrid(tags, forbid=lambda x, y: (y < 0.1) | (y > 0.9))

    assert any(p.box[1][0] == 0 for p in free.patches_at_level(1))  # touches the seam
    ny = away.level_shape(1)[1]
    for patch in away.patches_at_level(1):
        assert patch.box[1][0] > 0 and patch.box[1][1] < ny


def test_forbid_checks_its_mask() -> None:
    mesh = AMRHierarchy(base_shape=(16, 16), comm=WORLD)
    with pytest.raises(ValueError, match="shape"):
        mesh.regrid(middle_tags(mesh), forbid={1: np.zeros((3, 3), dtype=bool)})


def test_a_box_shorter_than_the_minimum_is_widened_where_there_is_room() -> None:
    tags = np.zeros((20, 20), dtype=bool)
    tags[0, 0] = True  # in a corner
    tags[10, 10] = True
    tags[10, 12] = True  # close to the one before

    assert cluster(tags, efficiency=0.1) == [((0, 1), (0, 1)), ((10, 11), (10, 13))]
    boxes = cluster(tags, efficiency=0.1, min_size=3)

    count = covered_cells((20, 20), boxes)
    assert count.max() == 1  # still disjoint
    assert np.all(count[tags] == 1)
    for (i0, i1), (j0, j1) in boxes:
        assert i1 - i0 >= 3 and j1 - j0 >= 3
    assert ((0, 3), (0, 3)) in boxes  # grown away from the corner

    # no room: allowed leaves a strip two cells wide
    allowed = np.zeros((20, 20), dtype=bool)
    allowed[:, 9:11] = True
    boxes = cluster(tags & allowed, efficiency=0.1, allowed=allowed, min_size=3)
    # the box is as wide as the strip allows (2 cells on y) and widened on x
    assert boxes == [((8, 11), (9, 11))]
    with pytest.raises(ValueError, match="min_size"):
        cluster(tags, min_size=0)
    with pytest.raises(ValueError, match="exceeds max_size"):
        cluster(tags, min_size=5, max_size=4)


def test_regrid_has_no_patch_narrower_than_the_minimum() -> None:
    mesh = AMRHierarchy(base_shape=(16, 16), comm=WORLD)
    tags = {0: np.zeros((16, 16), dtype=bool)}
    tags[0][8, 8] = True  # one cell: a 2 x 2 patch of level 1 without a minimum

    small = mesh.regrid(tags, buffer=0)
    wide = mesh.regrid(tags, buffer=0, min_patch_size=8)

    assert small.patches_at_level(1)[0].shape == (2, 2)
    for patch in wide.patches_at_level(1):
        assert min(patch.shape) >= 8
        assert patch.box[0][0] % 2 == 0 and patch.box[1][0] % 2 == 0  # aligned
    with pytest.raises(ValueError, match="min_patch_size"):
        mesh.regrid(tags, min_patch_size=0)
    with pytest.raises(ValueError, match="max_patch_size"):
        mesh.regrid(tags, min_patch_size=8, max_patch_size=4)


def test_placement_can_be_a_function_of_the_patch() -> None:
    last = SIZE - 1

    def where(level, box):
        cells = (box[0][1] - box[0][0]) * (box[1][1] - box[1][0])
        return tuple(range(SIZE)) if cells > 100 else last

    mesh = AMRHierarchy(
        base_shape=(16, 16),
        placement=where,
        base_boxes=[((0, 8), (0, 8)), ((8, 16), (0, 16))],
        comm=WORLD,
    )
    small = mesh.add_patch(level=1, box=((0, 8), (0, 8)))  # 64 cells: the last rank
    large = mesh.add_patch(level=1, box=((16, 32), (0, 16)))  # 256 cells: every rank
    explicit = mesh.add_patch(level=1, box=((16, 24), (16, 24)), ranks=0)  # a rank wins

    assert [p.ranks for p in mesh.patches_at_level(0)] == [(last,), tuple(range(SIZE))]
    assert small.ranks == (last,)
    assert large.ranks == tuple(range(SIZE))
    assert explicit.ranks == (0,)
    with pytest.raises(ValueError, match="placement"):
        AMRHierarchy(base_shape=(16, 16), placement="everywhere")


def test_large_patches_are_split_over_several_ranks() -> None:
    mesh = AMRHierarchy(
        base_shape=(16, 16),
        placement="balanced",
        max_cells_per_rank=100,
        base_boxes=[((0, 16), (0, 16))],
        comm=WORLD,
    )
    big = mesh.patches_at_level(0)[
        0
    ]  # 256 cells: three ranks of 100, at most all of them
    assert len(big.ranks) == min(SIZE, 3)
    assert big.ranks == tuple(
        range(len(big.ranks))
    )  # the least loaded, and all are equal

    small = mesh.add_patch(
        level=1, box=((0, 8), (0, 8))
    )  # 64 cells: one rank, the lightest
    assert len(small.ranks) == 1
    if SIZE > 3:
        assert small.ranks == (3,)

    new = mesh.regrid(middle_tags(mesh))  # the policy goes on in the new hierarchy
    assert new.max_cells_per_rank == 100 and new.placement == "balanced"
    for patch in new.patches_at_level(1):
        cells = patch.shape[0] * patch.shape[1]
        assert len(patch.ranks) == min(SIZE, max(1, -(-cells // 100)))
    with pytest.raises(ValueError, match="max_cells_per_rank"):
        AMRHierarchy(base_shape=(16, 16), max_cells_per_rank=0)


def test_without_a_limit_a_balanced_patch_stays_on_one_rank() -> None:
    mesh = AMRHierarchy(base_shape=(16, 16), placement="balanced", comm=WORLD)

    assert len(mesh.patches_at_level(0)[0].ranks) == 1
    assert mesh.max_cells_per_rank is None
