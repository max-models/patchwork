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
    with pytest.raises(ValueError, match="2D"):
        cluster(np.ones(4, dtype=bool))
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
        x, y = patch.local_cell_centers(mesh, "c")
        patch.field("c").local[...] = function(x, y)


def largest_error(mesh, function):
    error = 0.0
    for patch in mesh.local_patches:
        x, y = patch.local_node_coordinates(mesh, "u")
        error = max(error, float(np.abs(patch.field("u").local - function(x, y)).max()))
        x, y = patch.local_cell_centers(mesh, "c")
        error = max(error, float(np.abs(patch.field("c").local - function(x, y)).max()))
    return WORLD.allreduce(error, op=MPI.MAX)


def hierarchy(**options) -> AMRHierarchy:
    mesh = AMRHierarchy(base_shape=(32, 24), comm=WORLD, **options)
    mesh.allocate_field("u", centering="node", halo=1)
    mesh.allocate_field("c", halo=1)
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
    assert largest_error(mesh, lambda x, y: 1.0 + 2.0 * x + 0.0 * y) < 1e-12


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
