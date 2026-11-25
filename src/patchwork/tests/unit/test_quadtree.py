"""Quadtree regridding (``regrid(method="quadtree")``) and tagging by marker counts.

Runs serially and under ``mpirun -n N python -m pytest`` (see ``test_mpi.py``).
"""

from __future__ import annotations

import numpy as np
import pytest
from maybempi import MPI

from patchwork import (
    AMRHierarchy,
    function_tags,
    marker_tags,
    morton_ranks,
    quadtree_boxes,
)
from patchwork.mesh.regrid import combine_tags

WORLD = MPI.COMM_WORLD
SIZE = WORLD.Get_size()


def ring(x, y):
    return np.abs(np.hypot(x - 0.5, y - 0.5) - 0.3) < 0.03


def boxes_of(mesh, level):
    return sorted(patch.box for patch in mesh.patches_at_level(level))


def covered(mesh, level):
    cells = np.zeros(mesh.level_shape(level), dtype=bool)
    for box in boxes_of(mesh, level):
        cells[tuple(slice(lo, hi) for lo, hi in box)] = True
    return cells


@pytest.mark.parametrize("block_size", [4, 8, 16])
def test_quadtree_patches_are_aligned_blocks_whose_parents_are_blocks(
    block_size,
) -> None:
    mesh = AMRHierarchy(base_shape=(32, 32))
    tags = function_tags(mesh, ring, levels=3)
    new = mesh.regrid(tags, method="quadtree", block_size=block_size)
    assert new.max_level == 3
    for level in range(1, new.max_level + 1):
        for box in boxes_of(new, level):
            for lo, hi in box:  # one whole, aligned block of its level
                assert lo % block_size == 0
                assert hi - lo == block_size
            if level == 1:
                continue
            # inside one block of the level below: the patches form a tree
            parents = [
                parent
                for parent in boxes_of(new, level - 1)
                if all(
                    2 * plo <= lo and hi <= 2 * phi
                    for (lo, hi), (plo, phi) in zip(box, parent, strict=True)
                )
            ]
            assert len(parents) == 1
        # every tag of the level below is covered
        assert np.all(
            covered(new, level)[np.repeat(np.repeat(tags[level - 1], 2, 0), 2, 1)]
        )


def test_quadtree_blocks_far_away_stay_when_a_tag_goes() -> None:
    mesh = AMRHierarchy(base_shape=(32, 32))

    def blobs(both):
        def tag(x, y):
            near = np.hypot(x - 0.2, y - 0.2) < 0.06
            far = np.hypot(x - 0.8, y - 0.8) < 0.06
            return near | far if both else near

        return tag

    one = mesh.regrid(
        function_tags(mesh, blobs(False), levels=2), method="quadtree", block_size=8
    )
    two = mesh.regrid(
        function_tags(mesh, blobs(True), levels=2), method="quadtree", block_size=8
    )
    for level in (1, 2):
        near = [
            b for b in boxes_of(two, level) if b[0][1] <= one.level_shape(level)[0] // 2
        ]
        assert near == boxes_of(one, level)


def test_quadtree_moves_a_linear_field_exactly() -> None:
    mesh = AMRHierarchy(base_shape=(16, 16))
    mesh.allocate_field("u", centering="node")
    mesh.allocate_field("c")

    def plane(x, y):
        return 1.0 + 2.0 * x - 3.0 * y

    for patch in mesh.local_patches:
        patch.field("u").local[...] = plane(*patch.local_node_coordinates(mesh, "u"))
        patch.field("c").local[...] = plane(*patch.local_cell_centers(mesh, "c"))
    for tag in (lambda x, y: x < 0.4, lambda x, y: y > 0.7):
        mesh = mesh.regrid(
            function_tags(mesh, tag, levels=2), method="quadtree", block_size=4
        )
    assert mesh.max_level == 2
    for patch in mesh.local_patches:
        x, y = patch.local_node_coordinates(mesh, "u")
        np.testing.assert_allclose(patch.field("u").local, plane(x, y), atol=1e-12)
        x, y = patch.local_cell_centers(mesh, "c")
        np.testing.assert_allclose(patch.field("c").local, plane(x, y), atol=1e-12)


def test_quadtree_blocks_are_cut_at_the_edge_of_an_l_and_at_forbidden_cells() -> None:
    mesh = AMRHierarchy(
        base_shape=(16, 16),
        base_boxes=[((0, 8), (0, 16)), ((8, 16), (8, 16))],  # an L of area 3/4
    )
    tags = {0: np.ones((16, 16), dtype=bool)}
    new = mesh.regrid(tags, method="quadtree", block_size=12, nesting=0)
    inside = np.zeros((32, 32), dtype=bool)
    inside[:16, :] = True
    inside[16:, 16:] = True
    assert np.array_equal(covered(new, 1), inside)

    forbid = {1: np.zeros((32, 32), dtype=bool)}
    forbid[1][:, 30:] = True
    new = mesh.regrid(tags, method="quadtree", block_size=12, nesting=0, forbid=forbid)
    assert np.array_equal(covered(new, 1), inside & ~forbid[1])


def test_quadtree_keeps_the_nesting_margin_and_periodic_axes() -> None:
    mesh = AMRHierarchy(base_shape=(16, 16), periodic=(True, False))
    tags = function_tags(
        mesh, lambda x, y: (x < 0.05) & (np.abs(y - 0.5) < 0.05), levels=2
    )
    new = mesh.regrid(tags, method="quadtree", block_size=4, nesting=2)
    level1 = covered(new, 1)
    # the level-2 blocks at x = 0 need level 1 on both sides of the seam
    assert level1[0].any() and level1[-1].any()
    assert new.max_level == 2


@pytest.mark.parametrize("ndim", [1, 3])
def test_quadtree_in_one_and_three_dimensions(ndim) -> None:
    mesh = AMRHierarchy(base_shape=(8,) * ndim)
    tags = {0: np.zeros((8,) * ndim, dtype=bool)}
    tags[0][(3,) * ndim] = True
    new = mesh.regrid(tags, method="quadtree", block_size=4)
    for box in boxes_of(new, 1):
        assert all(lo % 4 == 0 and hi - lo == 4 for lo, hi in box)
    assert covered(new, 1)[(6,) * ndim]


def test_quadtree_boxes_is_usable_on_its_own() -> None:
    mesh = AMRHierarchy(base_shape=(16, 16))
    tags = combine_tags(mesh, {0: np.eye(16, dtype=bool)})
    boxes = quadtree_boxes(mesh, tags, block_size=8, buffer=0, nesting=0)
    assert boxes == {
        1: [
            ((0, 8), (0, 8)),
            ((8, 16), (8, 16)),
            ((16, 24), (16, 24)),
            ((24, 32), (24, 32)),
        ]
    }
    assert quadtree_boxes(mesh, {}, block_size=8, buffer=0, nesting=0) == {}


def test_quadtree_checks_its_arguments() -> None:
    mesh = AMRHierarchy(base_shape=(8, 8))
    tags = {0: np.ones((8, 8), dtype=bool)}
    with pytest.raises(ValueError, match="method"):
        mesh.regrid(tags, method="octopus")
    with pytest.raises(ValueError, match="max_patch_size"):
        mesh.regrid(tags, method="quadtree", max_patch_size=8)
    for size in (1, 3, 2.5):
        with pytest.raises(ValueError, match="block_size"):
            mesh.regrid(tags, method="quadtree", block_size=size)
    with pytest.raises(ValueError, match="negative"):
        mesh.regrid(tags, method="quadtree", buffer=-1)
    with pytest.raises(ValueError, match="min_patch_size"):
        mesh.regrid(tags, method="quadtree", min_patch_size=0)


def test_marker_tags_count_the_markers_of_every_rank() -> None:
    mesh = AMRHierarchy(base_shape=(4, 4), periodic=(True, False))
    rank = WORLD.Get_rank()
    # each rank holds a share of 9 markers in cell (1, 2), 3 in cell (3, 0)
    x = np.array([0.3] * 9 + [0.9] * 3)
    y = np.array([0.6] * 9 + [0.1] * 3)
    mine = slice(rank, None, SIZE)
    tags = marker_tags(
        mesh, x[mine] + 1.0, y[mine], per_cell=4, levels=2
    )  # wrapped in x
    expected = np.zeros((4, 4), dtype=bool)
    expected[1, 2] = True
    assert np.array_equal(tags[0], expected)
    assert tags[1].sum() == 1 and tags[1][2, 4]  # level 1: all 9 in one cell still
    assert marker_tags(mesh, x[mine], y[mine], per_cell=9, levels=1)[0].sum() == 0
    with pytest.raises(ValueError, match="coordinate"):
        marker_tags(mesh, x, per_cell=1, levels=1)
    with pytest.raises(ValueError, match="levels"):
        marker_tags(mesh, x, y, per_cell=1, levels=-1)


def test_marker_tags_skip_the_notch_of_an_l() -> None:
    mesh = AMRHierarchy(
        base_shape=(4, 4), base_boxes=[((0, 2), (0, 4)), ((2, 4), (2, 4))]
    )
    x, y = np.full(10, 0.8), np.full(10, 0.2)  # in the notch
    assert not marker_tags(mesh, x, y, per_cell=1, levels=1)[0].any()


def test_quadtree_adds_no_level_where_every_cell_is_forbidden() -> None:
    mesh = AMRHierarchy(base_shape=(8, 8))
    tags = {0: np.ones((8, 8), dtype=bool)}
    new = mesh.regrid(tags, method="quadtree", block_size=4, forbid=lambda x, y: x > -1)
    assert new.max_level == 0


def test_morton_ranks_follow_the_z_curve_with_equal_cells() -> None:
    # sixteen 2 x 2 blocks of a 8 x 8 level: the Z curve visits the quadrants in turn
    boxes = [
        ((2 * i, 2 * i + 2), (2 * j, 2 * j + 2)) for i in range(4) for j in range(4)
    ]
    ranks = morton_ranks({1: boxes}, 4)[1]
    for box, rank in zip(boxes, ranks, strict=True):
        (i0, _), (j0, _) = box
        assert rank == (i0 >= 4) + 2 * (j0 >= 4)  # the first axis is the lowest bit
    # a rank that holds the base already takes fewer patches
    loaded = morton_ranks({1: boxes}, 2, load=[32, 0])[1]
    assert loaded.count(0) == 4 and loaded.count(1) == 12
    # a patch and the finer patch over it are neighbours on the curve
    levels = {1: [((0, 4), (0, 4)), ((4, 8), (4, 8))], 2: [((0, 4), (0, 4))]}
    assert morton_ranks(levels, 2) == {1: [0, 1], 2: [0]}
    assert morton_ranks({1: boxes}, 1)[1] == [0] * 16
    assert morton_ranks({}, 3) == {}
    assert morton_ranks({1: [((0, 4),), ((4, 8),)]}, 2) == {1: [0, 1]}  # 1D
    with pytest.raises(ValueError, match="size"):
        morton_ranks({1: boxes}, 0)
    with pytest.raises(ValueError, match="load"):
        morton_ranks({1: boxes}, 2, load=[1.0])


def test_regrid_places_the_blocks_along_the_morton_curve() -> None:
    mesh = AMRHierarchy(base_shape=(16, 16), placement="morton")
    mesh.allocate_field("u", centering="node")
    tags = function_tags(mesh, ring, levels=2)
    new = mesh.regrid(tags, method="quadtree", block_size=8)
    assert new.placement == "morton"
    boxes = {level: [p.box for p in new.patches_at_level(level)] for level in (1, 2)}
    base = AMRHierarchy(base_shape=(16, 16), placement="morton").cells_on_rank
    expected = morton_ranks(boxes, SIZE, load=base)
    for level in (1, 2):
        patches = new.patches_at_level(level)
        assert [p.ranks for p in patches] == [(r,) for r in expected[level]]
    # the cells per rank differ by less than the largest patch
    load = new.cells_on_rank
    assert max(load) - min(load) <= max(256, max(base))
    # the next regrid keeps the placement; a patch added by hand is balanced
    again = new.regrid(tags, method="quadtree", block_size=8, fill_halos=True)
    assert again.placement == "morton"
    added = again.add_patch(3, ((40, 48), (40, 48)))
    assert len(added.ranks) == 1
    with pytest.raises(ValueError, match="morton"):
        AMRHierarchy(base_shape=(4, 4), placement="hilbert")
