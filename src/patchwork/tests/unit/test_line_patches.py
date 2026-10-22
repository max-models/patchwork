"""1D hierarchies: patches, halos, restriction, regridding and markers on a line.

The 1D counterparts of ``test_mesh.py``, ``test_halos.py``, ``test_regrid.py`` and
``test_markers.py``. Runs serially and under ``mpirun -n N python -m pytest`` (see
``test_mpi.py``), with the patches spread over the ranks there are.
"""

from __future__ import annotations

import numpy as np
import pytest
from maybempi import MPI

from patchwork import AMRHierarchy, cluster, function_tags, gradient_tags
from patchwork.nodal import CompositeNodes

WORLD = MPI.COMM_WORLD
SIZE = WORLD.Get_size()
RANK = WORLD.Get_rank()
SENTINEL = 1.0e9
HALO = 2

#: Level, box and ranks of the fine patches on a base of 16 cells.
PATCHES = (
    (1, ((0, 4),), SIZE - 1),  # at the lower wall
    (1, ((4, 14),), tuple(sorted({0, SIZE - 1}))),  # touches the one before
    (1, ((14, 24),), 1 % SIZE),  # and the next one
    (1, ((26, 32),), 0),  # at the upper wall
    (2, ((12, 24),), None),  # every rank
    (2, ((24, 40),), (SIZE - 1,)),  # over two level-1 patches
)


def hierarchy(
    centering: str = "cell", periodic: bool = False, halo: int = HALO, **field
) -> AMRHierarchy:
    mesh = AMRHierarchy(base_shape=(16,), periodic=(periodic,), comm=WORLD)
    for level, box, ranks in PATCHES:
        mesh.add_patch(level=level, box=box, ranks=ranks)
    mesh.allocate_field("u", halo=halo, centering=centering, **field)
    return mesh


def linear(x):
    return 1.0 + 2.0 * x


def wave(x):
    return 1.5 + 0.3 * np.cos(2.0 * np.pi * x)


def own_points(patch, nodal: bool) -> tuple[int, int]:
    ((lo, hi),) = patch.box
    return lo, hi + (1 if nodal else 0)


def contains(mesh: AMRHierarchy, level: int, i, nodal: bool):
    """Whether level points ``i`` are points of a patch (wrapped on a periodic line)."""
    n = mesh.level_shape(level)[0]
    candidates = (i % n, i % n + n) if mesh.periodic[0] else (i,)
    found = np.zeros(np.shape(i), dtype=bool)
    for patch in mesh.patches_at_level(level):
        lo, hi = own_points(patch, nodal)
        for ii in candidates:
            found |= (ii >= lo) & (ii < hi)
    return found


def storage_points(patch, name: str = "u"):
    """Level indices of every point of this rank's storage of field ``name``."""
    layout = patch.field(name).layout
    ((lo, _),) = patch.box
    ((b0, b1),) = layout.index_bounds
    (h,) = layout.halo
    return lo + np.arange(b0 - h, b1 + h)


def coordinates(mesh: AMRHierarchy, level: int, i, nodal: bool):
    (dx,) = mesh.spacing(level)
    return (i + (0.0 if nodal else 0.5)) * dx


def fill(mesh: AMRHierarchy, function, nodal: bool, name: str = "u") -> None:
    """``function`` on the patches' own points; the halos hold the sentinel."""
    for patch in mesh.local_patches:
        i = storage_points(patch, name)
        lo, hi = own_points(patch, nodal)
        own = (i >= lo) & (i < hi)
        storage = patch.field(name).local_with_halos
        storage[...] = SENTINEL
        storage[own] = function(coordinates(mesh, patch.level, i[own], nodal))


# The hierarchy ------------------------------------------------------------------ #


def test_a_line_is_a_hierarchy_of_one_axis() -> None:
    mesh = AMRHierarchy(base_shape=(8,), bounds=((-1.0, 3.0),), comm=WORLD)
    patch = mesh.add_patch(level=1, box=((4, 10),), ranks=0)

    assert mesh.ndim == 1
    assert mesh.periodic == (False,)
    assert mesh.spacing(1) == (0.25,)
    assert mesh.level_shape(2) == (32,)
    assert mesh.lower_bound == (-1.0,) and mesh.upper_bound == (3.0,)
    assert patch.shape == (6,) and patch.node_shape == (7,)
    assert patch.physical_bounds(mesh) == ((0.0, 1.5),)
    np.testing.assert_allclose(patch.node_coordinates(mesh)[0], np.arange(7) * 0.25)
    np.testing.assert_allclose(mesh.cell_volumes(1, np.arange(3)), 0.25)
    mesh.allocate_field("u")
    mesh.allocate_field("phi", centering="node")
    for local in mesh.local_patches:
        (x,) = local.local_cell_centers(mesh, "u")
        (xn,) = local.local_node_coordinates(mesh, "phi")
        assert x.shape == local.field("u").local.shape
        assert xn.shape == local.field("phi").local.shape
    assert AMRHierarchy((4,), comm=WORLD).bounds == ((0.0, 1.0),)


def test_patches_of_a_line_are_checked() -> None:
    mesh = AMRHierarchy(base_shape=(8,), comm=WORLD)
    mesh.add_patch(level=1, box=((4, 8),))
    with pytest.raises(ValueError, match="align"):
        mesh.add_patch(level=1, box=((9, 12),))
    with pytest.raises(ValueError, match="overlaps"):
        mesh.add_patch(level=1, box=((6, 10),))
    with pytest.raises(ValueError, match="not contained"):
        mesh.add_patch(level=2, box=((4, 12),))
    with pytest.raises(ValueError, match="patch box"):
        mesh.add_patch(level=1, box=((0, 2), (0, 2)))
    with pytest.raises(ValueError, match="one pair per axis"):
        AMRHierarchy(base_shape=(8,), bounds=((0.0, 1.0), (0.0, 1.0)))
    with pytest.raises(ValueError, match="one flag per axis"):
        AMRHierarchy(base_shape=(8,), periodic=(False, False))
    with pytest.raises(ValueError, match="r–z"):
        AMRHierarchy(base_shape=(8,), geometry="cylindrical")
    with pytest.raises(ValueError, match="one or two"):
        AMRHierarchy(base_shape=(4, 4, 4))


def test_locate_on_a_line() -> None:
    mesh = hierarchy()
    patches = mesh.patches
    x = np.array([0.01, 0.2, 0.25, 0.4, 0.55, 0.62, 0.99, 1.0, -0.1, 1.1, np.nan])
    found = mesh.locate(x)
    for point, number in zip(x[:8], found[:8], strict=True):
        patch = patches[number]
        ((x0, x1),) = patch.physical_bounds(mesh)
        assert x0 <= point <= x1
        # no finer patch holds it
        finer = mesh.locate(np.array([point]))
        assert finer[0] == number
    assert (
        patches[found[2]].level == 2
    )  # an edge belongs to the patch that starts there
    assert patches[found[7]].level == 1  # the upper face to the last cell
    np.testing.assert_array_equal(found[8:], -1)
    assert mesh.locate(x.reshape(1, -1)).shape == (1, x.size)
    with pytest.raises(ValueError, match="1 coordinate"):
        mesh.locate(x, x)

    wrapped = hierarchy(periodic=True)
    np.testing.assert_array_equal(
        wrapped.locate(np.array([1.3, -0.7, 0.3])), wrapped.locate(np.array([0.3] * 3))
    )


# Halos and restriction ---------------------------------------------------------- #


@pytest.mark.parametrize("centering", ["cell", "node"])
@pytest.mark.parametrize("periodic", [False, True], ids=["walls", "periodic"])
def test_every_halo_point_gets_its_value(centering, periodic) -> None:
    nodal = centering == "node"
    mesh = hierarchy(centering, periodic)
    function = wave if periodic else linear
    fill(mesh, function, nodal)

    mesh.exchange_halos("u")

    counts = {"copied": 0, "interpolated": 0}
    for patch in mesh.local_patches:
        level = patch.level
        i = storage_points(patch)
        storage = np.asarray(patch.field("u").local_with_halos)
        expected = function(coordinates(mesh, level, i, nodal))
        lo, hi = own_points(patch, nodal)
        ghost = (i < lo) | (i >= hi)
        n = mesh.level_shape(level)[0]
        outside = ((i < 0) | (i >= n + (1 if nodal else 0))) & (not periodic)
        copied = ghost & ~outside & contains(mesh, level, i, nodal)
        rest = ghost & ~outside & ~copied

        np.testing.assert_array_equal(storage[~ghost], expected[~ghost])
        np.testing.assert_allclose(
            storage[copied], expected[copied], rtol=0, atol=1e-12
        )
        assert np.all(storage[ghost & outside] == SENTINEL)
        if level == 0:
            assert not rest.any()
        tolerance = 0.02 if periodic else 1e-12  # linear interpolation is exact
        np.testing.assert_allclose(
            storage[rest], expected[rest], rtol=0, atol=tolerance
        )
        counts["copied"] += int(copied.sum())
        counts["interpolated"] += int(rest.sum())
    totals = {key: WORLD.allreduce(value, op=MPI.SUM) for key, value in counts.items()}
    assert all(totals.values()), totals


def restricted(mesh: AMRHierarchy, level: int, i, values, nodal: bool):
    """What restriction leaves at points ``i`` of ``level``: the finest values below.

    ``values(level, i)`` is what every level holds before restriction.
    """
    ratio = mesh.refinement_ratio
    result = values(level, i)
    if level == mesh.max_level:
        return result
    if nodal:
        under = contains(mesh, level + 1, i * ratio, nodal)
        finer = restricted(mesh, level + 1, i * ratio, values, nodal)
        return np.where(under, finer, result)
    children = [i * ratio + k for k in range(ratio)]
    under = np.all([contains(mesh, level + 1, c, nodal) for c in children], axis=0)
    finer = np.mean(
        [restricted(mesh, level + 1, c, values, nodal) for c in children], axis=0
    )
    return np.where(under, finer, result)


@pytest.mark.parametrize("centering", ["cell", "node"])
def test_restriction_goes_down_level_by_level(centering) -> None:
    nodal = centering == "node"
    mesh = hierarchy(centering)
    offsets = {0: SENTINEL, 1: 100.0, 2: 0.0}

    def values(level, i):
        return linear(coordinates(mesh, level, i, nodal)) + offsets[level]

    for patch in mesh.local_patches:
        i = storage_points(patch)
        patch.field("u").local_with_halos[...] = values(patch.level, i)

    mesh.restrict("u")

    for patch in mesh.local_patches:
        i = storage_points(patch)[HALO:-HALO]
        stored = np.asarray(patch.field("u").local_with_halos)[HALO:-HALO]
        expected = restricted(mesh, patch.level, i, values, nodal)
        np.testing.assert_allclose(stored, expected, rtol=0, atol=1e-9)


# Regridding ----------------------------------------------------------------------- #


def covered_cells(size: int, boxes) -> np.ndarray:
    count = np.zeros(size, dtype=int)
    for ((i0, i1),) in boxes:
        count[i0:i1] += 1
    return count


def test_clusters_on_a_line() -> None:
    tags = np.zeros(40, dtype=bool)
    tags[[2, 3, 4, 10, 11, 33]] = True
    tags[20:28] = True

    assert cluster(tags) == [((2, 5),), ((10, 12),), ((20, 28),), ((33, 34),)]
    assert cluster(tags, max_size=3)[2:5] == [((20, 22),), ((22, 25),), ((25, 28),)]
    # a box with a gap of one cell is kept at a low efficiency
    assert cluster(tags[:12], efficiency=0.3) == [((2, 12),)]
    widened = cluster(tags, min_size=4)
    assert all(hi - lo >= 4 for ((lo, hi),) in widened)
    assert covered_cells(40, widened).max() == 1
    allowed = np.ones(40, dtype=bool)
    allowed[3] = False
    assert ((3, 4),) not in cluster(tags, allowed=allowed)
    assert not covered_cells(40, cluster(tags, allowed=allowed))[3]


@pytest.mark.parametrize("seed", range(4))
def test_random_clusters_cover_the_tags_without_overlapping(seed) -> None:
    rng = np.random.default_rng(seed)
    tags = rng.random(200) < 0.15
    boxes = cluster(tags, efficiency=0.6, max_size=12)
    count = covered_cells(200, boxes)
    assert count.max() <= 1
    assert np.all(count[tags] == 1)
    assert all(hi - lo <= 12 for ((lo, hi),) in boxes)


def regrid_mesh(**settings) -> AMRHierarchy:
    mesh = AMRHierarchy(base_shape=(32,), comm=WORLD, **settings)
    mesh.allocate_field("c", halo=1)
    mesh.allocate_field("n", halo=1, centering="node")
    mesh.allocate_field("rho", halo=1, conservative=True)
    return mesh


def fill_all(mesh: AMRHierarchy, function) -> None:
    for name, nodal in (("c", False), ("n", True), ("rho", False)):
        fill(mesh, function, nodal, name)


def assert_properly_nested(mesh: AMRHierarchy, nesting: int) -> None:
    ratio = mesh.refinement_ratio
    for level in range(1, mesh.max_level + 1):
        below = covered_cells(
            mesh.level_shape(level - 1)[0],
            [p.box for p in mesh.patches_at_level(level - 1)],
        )
        n = below.size
        for patch in mesh.patches_at_level(level):
            ((lo, hi),) = patch.box
            assert lo % ratio == 0 and hi % ratio == 0
            for cell in range(lo // ratio - nesting, hi // ratio + nesting):
                if 0 <= cell < n:
                    assert below[cell], (level, patch.box)


@pytest.mark.parametrize("placement", ["all", "balanced"])
def test_regridding_follows_the_tags_and_moves_the_fields_exactly(placement) -> None:
    mesh = regrid_mesh(placement=placement)
    fill_all(mesh, linear)

    for centre in (0.3, 0.5, 0.7):
        tags = function_tags(mesh, lambda x, c=centre: abs(x - c) < 0.08, levels=3)
        mesh = mesh.regrid(tags, max_patch_size=16)

        assert mesh.max_level == 3
        assert_properly_nested(mesh, nesting=1)
        assert all(p.box[0][1] - p.box[0][0] <= 16 for p in mesh.patches if p.level)
        for level, level_tags in tags.items():
            fine = covered_cells(
                mesh.level_shape(level + 1)[0],
                [p.box for p in mesh.patches_at_level(level + 1)],
            )
            assert fine.reshape(-1, mesh.refinement_ratio).any(axis=1)[level_tags].all()
        # copied from old patches or interpolated from the new level below:
        # both exact for a linear function
        # (this rank's block: the halos are not filled)
        for name, nodal in (("c", False), ("n", True)):
            for patch in mesh.local_patches:
                ((lo, _),) = patch.box
                ((b0, b1),) = patch.field(name).layout.index_bounds
                i = lo + np.arange(b0, b1)
                stored = np.asarray(patch.field(name).local)
                expected = linear(coordinates(mesh, patch.level, i, nodal))
                np.testing.assert_allclose(stored, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("periodic", [False, True])
def test_a_conservative_field_keeps_its_integral_through_regrids(periodic) -> None:
    mesh = regrid_mesh(periodic=(periodic,), placement="balanced")

    for centre in (0.3, 0.45, 0.6, 0.6):
        # each level filled on its own: the coarse cells under a patch do not
        # hold the means of its cells until restriction
        fill(
            mesh,
            lambda x, c=centre: np.exp(-(((x - c) / 0.05) ** 2)) + 0.1 + 0.0 * x,
            False,
            "rho",
        )
        total = mesh.integral("rho")
        low = min(
            (float(np.min(p.field("rho").local)) for p in mesh.local_patches),
            default=np.inf,
        )
        tags = function_tags(mesh, lambda x, c=centre: abs(x - c) < 0.1, levels=2)
        mesh = mesh.regrid(tags)

        assert mesh.integral("rho") == pytest.approx(total, rel=1e-13)
        smallest = min(
            (float(np.min(p.field("rho").local)) for p in mesh.local_patches),
            default=np.inf,
        )
        assert (
            WORLD.allreduce(smallest, op=MPI.MIN)
            >= WORLD.allreduce(low, op=MPI.MIN) - 1e-14
        )


def test_the_integral_of_cell_and_node_fields_on_a_line() -> None:
    mesh = regrid_mesh()
    mesh = mesh.regrid(function_tags(mesh, lambda x: x > 0.6, levels=2))
    fill_all(mesh, linear)
    # the integral of 1 + 2x over [0, 1]; the trapezoidal rule is exact for it
    assert mesh.integral("c") == pytest.approx(2.0, rel=1e-13)
    assert mesh.integral("n") == pytest.approx(2.0, rel=1e-13)


def test_gradient_tags_find_a_step_on_every_rank() -> None:
    mesh = regrid_mesh()
    mesh = mesh.regrid(function_tags(mesh, lambda x: abs(x - 0.5) < 0.2, levels=1))
    fill_all(mesh, lambda x: np.where(x < 0.5, 0.0, 1.0))
    for name in ("c", "n"):
        tags = gradient_tags(mesh, name, 0.5)
        for level, values in tags.items():
            middle = mesh.level_shape(level)[0] // 2
            # cells: the two on either side of the step; nodes: the cell whose
            # upper node is the first one at 1
            expected = {middle - 1, middle} if name == "c" else {middle - 1}
            found = {int(k) for k in np.flatnonzero(values)}
            assert set().union(*WORLD.allgather(found)) == expected


# Markers ---------------------------------------------------------------------------- #


def markers(count: int = 400):
    """The same markers on every rank: positions, charges and ids, some on edges."""
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.random(count), np.arange(65) / 64.0, [1.0, 0.0, 0.5]])
    q = rng.normal(size=x.size)
    return x, q, np.arange(x.size)


def share(*arrays):
    """This rank's part of arrays that every rank has: every ``SIZE``-th marker."""
    return tuple(array[RANK::SIZE] for array in arrays)


def test_owner_is_a_rank_of_the_patch_and_the_same_on_every_rank() -> None:
    mesh = hierarchy("node", halo=1)
    x, _, _ = markers()
    owners = mesh.owner(x, "u")
    found = mesh.locate(x)
    patches = mesh.patches
    for owner, number in zip(owners, found, strict=True):
        assert owner in patches[number].ranks
    gathered = WORLD.allgather(owners)
    assert all(np.array_equal(gathered[0], other) for other in gathered)
    np.testing.assert_array_equal(mesh.owner(x, name="u"), owners)
    assert mesh.owner(np.array([1.5]), "u")[0] == -1


def test_migrate_moves_every_marker_to_its_owner_and_loses_none() -> None:
    mesh = hierarchy("node", halo=1)
    x, q, ids = share(*markers())

    x, q, ids = mesh.migrate(x, q, ids, name="u")

    assert np.all(mesh.owner(x, "u") == RANK)
    everyone = np.sort(np.concatenate(WORLD.allgather(ids)))
    np.testing.assert_array_equal(everyone, markers()[2])
    _, q_all, _ = markers()
    np.testing.assert_array_equal(q, q_all[ids])
    with pytest.raises(ValueError, match="outside"):
        mesh.migrate(np.array([2.0]), name="u")
    (kept,) = mesh.migrate(np.array([2.0]), name="u", outside="drop")
    assert kept.size == 0


@pytest.mark.parametrize("halo", [1, 2])
@pytest.mark.parametrize("periodic", [False, True])
def test_the_deposit_gives_the_charge_of_the_composite_nodes(halo, periodic) -> None:
    mesh = hierarchy("node", periodic=periodic, halo=halo)
    x, q, _ = markers()
    expected = CompositeNodes(mesh).deposit(x[None, :], q)
    mine_x, mine_q = mesh.migrate(*share(x, q), name="u")

    mesh.deposit(mine_x, mine_q, "u")
    mesh.accumulate_halos("u")

    deposited = CompositeNodes(mesh).from_patches("u")
    np.testing.assert_allclose(deposited, expected, rtol=0, atol=1e-12)
    assert deposited.sum() == pytest.approx(q.sum(), rel=1e-12)
    # added to, not replaced
    mesh.deposit(mine_x, mine_q, name="u", reset=False)
    with pytest.raises(TypeError, match="takes"):
        mesh.deposit(mine_x, mine_x, mine_q, "u")


def test_markers_follow_their_patch_after_a_regrid() -> None:
    mesh = regrid_mesh(placement="balanced")
    mesh = mesh.regrid(function_tags(mesh, lambda x: abs(x - 0.3) < 0.1, levels=2))
    x, q, ids = share(*markers())
    x, q, ids = mesh.migrate(x, q, ids, name="n")

    moved = mesh.regrid(function_tags(mesh, lambda x: abs(x - 0.7) < 0.1, levels=2))
    x, q, ids = moved.migrate(x, q, ids, name="n")

    assert np.all(moved.owner(x, "n") == RANK)
    everyone = np.sort(np.concatenate(WORLD.allgather(ids)))
    np.testing.assert_array_equal(everyone, markers()[2])


def test_forbidden_cells_of_a_line_are_never_covered() -> None:
    mesh = regrid_mesh()

    mesh = mesh.regrid(
        function_tags(mesh, lambda x: x < 0.6, levels=2),
        forbid=lambda x: (x > 0.2) & (x < 0.3),
        min_patch_size=4,
    )

    assert mesh.max_level == 2
    for patch in mesh.patches:
        if patch.level:
            ((x0, x1),) = patch.physical_bounds(mesh)
            assert x1 <= 0.2 or x0 >= 0.3, patch.box
            assert patch.box[0][1] - patch.box[0][0] >= 4
