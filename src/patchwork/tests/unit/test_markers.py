"""Markers on patches: who owns a marker, moving markers, the deposit of their charge.

Runs serially and under ``mpirun -n N python -m pytest`` (see ``test_mpi.py``). Every rank
makes the same global markers and keeps its share, so the checks compare with what one
rank computes for all of them (:meth:`CompositeNodes.deposit`).
"""

from __future__ import annotations

import numpy as np
import pytest
from maybempi import MPI

from patchwork import AMRHierarchy, function_tags
from patchwork.nodal import CompositeNodes

WORLD = MPI.COMM_WORLD
SIZE = WORLD.Get_size()
RANK = WORLD.Get_rank()

#: Level, box and ranks of the fine patches, on a 16 x 12 base periodic in y.
PATCHES = (
    (1, ((4, 14), (0, 24)), tuple(sorted({0, SIZE - 1}))),  # along the periodic axis
    (1, ((14, 24), (6, 18)), 1 % SIZE),  # touches the one before
    (1, ((0, 4), (8, 16)), SIZE - 1),  # at the lower x wall
    (1, ((26, 32), (18, 24)), 0),  # at the upper x wall and the periodic face
    (2, ((12, 24), (8, 20)), None),  # every rank
    (2, ((24, 32), (16, 24)), (SIZE - 1,)),  # over two level-1 patches
)


def hierarchy(halo: int = 1, placed: bool = True, **field) -> AMRHierarchy:
    mesh = AMRHierarchy(base_shape=(16, 12), periodic=(False, True), comm=WORLD)
    for level, box, ranks in PATCHES:
        mesh.add_patch(level=level, box=box, ranks=ranks if placed else None)
    mesh.allocate_field("q", centering="node", halo=halo, **field)
    return mesh


def markers(count: int = 700) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The same markers on every rank: positions, charges and ids (all in the domain).

    Random ones, and some on the edges: on the nodes of the finest lattice (the edges of
    the patches), on the upper face of the domain and across the periodic seam.
    """
    rng = np.random.default_rng(11)
    x = rng.random(count)
    y = rng.random(count)
    i = rng.integers(0, 65, 120)
    j = rng.integers(0, 49, 120)
    x = np.concatenate([x, i / 64.0, [1.0, 1.0, 0.0, 0.5, 0.5]])
    y = np.concatenate([y, j / 48.0, [0.3, 1.0, 0.0, 1.0 - 1e-12, 1e-12]])
    y = np.concatenate([y, y[:40] + 1.0, y[40:80] - 1.0])  # the same ones, a period off
    x = np.concatenate([x, x[:40], x[40:80]])
    q = rng.normal(size=x.size)
    return x, y, q, np.arange(x.size)


def share(*arrays):
    """This rank's part of arrays that every rank has: every ``SIZE``-th marker."""
    mine = np.arange(arrays[0].size) % SIZE == RANK
    return tuple(array[mine] for array in arrays)


def test_owner_is_a_rank_of_the_patch_and_the_same_on_every_rank() -> None:
    mesh = hierarchy()
    x, y, _, _ = markers()
    x = np.concatenate([x, [-0.2, 1.3, np.nan]])  # outside the domain
    y = np.concatenate([y, [0.5, 0.5, 0.5]])

    ranks = mesh.owner(x, y, "q")

    found = mesh.locate(x, y)
    assert ranks.shape == x.shape
    assert np.array_equal(ranks, WORLD.bcast(ranks, root=0))
    assert np.all(ranks[found < 0] == -1)
    assert np.all(found[ranks < 0] < 0)
    patches = mesh.patches
    for number, owner in zip(found, ranks, strict=True):
        if number >= 0:
            assert int(owner) in patches[number].ranks


def test_owner_follows_the_blocks_of_the_field() -> None:
    mesh = hierarchy()
    x, y, _, _ = markers()
    found = mesh.locate(x, y)
    ranks = mesh.owner(x, y, "q")

    # a patch on several ranks is split along its first axis: the owner is the rank
    # whose block holds the first index of the cell's lower node
    for number, patch in enumerate(mesh.patches):
        if len(patch.ranks) < 2:
            continue
        chosen = np.flatnonzero(found == number)
        cell = np.floor(x[chosen] / mesh.spacing(patch.level)[0]).astype(int)
        cell = np.minimum(cell, mesh.level_shape(patch.level)[0] - 1) - patch.box[0][0]
        order = np.argsort(cell, kind="stable")
        assert np.all(np.diff(ranks[chosen][order]) >= 0)  # ranks in order along x


def test_migrate_moves_every_marker_to_its_owner_and_loses_none() -> None:
    mesh = hierarchy()
    x, y, q, ids = markers()
    weights = np.stack([q, 2.0 * q, 3.0 * q], axis=1)

    nx, ny, nq, nweights, nids = mesh.migrate(*share(x, y, q, weights, ids), name="q")

    assert np.all(mesh.owner(nx, ny, "q") == RANK)
    # the arrays travel together
    assert np.array_equal(nx, x[nids]) and np.array_equal(ny, y[nids])
    assert np.array_equal(nq, q[nids]) and np.array_equal(nweights, weights[nids])
    everyone = np.sort(np.concatenate(WORLD.allgather(nids)))
    assert np.array_equal(everyone, ids)  # none lost, none twice
    # the checksum of the charge
    total = WORLD.allreduce(float(nq.sum()), op=MPI.SUM)
    assert total == pytest.approx(float(q.sum()), abs=1e-9)


def test_markers_outside_the_domain_are_an_error_or_dropped() -> None:
    mesh = hierarchy()
    x, y, q, ids = markers(50)
    x = np.concatenate([x, [-0.5, 1.5]])
    y = np.concatenate([y, [0.5, 0.5]])
    ids = np.arange(x.size)

    with pytest.raises(ValueError, match="2 markers are outside"):
        mesh.migrate(*share(x, y, ids), name="q")
    with pytest.raises(ValueError, match="outside must be"):
        mesh.migrate(*share(x, y, ids), name="q", outside="keep")
    nx, ny, nids = mesh.migrate(*share(x, y, ids), name="q", outside="drop")

    everyone = np.sort(np.concatenate(WORLD.allgather(nids)))
    assert np.array_equal(everyone, ids[:-2])


def test_migrate_checks_its_arrays_and_works_with_no_markers() -> None:
    mesh = hierarchy()
    with pytest.raises(ValueError, match="one row per marker"):
        mesh.migrate(np.zeros(3), np.zeros(3), np.zeros(4), name="q")
    with pytest.raises(ValueError, match="one-dimensional"):
        mesh.migrate(np.zeros((2, 2)), np.zeros((2, 2)), name="q")

    nx, ny, nq = mesh.migrate(np.empty(0), np.empty(0), np.empty(0), name="q")

    assert nx.size == ny.size == nq.size == 0


def test_markers_follow_their_patch_to_another_rank_after_a_regrid() -> None:
    mesh = hierarchy()
    x, y, q, ids = markers()
    nx, ny, nq, nids = mesh.migrate(*share(x, y, q, ids), name="q")
    tags = function_tags(
        mesh, lambda a, b: (a > 0.55) & (a < 0.9) & (b > 0.1) & (b < 0.5), levels=1
    )

    # every new patch goes to the last rank
    new = mesh.regrid(tags, placement=lambda level, box: (SIZE - 1,))

    stay = new.owner(nx, ny, "q") == RANK
    mx, my, mq, mids = new.migrate(nx, ny, nq, nids, name="q")
    assert np.all(new.owner(mx, my, "q") == RANK)
    assert np.array_equal(mx, x[mids]) and np.array_equal(mq, q[mids])
    everyone = np.sort(np.concatenate(WORLD.allgather(mids)))
    assert np.array_equal(everyone, ids)
    if SIZE > 1:  # some markers had to move: the markers over the new patches
        moved = WORLD.allreduce(int((~stay).sum()), op=MPI.SUM)
        assert moved > 0


@pytest.mark.parametrize(
    ("halo", "placed"),
    [
        (1, True),
        (2, True),
        (1, False),
    ],  # a halo of 2 does not fit blocks of the small patches
    ids=["placed", "wide halo", "all ranks"],
)
def test_the_deposit_gives_the_charge_of_the_composite_nodes(halo, placed) -> None:
    mesh = hierarchy(halo=halo, placed=placed)
    x, y, q, _ = markers()
    nodes = CompositeNodes(mesh)
    expected = nodes.deposit(np.array([x, y]), q)

    nx, ny, nq = mesh.migrate(*share(x, y, q), name="q")
    mesh.deposit(nx, ny, nq, "q")
    mesh.accumulate_halos("q")

    found = nodes.from_patches("q")
    assert np.abs(found - expected).max() < 1e-12
    # the total is kept: summed over every copy of every node, and the halos are empty
    own = sum(float(patch.field("q").local.sum()) for patch in mesh.local_patches)
    assert WORLD.allreduce(own, op=MPI.SUM) == pytest.approx(float(q.sum()), abs=1e-9)
    for patch in mesh.local_patches:
        array = patch.field("q")
        assert array.local_with_halos.sum() == pytest.approx(
            float(array.local.sum()), abs=1e-12
        )


def test_the_charge_ends_on_the_finest_patch_that_holds_a_node() -> None:
    mesh = hierarchy()
    x, y, q, _ = markers()
    nx, ny, nq = mesh.migrate(*share(x, y, q), name="q")
    mesh.deposit(nx, ny, nq, "q")
    mesh.accumulate_halos("q")

    # the level-2 patch ((12, 24), (8, 20)) is over the level-1 nodes 6..12 by 4..10, in the
    # indices of the level-1 patch ((4, 14), (0, 24)): 2..8 by 4..10. Every one of them is also
    # a node of the fine patch, which owns it
    coarse = next(
        p for p in mesh.patches if p.level == 1 and p.box == ((4, 14), (0, 24))
    )
    values = mesh.gather(coarse, "q")
    assert np.all(values[2:9, 4:11] == 0.0)
    assert np.abs(values).sum() > 0.0  # elsewhere the coarse patch keeps its charge
    # and the fine ones hold the charge of the markers in them
    fine = next(
        p for p in mesh.patches if p.level == 2 and p.box == ((12, 24), (8, 20))
    )
    assert np.abs(mesh.gather(fine, "q")).sum() > 0.0


def test_a_second_deposit_replaces_the_first_unless_asked_to_add() -> None:
    mesh = hierarchy()
    x, y, q, _ = markers()
    nx, ny, nq = mesh.migrate(*share(x, y, q), name="q")
    nodes = CompositeNodes(mesh)
    expected = nodes.deposit(np.array([x, y]), q)

    mesh.deposit(nx, ny, nq, "q")
    mesh.deposit(nx, ny, nq, "q")  # reset: the same
    mesh.accumulate_halos("q")
    once = nodes.from_patches("q")
    mesh.deposit(nx, ny, nq, "q")
    mesh.deposit(nx, ny, nq, "q", reset=False)  # twice the charge
    mesh.accumulate_halos("q")
    twice = nodes.from_patches("q")

    assert np.abs(once - expected).max() < 1e-12
    assert np.abs(twice - 2.0 * expected).max() < 1e-12


def test_markers_outside_the_domain_deposit_nothing() -> None:
    mesh = hierarchy()
    x, y, q, _ = markers(80)
    nx, ny, nq = mesh.migrate(*share(x, y, q), name="q")
    nodes = CompositeNodes(mesh)
    expected = nodes.deposit(np.array([x, y]), q)

    # outside the domain: no owner, nothing deposited, no error
    mesh.deposit(
        np.concatenate([nx, [-0.3, 1.4]]),
        np.concatenate([ny, [0.5, 0.5]]),
        np.concatenate([nq, [100.0, 100.0]]),
        "q",
    )
    mesh.accumulate_halos("q")

    assert np.abs(nodes.from_patches("q") - expected).max() < 1e-12


def test_a_marker_on_the_wrong_rank_is_refused() -> None:
    if SIZE == 1:
        pytest.skip("on one rank every marker is on its owner")
    mesh = hierarchy()
    x, y, q, _ = markers()

    with pytest.raises(ValueError, match="rank that owns them"):
        mesh.deposit(x, y, q, "q")  # every rank has all of them


def test_the_deposit_checks_the_field_and_its_arguments() -> None:
    mesh = hierarchy()
    mesh.allocate_field("cells")
    mesh.allocate_field("bare", centering="node", halo=0)
    x, y, q, _ = markers(20)

    with pytest.raises(KeyError, match="no field"):
        mesh.deposit(x, y, q, "missing")
    with pytest.raises(ValueError, match="node-centered"):
        mesh.deposit(x, y, q, "cells")
    with pytest.raises(ValueError, match="halo of at least one"):
        mesh.deposit(x, y, q, "bare")
    with pytest.raises(ValueError, match="charges"):
        mesh.deposit(x, y, q[:-1], "q")
    with pytest.raises(NotImplementedError, match="node-centered"):
        mesh.accumulate_halos("cells")


def test_a_replicated_field_has_no_owner_of_a_point() -> None:
    if SIZE == 1:
        pytest.skip("a patch on one rank is not replicated")
    mesh = hierarchy(split=None)
    x, y, _, _ = markers(20)

    with pytest.raises(ValueError, match="replicated"):
        mesh.owner(x, y, "q")


def test_the_owners_are_found_again_after_a_patch_is_added() -> None:
    mesh = AMRHierarchy(base_shape=(16, 16), comm=WORLD)
    mesh.allocate_field("q", centering="node")
    x, y = np.array([0.5]), np.array([0.5])
    before = mesh.owner(x, y, "q")

    mesh.add_patch(level=1, box=((8, 24), (8, 24)), ranks=SIZE - 1)
    after = mesh.owner(x, y, "q")

    assert after[0] == SIZE - 1
    assert before[0] in range(SIZE)
