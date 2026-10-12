"""Tests for the patch hierarchy and the cell-centered solver on it."""

from __future__ import annotations

import numpy as np
import pytest

from patchwork import AMRHierarchy, plot_patches, solve_poisson

ELL = [((0, 8), (0, 16)), ((8, 16), (8, 16))]


def test_the_base_level_can_be_several_boxes() -> None:
    mesh = AMRHierarchy(base_shape=(16, 16), base_boxes=ELL)

    assert [patch.box for patch in mesh.patches_at_level(0)] == ELL
    assert sum(patch.shape[0] * patch.shape[1] for patch in mesh) == 192
    with pytest.raises(ValueError, match="overlaps another base box"):
        AMRHierarchy(base_shape=(16, 16), base_boxes=[*ELL, ((4, 12), (4, 12))])
    with pytest.raises(ValueError, match="exceeds the base shape"):
        AMRHierarchy(base_shape=(16, 16), base_boxes=[((0, 8), (0, 20))])
    with pytest.raises(ValueError, match="at least one box"):
        AMRHierarchy(base_shape=(16, 16), base_boxes=[])


def test_a_patch_lies_within_the_patches_of_the_level_below() -> None:
    mesh = AMRHierarchy(base_shape=(16, 16), base_boxes=ELL)

    # over both base boxes, above the corner
    mesh.add_patch(level=1, box=((12, 20), (20, 28)))
    with pytest.raises(ValueError, match="not contained"):
        # reaches into the quadrant the domain leaves out
        mesh.add_patch(level=1, box=((12, 20), (8, 20)))
    with pytest.raises(ValueError, match="not contained"):
        # level 2 needs level 1 below all of it
        mesh.add_patch(level=2, box=((24, 44), (40, 56)))
    mesh.add_patch(level=2, box=((28, 36), (44, 52)))
    assert mesh.max_level == 2


def _solve_sine(mesh: AMRHierarchy) -> float:
    """Largest error of the cell-centered solve of sin(pi x) sin(pi y)."""

    def exact(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    mesh.allocate_field("rhs")
    for patch in mesh:
        x, y = patch.local_cell_centers(mesh)
        patch.field("rhs").local[...] = 2.0 * np.pi**2 * exact(x, y)
    solve_poisson(mesh, "rhs", potential="phi")
    return max(
        float(
            np.abs(
                patch.field("phi").local - exact(*patch.local_cell_centers(mesh))
            ).max()
        )
        for patch in mesh
    )


def test_the_cell_centered_solve_grounds_the_inner_edges_of_an_l() -> None:
    """sin(pi x) sin(pi y) vanishes on x = 0 and y = 0: the inner edges of the L."""
    errors = []
    for base in (16, 32):
        half = base // 2
        mesh = AMRHierarchy(
            base_shape=(base, base),
            bounds=((-1.0, 1.0), (-1.0, 1.0)),
            base_boxes=[((0, half), (0, base)), ((half, base), (half, base))],
        )
        errors.append(_solve_sine(mesh))

    assert errors[0] / errors[1] > 3.5
    assert errors[1] < 5e-3


def test_the_cell_centered_solve_takes_a_patch_over_two_base_boxes() -> None:
    mesh = AMRHierarchy(
        base_shape=(16, 16), bounds=((-1.0, 1.0), (-1.0, 1.0)), base_boxes=ELL
    )
    mesh.add_patch(level=1, box=((12, 20), (20, 28)))

    assert _solve_sine(mesh) < 2e-2
    # the base cells under the patch hold the mean of its cells, in both boxes
    fine = mesh.patches[-1].field("phi").local
    means = fine.reshape(4, 2, 4, 2).mean(axis=(1, 3))
    left, right = (patch.field("phi").local for patch in mesh.patches_at_level(0))
    assert left[6:8, 10:14] == pytest.approx(means[:2])
    assert right[0:2, 2:6] == pytest.approx(means[2:])


def test_patches_of_an_l_are_plotted() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    mesh = AMRHierarchy(base_shape=(16, 16), base_boxes=ELL)
    mesh.add_patch(level=1, box=((12, 20), (20, 28)))
    mesh.allocate_field("phi", centering="node")

    figure, axes = plot_patches(mesh, field="phi")

    assert len(axes.collections) >= 3  # one image per patch
    matplotlib.pyplot.close(figure)


def test_on_one_rank_every_patch_is_local_whatever_its_placement() -> None:
    for placement in ("all", "balanced"):
        mesh = AMRHierarchy(base_shape=(16, 16), base_boxes=ELL, placement=placement)
        patch = mesh.add_patch(level=1, box=((12, 20), (20, 28)))
        explicit = mesh.add_patch(level=1, box=((0, 8), (0, 8)), ranks=0)
        mesh.allocate_field("phi")

        assert all(p.ranks == (0,) and p.is_local for p in mesh.patches)
        assert mesh.local_patches == mesh.patches
        assert patch.comm is mesh.comm and explicit.comm is mesh.comm
        assert mesh.cells_on_rank == (192 + 64 + 64,)
        patch.field("phi").local[...] = 3.0
        assert np.array_equal(mesh.gather(patch, "phi"), np.full((8, 8), 3.0))
        assert np.array_equal(mesh.gather(patch, "phi", root=0), np.full((8, 8), 3.0))


def test_ranks_of_a_patch_are_checked() -> None:
    mesh = AMRHierarchy(base_shape=(16, 16))

    with pytest.raises(ValueError, match="not all ranks of a communicator of 1"):
        mesh.add_patch(level=1, box=((0, 8), (0, 8)), ranks=1)
    with pytest.raises(ValueError, match="distinct and not empty"):
        mesh.add_patch(level=1, box=((0, 8), (0, 8)), ranks=[0, 0])
    with pytest.raises(ValueError, match="distinct and not empty"):
        mesh.add_patch(level=1, box=((0, 8), (0, 8)), ranks=[])
    with pytest.raises(ValueError, match="placement must be"):
        AMRHierarchy(base_shape=(16, 16), placement="nearest")
    with pytest.raises(ValueError, match="one entry per base box"):
        AMRHierarchy(base_shape=(16, 16), base_boxes=ELL, base_ranks=[0])


def test_the_nodes_of_a_rank_come_with_their_coordinates() -> None:
    mesh = AMRHierarchy(base_shape=(4, 2), bounds=((0.0, 2.0), (0.0, 1.0)))
    mesh.allocate_field("phi", centering="node")
    (patch,) = mesh.patches

    x, y = patch.local_node_coordinates(mesh, "phi")

    assert x.shape == patch.field("phi").local.shape == (5, 3)
    assert x[:, 0] == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])
    assert y[0] == pytest.approx([0.0, 0.5, 1.0])
