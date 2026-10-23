"""The compiled kernels against their NumPy versions, and the node maps CompositeNodes keeps."""

from __future__ import annotations

import cunumpy as xp
import numpy as np
import pytest

from patchwork import AMRHierarchy
from patchwork.kernels.locate_points import kernel as locate_points
from patchwork.nodal import CompositeNodes

ELL = [((0, 8), (0, 16)), ((8, 16), (8, 16))]


def _pyccel_available() -> bool:
    try:
        with xp.kernels.use_host_kernel_implementation("pyccel"):
            AMRHierarchy(base_shape=(4, 4)).locate([0.5], [0.5])
    except Exception:  # noqa: BLE001 -- no Pyccel, or no compiler
        return False
    return True


IMPLEMENTATIONS = [
    "numpy",
    pytest.param(
        "pyccel",
        marks=pytest.mark.skipif(not _pyccel_available(), reason="needs Pyccel"),
    ),
]


def _meshes() -> dict[str, AMRHierarchy]:
    nested = AMRHierarchy(base_shape=(16, 12), bounds=((0.0, 2.0), (-1.0, 0.5)))
    nested.add_patch(level=1, box=((4, 20), (2, 18)))
    nested.add_patch(level=1, box=((20, 28), (6, 14)))  # touches the one before
    nested.add_patch(level=2, box=((12, 40), (8, 28)))  # over both
    periodic = AMRHierarchy(base_shape=(16, 12), periodic=(True, True))
    periodic.add_patch(level=1, box=((4, 14), (0, 24)))  # the whole periodic axis
    periodic.add_patch(level=1, box=((26, 32), (18, 24)))  # at the periodic faces
    periodic.add_patch(level=2, box=((52, 64), (36, 48)))
    ell = AMRHierarchy(base_shape=(16, 16), base_boxes=ELL, refinement_ratio=4)
    ell.add_patch(level=1, box=((24, 40), (40, 56)))  # over both base boxes
    line = AMRHierarchy(base_shape=(20,), bounds=((1.0, 3.0),), periodic=(True,))
    line.add_patch(level=1, box=((0, 12),))
    line.add_patch(level=2, box=((4, 16),))
    return {"nested": nested, "periodic": periodic, "ell": ell, "line": line}


def _points(mesh: AMRHierarchy) -> list[np.ndarray]:
    """Random points, some outside, on the finest lattice's lines, faces and NaN."""
    rng = np.random.default_rng(5)
    finest = mesh.level_shape(mesh.max_level)
    coordinates = []
    for axis, (low, high) in enumerate(mesh.bounds):
        length = high - low
        random = low + length * (1.6 * rng.random(600) - 0.3)
        lines = low + length * rng.integers(0, finest[axis] + 1, 200) / finest[axis]
        faces = np.array([low, high, low - length, high + length, np.nan, high])
        coordinates.append(np.concatenate([random, lines, faces]))
    if mesh.ndim == 2:  # the lines along one axis only, the other one random
        coordinates[1][600:700] = coordinates[1][:100]
    return coordinates


@pytest.mark.parametrize("name", ["nested", "periodic", "ell", "line"])
def test_the_compiled_locate_gives_what_the_numpy_one_gives(name: str) -> None:
    if not _pyccel_available():
        pytest.skip("needs Pyccel")
    mesh = _meshes()[name]
    points = _points(mesh)

    results = {}
    for implementation in ("numpy", "pyccel"):
        with xp.kernels.use_host_kernel_implementation(implementation):
            results[implementation] = mesh._cells(points)

    (found, cell, wrapped), (found_c, cell_c, wrapped_c) = results.values()
    assert np.array_equal(found, found_c)
    assert all(np.array_equal(a, b) for a, b in zip(cell, cell_c, strict=True))
    for a, b in zip(wrapped, wrapped_c, strict=True):
        assert np.allclose(a, b, rtol=0.0, atol=1e-14, equal_nan=True)
    assert (found >= 0).any() and (found < 0).any()


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("name", ["nested", "periodic", "ell", "line"])
def test_located_cells_are_the_leaf_cells_of_the_composite_mesh(
    name: str, implementation: str
) -> None:
    """The cell locate gives is the leaf cell that CompositeNodes puts the point in."""
    mesh = _meshes()[name]
    nodes = CompositeNodes(mesh)
    points = _points(mesh)
    with xp.kernels.use_host_kernel_implementation(implementation):
        found, cell, wrapped = mesh._cells(points)

    inside = found >= 0
    patches = mesh.patches
    ratio = mesh.refinement_ratio
    span = np.array([ratio ** (mesh.max_level - patches[n].level) for n in found])
    finest = mesh.level_shape(mesh.max_level)
    fine = tuple(  # the point's cell on the finest lattice
        np.minimum(((w[inside] - low) / h).astype(np.int64), n - 1)
        for w, (low, _), h, n in zip(
            wrapped, mesh.bounds, mesh.spacing(mesh.max_level), finest, strict=True
        )
    )
    leaf = nodes._leaf_cells()[fine]
    assert np.all(leaf >= 0)
    assert np.array_equal(nodes.cell_span[leaf], span[inside])
    for axis in range(mesh.ndim):
        assert np.array_equal(
            nodes.cell_lower[leaf, axis], cell[axis][inside] * span[inside]
        )
        low, high = mesh.bounds[axis]
        assert np.all((wrapped[axis][inside] >= low) & (wrapped[axis][inside] <= high))
    for number in np.unique(found[inside]):
        box = patches[number].box
        for axis, (lo, hi) in enumerate(box):
            chosen = cell[axis][found == number]
            assert np.all((chosen >= lo) & (chosen < hi))


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_locate_takes_no_points(implementation: str) -> None:
    mesh = _meshes()["nested"]
    with xp.kernels.use_host_kernel_implementation(implementation):
        assert mesh.locate(np.zeros(0), np.zeros(0)).shape == (0,)


def test_locate_uses_the_compiled_kernel_when_pyccel_is_installed() -> None:
    if not _pyccel_available():
        pytest.skip("needs Pyccel")
    _meshes()["nested"].locate([0.5], [0.0])
    assert locate_points.selected() == "pyccel"


@pytest.mark.parametrize("name", ["nested", "periodic", "ell", "line"])
def test_from_patches_reads_the_finest_and_the_upper_seam_copy(name: str) -> None:
    """Different values in every copy of a node: the one read is the old rule's."""
    mesh = _meshes()[name]
    mesh.allocate_field("u", centering="node")
    nodes = CompositeNodes(mesh)
    rng = np.random.default_rng(2)
    expected = np.zeros(nodes.num_nodes)
    for patch in mesh.patches:  # coarse to fine, the last copy written wins
        values = rng.random(patch.node_shape)
        patch.field("u").local[...] = values
        expected[nodes.patch_nodes(patch)] = values

    assert np.array_equal(nodes.from_patches("u"), expected)
    # and a second time, from the maps found at the first call
    assert np.array_equal(nodes.from_patches("u"), expected)


@pytest.mark.parametrize("name", ["nested", "periodic", "ell", "line"])
def test_patch_nodes_are_kept_and_read_only(name: str) -> None:
    mesh = _meshes()[name]
    mesh.allocate_field("u", centering="node")
    nodes = CompositeNodes(mesh)
    values = np.random.default_rng(4).random(nodes.num_nodes)

    for patch in mesh.patches:
        first = nodes.patch_nodes(patch)
        assert first.shape == patch.node_shape
        assert nodes.patch_nodes(patch) is first
        with pytest.raises(ValueError, match="read-only"):
            first[(0,) * mesh.ndim] = 0
    nodes.to_patches(values, "u")
    assert np.array_equal(nodes.from_patches("u"), values)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("name", ["nested", "periodic", "ell"])
def test_the_compiled_deposit_is_the_transposed_evaluation(
    name: str, implementation: str
) -> None:
    """CompositeNodes.deposit (a kernel in 2D) against the weights of cell_weights."""
    mesh = _meshes()[name]
    nodes = CompositeNodes(mesh)
    points = np.stack(_points(mesh))
    charges = np.random.default_rng(8).normal(size=points.shape[1])
    valid = ~np.isnan(points).any(axis=0)
    corners, weights = nodes.cell_weights(points[:, valid])
    expected = np.bincount(
        corners.reshape(-1),
        weights=(weights * charges[valid, None]).reshape(-1),
        minlength=nodes.num_nodes,
    )

    with xp.kernels.use_host_kernel_implementation(implementation):
        charge = nodes.deposit(points, charges)  # NaN points deposit nothing
        assert np.array_equal(nodes.deposit(points[:, :0], charges[:0]), 0 * charge)

    assert np.allclose(charge, expected, rtol=0.0, atol=1e-12)
    if name != "periodic":  # some points are outside
        assert np.any(weights.sum(axis=1) == 0.0)
