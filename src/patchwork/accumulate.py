"""Scatter-add of a node-centered deposit: every node's charge goes to one copy.

A particle code deposits the charge of its markers patch by patch (see
:meth:`AMRHierarchy.deposit <patchwork.mesh.AMRHierarchy.deposit>`). The result is not
yet the charge of the composite mesh (:class:`~patchwork.nodal.CompositeNodes`), for two
reasons:

1. a marker near the edge of a rank's block deposits into the halo of the block, and
2. a node of the composite mesh is held by several patches (the nodes of the cell
   corners that touch or nest, and the two copies of a node on a periodic seam), and
   each copy only has the charge of the markers whose cell is in its own patch.

:class:`Accumulation` fixes both. The halos are added into the neighbouring blocks with
mpiarray's ``accumulate_halos``. Then the charge of every copy of a node is moved to the
owner copy: the one in the finest patch that holds the node (the last one in
:attr:`AMRHierarchy.patches` among the patches of that level), which is the copy
:meth:`CompositeNodes.from_patches <patchwork.nodal.CompositeNodes.from_patches>` reads.
The other copies end up zero, so the sum over all copies is kept. The transport is one
``Alltoallv`` (:func:`mpiarray.migrate`) over the hierarchy's communicator.

The owner copy of every node is found with one table on the finest lattice of the whole
domain (one integer per lattice node, painted patch by patch from the coarsest).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mpiarray as mpa
import numpy as np

if TYPE_CHECKING:
    from patchwork.mesh import AMRHierarchy


class Accumulation:
    """Add the halos of node field ``name`` and move every node's charge to its owner copy.

    Built collectively (the ranks tell each other which block they hold), executed as
    often as needed. Collective over the hierarchy's communicator, as is :meth:`execute`.
    """

    def __init__(self, hierarchy: AMRHierarchy, name: str) -> None:
        """Find, for every node a rank holds, where its charge goes."""
        spec = hierarchy._field_specs[name]
        if spec["centering"] != "node":
            raise NotImplementedError("the scatter-add is for node-centered fields")
        self._hierarchy = hierarchy
        self._name = name
        ratio = hierarchy.refinement_ratio
        finest = hierarchy.max_level
        periodic = hierarchy.periodic
        lattice = np.asarray(hierarchy.level_shape(finest), dtype=np.int64)
        patches = hierarchy.patches
        ndim = hierarchy.ndim

        table = np.full(
            tuple(
                int(n) if wrap else int(n) + 1
                for n, wrap in zip(lattice, periodic, strict=True)
            ),
            -1,
            dtype=np.int32,
        )
        span = np.array([ratio ** (finest - patch.level) for patch in patches])
        lower = np.array([[lo for lo, _ in patch.box] for patch in patches])
        sizes = np.array([hierarchy.level_shape(patch.level) for patch in patches])
        extent = np.array([[hi - lo for lo, hi in patch.box] for patch in patches])
        for number, patch in enumerate(
            patches
        ):  # coarse to fine: the finest paints last
            axes = []
            for axis, (lo, hi) in enumerate(patch.box):
                index = np.arange(lo, hi + 1) * span[number]
                axes.append(index % lattice[axis] if periodic[axis] else index)
            table[np.ix_(*axes)] = number

        from patchwork.mesh import _gather_blocks

        blocks = hierarchy._schedule("blocks", name, _gather_blocks)
        # per local patch: the storage of the nodes that move, and where each goes
        self._moves: list[dict[str, Any]] = []
        for number, patch in enumerate(patches):
            if not patch.is_local:
                continue
            array = patch.field(name)
            node = np.meshgrid(
                *(np.arange(b0, b1) for b0, b1 in array.layout.index_bounds),
                indexing="ij",
            )
            index = [
                lower[number][axis] + node[axis] for axis in range(ndim)
            ]  # level indices
            key = [
                (index[axis] * span[number]) % lattice[axis]
                if periodic[axis]
                else index[axis] * span[number]
                for axis in range(ndim)
            ]
            owner = table[tuple(key)].astype(np.int64)
            if (owner < 0).any():
                raise RuntimeError(
                    "a node of a patch is not a node of the composite mesh"
                )
            target = []
            for axis in range(ndim):
                at = key[axis] // span[owner]
                if periodic[axis]:
                    # a node on the seam has two copies in a patch that spans it, at
                    # the start and at the end of the axis: the upper one is the owner
                    # (the one CompositeNodes.from_patches reads, the last it writes)
                    above = at + sizes[owner, axis]
                    inside = above <= lower[owner, axis] + extent[owner, axis]
                    at = np.where(inside, above, at)
                    at = np.where(at < lower[owner, axis], at + sizes[owner, axis], at)
                target.append(at - lower[owner, axis])
            if any(
                ((target[axis] < 0) | (target[axis] > extent[owner, axis])).any()
                for axis in range(ndim)
            ):
                raise RuntimeError("the owner copy of a node is outside its patch")
            moving = owner != number
            for axis in range(ndim):
                moving |= target[axis] != node[axis]
            where = np.nonzero(moving)
            owners = owner[where]
            at = [values[where] for values in target]
            destination = np.empty(owners.size, dtype=np.int64)
            for held in np.unique(owners):
                members, replicated = blocks[int(held)]
                if replicated:
                    raise ValueError(
                        f"field {name!r} is replicated (split=None) on the several ranks "
                        f"of a patch: allocate it with split=0 to accumulate"
                    )
                chosen = np.flatnonzero(owners == held)
                for rank, bounds in members:
                    inside = np.ones(chosen.size, dtype=bool)
                    for values, (a0, a1) in zip(at, bounds, strict=True):
                        inside &= (values[chosen] >= a0) & (values[chosen] < a1)
                    destination[chosen[inside]] = rank
            halo = array.layout.halo
            self._moves.append(
                {
                    "patch": number,
                    "array": array,
                    "storage": tuple(
                        values + h for values, h in zip(where, halo, strict=True)
                    ),
                    "destination": destination,
                    "owner": owners.astype(np.int64),
                    "node": tuple(values.astype(np.int64) for values in at),
                }
            )
        # where the received charge goes: the block of every local patch, as the
        # offset from a node's index in the patch to its place in the storage
        self._local: dict[int, tuple[Any, tuple[int, ...]]] = {}
        for number, patch in enumerate(patches):
            if patch.is_local:
                array = patch.field(name)
                layout = array.layout
                offset = tuple(
                    h - b0
                    for (b0, _), h in zip(layout.index_bounds, layout.halo, strict=True)
                )
                self._local[number] = (array, offset)

    def execute(self) -> None:
        """Add the halos and move the charge. Collective over the hierarchy's communicator."""
        for patch in self._hierarchy.local_patches:
            patch.field(self._name).accumulate_halos()
        ndim = self._hierarchy.ndim
        destination, patch_number, charge = [], [], []
        nodes: list[list[Any]] = [[] for _ in range(ndim)]
        for move in self._moves:
            storage = move["array"].local_with_halos
            charge.append(np.asarray(storage[move["storage"]], dtype=np.float64))
            storage[move["storage"]] = 0.0  # the copy is emptied before the sum is made
            destination.append(move["destination"])
            patch_number.append(move["owner"])
            for axis in range(ndim):
                nodes[axis].append(move["node"][axis])

        def joined(parts: list[Any], dtype: Any) -> Any:
            return np.concatenate(parts) if parts else np.empty(0, dtype=dtype)

        received = mpa.migrate(
            joined(destination, np.int64),
            joined(patch_number, np.int64),
            *(joined(parts, np.int64) for parts in nodes),
            joined(charge, np.float64),
            comm=self._hierarchy.comm,
        )
        number, *index, values = (np.asarray(part) for part in received)
        for held in np.unique(number):
            array, offset = self._local[int(held)]
            chosen = number == held
            np.add.at(
                array.local_with_halos,
                tuple(
                    values_[chosen] + o
                    for values_, o in zip(index, offset, strict=True)
                ),
                values[chosen],
            )
