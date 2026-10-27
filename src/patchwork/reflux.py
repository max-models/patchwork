r"""Conservative explicit time steps on a patch hierarchy: refluxing.

A model with a conserved cell-centered quantity ``u`` (an amount per volume, such
as a density or an energy density) advances it patch by patch with the fluxes
through the faces of its cells,

.. math::

    V_i\,\frac{u_i^{n+1} - u_i^n}{\Delta t} = -\sum_{f \in \partial i} A_f\,F_f\,n_{f,i},

where :math:`F_f` is the flux density through face :math:`f` along its axis,
:math:`A_f` its area and :math:`V_i` the cell's volume (``AMRHierarchy.face_areas``
and ``cell_volumes``: in r–z the faces and cells are rings). Each patch computes
its fluxes from its own cells and its halo (:meth:`AMRHierarchy.exchange_halos
<patchwork.mesh.AMRHierarchy.exchange_halos>`). Where a patch meets a coarser
level, the coarse cell next to it takes the flux of its coarse face, while the
fine cells on the other side take the fluxes of their fine faces. The two do not
agree, and the :meth:`~patchwork.mesh.AMRHierarchy.integral` of ``u`` drifts.

:class:`FluxRegister` restores it (Berger and Colella, *J. Comput. Phys.* 82, 1989).
For every face of a coarse leaf cell on the edge of a finer level, it records the
coarse flux and the sum of the fine fluxes through it, each times its area and time
step. Fine levels may take several time steps for one of the coarse level
(*subcycling*): their fluxes add up over the steps. :meth:`FluxRegister.reflux`
then corrects the coarse cell by the difference, so that it sees the fine
fluxes instead of its own, and gives the coarse cells under the finer patches
the mean of their fine cells (:meth:`~patchwork.mesh.AMRHierarchy.restrict`).
Every face of the leaf cells then carries one flux, and the integral changes only
through the faces of the domain: to round-off, in the Cartesian plane and in r–z.

Without subcycling, one step of all levels::

    register = FluxRegister(mesh)
    mesh.exchange_halos("n")
    register.update("n", fluxes, dt)  # every patch: u -= dt / V * div(A F)
    register.reflux("n")  # the coarse cells next to finer patches; restrict

With subcycling, a level takes its step, then the finer one takes ``ratio``
steps (recursively), and the level is refluxed once they are in step again::

    def advance(level, dt):
        mesh.exchange_halos("n", level=level, below=...)  # in time, see below
        register.update("n", fluxes, dt, level=level)
        if level < mesh.max_level:
            for _ in range(ratio):
                advance(level + 1, dt / ratio)
            register.reflux("n", level=level)

The halos of a subcycling level interpolate the level below at the time in between
from a field that holds it (``exchange_halos(name, level=, below=)``), e.g. the
linear interpolation of the coarse values before and after its step.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import scope_profiler as sp
from maybempi import MPI

from patchwork._profiling import PROFILE_TAGS
from patchwork.cells import CompositeCells

if TYPE_CHECKING:
    from patchwork.mesh import AMRHierarchy, Patch

#: Face fluxes of a patch: ``fluxes(patch)`` returns the flux densities through
#: the faces normal to each axis of this rank's block of the patch, positive
#: along the axis: in 2D ``(fx, fy)`` of shapes ``(nx + 1, ny)`` and
#: ``(nx, ny + 1)`` for a block of ``(nx, ny)`` cells; on a line ``(fx,)`` of
#: shape ``(nx + 1,)``.
Fluxes = Callable[["Patch"], tuple[Any, ...]]


class FluxRegister:
    """Matches the fluxes of coarse and fine levels at their interfaces (refluxing).

    Built for one hierarchy (after a regrid, build one for the new hierarchy,
    with no step under way). One register serves any number of fields: it
    keeps what it records per field name. See the module docstring.

    The coarse–fine faces are the faces between two leaf cells of different
    levels (:class:`~patchwork.cells.CompositeCells`): a level need not be
    properly nested in the one below, and periodic axes and domains that are not
    rectangles are covered. Under MPI the coarse and fine cells of a face may be
    on different ranks: :meth:`reflux` sums the records over the hierarchy's
    communicator (one ``allreduce`` of a value per coarse face).

    Attributes:
        hierarchy: The patch hierarchy.
        num_faces: The number of coarse faces on the edges of finer levels.
    """

    @staticmethod
    def _by_patch(
        cells: CompositeCells,
        leaf: np.ndarray,
        entry: np.ndarray,
        axis: np.ndarray,
        side: np.ndarray,
    ) -> dict[int, tuple[np.ndarray, ...]]:
        """Faces of leaf cells grouped by patch, in the patch's own cell indices."""
        result = {}
        boxes = [patch.box for patch in cells.hierarchy.patches]
        for number in np.unique(cells.patch[leaf]):
            chosen = cells.patch[leaf] == number
            origin = np.array([lo for lo, _ in boxes[number]], dtype=np.int64)
            result[int(number)] = (
                entry[chosen],
                cells.index[leaf[chosen]] - origin,
                axis[chosen],
                side[chosen],
            )
        return result

    def __init__(self, hierarchy: AMRHierarchy, cells: CompositeCells | None = None):
        """Find the coarse–fine faces of ``hierarchy`` (``cells``: its leaf cells)."""
        self.hierarchy = hierarchy
        cells = CompositeCells(hierarchy) if cells is None else cells
        # a key per coarse face: (coarse cell, axis, side of the cell)
        per_cell = 2 * hierarchy.ndim
        keys, fine_cells, fine_axes, fine_sides = [], [], [], []
        for axis in range(hierarchy.ndim):
            faces = cells.faces(axis)
            low, high = faces["low"], faces["high"]
            mixed = cells.span[low] != cells.span[high]
            low, high = low[mixed], high[mixed]
            fine_low = cells.span[low] < cells.span[high]
            coarse = np.where(fine_low, high, low)
            # the side of the cell the face is on: 1 above it, 0 below
            coarse_side = (~fine_low).astype(np.int64)
            keys.append(coarse * per_cell + axis * 2 + coarse_side)
            fine_cells.append(np.where(fine_low, low, high))
            fine_axes.append(np.full(low.size, axis, dtype=np.int64))
            fine_sides.append(fine_low.astype(np.int64))
        unique, entry = np.unique(np.concatenate(keys), return_inverse=True)
        coarse_cells = unique // per_cell
        self.num_faces = int(unique.size)
        patches = hierarchy.patches
        levels = np.array([patch.level for patch in patches], dtype=np.int64)
        self._coarse_level = levels[cells.patch[coarse_cells]]
        # per patch: (entries, local indices (n, ndim), axis, side) of its coarse
        # faces and of its fine faces
        self._coarse = self._by_patch(
            cells,
            coarse_cells,
            np.arange(unique.size),
            (unique % per_cell) // 2,
            unique % 2,
        )
        self._fine = self._by_patch(
            cells,
            np.concatenate(fine_cells),
            entry.reshape(-1),
            np.concatenate(fine_axes),
            np.concatenate(fine_sides),
        )
        self._records: dict[str, np.ndarray] = {}

    def _record(self, name: str) -> np.ndarray:
        if name not in self._records:
            dtype = np.result_type(self.hierarchy._field_specs[name]["dtype"], float)
            self._records[name] = np.zeros(self.num_faces, dtype=dtype)
        return self._records[name]

    @sp.profile("patchwork.flux_update", tags=PROFILE_TAGS)
    def update(
        self,
        name: str,
        fluxes: Fluxes,
        dt: float,
        *,
        level: int | None = None,
        scale: float | None = None,
    ) -> None:
        """Advance field ``name`` by the divergence of face fluxes, and record them.

        For every patch this rank holds (of ``level`` only, if given), the
        cells of its block take ``u -= dt / V * sum(A F n)`` with the flux
        densities ``fluxes(patch)`` (see :data:`Fluxes`). The fluxes through
        coarse–fine faces are recorded times ``scale`` (default ``dt``) for
        :meth:`reflux`. A multi-stage method records each stage with its
        weight: the two stages of Heun's method (SSP-RK2) with ``scale=dt / 2``.

        Local to this rank: no communication. The fluxes of a face that two
        patches of the same level, or two ranks of one patch, share must agree
        (as they do when computed alike from the values the halos give).
        """
        hierarchy = self.hierarchy
        if hierarchy.centering(name) != "cell":
            raise ValueError(f"field {name!r} is not cell-centered")
        records = self._record(name)
        weight = dt if scale is None else scale
        ndim = hierarchy.ndim
        column = (-1,) + (1,) * (ndim - 1)
        for number, patch in enumerate(hierarchy.patches):
            if not patch.is_local or (level is not None and patch.level != level):
                continue
            array = patch.field(name)
            bounds = array.layout.index_bounds
            low = np.array([lo for lo, _ in bounds], dtype=np.int64)
            high = np.array([hi for _, hi in bounds], dtype=np.int64)
            block = tuple(int(n) for n in high - low)
            flux = tuple(np.asarray(f) for f in fluxes(patch))
            shapes = tuple(
                tuple(n + (a == axis) for a, n in enumerate(block))
                for axis in range(ndim)
            )
            if tuple(f.shape for f in flux) != shapes:
                raise ValueError(
                    f"the fluxes of a block of {block} cells have the shapes "
                    f"{shapes}, got {tuple(f.shape for f in flux)}"
                )
            i0 = patch.box[0][0]
            (b0, b1) = bounds[0]
            through = []
            for axis in range(ndim):
                faces_i = i0 + np.arange(b0, b1 + (axis == 0))
                area = hierarchy.face_areas(patch.level, axis, faces_i)
                through.append(flux[axis] * area.reshape(column))
            net = sum(np.diff(t, axis=axis) for axis, t in enumerate(through))
            volume = hierarchy.cell_volumes(patch.level, i0 + np.arange(b0, b1))
            array.local[...] -= dt * net / volume.reshape(column)
            # a patch on several ranks that all hold all of it records once
            if not array.layout.split and patch.comm.Get_rank() != 0:
                continue
            for faces, sign in ((self._coarse, -1.0), (self._fine, 1.0)):
                if number not in faces:
                    continue
                entry, index, axis, side = faces[number]
                mine = np.all((index >= low) & (index < high), axis=1)
                local, axis, side = index[mine] - low, axis[mine], side[mine]
                value = np.empty(local.shape[0], dtype=records.dtype)
                for normal in range(ndim):
                    on = axis == normal
                    at = local[on].copy()
                    at[:, normal] += side[on]
                    value[on] = through[normal][tuple(at.T)]
                np.add.at(records, entry[mine], sign * weight * value)

    @sp.profile("patchwork.reflux", tags=PROFILE_TAGS)
    def reflux(self, name: str, *, level: int | None = None) -> None:
        """Correct the coarse cells next to finer levels, and restrict onto them.

        Each coarse leaf cell on the edge of a finer level takes, for each of
        its faces there, the recorded sum of the fine fluxes instead of its own
        flux (divided by its volume), and the records of those faces are
        cleared. Then the cells under the finer patches take the mean of their
        fine cells. With ``level``, for the coarse cells of that level only
        (when the finer levels have caught up with it), and the restriction
        onto that level only. Collective over the hierarchy's communicator.
        """
        hierarchy = self.hierarchy
        records = self._record(name)
        chosen = (
            np.ones(self.num_faces, dtype=bool)
            if level is None
            else self._coarse_level == level
        )
        # a copy: a serial allreduce may hand back the array itself
        total = np.array(hierarchy.comm.allreduce(records, op=MPI.SUM))
        records[chosen] = 0.0
        for number, patch in enumerate(hierarchy.patches):
            if not patch.is_local or number not in self._coarse:
                continue
            if level is not None and patch.level != level:
                continue
            array = patch.field(name)
            bounds = array.layout.index_bounds
            low = np.array([lo for lo, _ in bounds], dtype=np.int64)
            high = np.array([hi for _, hi in bounds], dtype=np.int64)
            entry, index, _, side = self._coarse[number]
            mine = np.all((index >= low) & (index < high), axis=1)
            entry, index, side = entry[mine], index[mine], side[mine]
            volume = hierarchy.cell_volumes(patch.level, patch.box[0][0] + index[:, 0])
            # the face above the cell carries the flux out of it, the one below in
            change = np.where(side == 1, -1.0, 1.0) * total[entry] / volume
            np.add.at(array.local, tuple((index - low).T), change)
        hierarchy.restrict(name, level=level)

    def reset(self, name: str | None = None) -> None:
        """Clear the records of field ``name`` (of every field without it)."""
        for key in self._records if name is None else [name]:
            if key in self._records:
                self._records[key][...] = 0.0
