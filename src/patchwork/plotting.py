"""Matplotlib plots for AMR patch layouts and patch-local fields."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from patchwork.mesh import AMRHierarchy


def plot_patches(
    hierarchy: AMRHierarchy,
    *,
    field: str | None = None,
    show_grid: bool = True,
    ax: Any = None,
    cmap: str = "viridis",
    root: int = 0,
    grid_linewidth: float | None = None,
    grid_alpha: float | None = None,
) -> tuple[Any, Any] | None:
    """Plot patch extents and cell grids, optionally colored by a field.

    With ``show_grid=False`` and no ``field`` only the patch outlines are drawn,
    e.g. over an image that is already on ``ax``. ``grid_linewidth`` and
    ``grid_alpha`` set the cell lines of every level; by default they are thin
    and faint, more so on the base level, which suits a large figure. A small
    one, e.g. an animation's frames, may need stronger lines.

    When ``field`` is provided, each patch field is gathered collectively and
    plotted on ``root`` only. Call this function on every rank in the patch
    communicator. Layout-only plots are also returned on ``root`` only.
    Matplotlib is imported only when this function is called.

    Returns ``(figure, axes)`` on the plotting rank and ``None`` elsewhere.
    """
    import numpy as np

    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise ImportError(
            "plotting requires matplotlib; install patchwork[plot]"
        ) from exc

    rank = hierarchy.comm.Get_rank()
    gathered: dict[int, Any] = {}
    if field is not None:
        for patch in hierarchy:
            gathered[id(patch)] = hierarchy.gather(patch, field, root=root)
    if rank != root:
        return None

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    else:
        fig = ax.figure

    palette = plt.get_cmap("tab10")
    mappable = None
    # the colour scale shared by all patches; set when a field is drawn
    limits = (0.0, 1.0)
    nodal = field is not None and hierarchy.centering(field) == "node"
    if field is not None:
        # one colour scale for all patches
        limits = (
            min(float(np.nanmin(values)) for values in gathered.values()),
            max(float(np.nanmax(values)) for values in gathered.values()),
        )
    for patch in hierarchy:
        ((x0, x1), (y0, y1)) = patch.physical_bounds(hierarchy)
        color = palette(patch.level % 10)
        if field is not None:
            values = gathered[id(patch)]
            dx, dy = hierarchy.spacing(patch.level)
            x_edges = x0 + np.arange(patch.shape[0] + 1) * dx
            y_edges = y0 + np.arange(patch.shape[1] + 1) * dy
            # node values sit on the cell corners and are shaded between them
            mappable = ax.pcolormesh(
                x_edges,
                y_edges,
                values.T,
                shading="gouraud" if nodal else "flat",
                cmap=cmap,
                vmin=limits[0],
                vmax=limits[1],
                zorder=2 * patch.level + 1,
            )
        elif patch.level > 0 and show_grid:
            # Mask the covered coarse cells so only the finest active grid is visible.
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    facecolor="white",
                    edgecolor="none",
                    zorder=2 * patch.level,
                )
            )
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor=color,
                linewidth=2.0 if patch.level else 1.2,
                linestyle="--" if patch.level == 0 else "-",
                zorder=2 * patch.level + 2,
            )
        )
        if show_grid:
            dx, dy = hierarchy.spacing(patch.level)
            alpha = grid_alpha
            if alpha is None:
                alpha = 0.62 if patch.level else 0.24
            linewidth = grid_linewidth
            if linewidth is None:
                linewidth = 0.42 if patch.level else 0.32
            ax.vlines(
                x0 + np.arange(patch.shape[0] + 1) * dx,
                y0,
                y1,
                colors=[color],
                alpha=alpha,
                linewidth=linewidth,
                zorder=2 * patch.level + 2,
            )
            ax.hlines(
                y0 + np.arange(patch.shape[1] + 1) * dy,
                x0,
                x1,
                colors=[color],
                alpha=alpha,
                linewidth=linewidth,
                zorder=2 * patch.level + 2,
            )

    (xmin, xmax), (ymin, ymax) = hierarchy.bounds
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel="x", ylabel="y")
    ax.set_aspect("equal", adjustable="box")
    levels = range(hierarchy.max_level + 1)
    handles = [
        Line2D([0], [0], color=palette(level % 10), lw=2, label=f"level {level}")
        for level in levels
        if hierarchy.patches_at_level(level)
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)
    if mappable is not None:
        fig.colorbar(mappable, ax=ax, label=field)
    ax.set_title(
        "Patchwork AMR hierarchy" if field is None else f"Patchwork field: {field}"
    )
    return fig, ax
