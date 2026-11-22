"""Matplotlib plots for AMR patch layouts and patch-local fields."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from patchwork.mesh.hierarchy import AMRHierarchy


def _plot_line(
    hierarchy: AMRHierarchy,
    ax: Any,
    field: str | None,
    gathered: dict[int, Any],
    show_grid: bool,
    palette: Callable[[int], Any],
    grid_alpha: float | None,
    scale: float,
    label: Callable[[int], str | None],
) -> None:
    """`plot_patches` of a 1D hierarchy."""
    import numpy as np

    nodal = field is not None and hierarchy.centering(field) == "node"
    labelled: set[int] = set()
    for patch in hierarchy:
        ((x0, x1),) = patch.physical_bounds(hierarchy)
        x0, x1 = x0 * scale, x1 * scale
        color = palette(patch.level)
        (edges,) = patch.node_coordinates(hierarchy)
        edges = np.asarray(edges) * scale
        entry = label(patch.level) if patch.level not in labelled else None
        labelled.add(patch.level)
        if field is None:
            ax.plot(
                [x0, x1], [patch.level, patch.level], color=color, lw=6.0, label=entry
            )
            if show_grid:
                alpha = 0.8 if grid_alpha is None else grid_alpha
                ax.vlines(
                    edges,
                    patch.level - 0.2,
                    patch.level + 0.2,
                    colors=[color],
                    alpha=alpha,
                    linewidth=0.8,
                )
            continue
        values = np.asarray(gathered[id(patch)])
        x = edges if nodal else 0.5 * (edges[:-1] + edges[1:])
        if patch.level:
            ax.axvspan(x0, x1, color=color, alpha=0.08, lw=0)
        ax.plot(
            x,
            values,
            color=color,
            marker="." if show_grid else None,
            lw=1.5,
            zorder=2 * patch.level + 2,
            label=entry,
        )
    ((xmin, xmax),) = hierarchy.bounds
    ax.set_xlim(xmin * scale, xmax * scale)
    ax.set_xlabel("x")
    if field is None:
        ax.set_ylabel("level")
        ax.set_yticks(range(hierarchy.max_level + 1))
        ax.set_ylim(-0.5, hierarchy.max_level + 0.5)
    else:
        ax.set_ylabel(field)
    if any(label(level) is not None for level in labelled):
        ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title(
        "Patchwork AMR hierarchy" if field is None else f"Patchwork field: {field}"
    )


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
    transpose: bool = False,
    scale: float = 1.0,
    mask_covered: bool = True,
    colors: Sequence[Any] | None = None,
    labels: Sequence[str | None] | None = None,
    title: str | None = None,
) -> tuple[Any, Any] | None:
    """Plot patch extents and cell grids, optionally colored by a field.

    With ``show_grid=False`` and no ``field`` only the patch outlines are drawn,
    e.g. over an image that is already on ``ax``. ``grid_linewidth`` and
    ``grid_alpha`` set the cell lines of every level; by default they are thin
    and faint, more so on the base level, which suits a large figure. A small
    one, e.g. an animation's frames, may need stronger lines.

    ``transpose`` draws the hierarchy's second axis horizontally (e.g. ``x`` of an
    ``(r, x)`` cylindrical hierarchy), ``scale`` multiplies the coordinates (``1e3``
    for millimetres from metres). Without ``field``, the cells of a coarser level
    under a finer patch are hidden; ``mask_covered=False`` keeps them, so every
    level's grid shows in full. ``colors`` and ``labels`` give each level its colour
    and legend entry (``None`` leaves a level out of the legend; by default ``tab10``
    and "level n"; the entries are the labels of the plotted artists, so a later
    ``ax.legend()`` keeps them), ``title`` the axes' title (``""`` for none).

    A 1D hierarchy is drawn as a line plot: without ``field`` one bar per patch
    at the height of its level (with its cell edges if ``show_grid``); with
    ``field`` the values of every patch against ``x``, the finer patches on top
    and shaded over their extent.

    When ``field`` is provided, each patch field is gathered collectively and
    plotted on ``root`` only. Call this function on every rank in the patch
    communicator. Layout-only plots are also returned on ``root`` only.
    Matplotlib is imported only when this function is called.

    Returns ``(figure, axes)`` on the plotting rank and ``None`` elsewhere.
    """
    import numpy as np

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise ImportError(
            "plotting requires matplotlib; install patchwork[plot]"
        ) from exc

    if transpose and hierarchy.ndim != 2:
        raise ValueError("transpose applies to 2D hierarchies only")
    rank = hierarchy.comm.Get_rank()
    gathered: dict[int, Any] = {}
    if field is not None:
        for patch in hierarchy:
            gathered[id(patch)] = hierarchy.gather(patch, field, root=root)
    if rank != root:
        return None

    if ax is None:
        size = (8, 6) if hierarchy.ndim == 2 else (8, 3.5)
        fig, ax = plt.subplots(figsize=size, constrained_layout=True)
    else:
        fig = ax.figure

    tab10 = plt.get_cmap("tab10")

    def palette(level: int) -> Any:
        """The colour of ``level``."""
        if colors is not None:
            return colors[level]
        return tab10(level % 10)

    def label(level: int) -> str | None:
        """The legend entry of ``level``."""
        return f"level {level}" if labels is None else labels[level]

    if hierarchy.ndim == 1:
        _plot_line(
            hierarchy, ax, field, gathered, show_grid, palette, grid_alpha, scale, label
        )
        if title is not None:
            ax.set_title(title)
        return fig, ax

    def rectangle(x0: float, x1: float, y0: float, y1: float, **kwargs: Any) -> None:
        """Add the box ``[x0, x1] x [y0, y1]`` of the hierarchy to the axes."""
        if transpose:
            x0, x1, y0, y1 = y0, y1, x0, x1
        ax.add_patch(
            Rectangle(
                (x0 * scale, y0 * scale),
                (x1 - x0) * scale,
                (y1 - y0) * scale,
                **kwargs,
            )
        )

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
    labelled: set[int] = set()
    for patch in hierarchy:
        ((x0, x1), (y0, y1)) = patch.physical_bounds(hierarchy)
        color = palette(patch.level)
        x_edges, y_edges = (
            edges * scale for edges in patch.node_coordinates(hierarchy)
        )
        if field is not None:
            values = np.asarray(gathered[id(patch)])
            # node values sit on the cell corners and are shaded between them
            mappable = ax.pcolormesh(
                *(
                    (y_edges, x_edges, values)
                    if transpose
                    else (x_edges, y_edges, values.T)
                ),
                shading="gouraud" if nodal else "flat",
                cmap=cmap,
                vmin=limits[0],
                vmax=limits[1],
                zorder=2 * patch.level + 1,
            )
        elif patch.level > 0 and show_grid and mask_covered:
            # Mask the covered coarse cells so only the finest active grid is visible.
            rectangle(
                x0,
                x1,
                y0,
                y1,
                facecolor="white",
                edgecolor="none",
                zorder=2 * patch.level,
            )
        # the level's legend entry: its first patch's cell lines, or else its outline
        entry = label(patch.level) if patch.level not in labelled else None
        labelled.add(patch.level)
        rectangle(
            x0,
            x1,
            y0,
            y1,
            label=None if show_grid else entry,
            fill=False,
            edgecolor=color,
            linewidth=2.0 if patch.level else 1.2,
            linestyle="--" if patch.level == 0 else "-",
            zorder=2 * patch.level + 2,
        )
        if show_grid:
            alpha = grid_alpha
            if alpha is None:
                alpha = 0.62 if patch.level else 0.24
            linewidth = grid_linewidth
            if linewidth is None:
                linewidth = 0.42 if patch.level else 0.32
            style = {
                "colors": [color],
                "alpha": alpha,
                "linewidth": linewidth,
                "zorder": 2 * patch.level + 2,
            }
            # the lines across the first axis, then those across the second
            first, second = (
                (ax.hlines, ax.vlines) if transpose else (ax.vlines, ax.hlines)
            )
            first(x_edges, y_edges[0], y_edges[-1], label=entry, **style)
            second(y_edges, x_edges[0], x_edges[-1], **style)

    (xmin, xmax), (ymin, ymax) = hierarchy.bounds
    limits_x, limits_y = (xmin * scale, xmax * scale), (ymin * scale, ymax * scale)
    names = ("x", "y")
    if transpose:
        limits_x, limits_y = limits_y, limits_x
        names = ("y", "x")
    ax.set(xlim=limits_x, ylim=limits_y, xlabel=names[0], ylabel=names[1])
    ax.set_aspect("equal", adjustable="box")
    if any(label(level) is not None for level in labelled):
        legend = ax.legend(loc="upper right", framealpha=0.9)
        for handle in legend.legend_handles:
            # the faint cell lines as solid lines in the legend
            handle.set_alpha(1.0)
            handle.set_linewidth(2.0)
    if mappable is not None:
        fig.colorbar(mappable, ax=ax, label=field)
    if title is None:
        title = (
            "Patchwork AMR hierarchy" if field is None else f"Patchwork field: {field}"
        )
    ax.set_title(title)
    return fig, ax
