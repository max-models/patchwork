"""The figures of the MPI communication guide: ``docs/public/figures/mpi/*.svg``.

Every figure is a schematic of how patchwork moves data between ranks: who holds which
patch, the blocks and halos of a patch, the copy plan behind the halo exchange, where every
halo cell comes from, the coarse buffer of the interpolation, the restriction, the deposit
with its migration and accumulation, and the transfer of a regrid. Where a figure shows a
classification of cells (the sources of the halo cells, the cells a regrid copies), it is
computed with patchwork's own geometry, on a hierarchy built in the script, and checked
against the schedule patchwork builds for it.

Rerun with ``make docs-figures`` or ``python docs/mpi_figures.py`` (needs matplotlib).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402

from patchwork import AMRHierarchy  # noqa: E402
from patchwork.mesh.halos import HaloExchange, _frame, _overlap, _Points, _shift  # noqa: E402
from patchwork.mesh.regrid import transfer  # noqa: E402

OUTPUT = Path(__file__).parent / "public" / "figures" / "mpi"

#: One colour per rank.
RANK = ["#4C72B0", "#DD8452", "#55A868", "#8172B2"]
#: One colour per source of a halo cell (and of a cell of a new patch).
SOURCE = {
    "same": "#2A9D8F",
    "periodic": "#2A9D8F",
    "interpolated": "#E9C46A",
    "neighbour": "#B08BD6",
    "outside": "#E0E0E0",
}
INK = "#333333"
FAINT = "#C8C8C8"

plt.rcParams.update(
    {
        "svg.fonttype": "none",
        "font.size": 10,
        "font.family": "sans-serif",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "text.color": INK,
    }
)


# Helpers ------------------------------------------------------------------------------- #


def chunks(length: int, count: int) -> list[tuple[int, int]]:
    """How mpiarray splits an axis of ``length`` over ``count`` ranks (the first get one more)."""
    base, extra = divmod(length, count)
    bounds = []
    for i in range(count):
        start = i * base + min(i, extra)
        bounds.append((start, start + base + (1 if i < extra else 0)))
    return bounds


def rect(
    ax: Any, x0: float, y0: float, x1: float, y1: float, **style: Any
) -> Rectangle:
    """A rectangle from its corners."""
    style.setdefault("linewidth", 0)
    patch = Rectangle((x0, y0), x1 - x0, y1 - y0, **style)
    ax.add_patch(patch)
    return patch


def box_rect(ax: Any, box: Any, scale: float, **style: Any) -> Rectangle:
    """A half-open index box, drawn in units of ``1 / scale`` cells."""
    (i0, i1), (j0, j1) = box
    return rect(ax, i0 / scale, j0 / scale, i1 / scale, j1 / scale, **style)


def grid(ax: Any, box: Any, scale: float, color: str = FAINT, lw: float = 0.4) -> None:
    """The cell lines of a box, in units of ``1 / scale`` cells."""
    (i0, i1), (j0, j1) = box
    for i in range(i0, i1 + 1):
        ax.plot([i / scale] * 2, [j0 / scale, j1 / scale], color=color, lw=lw, zorder=1)
    for j in range(j0, j1 + 1):
        ax.plot([i0 / scale, i1 / scale], [j / scale] * 2, color=color, lw=lw, zorder=1)


def arrow(
    ax: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = INK,
    lw: float = 1.2,
    rad: float = 0.0,
    style: str = "-|>",
    **kw: Any,
) -> None:
    """An arrow from ``start`` to ``end`` (``rad`` bends it)."""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            shrinkA=0,
            shrinkB=0,
            connectionstyle=f"arc3,rad={rad}",
            mutation_scale=12,
        ),
        zorder=6,
        **kw,
    )


def note(ax: Any, x: float, y: float, text: str, **kw: Any) -> None:
    """A text label, centred by default."""
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    kw.setdefault("fontsize", 9)
    kw.setdefault("zorder", 7)
    ax.text(x, y, text, **kw)


def card(ax: Any, x: float, y: float, w: float, h: float, **style: Any) -> None:
    """A rounded box (for the diagrams without a grid)."""
    style.setdefault("facecolor", "white")
    style.setdefault("edgecolor", INK)
    style.setdefault("linewidth", 1.0)
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.15", **style)
    )


def rank_squares(ax: Any, x: float, y: float, ranks: Any, size: float = 0.42) -> None:
    """A row of small squares, one per rank in its colour, starting at ``(x, y)``."""
    for k, rank in enumerate(ranks):
        rect(
            ax,
            x + k * (size + 0.1),
            y,
            x + k * (size + 0.1) + size,
            y + size,
            facecolor=RANK[rank],
            edgecolor="white",
            linewidth=0.8,
        )
        note(
            ax,
            x + k * (size + 0.1) + size / 2,
            y + size / 2,
            str(rank),
            color="white",
            fontsize=8,
        )


def blank(ax: Any, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    """Equal aspect, no frame."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def save(fig: Any, name: str) -> None:
    """Write the figure as SVG into the output directory and close it."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT / f"{name}.svg", format="svg", bbox_inches="tight", pad_inches=0.05
    )
    plt.close(fig)
    print(f"wrote {OUTPUT / name}.svg")


# 1. Who holds what: ranks, patches and their communicators ----------------------------- #


def figure_ranks() -> None:
    """Who holds what: the blocks of three patches by rank, and the communicators."""
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1, 1.25]}
    )
    base = ((0, 16), (0, 16))
    p1 = ((8, 24), (4, 20))  # level 1, ranks 1 and 2
    p2 = ((24, 40), (16, 32))  # level 2, rank 3
    grid(left, base, 1)
    for rank, (a, b) in enumerate(chunks(16, 4)):
        rect(left, a, 0, b, 16, facecolor=RANK[rank], alpha=0.18)
        note(left, (a + b) / 2, 15.2, f"rank {rank}", color=RANK[rank], fontsize=9)
    for rank, (a, b) in zip((1, 2), chunks(16, 2), strict=True):
        (i0, _), (j0, j1) = p1
        rect(
            left,
            (i0 + a) / 2,
            j0 / 2,
            (i0 + b) / 2,
            j1 / 2,
            facecolor=RANK[rank],
            alpha=0.45,
        )
        note(
            left,
            (i0 + (a + b) / 2) / 2,
            (j0 + 1.5) / 2,
            f"rank {rank}",
            color="white",
            fontsize=8,
        )
    grid(left, p1, 2, lw=0.3)
    box_rect(left, p2, 4, facecolor=RANK[3], alpha=0.65)
    grid(left, p2, 4, lw=0.25)
    note(left, 8, 5.2, "rank 3", color="white", fontsize=8)
    for box, scale in ((base, 1), (p1, 2), (p2, 4)):
        box_rect(left, box, scale, facecolor="none", edgecolor=INK, linewidth=1.4)
    note(left, 8, 11.3, "level 1: ranks 1, 2", fontsize=9, color=INK)
    note(left, 2.5, 0.8, "level 0: all ranks", fontsize=9, ha="left")
    note(left, 8, 7.0, "level 2", fontsize=8, color="white")
    blank(left, (-0.3, 16.3), (-0.3, 16.3))
    left.set_title(
        "the patches, coloured by the rank that holds each block", fontsize=10
    )

    blank(right, (0, 11), (0, 7.6))
    card(right, 2.9, 6.1, 5.2, 1.2, facecolor="#F6F6F6")
    note(right, 5.5, 6.95, "the hierarchy's communicator  (comm)", fontsize=9.5)
    rank_squares(right, 4.45, 6.25, range(4))
    columns = [
        (
            0.2,
            "patch 0, level 0",
            "ranks (0, 1, 2, 3)",
            "comm = the hierarchy's",
            (0, 1, 2, 3),
        ),
        (3.8, "patch 1, level 1", "ranks (1, 2)", "comm = comm.Split(…)", (1, 2)),
        (7.4, "patch 2, level 2", "ranks (3,)", "comm = COMM_SELF", (3,)),
    ]
    for x, title, ranks, how, members in columns:
        card(right, x, 2.0, 3.4, 2.9)
        note(right, x + 1.7, 4.5, title, fontsize=9.5, fontweight="bold")
        note(right, x + 1.7, 3.95, ranks, fontsize=9)
        note(right, x + 1.7, 3.45, how, fontsize=8, family="monospace")
        rank_squares(right, x + 1.7 - len(members) * 0.26, 2.35, members)
        arrow(right, (5.5, 6.1), (x + 1.7, 4.9), color="#888888", lw=1.0)
    note(
        right,
        5.5,
        1.2,
        "on a rank outside a patch's ranks: patch.comm is None, patch.is_local is False,\n"
        "and the patch holds no data there (the hierarchy itself is the same on every rank)",
        fontsize=8.5,
        color="#555555",
    )
    note(
        right,
        5.5,
        0.35,
        "fields: one mpiarray array per patch, split along the first axis over the patch's ranks",
        fontsize=8.5,
        color="#555555",
    )
    right.set_title(
        "one communicator per group of ranks, made once and shared", fontsize=10
    )
    save(fig, "ranks")


# 2. Blocks and halos of one patch ------------------------------------------------------- #


def figure_blocks() -> None:
    """One patch on three ranks: its blocks, the two kinds of halo, one rank's storage."""
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    nx, ny, h = 12, 8, 1
    blocks = chunks(nx, 3)
    # the halo ring at the patch walls: patchwork's copy plan fills it
    rect(left, -h, -h, nx + h, ny + h, facecolor=SOURCE["same"], alpha=0.35)
    rect(left, 0, 0, nx, ny, facecolor="white")
    for rank, (a, b) in enumerate(blocks):
        rect(left, a, 0, b, ny, facecolor=RANK[rank], alpha=0.3)
        note(left, (a + b) / 2, ny / 2, f"rank {rank}\nblock [{a}, {b})", fontsize=9)
    # the halo columns of rank 1 along the seams: mpiarray fills them
    a, b = blocks[1]
    for x0, x1 in ((a - h, a), (b, b + h)):
        rect(
            left,
            x0,
            -h,
            x1,
            ny + h,
            facecolor=SOURCE["neighbour"],
            alpha=0.75,
            hatch="//",
            edgecolor="white",
        )
    grid(left, ((-h, nx + h), (-h, ny + h)), 1, color="#AAAAAA", lw=0.4)
    rect(left, 0, 0, nx, ny, facecolor="none", edgecolor=INK, linewidth=1.5)
    for x in (a, b):
        left.plot([x, x], [0, ny], color=INK, lw=1.0, ls="--")
    note(
        left,
        nx / 2,
        -h - 0.9,
        "halo at the patch's walls: filled by patchwork's copy plan",
        fontsize=9,
        color="#1F7A70",
    )
    note(
        left,
        nx / 2,
        -h - 1.7,
        "receiving regions along x:  rank 0 [-1, 4)   rank 1 [4, 8)   rank 2 [8, 13)",
        fontsize=8.5,
        family="monospace",
    )
    arrow(left, (a - 2.6, ny + 2.0), (a - 0.5, ny + 0.6), color="#7B5FB0", rad=0.2)
    note(
        left,
        a - 2.6,
        ny + 2.4,
        "halo between blocks: mpiarray's update_halos",
        fontsize=9,
        color="#7B5FB0",
        ha="left",
    )
    blank(left, (-1.6, nx + 1.6), (-3.1, ny + 3.0))
    left.set_title(
        "a 12 × 8 patch with halo 1 on three ranks (split along x)", fontsize=10
    )

    # the storage of rank 1: local_with_halos, (6, 10)
    w = b - a
    rect(right, 0, 0, w + 2 * h, ny + 2 * h, facecolor=SOURCE["same"], alpha=0.35)
    rect(right, h, h, w + h, ny + h, facecolor=RANK[1], alpha=0.3)
    for x0, x1 in ((0, h), (w + h, w + 2 * h)):
        rect(
            right,
            x0,
            0,
            x1,
            ny + 2 * h,
            facecolor=SOURCE["neighbour"],
            alpha=0.75,
            hatch="//",
            edgecolor="white",
        )
    grid(right, ((0, w + 2 * h), (0, ny + 2 * h)), 1, color="#AAAAAA", lw=0.4)
    rect(right, h, h, w + h, ny + h, facecolor="none", edgecolor=INK, linewidth=1.5)
    note(right, h + w / 2, h + ny / 2, "own block\n(.local)", fontsize=9)
    note(right, h + w / 2, ny + 1.5 * h, "from the plan", fontsize=8, color="#1F7A70")
    note(right, h + w / 2, 0.5 * h, "from the plan", fontsize=8, color="#1F7A70")
    note(
        right,
        0.5 * h,
        h + ny / 2,
        "from rank 0",
        fontsize=8,
        rotation=90,
        color="#5A3F8F",
    )
    note(
        right,
        w + 1.5 * h,
        h + ny / 2,
        "from rank 2",
        fontsize=8,
        rotation=90,
        color="#5A3F8F",
    )
    note(
        right,
        (w + 2 * h) / 2,
        -1.1,
        "layout.index_bounds = ((4, 8), (0, 8))\nlayout.halo = (1, 1)\nlocal_with_halos.shape = (6, 10)",
        fontsize=8.5,
        family="monospace",
        va="top",
    )
    blank(right, (-0.5, w + 2 * h + 0.5), (-3.6, ny + 2 * h + 0.5))
    right.set_title("what rank 1 stores", fontsize=10)
    save(fig, "blocks")


# 3. The copy plan: build once, execute as one Alltoallv ------------------------------- #


def figure_copy_plan() -> None:
    """The copy plan: the pieces of one copy at the build, and the Alltoallv of an execution."""
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10.5, 8.2), gridspec_kw={"height_ratios": [1.1, 1.05]}
    )
    # build: a source box on array A (ranks 0, 1) goes to a target box on array B (ranks 2, 3)
    blank(top, (0, 23), (0, 9.2))
    note(
        top,
        0.3,
        8.8,
        "build (once): where each copy's boxes meet the blocks",
        fontsize=10,
        ha="left",
        fontweight="bold",
    )
    # array A: 8 x 4 cells, split along x over ranks 0 and 1
    ax0, ay0 = 1.0, 2.6
    for rank, (a, b) in enumerate(chunks(8, 2)):
        rect(top, ax0 + a, ay0, ax0 + b, ay0 + 4, facecolor=RANK[rank], alpha=0.3)
        note(
            top,
            ax0 + (a + b) / 2,
            ay0 + 4.4,
            f"rank {rank}",
            fontsize=8.5,
            color=RANK[rank],
        )
    for i in range(9):
        top.plot([ax0 + i] * 2, [ay0, ay0 + 4], color="#AAAAAA", lw=0.4)
    for j in range(5):
        top.plot([ax0, ax0 + 8], [ay0 + j] * 2, color="#AAAAAA", lw=0.4)
    rect(
        top, ax0, ay0, ax0 + 8, ay0 + 4, facecolor="none", edgecolor=INK, linewidth=1.3
    )
    rect(
        top,
        ax0 + 2,
        ay0 + 1,
        ax0 + 7,
        ay0 + 3,
        facecolor="none",
        edgecolor="#C0392B",
        linewidth=2.0,
    )
    note(
        top, ax0 + 4, ay0 - 0.5, "source array A, its box ((2, 7), (1, 3))", fontsize=9
    )
    # array B: 6 x 4 cells with halo 1 at (13, 1.5), split along x over ranks 2 and 3
    bx0, by0 = 14.0, 2.6
    rect(top, bx0 - 1, by0 - 1, bx0 + 7, by0 + 5, facecolor="#EEEEEE")
    for rank, (a, b) in zip((2, 3), chunks(6, 2), strict=True):
        rect(top, bx0 + a, by0, bx0 + b, by0 + 4, facecolor=RANK[rank], alpha=0.3)
        note(
            top,
            bx0 + (a + b) / 2,
            by0 + 5.4,
            f"rank {rank}",
            fontsize=8.5,
            color=RANK[rank],
        )
    for i in range(-1, 8):
        top.plot([bx0 + i] * 2, [by0 - 1, by0 + 5], color="#AAAAAA", lw=0.4)
    for j in range(-1, 6):
        top.plot([bx0 - 1, bx0 + 7], [by0 + j] * 2, color="#AAAAAA", lw=0.4)
    rect(
        top, bx0, by0, bx0 + 6, by0 + 4, facecolor="none", edgecolor=INK, linewidth=1.3
    )
    rect(
        top,
        bx0 - 1,
        by0 + 1,
        bx0 + 4,
        by0 + 3,
        facecolor="none",
        edgecolor="#C0392B",
        linewidth=2.0,
    )
    note(
        top,
        bx0 + 3,
        by0 - 1.5,
        "target array B (halo 1), its box ((-1, 4), (1, 3))",
        fontsize=9,
    )
    # pieces: source block of rank 0 (x 2..4) -> target x -1..1 -> rank 2's region [-1, 3)
    # source block of rank 1 (x 4..7) -> target x 1..4 -> rank 2 (1..3) and rank 3 (3..4)
    arrow(
        top,
        (ax0 + 3.0, ay0 + 2.0),
        (bx0 + 0.0, by0 + 2.0),
        color=RANK[0],
        lw=1.6,
        rad=-0.25,
    )
    arrow(
        top,
        (ax0 + 5.0, ay0 + 2.4),
        (bx0 + 2.0, by0 + 2.4),
        color=RANK[1],
        lw=1.6,
        rad=-0.15,
    )
    arrow(
        top,
        (ax0 + 6.5, ay0 + 1.6),
        (bx0 + 3.5, by0 + 1.6),
        color=RANK[1],
        lw=1.6,
        rad=0.1,
    )
    note(top, 11.5, 8.2, "the pieces of this copy", fontsize=8.5, color="#555555")
    note(top, 11.5, 7.7, "0 → 2: 2 × 2 cells", fontsize=8.5, color=RANK[0])
    note(top, 11.5, 7.2, "1 → 2: 2 × 2 cells", fontsize=8.5, color=RANK[1])
    note(top, 11.5, 6.7, "1 → 3: 1 × 2 cells", fontsize=8.5, color=RANK[1])

    # execute: pack, Alltoallv, unpack
    blank(bottom, (0, 23), (0, 8.4))
    note(
        bottom,
        0.3,
        8.0,
        "execute (every call): pack, one Alltoallv over the hierarchy's communicator, unpack",
        fontsize=10,
        ha="left",
        fontweight="bold",
    )
    lanes = {0: 6.3, 1: 4.9, 2: 3.5, 3: 2.1}
    # send buffers (left), laid out by destination rank; receive buffers (right), by source rank
    sends = {0: [(2, 4)], 1: [(2, 4), (3, 2)], 2: [], 3: []}
    receives = {0: [], 1: [], 2: [(0, 4), (1, 4)], 3: [(1, 2)]}
    for rank, y in lanes.items():
        note(
            bottom,
            0.9,
            y + 0.4,
            f"rank {rank}",
            fontsize=9,
            color=RANK[rank],
            fontweight="bold",
        )
        x = 2.2
        note(
            bottom,
            2.2,
            y + 1.05,
            "send buffer",
            fontsize=7.5,
            ha="left",
            color="#777777",
        ) if rank == 0 else None
        for other, count in sends[rank]:
            rect(
                bottom,
                x,
                y,
                x + count * 0.55,
                y + 0.8,
                facecolor=RANK[other],
                alpha=0.75,
                edgecolor="white",
                linewidth=1,
            )
            note(
                bottom,
                x + count * 0.275,
                y + 0.4,
                f"to {other}",
                fontsize=7.5,
                color="white",
            )
            x += count * 0.55
        if not sends[rank]:
            note(
                bottom,
                2.2,
                y + 0.4,
                "nothing to send",
                fontsize=7.5,
                ha="left",
                color="#999999",
            )
        x = 14.0
        note(
            bottom,
            14.0,
            y + 1.05,
            "receive buffer",
            fontsize=7.5,
            ha="left",
            color="#777777",
        ) if rank == 0 else None
        for other, count in receives[rank]:
            rect(
                bottom,
                x,
                y,
                x + count * 0.55,
                y + 0.8,
                facecolor=RANK[other],
                alpha=0.75,
                edgecolor="white",
                linewidth=1,
            )
            note(
                bottom,
                x + count * 0.275,
                y + 0.4,
                f"from {other}",
                fontsize=7.5,
                color="white",
            )
            x += count * 0.55
        if not receives[rank]:
            note(
                bottom,
                14.0,
                y + 0.4,
                "nothing to receive",
                fontsize=7.5,
                ha="left",
                color="#999999",
            )
    # the Alltoallv: a band in the middle, with the three messages
    rect(bottom, 7.6, 1.6, 12.6, 7.6, facecolor="#F3F3F3")
    note(bottom, 10.1, 7.3, "Alltoallv", fontsize=10, fontweight="bold")
    note(
        bottom,
        10.1,
        6.6,
        "counts and offsets per rank,\nfixed at the build",
        fontsize=7.5,
        color="#555555",
    )
    arrow(bottom, (7.6, lanes[0] + 0.4), (12.6, lanes[2] + 0.6), color=RANK[0], lw=1.6)
    arrow(bottom, (7.6, lanes[1] + 0.5), (12.6, lanes[2] + 0.2), color=RANK[1], lw=1.6)
    arrow(bottom, (7.6, lanes[1] + 0.3), (12.6, lanes[3] + 0.4), color=RANK[1], lw=1.6)
    note(
        bottom,
        10.1,
        1.9,
        "a rank also sends to itself through the same buffers;\non one rank the receive buffer is the send buffer",
        fontsize=7.5,
        color="#555555",
    )
    # unpack
    rect(
        bottom,
        18.4,
        1.9,
        22.8,
        7.1,
        facecolor="none",
        edgecolor="#AAAAAA",
        linewidth=0.8,
        linestyle="--",
    )
    note(bottom, 20.6, 6.6, "unpack", fontsize=9, fontweight="bold")
    note(
        bottom,
        20.6,
        4.4,
        "each piece goes to its place\nin target.local_with_halos,\nin the order of the copies:\nwhere boxes overlap,\nthe last copy wins",
        fontsize=7.5,
    )
    arrow(
        bottom, (16.5, lanes[2] + 0.4), (18.4, lanes[2] + 0.4), color="#888888", lw=1.0
    )
    arrow(
        bottom, (15.2, lanes[3] + 0.4), (18.4, lanes[3] + 0.4), color="#888888", lw=1.0
    )
    note(
        bottom,
        11.5,
        0.5,
        "the sources are read before any target is written, so a copy from an array to itself is safe",
        fontsize=8.5,
        color="#555555",
    )
    save(fig, "copy_plan")


# 4. Where every halo cell comes from, on a real hierarchy ----------------------------- #


def halo_sources(
    mesh: AMRHierarchy, patch: Any, name: str, width: int
) -> dict[tuple[int, int], str]:
    """The source of every halo cell of ``patch``: patchwork's rules, cell by cell."""
    nodal = mesh.centering(name) == "node"
    points = _Points(mesh, nodal)
    own = points.own(patch)
    kinds: dict[tuple[int, int], str] = {}
    for frame in _frame(own, width):
        for i in range(*frame[0]):
            for j in range(*frame[1]):
                inside = points.in_domain(patch.level, np.array([i]), np.array([j]))[0]
                kinds[(i, j)] = (
                    ("interpolated" if patch.level else "outside")
                    if inside
                    else "outside"
                )
    for other in mesh.patches_at_level(patch.level):
        for shift in points.shifts(patch.level):
            if other is patch and not any(shift):
                continue
            theirs = _shift(points.own(other), shift)
            for frame in _frame(own, width):
                part = _overlap(frame, theirs)
                if part is None:
                    continue
                for i in range(*part[0]):
                    for j in range(*part[1]):
                        kinds[(i, j)] = "periodic" if any(shift) else "same"
    return kinds


def check_halo_sources(mesh: AMRHierarchy, patch: Any, name: str, kinds: dict) -> None:
    """The classification agrees with the schedule patchwork builds (serial run)."""
    exchange = HaloExchange(mesh, name)
    array = patch.field(name)
    copied = sum(
        np.prod([e - s for s, e in target_box])
        for _, _, target, target_box in exchange._plan._copies
        if target is array
    )
    interpolated = sum(
        item.index[0].size for item in exchange._interpolations if item.array is array
    )
    counts = {kind: sum(1 for k in kinds.values() if k == kind) for kind in SOURCE}
    assert copied == counts["same"] + counts["periodic"], (copied, counts)
    assert interpolated == counts["interpolated"], (interpolated, counts)


def figure_halo_sources() -> None:
    """The halo cells of a patch coloured by their source, on a periodic hierarchy."""
    mesh = AMRHierarchy(base_shape=(16, 16), periodic=(True, False))
    p1 = mesh.add_patch(level=1, box=((0, 8), (0, 24)))
    p2 = mesh.add_patch(level=1, box=((24, 32), (0, 16)))
    mesh.allocate_field("u", halo=1)
    kinds = halo_sources(mesh, p2, "u", 1)
    check_halo_sources(mesh, p2, "u", kinds)

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    grid(ax, ((0, 16), (0, 16)), 1)
    for patch in (p1, p2):
        grid(ax, patch.box, 2, color="#DDDDDD", lw=0.3)
    for (i, j), kind in kinds.items():
        style = dict(facecolor=SOURCE[kind], edgecolor="white", linewidth=0.5, zorder=3)
        if kind == "periodic":
            style.update(hatch="////", edgecolor="white")
        rect(ax, i / 2, j / 2, (i + 1) / 2, (j + 1) / 2, **style)
    for patch in (p1, p2):
        box_rect(
            ax, patch.box, 2, facecolor="none", edgecolor=INK, linewidth=1.5, zorder=4
        )
    rect(ax, 0, 0, 16, 16, facecolor="none", edgecolor=INK, linewidth=1.0, zorder=4)
    # the seam between p2's two ranks (split along x: level-1 cells 24..28 and 28..32)
    ax.plot([14, 14], [0, 8], color=INK, lw=1.0, ls="--", zorder=5)
    rect(ax, 13.5, 0, 14.5, 8, facecolor=SOURCE["neighbour"], alpha=0.45, zorder=3)
    note(ax, 2.0, 12.6, "patch 1, level 1", fontsize=9)
    note(ax, 13.9, 9.3, "patch 2, level 1, on ranks a | b", fontsize=9)
    note(ax, 8.0, 14.6, "level 0: the base patch", fontsize=9, color="#777777")
    # periodic images: the halo at x = 32 (level 1) is the column x = 0 of patch 1
    rect(
        ax,
        0,
        0,
        0.5,
        8,
        facecolor="none",
        edgecolor="#1F7A70",
        linewidth=1.4,
        linestyle=":",
        zorder=5,
    )
    for y in (2.0, 6.0):
        arrow(ax, (0.0, y), (-1.2, y), color="#1F7A70", lw=1.2, style="-")
        arrow(ax, (17.4, y), (16.6, y), color="#1F7A70", lw=1.2)
    note(ax, -0.6, 4.0, "wraps\naround", fontsize=8, color="#1F7A70")
    note(ax, 17.0, 4.0, "wraps\naround", fontsize=8, color="#1F7A70")
    note(
        ax,
        8.0,
        10.6,
        "periodic in x: the halo column at x = 32 (level 1) is the dotted column x = 0 of patch 1",
        fontsize=8.5,
        color="#1F7A70",
    )
    legend = [
        Patch(facecolor=SOURCE["same"], label="a patch of the same level (copy plan)"),
        Patch(
            facecolor=SOURCE["periodic"],
            hatch="////",
            edgecolor="white",
            label="its periodic image (copy plan)",
        ),
        Patch(
            facecolor=SOURCE["interpolated"],
            label="interpolated from the level below (copy plan into a buffer, then local)",
        ),
        Patch(
            facecolor=SOURCE["neighbour"],
            alpha=0.6,
            label="the other block of the patch (mpiarray update_halos)",
        ),
        Patch(
            facecolor=SOURCE["outside"],
            label="outside the domain: left to the boundary conditions",
        ),
    ]
    ax.legend(
        handles=legend,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8.5,
        frameon=False,
        title="the halo cells of patch 2",
        title_fontsize=9,
    )
    blank(ax, (-1.6, 17.8), (-0.8, 16.5))
    save(fig, "halo_sources")


# 5. The coarse buffer of the interpolation ------------------------------------------- #


def figure_interpolation() -> None:
    """The coarse buffer of the interpolation, with two stencils (one clipped at a wall)."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    # coarse cells 10..14 x 0..4 of level 0; the fine patch starts at level-1 cell 24 (x = 12)
    grid(ax, ((10, 14), (0, 4)), 1, color="#BBBBBB", lw=0.6)
    for i in range(10, 14):
        for j in range(4):
            ax.plot(i + 0.5, j + 0.5, "o", color="#777777", ms=5, zorder=4)
    rect(ax, 12, 0, 14, 4, facecolor="white", zorder=2)
    grid(ax, ((24, 28), (0, 8)), 2, color="#DDDDDD", lw=0.4)
    rect(ax, 12, 0, 14, 4, facecolor="none", edgecolor=INK, linewidth=1.5, zorder=5)
    ax.plot([10, 14], [0, 0], color=INK, lw=1.5, zorder=5)
    note(ax, 13.0, 4.3, "fine patch (level 1)", fontsize=9)
    note(ax, 12.0, -0.45, "wall of the domain (y = 0)", fontsize=8.5, color="#777777")
    # fine halo cells x in [11.5, 12), j = 0..7 (level-1 index 23), with their stencils
    buffer_box = (
        (11, 13),
        (0, 4),
    )  # lower corners 11..11 and 0..2 (clipped at the wall) + 2
    rect(
        ax,
        *(buffer_box[0][0], buffer_box[1][0], buffer_box[0][1], buffer_box[1][1]),
        facecolor=SOURCE["interpolated"],
        alpha=0.25,
        zorder=1,
    )
    rect(
        ax,
        11,
        0,
        13,
        4,
        facecolor="none",
        edgecolor="#B5891A",
        linewidth=1.6,
        linestyle="--",
        zorder=5,
    )
    for jf in range(8):
        y0 = jf / 2
        rect(
            ax,
            11.5,
            y0,
            12.0,
            y0 + 0.5,
            facecolor=SOURCE["interpolated"],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    # two stencils: a fine cell inside, and one at the wall (clipped: linear extrapolation)
    for jf, color in ((3, "#8E6B00"), (0, "#C0392B")):
        position = (jf + 0.5) / 2 - 0.5
        lower = int(np.floor(position))
        lower = min(max(lower, 0), 16 - 2)
        fx, fy = 11.75, jf / 2 + 0.25
        ax.plot(fx, fy, "s", color=color, ms=6, zorder=6)
        for ci in (11, 12):
            for cj in (lower, lower + 1):
                ax.plot([fx, ci + 0.5], [fy, cj + 0.5], color=color, lw=1.0, zorder=5)
                ax.plot(
                    ci + 0.5,
                    cj + 0.5,
                    "o",
                    color=color,
                    ms=6,
                    zorder=6,
                    markerfacecolor="white",
                )
    note(
        ax,
        9.95,
        2.0,
        "fine halo cell (23, 3):\nX = 11.25, Y = 1.25\nweights ¾·¼, ¼·¼, ...\nof the four coarse cells",
        fontsize=8.5,
        ha="right",
        color="#8E6B00",
    )
    note(
        ax,
        9.95,
        0.35,
        "fine halo cell (23, 0) at the wall:\nY = −0.25 → the two inside\ncoarse points, extrapolated",
        fontsize=8.5,
        ha="right",
        color="#C0392B",
    )
    note(
        ax,
        12.0,
        4.85,
        "buffer box ((11, 13), (0, 4)): a replicated array on the fine patch's ranks,\nfilled by the copy plan from the coarse patches that overlap it",
        fontsize=8.5,
        color="#8E6B00",
    )
    note(
        ax,
        14.9,
        2.0,
        "after the Alltoallv,\neach rank of the fine patch\ncomputes the weighted sums\nfor the halo cells it stores\n(no further messages)",
        fontsize=8.5,
        ha="left",
        color="#555555",
    )
    blank(ax, (6.5, 17.5), (-0.9, 5.4))
    save(fig, "interpolation")


# 6. Restriction ---------------------------------------------------------------------- #


def figure_restriction() -> None:
    """A fine patch restricted through a replicated buffer onto a coarse patch on two ranks."""
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    # coarse patch 8 x 6 on ranks 0 and 1 (x blocks 0..4, 4..8), at x offset 14
    cx = 14.0
    for rank, (a, b) in enumerate(chunks(8, 2)):
        rect(ax, cx + a, 0, cx + b, 6, facecolor=RANK[rank], alpha=0.25)
        note(ax, cx + (a + b) / 2, 6.4, f"rank {rank}", fontsize=9, color=RANK[rank])
    for i in range(9):
        ax.plot([cx + i] * 2, [0, 6], color="#BBBBBB", lw=0.5)
    for j in range(7):
        ax.plot([cx, cx + 8], [j] * 2, color="#BBBBBB", lw=0.5)
    rect(ax, cx, 0, cx + 8, 6, facecolor="none", edgecolor=INK, linewidth=1.5)
    ax.plot([cx + 4] * 2, [0, 6], color=INK, lw=1.0, ls="--")
    # the cells under the fine patch: coarse 2..6 x 1..5, written by the rank whose block holds them
    for rank, (a, b) in enumerate(((2, 4), (4, 6))):
        rect(ax, cx + a, 1, cx + b, 5, facecolor=RANK[rank], alpha=0.55)
    rect(ax, cx + 2, 1, cx + 6, 5, facecolor="none", edgecolor="#8E6B00", linewidth=1.5)
    note(
        ax,
        cx + 4,
        -0.6,
        "coarse patch (level 0): each rank writes\nthe cells of its own block only",
        fontsize=9,
    )
    # the fine patch 8 x 8 (level 1), on rank 2, at x offset 0
    for i in range(9):
        ax.plot([i * 0.5] * 2, [1, 5], color="#BBBBBB", lw=0.5)
    for j in range(9):
        ax.plot([0, 4], [1 + j * 0.5] * 2, color="#BBBBBB", lw=0.5)
    rect(ax, 0, 1, 4, 5, facecolor=RANK[2], alpha=0.3)
    rect(ax, 0, 1, 4, 5, facecolor="none", edgecolor=INK, linewidth=1.5)
    for i in range(5):
        ax.plot([i] * 2, [1, 5], color=INK, lw=0.8)
    for j in range(5):
        ax.plot([0, 4], [1 + j] * 2, color=INK, lw=0.8)
    note(ax, 2, 6.4, "rank 2", fontsize=9, color=RANK[2])
    note(
        ax,
        2,
        -0.6,
        "fine patch (level 1), 8 × 8 cells:\nthe 2 × 2 groups of a coarse cell",
        fontsize=9,
    )
    # the buffer: a replicated 8 x 8 fine array on ranks 0 and 1
    bx = 6.5
    rect(ax, bx, 1, bx + 4, 5, facecolor="#F3E3B8")
    for i in range(9):
        ax.plot([bx + i * 0.5] * 2, [1, 5], color="#D9C58F", lw=0.5)
    for j in range(9):
        ax.plot([bx, bx + 4], [1 + j * 0.5] * 2, color="#D9C58F", lw=0.5)
    rect(
        ax,
        bx,
        1,
        bx + 4,
        5,
        facecolor="none",
        edgecolor="#B5891A",
        linewidth=1.5,
        linestyle="--",
    )
    note(
        ax,
        bx + 2,
        6.4,
        "buffer on ranks 0 and 1 (replicated)",
        fontsize=9,
        color="#8E6B00",
    )
    note(
        ax,
        bx + 2,
        -0.6,
        "the fine values, all of them\non each rank of the coarse patch",
        fontsize=9,
        color="#8E6B00",
    )
    arrow(ax, (4.1, 3), (bx - 0.1, 3), color="#8E6B00", lw=1.8)
    note(ax, 5.25, 3.5, "copy plan\n(Alltoallv)", fontsize=8.5, color="#8E6B00")
    arrow(ax, (bx + 4.1, 3), (cx + 1.9, 3), color="#8E6B00", lw=1.8)
    note(
        ax,
        12.25,
        3.7,
        "mean of each 2 × 2\n(volume-weighted in r–z),\nlocal",
        fontsize=8.5,
        color="#8E6B00",
    )
    note(
        ax,
        11.0,
        7.6,
        "finest level first, one plan per fine level, so that a level passes on what it received from the one above",
        fontsize=9,
        color="#555555",
    )
    blank(ax, (-0.5, 22.5), (-1.5, 8.0))
    save(fig, "restriction")


# 7. Deposit: migrate, scatter into block and halo, accumulate -------------------------- #


def figure_deposit() -> None:
    """Migration of markers, the deposit into block and halo, and the owner copies."""
    fig, axes = plt.subplots(
        1, 3, figsize=(12.0, 4.6), gridspec_kw={"width_ratios": [1.1, 1, 1]}
    )
    left, middle, right = axes

    # (a) markers coloured by the rank that owns them
    mesh = AMRHierarchy(base_shape=(8, 8))
    fine = mesh.add_patch(level=1, box=((6, 14), (4, 12)))
    ranks = {
        0: chunks(8, 2),
        1: [(0, 8)],
    }  # base on ranks 0, 1 (x blocks); fine patch on rank 2

    def owner(x: float, y: float) -> int:
        (i0, _), (j0, j1) = fine.box
        if i0 / 2 <= x < 7 and j0 / 2 <= y < j1 / 2:
            return 2
        return 0 if int(x) < 4 else 1

    grid(left, ((0, 8), (0, 8)), 1)
    for rank, (a, b) in enumerate(ranks[0]):
        rect(left, a, 0, b, 8, facecolor=RANK[rank], alpha=0.15)
    box_rect(left, fine.box, 2, facecolor=RANK[2], alpha=0.3)
    grid(left, fine.box, 2, lw=0.3)
    box_rect(left, fine.box, 2, facecolor="none", edgecolor=INK, linewidth=1.3)
    rect(left, 0, 0, 8, 8, facecolor="none", edgecolor=INK, linewidth=1.0)
    rng = np.random.default_rng(3)
    xs, ys = rng.uniform(0, 8, 70), rng.uniform(0, 8, 70)
    for x, y in zip(xs, ys, strict=True):
        left.plot(x, y, "o", color=RANK[owner(x, y)], ms=4, zorder=5)
    # markers that moved: drawn where they were, with an arrow to the rank that owns them
    for (x0, y0), (x1, y1) in (
        ((2.4, 6.2), (6.7, 7.4)),
        ((6.6, 1.2), (4.6, 5.2)),
        ((5.2, 2.6), (2.0, 2.0)),
    ):
        left.plot(
            x0,
            y0,
            "o",
            color=RANK[owner(x1, y1)],
            ms=4,
            zorder=5,
            markerfacecolor="white",
        )
        arrow(left, (x0, y0), (x1, y1), color="#888888", lw=1.0, rad=0.2)
    note(left, 2, 8.4, "rank 0", fontsize=9, color=RANK[0])
    note(left, 6, 8.4, "rank 1", fontsize=9, color=RANK[1])
    note(left, 5, 4.7, "rank 2", fontsize=9, color=RANK[2])
    note(
        left,
        4,
        -1.1,
        "owner: the rank of the block of the finest patch\nwhose cell holds the marker; migrate moves\neach marker there (one Alltoallv per array)",
        fontsize=8.5,
    )
    blank(left, (-0.3, 8.3), (-2.2, 8.9))
    left.set_title("(a) migrate", fontsize=10)

    # (b) a marker next to a seam deposits into the halo nodes of its block
    nx, ny = 6, 5
    seam = 3
    for rank, (a, b) in enumerate(((0, seam), (seam, nx))):
        rect(middle, a, 0, b, ny, facecolor=RANK[rank], alpha=0.18)
    grid(middle, ((0, nx), (0, ny)), 1, color="#BBBBBB", lw=0.5)
    rect(middle, 0, 0, nx, ny, facecolor="none", edgecolor=INK, linewidth=1.3)
    middle.plot([seam, seam], [0, ny], color=INK, lw=1.0, ls="--")
    # node blocks along x: split of nx + 1 = 7 nodes over 2 ranks -> 0..4 and 4..7
    (a0, a1), (b0, b1) = chunks(nx + 1, 2)
    for i in range(nx + 1):
        for j in range(ny + 1):
            mine = a0 <= i < a1
            middle.plot(
                i,
                j,
                "o",
                ms=6 if mine else 5,
                color=RANK[0] if mine else "white",
                markeredgecolor=RANK[0] if mine or i == a1 else "#AAAAAA",
                zorder=5,
            )
    px, py = 3.7, 2.4
    middle.plot(px, py, "*", color="#C0392B", ms=14, zorder=6)
    for ci in (3, 4):
        for cj in (2, 3):
            arrow(middle, (px, py), (ci, cj), color="#C0392B", lw=1.0)
    rect(
        middle,
        a1 - 0.5,
        -0.5,
        a1 + 0.5,
        ny + 0.5,
        facecolor=SOURCE["neighbour"],
        alpha=0.35,
        zorder=2,
    )
    note(middle, 1.5, ny + 0.6, "rank 0's nodes", fontsize=9, color=RANK[0])
    note(middle, 5.0, ny + 0.6, "rank 1's nodes", fontsize=9, color=RANK[1])
    note(middle, a1, -0.9, "halo nodes of rank 0", fontsize=8.5, color="#5A3F8F")
    note(
        middle,
        nx / 2,
        -1.9,
        "a marker of rank 0 near the seam: two of its\nfour corners are halo nodes; accumulate_halos\nadds them into rank 1's nodes (Isend/Irecv)",
        fontsize=8.5,
    )
    blank(middle, (-0.8, nx + 0.8), (-2.9, ny + 1.0))
    middle.set_title("(b) deposit into block and halo", fontsize=10)

    # (c) the copies of a node: coarse-fine edge and periodic seam
    grid(right, ((0, 6), (0, 4)), 1, color="#BBBBBB", lw=0.5)
    rect(right, 2, 0, 5, 3, facecolor="white", zorder=2)
    grid(right, ((4, 10), (0, 6)), 2, color="#CCCCCC", lw=0.4)
    rect(right, 2, 0, 5, 3, facecolor="none", edgecolor=INK, linewidth=1.3, zorder=4)
    rect(right, 0, 0, 6, 4, facecolor="none", edgecolor=INK, linewidth=1.0, zorder=4)
    for i in range(7):
        for j in range(5):
            right.plot(i, j, "o", color="#777777", ms=6, zorder=5)
    for i in range(4, 11):
        for j in range(7):
            right.plot(i / 2, j / 2, "o", color=RANK[2], ms=3.5, zorder=6)
    # shared nodes: the coarse copy goes to the fine copy
    for i, j in ((2, 1), (3, 0), (4, 3), (5, 2)):
        arrow(
            right, (i - 0.35, j + 0.35), (i - 0.05, j + 0.05), color="#C0392B", lw=1.4
        )
    # periodic seam: x = 0 and x = 6 are copies, the upper one owns
    for j in (1, 2, 3):
        arrow(
            right,
            (0.0, j + 0.5),
            (5.9, j + 0.5),
            color="#1F7A70",
            lw=0.9,
            rad=-0.12,
            style="-|>",
            alpha=0.8,
        )
    note(right, 3.5, 3.4, "fine patch", fontsize=9, color=RANK[2])
    note(
        right,
        3,
        5.0,
        "periodic x: the copy at x = 0 gives its charge\nto the one at x = L (the upper one owns)",
        fontsize=8.5,
        color="#1F7A70",
    )
    note(
        right,
        3,
        -1.1,
        "a node held by several patches: all copies\nsend their charge to the finest patch's copy\n(one mpiarray.migrate); the others end up zero",
        fontsize=8.5,
    )
    blank(right, (-0.6, 6.6), (-2.2, 5.7))
    right.set_title("(c) accumulate to the owner copy", fontsize=10)
    save(fig, "deposit")


# 8. Regrid: what a transfer copies and what it interpolates -------------------------- #


def figure_regrid() -> None:
    """The cells a regrid copies from the old patches and the ones it interpolates."""
    old = AMRHierarchy(base_shape=(16, 16))
    old.add_patch(level=1, box=((4, 16), (4, 16)))
    old.add_patch(level=1, box=((16, 24), (8, 20)))
    old.allocate_field("u", halo=1)
    for patch in old.patches_at_level(1):
        patch.field("u").local[...] = 1.0
    new = AMRHierarchy(base_shape=(16, 16))
    boxes = [((8, 20), (6, 18)), ((20, 28), (10, 22))]
    for box in boxes:
        new.add_patch(level=1, box=box)
    new.allocate_field("u", halo=1)
    transfer(
        old, new, "u"
    )  # the copied cells keep the 1, the interpolated ones get the base's 0

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    grid(ax, ((0, 16), (0, 16)), 1)
    for patch in new.patches_at_level(1):
        values = np.asarray(patch.field("u").local)
        (i0, _), (j0, _) = patch.box
        for (a, b), value in np.ndenumerate(values):
            color = SOURCE["same"] if value > 0.5 else SOURCE["interpolated"]
            rect(
                ax,
                (i0 + a) / 2,
                (j0 + b) / 2,
                (i0 + a + 1) / 2,
                (j0 + b + 1) / 2,
                facecolor=color,
                edgecolor="white",
                linewidth=0.3,
                zorder=2,
            )
        box_rect(
            ax, patch.box, 2, facecolor="none", edgecolor=INK, linewidth=1.6, zorder=4
        )
    for patch in old.patches_at_level(1):
        box_rect(
            ax,
            patch.box,
            2,
            facecolor="none",
            edgecolor="#777777",
            linewidth=1.4,
            linestyle="--",
            zorder=4,
        )
    rect(ax, 0, 0, 16, 16, facecolor="none", edgecolor=INK, linewidth=1.0)
    legend = [
        Line2D([], [], color="#777777", ls="--", lw=1.4, label="old level-1 patches"),
        Line2D([], [], color=INK, lw=1.6, label="new level-1 patches"),
        Patch(
            facecolor=SOURCE["same"],
            label="copied from an old patch of the level (copy plan)",
        ),
        Patch(
            facecolor=SOURCE["interpolated"],
            label="interpolated from the new level below",
        ),
    ]
    ax.legend(
        handles=legend,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8.5,
        frameon=False,
    )
    note(
        ax,
        8,
        -0.9,
        "tags: one Allreduce (max) per level;  transfer: one copy plan per level, coarsest first",
        fontsize=9,
        color="#555555",
    )
    blank(ax, (-0.5, 16.5), (-1.6, 16.5))
    save(fig, "regrid")


if __name__ == "__main__":
    figure_ranks()
    figure_blocks()
    figure_copy_plan()
    figure_halo_sources()
    figure_interpolation()
    figure_restriction()
    figure_deposit()
    figure_regrid()
