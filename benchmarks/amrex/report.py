"""Markdown tables of the benchmark results: ``python report.py results/*.jsonl``."""

from __future__ import annotations

import json
import sys


def ms(value: float | None) -> str:
    """Seconds as milliseconds, three significant digits."""
    return "" if value is None else f"{1e3 * value:.3g}"


def label(row: dict) -> str:
    """The code and its variant."""
    if row["code"] == "amrex":
        extra = f" bf={row['blocking_factor']}" if "blocking_factor" in row else ""
        return "AMReX" + extra
    variant = row.get("backend", "")
    if row.get("mass") == "consistent":
        variant += ", consistent"
    if row.get("case") == "B3" and row.get("coarse_fine"):
        variant += f", {row['coarse_fine']}"
    return f"patchwork {variant}".strip()


def table(header: list[str], rows: list[list]) -> str:
    """A Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    """Print the tables."""
    rows = [json.loads(line) for path in sys.argv[1:] for line in open(path)]
    out = []
    for case, title in (
        ("B1", "B1: 2D node-centered Poisson"),
        ("B2", "B2: 3D node-centered Poisson"),
        ("B3", "B3: 2D cell-centered, variable coefficient"),
    ):
        body = []
        for r in (r for r in rows if r["case"] == case):
            if "failed" in r:
                body.append(
                    [r["n_cell"], label(r), "", "", "", "", "", "", str(r["failed"])]
                )
                continue
            body.append(
                [
                    r["n_cell"],
                    label(r),
                    r["points"],
                    ms(r["build_s"]),
                    ms(r["first_solve_s"]),
                    ms(r["solve_s"]),
                    f"{r['error']:.2e}",
                    f"{r['peak_mb']:.0f}",
                    r.get("iterations", ""),
                ]
            )
        if body:
            out += [
                f"## {title}",
                "",
                table(
                    [
                        "base",
                        "code",
                        "points",
                        "build ms",
                        "first solve ms",
                        "solve ms",
                        "max error",
                        "peak MB",
                        "iterations / note",
                    ],
                    body,
                ),
                "",
            ]
    body = []
    for r in (r for r in rows if r["case"] == "B4"):
        body.append(
            [
                r["ndim"],
                r["n_cell"],
                r["tile"],
                label(r),
                r.get("fine_patches", ""),
                ms(r.get("first_s")),
                ms(r.get("fill_s")),
                ms(r.get("restrict_s")),
                r.get("failed", ""),
            ]
        )
    if body:
        out += [
            "## B4: halos and restriction (one rank)",
            "",
            table(
                [
                    "dim",
                    "base",
                    "tile",
                    "code",
                    "fine patches",
                    "first (schedule) ms",
                    "fill ms",
                    "restrict ms",
                    "note",
                ],
                body,
            ),
            "",
        ]
    body = [
        [
            label(r),
            ms(r.get("regrid_s")),
            r.get("levels", ""),
            r.get("patches", ""),
            r.get("cells", ""),
            r.get("failed", ""),
        ]
        for r in rows
        if r["case"] == "B5"
    ]
    if body:
        out += [
            "## B5: tag and regrid the moving charge (per step)",
            "",
            table(["code", "regrid ms", "levels", "patches", "cells", "note"], body),
            "",
        ]
    body = [
        [
            r["ndim"],
            r["count"],
            label(r),
            ms(r.get("migrate_s")),
            ms(r.get("deposit_s")),
            ms(r.get("gather_s")),
            ms(r.get("sort_s")),
            r.get("failed", ""),
        ]
        for r in rows
        if r["case"] == "B6"
    ]
    if body:
        out += [
            "## B6: markers (one rank)",
            "",
            table(
                [
                    "dim",
                    "markers",
                    "code",
                    "migrate ms",
                    "deposit ms",
                    "gather ms",
                    "sort ms",
                    "note",
                ],
                body,
            ),
            "",
        ]
    body = []
    for r in (r for r in rows if r["case"] == "B7"):
        if "failed" in r:
            body.append([r["what"], r["ranks"], label(r), "", "", "", str(r["failed"])])
        elif r["what"] == "poisson3d":
            body.append(
                [
                    r["what"],
                    r["ranks"],
                    label(r),
                    ms(r["build_s"]),
                    ms(r["solve_s"]),
                    f"err {r['error']:.1e}",
                    "",
                ]
            )
        elif r["what"] == "halo3d":
            body.append(
                [
                    r["what"],
                    r["ranks"],
                    label(r),
                    ms(r.get("first_s")),
                    ms(r["fill_s"]),
                    f"restrict {ms(r['restrict_s'])}",
                    "",
                ]
            )
        else:
            body.append(
                [
                    r["what"],
                    r["ranks"],
                    label(r),
                    ms(r["migrate_s"]),
                    ms(r["deposit_s"]),
                    f"gather {ms(r['gather_s'])}",
                    "",
                ]
            )
    if body:
        out += [
            "## B7: strong scaling",
            "",
            "poisson3d: build and solve; halo3d: schedule and fill; markers2d: "
            "migrate and deposit.",
            "",
            table(["what", "ranks", "code", "a ms", "b ms", "c", "note"], body),
            "",
        ]
    print("\n".join(out))


if __name__ == "__main__":
    main()
