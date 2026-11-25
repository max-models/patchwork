"""Run `bench_quadtree.py` for several meshes and placements and print a table of the times.

    python benchmarks/quadtree/run.py --out /tmp/quadtree            # serial, with the solve
    python benchmarks/quadtree/run.py --out /tmp/quadtree --ranks 4  # halos, deposit, regrid

Every configuration runs ``--repeats`` times under ``scope-profiler run``; the table gives
the median over the repeats of the mean time per step of each region (ms; under MPI on the
rank where it is largest), and the mesh of
the JSON line of the run: patches, refined cells, the share of patches a regrid keeps and
the load imbalance (largest cells per rank over the mean) and the marker imbalance (largest
markers per rank over the mean, after `migrate`). ``morton, markers`` weighs the Morton cuts
with the markers, a marker costing a tenth of a cell. Profiles are written to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
BENCH = HERE / "bench_quadtree.py"
PROFILER = Path(sys.executable).with_name("scope-profiler")
PARTS = ("tag", "regrid", "migrate", "deposit", "solver", "solve", "halos", "gather")

#: the placements compared under MPI, and their arguments
PLACEMENTS = {
    "balanced": ["--placement", "balanced"],
    "morton": ["--placement", "morton"],
    "morton, markers": ["--placement", "morton", "--marker-cost", "0.1"],
}

#: (label, arguments) of the meshes compared
MESHES = [
    ("cluster", ["--method", "cluster"]),
    ("cluster, max 16", ["--method", "cluster", "--max-patch-size", "16"]),
    ("quadtree 8", ["--method", "quadtree", "--block-size", "8"]),
    ("quadtree 16", ["--method", "quadtree", "--block-size", "16"]),
    ("quadtree 32", ["--method", "quadtree", "--block-size", "32"]),
]


def run_once(arguments: list[str], ranks: int, output: Path) -> tuple[dict, dict]:
    """One profiled run: the JSON line of the mesh and the mean ms per step of each part."""
    command = [str(PROFILER), "run", "-o", str(output), str(BENCH), *arguments]
    if ranks > 1:
        command = ["mpiexec", "-n", str(ranks), "--oversubscribe", *command]
    env = os.environ | {
        "OMPI_ALLOW_RUN_AS_ROOT": "1",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
        "OMP_NUM_THREADS": "1",
    }
    result = subprocess.run(
        command, capture_output=True, text=True, env=env, check=True
    )
    mesh = next(
        json.loads(line)
        for line in result.stdout.splitlines()
        if line.startswith("{") and '"method"' in line
    )
    times: dict[str, float] = {}
    for rank in range(ranks):  # the slowest rank's mean per step of each part
        inspected = subprocess.run(
            [str(PROFILER), "inspect", str(output), "--json", "--top", "50"]
            + (["--ranks", str(rank)] if ranks > 1 else []),
            capture_output=True,
            text=True,
            check=True,
        )
        for region in json.loads(inspected.stdout)["regions"]["items"]:
            name = region["name"]
            if name.startswith("quadtree_bench."):
                part = name.split(".", 1)[1]
                times[part] = max(times.get(part, 0.0), 1e3 * region["avg"])
    return mesh, times


def main() -> None:
    """Run every configuration and print the table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--placements", nargs="+", default=None)
    args, bench = parser.parse_known_args()
    args.out.mkdir(parents=True, exist_ok=True)
    placements = args.placements or (
        list(PLACEMENTS) if args.ranks > 1 else ["balanced"]
    )
    extra = list(bench) + ([] if args.ranks == 1 else ["--no-solve"])

    header = [
        "mesh",
        "placement",
        "patches",
        "refined",
        "kept",
        "imbalance",
        "markers",
        *PARTS,
        "total",
    ]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for label, arguments in MESHES:
        for placement in placements:
            runs = []
            for repeat in range(args.repeats):
                name = f"{label}_{placement}_{args.ranks}_{repeat}".replace(", ", "_")
                name = name.replace(" ", "") + ".h5"
                runs.append(
                    run_once(
                        [*arguments, *PLACEMENTS[placement], *extra],
                        args.ranks,
                        args.out / name,
                    )
                )
            mesh = runs[0][0]
            times = {
                part: float(np.median([t.get(part, 0.0) for _, t in runs]))
                for part in PARTS
            }
            total = sum(times.values())
            cells = [
                f"{mesh['patches']:.0f}",
                f"{mesh['refined_cells']:.0f}",
                f"{mesh['kept']:.0%}",
                f"{mesh['load_imbalance']:.2f}",
                f"{mesh['marker_imbalance']:.2f}",
                *(f"{times[part]:.1f}" if times[part] else "-" for part in PARTS),
                f"{total:.1f}",
            ]
            print(f"| {label} | {placement} | " + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    main()
