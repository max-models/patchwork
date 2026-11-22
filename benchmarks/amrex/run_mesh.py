"""Run the benchmarks B4-B7 (patchwork and AMReX) and append JSON lines.

Usage: ``python run_mesh.py --amrex2 <build2> --amrex3 <build3> --out results.jsonl``
with the directories of the AMReX drivers (``bench_mesh``, ``bench_particles``,
``bench_solve``). ``--max-ranks`` caps B7 (the machine's cores).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from run_solve import run

HERE = Path(__file__).parent
MPIEXEC = [
    "mpiexec",
    "--allow-run-as-root",
    "--oversubscribe",
    "--bind-to",
    "core",
    "-n",
]


def cases(args: argparse.Namespace) -> list[tuple[dict, list[str]]]:
    """(label, command) of every run."""
    py = sys.executable
    out = []
    amrex = {2: Path(args.amrex2), 3: Path(args.amrex3)}
    # B4: halos and restriction
    for ndim, n, tiles in ((2, 128, (32, 16, 8, 4)), (3, 32, (16, 8, 4))):
        for tile in tiles:
            label = {"case": "B4", "ndim": ndim, "n_cell": n, "tile": tile}
            out.append(
                (
                    label | {"code": "amrex"},
                    [
                        str(amrex[ndim] / "bench_mesh"),
                        "mode=halo",
                        f"n_cell={n}",
                        f"tile={tile}",
                        "tiny_profiler.enabled=0",
                    ],
                )
            )
            out.append(
                (
                    label | {"code": "patchwork"},
                    [
                        py,
                        str(HERE / "bench_mesh.py"),
                        "--mode",
                        "halo",
                        "--ndim",
                        str(ndim),
                        "--n",
                        str(n),
                        "--tile",
                        str(tile),
                    ],
                )
            )
    # B5: regridding the moving charge
    label = {"case": "B5", "ndim": 2}
    for factor in (2, 8):
        out.append(
            (
                label | {"code": "amrex", "blocking_factor": factor},
                [
                    str(amrex[2] / "bench_mesh"),
                    "mode=regrid",
                    f"blocking_factor={factor}",
                    "tiny_profiler.enabled=0",
                ],
            )
        )
    out.append(
        (
            label | {"code": "patchwork"},
            [py, str(HERE / "bench_mesh.py"), "--mode", "regrid"],
        )
    )
    # B6: markers, one rank
    for ndim, n in ((2, 64), (3, 32)):
        for count in (10**5, 10**6, 10**7):
            repeats = 3 if count >= 10**7 else 5
            label = {
                "case": "B6",
                "ndim": ndim,
                "n_cell": n,
                "count": count,
                "ranks": 1,
            }
            out.append(
                (
                    label | {"code": "amrex"},
                    [
                        str(amrex[ndim] / "bench_particles"),
                        f"n_cell={n}",
                        f"count={count}",
                        f"repeats={repeats}",
                        "tiny_profiler.enabled=0",
                    ],
                )
            )
            out.append(
                (
                    label | {"code": "patchwork"},
                    [
                        py,
                        str(HERE / "bench_particles.py"),
                        "--ndim",
                        str(ndim),
                        "--n",
                        str(n),
                        "--count",
                        str(count),
                        "--repeats",
                        str(repeats),
                    ],
                )
            )
    # B7: strong scaling of B2 (3D Poisson), B4 (halos) and B6 (markers)
    ranks = [r for r in (1, 2, 4, 8, 16, 32, 64) if r <= args.max_ranks]
    for r in ranks:
        mpi = [*MPIEXEC, str(r)]
        label = {"case": "B7", "ranks": r}
        out.append(
            (
                label | {"what": "poisson3d", "code": "amrex"},
                [
                    *mpi,
                    str(amrex[3] / "bench_solve"),
                    "mode=nodal",
                    "n_cell=64",
                    "max_grid_size=32",
                    "tiny_profiler.enabled=0",
                ],
            )
        )
        out.append(
            (
                label | {"what": "poisson3d", "code": "patchwork"},
                [
                    *mpi,
                    py,
                    str(HERE / "bench_solve.py"),
                    "--ndim",
                    "3",
                    "--n",
                    "64",
                    "--backend",
                    "petsc",
                    "--placement",
                    "balanced",
                    "--repeats",
                    "3",
                ],
            )
        )
        out.append(
            (
                label | {"what": "halo3d", "code": "amrex"},
                [
                    *mpi,
                    str(amrex[3] / "bench_mesh"),
                    "mode=halo",
                    "n_cell=64",
                    "tile=16",
                    "tiny_profiler.enabled=0",
                ],
            )
        )
        out.append(
            (
                label | {"what": "halo3d", "code": "patchwork"},
                [
                    *mpi,
                    py,
                    str(HERE / "bench_mesh.py"),
                    "--mode",
                    "halo",
                    "--ndim",
                    "3",
                    "--n",
                    "64",
                    "--tile",
                    "16",
                ],
            )
        )
        out.append(
            (
                label | {"what": "markers2d", "code": "amrex"},
                [
                    *mpi,
                    str(amrex[2] / "bench_particles"),
                    "n_cell=128",
                    "count=4000000",
                    "repeats=3",
                    "tiny_profiler.enabled=0",
                ],
            )
        )
        out.append(
            (
                label | {"what": "markers2d", "code": "patchwork"},
                [
                    *mpi,
                    py,
                    str(HERE / "bench_particles.py"),
                    "--n",
                    "128",
                    "--count",
                    "4000000",
                    "--repeats",
                    "3",
                ],
            )
        )
    return [c for c in out if c[0]["case"] in args.cases]


def main() -> None:
    """Run all cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amrex2", required=True)
    parser.add_argument("--amrex3", required=True)
    parser.add_argument("--out", default="results_mesh.jsonl")
    parser.add_argument("--cases", nargs="*", default=["B4", "B5", "B6", "B7"])
    parser.add_argument("--max-ranks", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    with open(args.out, "a") as out:
        for label, command in cases(args):
            result = dict(label)
            result.update(run(command, args.timeout))
            out.write(json.dumps(result) + "\n")
            out.flush()
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
