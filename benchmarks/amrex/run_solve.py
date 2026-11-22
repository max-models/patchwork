"""Run the solver benchmarks B1-B3 (patchwork and AMReX) and append JSON lines.

Usage: ``python run_solve.py --amrex2 <build2>/bench_solve --amrex3 <build3>/bench_solve
--out results.jsonl``. Each case runs in its own process (peak memory per case) with a
time limit; a case that runs over it or fails is recorded with its error.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

# (case, ndim, base cells)
SIZES = {
    "B1": (2, [32, 64, 128, 256, 512]),
    "B2": (3, [16, 32, 48, 64]),
    "B3": (2, [32, 64, 128, 256, 512]),
}
# patchwork backends (and mass) per case; LU stops where its fill-in gets too large
PATCHWORK = {
    "B1": [
        ("lu", "lumped"),
        ("lu", "consistent"),
        ("multigrid", "lumped"),
        ("amg", "lumped"),
        ("petsc", "lumped"),
    ],
    "B2": [
        ("lu", "lumped"),
        ("amg", "lumped"),
        ("petsc", "lumped"),
        ("multigrid", "lumped"),
    ],
    "B3": [("lu", "symmetric"), ("lu", "conservative")],
}
LU_LIMIT = {"B2": 32}


def run(command: list[str], timeout: float) -> dict:
    """One case; its JSON line, or the failure."""
    try:
        done = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return {"failed": f"timeout {timeout} s"}
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    if done.returncode or not lines:
        return {"failed": (done.stderr or done.stdout).strip().splitlines()[-1:]}
    return json.loads(lines[-1])


def main() -> None:
    """Run all cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amrex2", required=True)
    parser.add_argument("--amrex3", required=True)
    parser.add_argument("--out", default="results_solve.jsonl")
    parser.add_argument("--cases", nargs="*", default=list(SIZES))
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args()

    with open(args.out, "a") as out:
        for case in args.cases:
            ndim, sizes = SIZES[case]
            mode = "cell" if case == "B3" else "nodal"
            amrex = args.amrex2 if ndim == 2 else args.amrex3
            for n in sizes:
                runs = [
                    (
                        {"code": "amrex"},
                        [
                            amrex,
                            f"mode={'abec' if mode == 'cell' else 'nodal'}",
                            f"n_cell={n}",
                            "tiny_profiler.enabled=0",
                        ],
                    )
                ]
                for backend, variant in PATCHWORK[case]:
                    # B3's variant is the coarse-fine treatment, the others' the mass
                    option = "--coarse-fine" if case == "B3" else "--mass"
                    if backend == "lu" and n > LU_LIMIT.get(case, 10**9):
                        continue
                    runs.append(
                        (
                            {
                                "code": "patchwork",
                                "backend": backend,
                                ("coarse_fine" if case == "B3" else "mass"): variant,
                            },
                            [
                                sys.executable,
                                str(HERE / "bench_solve.py"),
                                "--mode",
                                mode,
                                "--ndim",
                                str(ndim),
                                "--n",
                                str(n),
                                "--backend",
                                backend,
                                option,
                                variant,
                            ],
                        )
                    )
                for label, command in runs:
                    result = {"case": case, "ndim": ndim, "n_cell": n, **label}
                    result.update(run(command, args.timeout))
                    out.write(json.dumps(result) + "\n")
                    out.flush()
                    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
