r"""patchwork's scaling campaign: strong and weak scaling of the benchmarks on a cluster.

The cases are the drivers of ``benchmarks/amrex`` and ``bench_mhd.py`` here, each run as
its own ``srun``/``mpiexec`` with one JSON line, labelled and appended to a results file:

- ``B2``: the 3D node-centered Poisson solve on three levels (`NodalPoisson`, PETSc with
  BoomerAMG; setup and solve times, peak memory);
- ``B4`` (2D and 3D): halos and restriction on many fine patches (``placement="balanced"``);
- ``B6``: markers (migrate, deposit, gather, sort; needs mpiparticles);
- ``MHD``: a step and a regrid of carafe's `MHD2D` on the Orszag-Tang vortex (needs carafe).

Strong scaling keeps each case's size; weak scaling grows it with the ranks (the cells per
axis by ``P^(1/d)``, the markers by ``P``), so the work per rank stays about the same.

Three commands::

    # sbatch scripts, one per node count (slurm-script-generator), and a submit script
    python campaign.py jobs --nodes 1 2 4 8 --tasks-per-node 72 --cluster pitagora \
        --account <account> --setup "source ~/max-models/.venv/bin/activate" --out jobs/

    # the same runs here with mpiexec, small (checks the scripts and the drivers)
    python campaign.py local --ranks 1 2 4 --small --out results/local.jsonl

    # efficiency tables (Markdown) from the results
    python campaign.py report results/local.jsonl > RESULTS.md

Run from this directory. Times are the slowest rank's (the drivers put barriers around
what they time).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
AMREX = HERE.parent / "amrex"


@dataclass(frozen=True)
class Case:
    """A benchmark: its driver, its arguments at a size, and the times it reports."""

    name: str
    script: Path
    ndim: int
    size: int  # the base cells per axis of the strong-scaling run
    small: int  # the same for ``--small``
    multiple: int  # the sizes are rounded to a multiple of this
    metrics: tuple[str, ...]
    extra: tuple[str, ...] = ()
    markers: int = 0  # B6: the markers of the strong-scaling run

    def arguments(self, n: int, markers: int) -> list[str]:
        """The driver's arguments for ``n`` base cells per axis (and ``markers``)."""
        args = [*self.extra, "--n", str(n)]
        if self.markers:
            args += ["--count", str(markers)]
        return args


CASES = {
    case.name: case
    for case in (
        Case(
            "B2",
            AMREX / "bench_solve.py",
            3,
            64,
            16,
            8,
            ("build_s", "solve_s", "peak_mb"),
            ("--mode", "nodal", "--ndim", "3", "--nlevels", "3", "--backend", "petsc"),
        ),
        Case(
            "B4-2d",
            AMREX / "bench_mesh.py",
            2,
            512,
            64,
            64,
            ("first_s", "fill_s", "restrict_s"),
            (
                "--mode",
                "halo",
                "--ndim",
                "2",
                "--tile",
                "16",
                "--placement",
                "balanced",
            ),
        ),
        Case(
            "B4-3d",
            AMREX / "bench_mesh.py",
            3,
            64,
            16,
            16,
            ("first_s", "fill_s", "restrict_s"),
            ("--mode", "halo", "--ndim", "3", "--tile", "8", "--placement", "balanced"),
        ),
        Case(
            "B6",
            AMREX / "bench_particles.py",
            2,
            128,
            32,
            16,
            ("migrate_s", "deposit_s", "gather_s", "sort_s"),
            ("--ndim", "2", "--nlevels", "3"),
            markers=1_000_000,
        ),
        Case(
            "MHD",
            HERE / "bench_mhd.py",
            2,
            256,
            32,
            16,
            ("step_s", "regrid_s", "peak_mb"),
            ("--levels", "2", "--placement", "balanced"),
        ),
    )
}


def sized(case: Case, ranks: int, scaling: str, small: bool) -> tuple[int, int]:
    """The base cells per axis and the markers of ``case`` on ``ranks`` ranks."""
    n = case.small if small else case.size
    markers = case.markers // (100 if small else 1)
    if scaling == "weak":
        grown = n * ranks ** (1.0 / case.ndim)
        n = max(case.multiple, case.multiple * round(grown / case.multiple))
        markers *= ranks
    return n, markers


def runs(
    names: list[str],
    ranks: list[int],
    scalings: list[str],
    small: bool,
    backend: str = "petsc",
) -> list[tuple[dict, list[str]]]:
    """(label, driver command without the launcher) of every run.

    ``backend``: the solver backend of B2 (``"petsc"``: BoomerAMG; ``"amg"``: pyamg).
    """
    out = []
    for name in names:
        case = CASES[name]
        for scaling in scalings:
            for count in ranks:
                n, markers = sized(case, count, scaling, small)
                label = {"case": name, "scaling": scaling, "ranks": count, "n": n}
                command = [str(case.script), *case.arguments(n, markers)]
                if "--backend" in command:
                    command[command.index("--backend") + 1] = backend
                out.append((label, command))
    return out


# -- sbatch scripts ------------------------------------------------------------------


def jobs(args: argparse.Namespace) -> None:
    """One sbatch script per node count, and ``submit.sh`` for all of them."""
    from slurm_script_generator.slurm_script import SlurmScript

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = Path(args.results).resolve()
    scripts = []
    for nodes in args.nodes:
        if nodes == 1:
            # within a node: 1, 2, 4, ... ranks up to the full node
            ranks = [2**k for k in range(int(math.log2(args.tasks_per_node)) + 1)]
            if ranks[-1] != args.tasks_per_node:
                ranks.append(args.tasks_per_node)
        else:
            ranks = [nodes * args.tasks_per_node]
        commands = [
            "set -u",
            "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
            f"cd {shlex.quote(str(HERE))}",
            f"mkdir -p {shlex.quote(str(results.parent))}",
        ]
        if args.setup:
            commands.insert(1, args.setup)
        for label, command in runs(
            args.cases, ranks, args.scaling, False, args.backend
        ):
            label = {**label, "nodes": nodes, "tasks_per_node": args.tasks_per_node}
            launch = (
                f"srun --ntasks={label['ranks']} --cpu-bind=cores "
                f"timeout {args.limit} python {' '.join(map(shlex.quote, command))}"
            )
            commands.append(
                f"{launch} | python {shlex.quote(str(HERE / 'campaign.py'))} label "
                f"{shlex.quote(json.dumps(label))} >> {shlex.quote(str(results))}"
            )
        options = {
            "job_name": f"patchwork-scaling-{nodes}n",
            "nodes": nodes,
            "ntasks_per_node": args.tasks_per_node,
            "cpus_per_task": 1,
            "time": args.time,
            "output": f"patchwork-scaling-{nodes}n-%j.out",
        }
        if args.account:
            options["account"] = args.account
        if args.partition:
            options["partition"] = args.partition
        if args.qos:
            options["qos"] = args.qos
        if args.cluster:
            from slurm_script_generator.clusters import apply_cluster

            apply_cluster(args.cluster, options, uses_gpu=False)
        script = SlurmScript(
            **options,
            modules=list(args.module),
            custom_commands=commands,
        )
        path = out / f"scaling_{nodes}n.sh"
        path.write_text(script.generate_script())
        scripts.append(path.name)
        print(f"{path}: {len(commands) - 4} runs on {nodes} node(s)")
    submit = out / "submit.sh"
    submit.write_text(
        '#!/bin/bash\nset -e\ncd "$(dirname "$0")"\n'
        + "".join(f"sbatch {name}\n" for name in scripts)
    )
    submit.chmod(0o755)
    print(f"{submit}: submits them; the results go to {results}")


# -- local runs -----------------------------------------------------------------------


def local(args: argparse.Namespace) -> None:
    """The runs with mpiexec here, one after the other, appended to ``--out``."""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    launcher = ["mpiexec", "--oversubscribe"]
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        launcher.append("--allow-run-as-root")
    failed = 0
    for label, command in runs(
        args.cases, args.ranks, args.scaling, args.small, args.backend
    ):
        full = [*launcher, "-n", str(label["ranks"]), sys.executable, *command]
        start = time.perf_counter()
        try:
            result = subprocess.run(
                full,
                capture_output=True,
                text=True,
                timeout=args.limit,
                check=False,
                env=env,
            )
            lines = [
                line for line in result.stdout.splitlines() if line.startswith("{")
            ]
            record = {**label, **json.loads(lines[-1])} if lines else None
            reason = result.stderr.strip().splitlines()[-1:] if not lines else []
        except subprocess.TimeoutExpired:
            record, reason = None, [f"over {args.limit} s"]
        if record is None:
            failed += 1
            record = {**label, "failed": True, "reason": " ".join(reason)[-300:]}
        record["wall_s"] = time.perf_counter() - start
        with out.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        status = "failed: " + record["reason"] if record.get("failed") else "ok"
        print(
            f"{label['case']:6s} {label['scaling']:6s} {label['ranks']:4d} ranks "
            f"n={label['n']:4d}  {record['wall_s']:6.1f} s  {status}"
        )
    if failed:
        sys.exit(f"{failed} run(s) failed")


def label(args: argparse.Namespace) -> None:
    """Add the label to the JSON lines on stdin (the drivers' output on a cluster)."""
    extra = json.loads(args.label)
    for line in sys.stdin:
        if line.startswith("{"):
            print(json.dumps({**extra, **json.loads(line)}), flush=True)


# -- report ---------------------------------------------------------------------------


def efficiency(scaling: str, base: dict, record: dict, metric: str) -> float | None:
    """Parallel efficiency of ``record`` against the run on the fewest ranks."""
    t0, t = base.get(metric), record.get(metric)
    if not t0 or not t:
        return None
    if scaling == "strong":
        return t0 * base["ranks"] / (t * record["ranks"])
    return t0 / t


def report(args: argparse.Namespace) -> None:
    """Markdown tables: time and efficiency per case, scaling and rank count."""
    records = []
    for path in args.results:
        with open(path) as handle:
            records += [json.loads(line) for line in handle if line.strip()]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        groups[(record["case"], record["scaling"])].append(record)
    print("# patchwork scaling\n")
    print(
        f"{len(records)} runs from {', '.join(map(str, args.results))}. Times in ms "
        "(the slowest rank's), efficiency against the fewest ranks: strong "
        "`t0 P0 / (t P)`, weak `t0 / t`.\n"
    )
    for (name, scaling), group in sorted(groups.items()):
        case = CASES.get(name)
        metrics = case.metrics if case else ()
        group = sorted(group, key=lambda r: r["ranks"])
        ok = [r for r in group if not r.get("failed")]
        print(f"## {name}, {scaling} scaling\n")
        timed = [m for m in metrics if m.endswith("_s")]
        header = ["ranks", "n", *(m[:-2] + " [ms]" for m in timed)]
        header += [m[:-2] + " eff." for m in timed]
        if "peak_mb" in metrics:
            header.append("peak MB")
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        for record in group:
            if record.get("failed"):
                print(
                    f"| {record['ranks']} | {record['n']} | failed: "
                    f"{record.get('reason', '')[:80]} |"
                )
                continue
            row = [str(record["ranks"]), str(record["n"])]
            row += [f"{1e3 * record[m]:.1f}" if m in record else "" for m in timed]
            for m in timed:
                e = efficiency(scaling, ok[0], record, m) if ok else None
                row.append(f"{e:.2f}" if e is not None else "")
            if "peak_mb" in metrics:
                row.append(f"{record.get('peak_mb', 0):.0f}")
            print("| " + " | ".join(row) + " |")
        print()


def main() -> None:
    """Parse the command and run it."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
        sub.add_argument(
            "--scaling",
            nargs="+",
            default=["strong", "weak"],
            choices=["strong", "weak"],
        )
        sub.add_argument("--limit", type=int, default=1800, help="seconds per run")
        sub.add_argument("--backend", default="petsc", help="B2's solver backend")

    sub = commands.add_parser("jobs", help="write the sbatch scripts")
    common(sub)
    sub.add_argument("--nodes", type=int, nargs="+", default=[1, 2, 4, 8])
    sub.add_argument("--tasks-per-node", type=int, required=True)
    sub.add_argument("--time", default="04:00:00")
    sub.add_argument("--account")
    sub.add_argument("--partition")
    sub.add_argument("--qos")
    sub.add_argument("--cluster", help="a slurm-script-generator preset")
    sub.add_argument("--module", action="append", default=[])
    sub.add_argument("--setup", help="a shell line before the runs (a venv, say)")
    sub.add_argument("--results", default="results/scaling.jsonl")
    sub.add_argument("--out", default="jobs")
    sub.set_defaults(run=jobs)

    sub = commands.add_parser("local", help="run here with mpiexec")
    common(sub)
    sub.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4])
    sub.add_argument("--small", action="store_true", help="the small sizes")
    sub.add_argument("--out", default="results/local.jsonl")
    sub.set_defaults(run=local)

    sub = commands.add_parser("label", help="label the JSON lines on stdin")
    sub.add_argument("label")
    sub.set_defaults(run=label)

    sub = commands.add_parser("report", help="Markdown tables of results")
    sub.add_argument("results", nargs="+")
    sub.set_defaults(run=report)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
