# patchwork on a cluster: strong and weak scaling

`campaign.py` runs patchwork's benchmarks over many ranks and nodes and turns the results
into efficiency tables. The cases:

| Case | Driver | What is timed |
|---|---|---|
| B2 | `../amrex/bench_solve.py --mode nodal --ndim 3` | `NodalPoisson` on three levels: setup (numbering, assembly, AMG setup) and solve; peak memory |
| B4-2d, B4-3d | `../amrex/bench_mesh.py --mode halo` | `exchange_halos` (first call with the schedule, then repeated) and `restrict` on many fine patches |
| B6 | `../amrex/bench_particles.py` | markers: migrate, deposit, gather, sort (mpiparticles) |
| MHD | `bench_mhd.py` | a step of carafe's `MHD2D` (Orszag-Tang, three levels regridded on the current) and a regrid |

Strong scaling keeps the size of a case (the sizes in `CASES`); weak scaling grows the cells
per axis by `P^(1/d)` and the markers by `P`. A run is its own `srun` (or `mpiexec`) with a
time limit and prints one JSON line, which `campaign.py label` tags with the case, the
scaling, the ranks and the nodes and appends to the results file.

## On the cluster

Generate the scripts on the cluster (they `cd` to this directory), in an environment with
patchwork (`mpi`, `pyccel`, `petsc` extras), mpiparticles, carafe and slurm-script-generator:

```sh
python campaign.py jobs --nodes 1 2 4 8 --tasks-per-node 72 \
    --cluster pitagora --account <account> \
    --setup "source /path/to/max-models/.venv/bin/activate" --out jobs/
jobs/submit.sh                       # one job per node count
python campaign.py report results/scaling.jsonl > RESULTS.md
```

The one-node job runs 1, 2, 4, ... ranks up to the full node (the scaling inside a node);
the others run every case on all their cores. `--cluster` takes slurm-script-generator's
presets (partition and QOS); `--partition`, `--qos`, `--module` and `--time` set them by
hand. `--cases` and `--scaling` choose a subset.

## Here

```sh
python campaign.py local --ranks 1 2 4 --small --backend amg --out results/local.jsonl
python campaign.py report results/local.jsonl
```

runs the same commands with `mpiexec`, at small sizes (`--small`), to check the drivers and the
scripts before a cluster job (a few seconds per run). `--backend amg` uses pyamg for B2
where PETSc is not installed.

## What the results are for

They set the targets of the open performance steps of `plans/patchwork_plan.md` (todo
repo): 8n (the solver setup distributed over the ranks), what is left of 8k (halos and
restriction without a Python loop per patch), 8g (setup and memory of the solvers) and later
8i (GPU). A profile of one run shows where a case spends its time:

```sh
scope-profiler run --enable-tag patchwork --enable-tag carafe -o prof.h5 \
    bench_mhd.py --n 128 --levels 2
scope-profiler inspect prof.h5 --sort total --top 15
```

One core, the quick case of carafe's `examples/plasmoid_reconnection.py` (25 patches,
18k leaf cells): HLLD in NumPy 44% of a step, `FluxRegister.update` (once per conserved
field) 26%, halos and restriction 13%; the refined run then costs 1.6-2x the uniform mesh
of the finest level per step, with 55% of its cells.
