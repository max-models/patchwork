# Quadtree patches against clustered ones

`bench_quadtree.py` runs a particle-in-cell cycle on a mesh that follows a moving beam of
markers: tag (`marker_tags`), regrid, migrate, deposit, build the solver, solve, exchange
the halos and gather, each a scope-profiler region. `run.py` runs it for clustered patches
(`method="cluster"`, with and without `max_patch_size`) and quadtree blocks of 8, 16 and 32
cells, and prints a table; under MPI it compares `placement="balanced"` with
`placement="morton"` (without the solve, which the LU backend does not distribute).

```sh
python benchmarks/quadtree/run.py --out /tmp/quadtree               # serial, with the solve
python benchmarks/quadtree/run.py --out /tmp/quadtree --ranks 4     # halos, deposit, regrid
python benchmarks/quadtree/run.py --out /tmp/quadtree --steps 40 --base 128   # any option of bench_quadtree.py
```

`RESULTS.md` has the numbers of the last run.
