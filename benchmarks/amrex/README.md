# patchwork against AMReX

The same cases in patchwork (Python scripts here) and AMReX (C++ drivers in `cpp/`), on the
same boxes, one JSON line per run. `RESULTS.md` has the numbers of the last run.

| Case | What | patchwork | AMReX |
|---|---|---|---|
| B1 | 2D node-centered Poisson, 3 levels | `NodalPoisson` (`lu`, `multigrid`, `amg`, `petsc`) | `MLNodeLaplacian` |
| B2 | the same in 3D | `NodalPoisson` (`lu`, `amg`, `petsc`, `multigrid`) | `MLNodeLaplacian` |
| B3 | 2D cell-centered `-div(a grad u) + u`, variable `a` | `CellOperator` | `MLABecLaplacian` |
| B4 | halos and restriction, 16 to 1024 fine patches | `exchange_halos`, `restrict` | `FillPatchTwoLevels`, `average_down` |
| B5 | tagging and regridding a moving charge | `gradient_tags`, `regrid` | `AmrMesh::MakeNewGrids`, `FillPatch` |
| B6 | markers: migrate, deposit, gather, sort | mpiparticles `ParticleStore` + patchwork | `AmrParticleContainer` |
| B7 | B2, B4, B6 on several ranks | `placement="balanced"` | AMReX's default distribution |

The solves (B1-B3) have a Gaussian `exp(-|x - 0.5|^2 / 0.01)` as the exact solution with
zero Dirichlet faces; each finer level covers the middle half of the one below. The error is
the max norm against the exact solution over the leaf points (the cells under a finer level
are left out on the AMReX side: they hold the mean of the fine cells).

## Build AMReX and the drivers

```sh
git clone --depth 1 --branch 25.11 https://github.com/AMReX-Codes/amrex.git
for d in 2 3; do
  cmake -S amrex -B amrex/build$d -DAMReX_SPACEDIM=$d -DAMReX_MPI=ON -DAMReX_OMP=OFF \
    -DAMReX_FORTRAN=OFF -DAMReX_EB=OFF -DAMReX_TINY_PROFILE=ON -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$PWD/amrex-install$d
  cmake --build amrex/build$d -j --target install
  cmake -S cpp -B drv$d -DCMAKE_BUILD_TYPE=Release \
    -DAMReX_DIR=$PWD/amrex-install$d/lib/cmake/AMReX
  cmake --build drv$d -j
done
```

patchwork with the `mpi`, `pyccel`, `amg` and `petsc` extras, and `mpiparticles`.

## Run

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python run_solve.py --amrex2 drv2/bench_solve --amrex3 drv3/bench_solve --out results/solve.jsonl
python run_mesh.py --amrex2 drv2 --amrex3 drv3 --max-ranks 4 --out results/mesh.jsonl
python report.py results/solve.jsonl results/mesh.jsonl > RESULTS.md
```

Each run is its own process (peak memory per run) with a time limit; a run over it is
recorded as failed. The whole set takes about half an hour on four cores.
