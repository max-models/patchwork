# Results

Run 2026-11-22 on a 4-core container (Intel Xeon @ 2.10 GHz, one thread per rank, 15 GB),
AMReX 25.11 (CMake, MPI, no OpenMP or GPU, `-O3` Release), GCC 13.3, Open MPI 4.1.6,
PETSc 3.26.1 with hypre, patchwork at 51c2920 with its compiled (Pyccel) kernels,
mpiparticles 0.1.0. AMReX's MLMG stops at a relative residual of 1e-10; patchwork's
iterative backends at their defaults (the errors match LU's). Every solve has a new source,
so no backend starts from the last answer. The patchwork peak memory includes about 83 MB of
interpreter and imports. Times are medians of 5 (3 for the largest) repeats of one run per
case, not of several runs: differences under about 20% are within the noise.

## Findings

**Solves (B1-B3), one core.**

- *2D, repeated solves:* patchwork's LU solves 10-28x faster than MLMG (512 base, 657k
  unknowns: 120 ms against 1.15 s), at the same error, but takes 6.5 s to factorize; it is
  ahead from about the seventh solve on the same mesh (PIC with a fixed mesh), behind for
  one solve per mesh (regridding every step). The PETSc backend (BoomerAMG) is 1.2-1.7x
  faster per solve than MLMG in 2D, the `multigrid` backend as fast, `amg` 1.4x slower.
- *3D:* the `multigrid` backend (patchwork's geometric multigrid over the patches) already
  works in 3D and solves as fast as MLMG (64 base, 752k unknowns: 1.37 s against 1.44 s);
  BoomerAMG is 3.3x slower, `amg` 1.7x. LU stops at 32 (22 s, 1.1 GB).
- *Setup and memory are the gap, not the solve:* at 650-750k unknowns building a patchwork
  solver takes 1.8-2.4 s in 2D (6.5 s for LU) and 6.3-9.8 s in 3D, 2-5 times a solve there
  (AMReX: 10 ms, plus a first solve that costs about one solve), and patchwork peaks at
  0.75-2.0 GB where AMReX needs 140 MB (5-14x).
- *Cell-centered (B3):* with `coarse_fine="conservative"` patchwork has AMReX's error (3.60e-6
  against 3.56e-6 at 512) and solves 2x faster after an 8.4 s factorization; the default
  `"symmetric"` is 3-4x less accurate in the max norm (the coarse-fine faces, not the
  coefficient average: the arithmetic mean and a constant coefficient give the same error).
- *Discretization:* the lumped (five-point) nodal Laplacian has 1.4x AMReX's error, the
  consistent (Q1) one 0.6x; AMReX's nodal stencil sits between them.

**Mesh operations (B4, B5), one core.**

- `exchange_halos` and `restrict` cost 6-33x AMReX's `FillPatchTwoLevels` and
  `average_down`, about 15 µs per fine patch in 2D and 50 µs in 3D: Python overhead per
  patch (0.66 ms against 15 ms for 1024 patches).
- Building the halo schedule grows faster than the number of patches: 12 ms for 16 patches,
  0.8 s for 256, 9.9 s for 1024 (2D); 4.7 s for 512 patches in 3D. AMReX builds its
  communication metadata in well under a millisecond. This is quadratic and the first thing
  to fix.
- Tagging and regridding the moving charge: 9 ms per step against 0.22 ms (40x), to the same
  depth, with fewer cells covered (5592 against 5940 with blocking factor 2, 8768 with 8).

**Markers (B6), one core.** Deposit and gather are 5-9x slower than AMReX's
`ParticleToMesh`/`MeshToParticle` (1e7 markers in 2D: 1.5 s against 0.2 s), sorting 2-3x;
migrating on one rank is faster (mpiparticles only tests ownership; AMReX re-bins every
marker into its box). AMReX deposits each level on its own: the charge of markers near a
coarse-fine edge that falls outside the level's boxes is lost (0.3% of it in the 1e5 run)
unless the code sums it onto the coarse level itself; patchwork's `accumulate_halos` does
that.

**Scaling (B7), 1 to 4 ranks.** 3D Poisson solve: AMReX 2.6x on 4 ranks, patchwork's PETSc
backend 2.3x, but its 8.5-9.6 s build does not scale (assembled on every rank). Halos: AMReX
2.1x, patchwork 1.7x, and the schedule build grows with the ranks (121 to 181 ms). 4M
markers: deposit scales 4.6x in patchwork (3.2x in AMReX) but stays 5x slower; migrate does
not scale (106 to 123 ms). Beyond 4 ranks, and on GPUs, not measured.

**Decisions this suggests.**

- 8g: matrix-free multigrid in 3D is not needed for speed (the `multigrid` backend matches
  MLMG); what is missing is a cheaper setup and less memory: Galerkin products and assembled
  matrices instead of stencils.
- New: the halo schedule build is quadratic in the patches; the per-patch Python overhead of
  halos, restriction and regridding is 6-40x AMReX's.
- `coarse_fine="conservative"` as `CellOperator`'s default: same cost, AMReX's accuracy.
- Markers: the deposit and gather (mpifield's kernels called patch by patch) are the 5-9x.

## B1: 2D node-centered Poisson

| base | code | points | build ms | first solve ms | solve ms | max error | peak MB | iterations / note |
|---|---|---|---|---|---|---|---|---|
| 32 | AMReX | 3267 | 0.197 | 4.23 | 3.92 | 2.74e-03 | 34 | 8 |
| 32 | patchwork lu | 2689 | 12.8 | 0.344 | 0.141 | 3.02e-03 | 89 |  |
| 32 | patchwork lu, consistent | 2689 | 17.5 | 0.461 | 0.281 | 1.29e-03 | 92 |  |
| 32 | patchwork multigrid | 2689 | 13.6 | 4.1 | 3.25 | 3.02e-03 | 90 |  |
| 32 | patchwork amg | 2689 | 29.9 | 5.01 | 4.42 | 3.02e-03 | 96 |  |
| 32 | patchwork petsc | 2689 | 43.9 | 2.43 | 2.57 | 3.02e-03 | 118 |  |
| 64 | AMReX | 12675 | 0.225 | 12.8 | 12.2 | 6.14e-04 | 35 | 8 |
| 64 | patchwork lu | 10497 | 45.2 | 1.16 | 0.768 | 7.56e-04 | 103 |  |
| 64 | patchwork lu, consistent | 10497 | 73.4 | 1.8 | 1.29 | 3.22e-04 | 109 |  |
| 64 | patchwork multigrid | 10497 | 56.4 | 12.6 | 11.3 | 7.56e-04 | 99 |  |
| 64 | patchwork amg | 10497 | 59.3 | 15.4 | 14.9 | 7.56e-04 | 104 |  |
| 64 | patchwork petsc | 10497 | 59.3 | 6.96 | 7.87 | 7.56e-04 | 127 |  |
| 128 | AMReX | 50700 | 0.5 | 53.8 | 51.3 | 1.44e-04 | 40 | 9 |
| 128 | patchwork lu | 41473 | 208 | 5.07 | 3.38 | 1.89e-04 | 155 |  |
| 128 | patchwork lu, consistent | 41473 | 244 | 6.47 | 4.21 | 8.05e-05 | 185 |  |
| 128 | patchwork multigrid | 41473 | 140 | 50.8 | 51.3 | 1.89e-04 | 132 |  |
| 128 | patchwork amg | 41473 | 158 | 85.1 | 78.9 | 1.89e-04 | 134 |  |
| 128 | patchwork petsc | 41473 | 146 | 38 | 41.8 | 1.89e-04 | 164 |  |
| 256 | AMReX | 202800 | 2.01 | 289 | 278 | 3.49e-05 | 60 | 8 |
| 256 | patchwork lu | 164865 | 970 | 24.7 | 20.4 | 4.71e-05 | 389 |  |
| 256 | patchwork lu, consistent | 164865 | 1.5e+03 | 36.9 | 32.7 | 2.01e-05 | 521 |  |
| 256 | patchwork multigrid | 164865 | 447 | 218 | 209 | 4.71e-05 | 262 |  |
| 256 | patchwork amg | 164865 | 471 | 323 | 305 | 4.71e-05 | 263 |  |
| 256 | patchwork petsc | 164865 | 474 | 150 | 163 | 4.71e-05 | 309 |  |
| 512 | AMReX | 811200 | 10.2 | 972 | 1.15e+03 | 8.57e-06 | 139 | 9 |
| 512 | patchwork lu | 657409 | 6.45e+03 | 120 | 120 | 1.18e-05 | 1419 |  |
| 512 | patchwork lu, consistent | 657409 | 8.66e+03 | 163 | 166 | 5.03e-06 | 1869 |  |
| 512 | patchwork multigrid | 657409 | 2.38e+03 | 1.09e+03 | 1.07e+03 | 1.18e-05 | 790 |  |
| 512 | patchwork amg | 657409 | 1.79e+03 | 1.57e+03 | 1.57e+03 | 1.18e-05 | 748 |  |
| 512 | patchwork petsc | 657409 | 1.77e+03 | 776 | 864 | 1.18e-05 | 889 |  |

## B2: 3D node-centered Poisson

| base | code | points | build ms | first solve ms | solve ms | max error | peak MB | iterations / note |
|---|---|---|---|---|---|---|---|---|
| 16 | AMReX | 14739 | 0.278 | 24.1 | 23.5 | 1.55e-02 | 36 | 8 |
| 16 | patchwork lu | 13281 | 266 | 4.04 | 2.57 | 1.17e-02 | 133 |  |
| 16 | patchwork amg | 13281 | 116 | 21.3 | 21.2 | 1.17e-02 | 117 |  |
| 16 | patchwork petsc | 13281 | 151 | 50.8 | 42.3 | 1.17e-02 | 142 |  |
| 16 | patchwork multigrid | 13281 | 91.9 | 17.5 | 15.1 | 1.17e-02 | 117 |  |
| 32 | AMReX | 107811 | 1.1 | 190 | 179 | 3.58e-03 | 48 | 8 |
| 32 | patchwork lu | 97985 | 2.2e+04 | 105 | 107 | 3.06e-03 | 1090 |  |
| 32 | patchwork amg | 97985 | 668 | 134 | 179 | 3.06e-03 | 328 |  |
| 32 | patchwork petsc | 97985 | 1.02e+03 | 475 | 435 | 3.06e-03 | 341 |  |
| 32 | patchwork multigrid | 97985 | 716 | 133 | 123 | 3.06e-03 | 328 |  |
| 48 | AMReX | 352947 | 3.37 | 681 | 614 | 1.53e-03 | 79 | 8 |
| 48 | patchwork amg | 321697 | 2.55e+03 | 707 | 636 | 1.37e-03 | 897 |  |
| 48 | patchwork petsc | 321697 | 3.51e+03 | 1.73e+03 | 2.09e+03 | 1.37e-03 | 919 |  |
| 48 | patchwork multigrid | 321697 | 2.42e+03 | 517 | 529 | 1.37e-03 | 897 |  |
| 64 | AMReX | 823875 | 7.93 | 1.45e+03 | 1.44e+03 | 8.44e-04 | 138 | 8 |
| 64 | patchwork amg | 752001 | 6.59e+03 | 2.41e+03 | 2.42e+03 | 7.69e-04 | 1944 |  |
| 64 | patchwork petsc | 752001 | 9.82e+03 | 5.33e+03 | 4.72e+03 | 7.69e-04 | 2021 |  |
| 64 | patchwork multigrid | 752001 | 6.33e+03 | 1.46e+03 | 1.37e+03 | 7.69e-04 | 1944 |  |

## B3: 2D cell-centered, variable coefficient

| base | code | points | build ms | first solve ms | solve ms | max error | peak MB | iterations / note |
|---|---|---|---|---|---|---|---|---|
| 32 | AMReX | 3072 | 0.479 | 2.49 | 2.15 | 9.08e-04 | 34 | 9 |
| 32 | patchwork lu, symmetric | 2560 | 18.3 | 0.592 | 0.427 | 2.60e-03 | 90 |  |
| 32 | patchwork lu, conservative | 2560 | 18.9 | 0.539 | 0.381 | 9.32e-04 | 90 |  |
| 64 | AMReX | 12288 | 0.597 | 5.94 | 5.4 | 2.22e-04 | 35 | 7 |
| 64 | patchwork lu, symmetric | 10240 | 56.4 | 1.65 | 1.33 | 6.50e-04 | 107 |  |
| 64 | patchwork lu, conservative | 10240 | 67.5 | 2.17 | 1.7 | 2.28e-04 | 101 |  |
| 128 | AMReX | 49152 | 1.45 | 26.1 | 20.3 | 5.63e-05 | 41 | 7 |
| 128 | patchwork lu, symmetric | 40960 | 393 | 12.2 | 9.15 | 1.63e-04 | 175 |  |
| 128 | patchwork lu, conservative | 40960 | 321 | 10.7 | 7.09 | 5.71e-05 | 152 |  |
| 256 | AMReX | 196608 | 4.68 | 112 | 90.6 | 1.42e-05 | 64 | 7 |
| 256 | patchwork lu, symmetric | 163840 | 1.65e+03 | 51.9 | 48.2 | 4.06e-05 | 443 |  |
| 256 | patchwork lu, conservative | 163840 | 1.58e+03 | 51.9 | 47.2 | 1.44e-05 | 365 |  |
| 512 | AMReX | 786432 | 18.7 | 466 | 434 | 3.56e-06 | 155 | 7 |
| 512 | patchwork lu, symmetric | 655360 | 9.75e+03 | 237 | 230 | 1.39e-05 | 1449 |  |
| 512 | patchwork lu, conservative | 655360 | 8.37e+03 | 198 | 204 | 3.60e-06 | 1207 |  |

## B4: halos and restriction (one rank)

| dim | base | tile | code | fine patches | first (schedule) ms | fill ms | restrict ms | note |
|---|---|---|---|---|---|---|---|---|
| 2 | 128 | 32 | AMReX | 16 |  | 0.0147 | 0.0387 |  |
| 2 | 128 | 32 | patchwork | 16 | 12.2 | 0.39 | 0.353 |  |
| 2 | 128 | 16 | AMReX | 64 |  | 0.0435 | 0.0535 |  |
| 2 | 128 | 16 | patchwork | 64 | 58.2 | 1.05 | 0.712 |  |
| 2 | 128 | 8 | AMReX | 256 |  | 0.21 | 0.12 |  |
| 2 | 128 | 8 | patchwork | 256 | 796 | 4.76 | 2.88 |  |
| 2 | 128 | 4 | AMReX | 1024 |  | 0.659 | 0.29 |  |
| 2 | 128 | 4 | patchwork | 1024 | 9.91e+03 | 15 | 8.67 |  |
| 3 | 32 | 16 | AMReX | 8 |  | 0.103 | 0.0438 |  |
| 3 | 32 | 16 | patchwork | 8 | 11.9 | 0.603 | 0.526 |  |
| 3 | 32 | 8 | AMReX | 64 |  | 0.211 | 0.06 |  |
| 3 | 32 | 8 | patchwork | 64 | 135 | 3.25 | 1.28 |  |
| 3 | 32 | 4 | AMReX | 512 |  | 1.3 | 0.204 |  |
| 3 | 32 | 4 | patchwork | 512 | 4.67e+03 | 26.9 | 6.78 |  |

## B5: tag and regrid the moving charge (per step)

| code | regrid ms | levels | patches | cells | note |
|---|---|---|---|---|---|
| AMReX bf=2 | 0.222 | 5 | 15 | 5940 |  |
| AMReX bf=8 | 0.222 | 5 | 13 | 8768 |  |
| patchwork | 8.98 | 5 | 11 | 5592 |  |

## B6: markers (one rank)

| dim | markers | code | migrate ms | deposit ms | gather ms | sort ms | note |
|---|---|---|---|---|---|---|---|
| 2 | 100000 | AMReX | 6.41 | 1.23 | 1.06 | 2.5 |  |
| 2 | 100000 | patchwork | 1.11 | 9.88 | 10 | 8.22 |  |
| 2 | 1000000 | AMReX | 63.7 | 19.4 | 24.5 | 51.1 |  |
| 2 | 1000000 | patchwork | 17.4 | 108 | 112 | 108 |  |
| 2 | 10000000 | AMReX | 604 | 204 | 183 | 613 |  |
| 2 | 10000000 | patchwork | 276 | 1.48e+03 | 1.33e+03 | 1.59e+03 |  |
| 3 | 100000 | AMReX | 9.48 | 2.18 | 1.79 | 3.9 |  |
| 3 | 100000 | patchwork | 1.86 | 16 | 14.7 | 11.4 |  |
| 3 | 1000000 | AMReX | 88.9 | 28.6 | 27.1 | 59.2 |  |
| 3 | 1000000 | patchwork | 21.9 | 144 | 139 | 147 |  |
| 3 | 10000000 | AMReX | 841 | 303 | 304 | 974 |  |
| 3 | 10000000 | patchwork | 348 | 1.72e+03 | 1.54e+03 | 1.85e+03 |  |

## B7: strong scaling

poisson3d: build and solve; halo3d: schedule and fill; markers2d: migrate and deposit.

| what | ranks | code | a ms | b ms | c | note |
|---|---|---|---|---|---|---|
| poisson3d | 1 | AMReX | 8.7 | 1.4e+03 | err 8.4e-04 |  |
| poisson3d | 1 | patchwork petsc | 9.57e+03 | 5.49e+03 | err 7.7e-04 |  |
| halo3d | 1 | AMReX |  | 0.6 | restrict 0.45 |  |
| halo3d | 1 | patchwork | 121 | 4.43 | restrict 4.27 |  |
| markers2d | 1 | AMReX | 231 | 84.3 | gather 73.8 |  |
| markers2d | 1 | patchwork | 106 | 634 | gather 606 |  |
| poisson3d | 2 | AMReX | 5.08 | 853 | err 8.4e-04 |  |
| poisson3d | 2 | patchwork petsc | 9.12e+03 | 3.16e+03 | err 7.7e-04 |  |
| halo3d | 2 | AMReX |  | 0.448 | restrict 0.289 |  |
| halo3d | 2 | patchwork | 169 | 4.65 | restrict 4.65 |  |
| markers2d | 2 | AMReX | 124 | 43.1 | gather 36.9 |  |
| markers2d | 2 | patchwork | 249 | 299 | gather 269 |  |
| poisson3d | 4 | AMReX | 2.79 | 531 | err 8.4e-04 |  |
| poisson3d | 4 | patchwork petsc | 8.49e+03 | 2.4e+03 | err 7.7e-04 |  |
| halo3d | 4 | AMReX |  | 0.284 | restrict 0.171 |  |
| halo3d | 4 | patchwork | 181 | 2.61 | restrict 3.34 |  |
| markers2d | 4 | AMReX | 87.5 | 26.1 | gather 24.3 |  |
| markers2d | 4 | patchwork | 123 | 139 | gather 125 |  |
