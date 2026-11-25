# Results

2026-11-25, a cloud container with 4 cores (Intel Xeon, 2.1 GHz), one thread per rank,
`bench_quadtree.py` defaults: a 64 x 64 base, 3 levels, 200,000 markers (four fifths in a
Gaussian beam of width 0.06 moving by 0.01 per step), `per_cell=20`, 20 timed steps. Times
are the median of 3 runs of the mean ms per step; under MPI on the rank where the part is
largest. "kept" is the share of the patches of a regrid that were there before it. "tag"
counts the markers on one level more than it tags, for the cost of `morton, markers`.

## Serial, with the solve

With the halo geometry of kept patches reused (patchwork 10c; the numbers in brackets are
the halos without it, from the first run of this benchmark).

| mesh | patches | refined cells | kept | tag | regrid | migrate | deposit | solver | solve | halos | gather | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cluster | 8 | 27,864 | 0% | 106.2 | 10.7 | 20.3 | 55.6 | 113.7 | 5.9 | 8.1 (4.7) | 30.1 | 351 |
| cluster, `max_patch_size=16` | 134 | 27,864 | 2% | 101.5 | 85.5 | 38.7 | 130.8 | 124.3 | 12.5 | 120.7 (100.0) | 74.6 | 689 |
| quadtree, block 8 | 386 | 24,714 | 92% | 105.6 | 284.7 | 65.1 | 229.2 | 184.0 | 23.9 | 273.7 (295.5) | 134.7 | 1301 |
| quadtree, block 16 | 121 | 30,925 | 91% | 100.0 | 66.6 | 37.2 | 113.7 | 130.8 | 11.4 | 78.2 (94.5) | 68.2 | 606 |
| quadtree, block 32 | 49 | 50,330 | 94% | 102.1 | 30.4 | 29.4 | 84.9 | 199.4 | 12.1 | 28.4 (43.0) | 50.9 | 538 |

## 4 ranks, without the solve

`morton, markers` weighs the Morton cuts by `cost`: the markers of `marker_counts`, a marker
costing a tenth of a cell.

| mesh | placement | cells imbalance | markers imbalance | tag | regrid | migrate | deposit | halos | gather | total |
|---|---|---|---|---|---|---|---|---|---|---|
| cluster | balanced | 2.18 | 3.04 | 45.5 | 8.0 | 10.6 | 59.7 | 5.5 | 19.2 | 149 |
| cluster | morton | 2.19 | 3.04 | 46.4 | 9.8 | 10.9 | 62.1 | 5.6 | 23.6 | 159 |
| cluster | morton, markers | 2.19 | 3.04 | 46.7 | 9.5 | 10.1 | 53.1 | 5.6 | 19.8 | 145 |
| cluster, `max_patch_size=16` | balanced | 1.01 | 1.36 | 41.2 | 52.9 | 18.3 | 62.8 | 124.3 | 17.4 | 317 |
| cluster, `max_patch_size=16` | morton | 1.01 | 1.35 | 41.9 | 49.9 | 20.8 | 61.4 | 110.7 | 18.9 | 303 |
| cluster, `max_patch_size=16` | morton, markers | 1.11 | 1.22 | 40.5 | 55.3 | 19.3 | 60.7 | 128.7 | 17.2 | 322 |
| quadtree, block 8 | balanced | 1.00 | 1.31 | 40.8 | 215.3 | 28.7 | 92.6 | 395.5 | 23.4 | 796 |
| quadtree, block 8 | morton | 1.00 | 1.26 | 46.4 | 199.5 | 29.9 | 89.4 | 267.4 | 26.6 | 659 |
| quadtree, block 8 | morton, markers | 1.09 | 1.16 | 46.2 | 220.1 | 31.7 | 90.8 | 297.3 | 29.7 | 716 |
| quadtree, block 16 | balanced | 1.02 | 1.28 | 44.4 | 62.6 | 20.8 | 69.0 | 131.4 | 18.4 | 347 |
| quadtree, block 16 | morton | 1.02 | 1.35 | 46.9 | 63.2 | 27.2 | 63.2 | 105.0 | 20.5 | 326 |
| quadtree, block 16 | morton, markers | 1.13 | 1.24 | 48.6 | 67.4 | 21.0 | 68.1 | 104.0 | 18.0 | 327 |
| quadtree, block 32 | balanced | 1.02 | 1.28 | 42.6 | 23.3 | 16.4 | 58.0 | 44.9 | 13.9 | 199 |
| quadtree, block 32 | morton | 1.02 | 1.55 | 44.6 | 21.3 | 17.0 | 61.3 | 51.5 | 16.4 | 212 |
| quadtree, block 32 | morton, markers | 1.18 | 1.41 | 46.7 | 26.6 | 19.3 | 57.9 | 40.3 | 16.3 | 207 |

Imbalance: the largest cells (base level counted) or markers (after `migrate`) per rank over
the mean.

## What they say

- **The cost follows the number of patches, not how they are placed.** Regrid, halos,
  deposit and gather grow with the patches: 8 clusters take 351 ms per step, 386 blocks of 8
  cells 1301 ms. At about the same number of patches the quadtree is as fast as clusters
  (block 16, 121 patches: 606 ms; clusters cut to 16 cells, 134 patches: 689 ms).
- **Reusing the halo geometry of kept patches saves 7-34% of the halos** (block 8: 296 →
  274 ms, block 16: 95 → 78 ms, block 32: 43 → 28 ms). Not more, for two reasons: about
  a third of the patches still find their geometry again (kept patches whose neighbours or
  coarse patches changed, and new ones), and building the `mpiarray.CopyPlan` of the
  exchange from the copies is now the largest part (about 100 of 250 ms at 386 patches:
  per-copy Python and a pickled `allgather`), and it is built for every new hierarchy. Next:
  a cheaper `CopyPlan` in mpiarray (patchwork 8k). The clusters gain nothing, their patches are
  recut every regrid, and pay a few ms for checking the signatures (8 clusters: 4.7 → 8.1 ms).
- **Large blocks refine many cells**: blocks of 32 cover 50,330 cells for the 27,864 the
  clusters need; the solver setup grows with them (199 ms against 114 ms).
- **Morton placement**: on one node within the noise of `balanced` for most meshes (about
  10% between repeats); the halos of blocks of 8 are faster with it (267 against 396 ms),
  where the patches are small and many.
- **Weighing the cuts by the markers** (`morton, markers`) lowers the marker imbalance (blocks
  of 8: 1.26 → 1.16, 16: 1.35 → 1.24, 32: 1.55 → 1.41, clusters cut to 16: 1.35 → 1.22) for a
  higher cell imbalance (1.09-1.18), as asked. The step time does not change measurably on
  four ranks: with 200,000 markers the deposit and gather take 60-120 ms against the cells'
  halos and regrid. It pays where the markers dominate (many markers per cell, the push of a
  real simulation, many nodes). The 7 clusters cannot be balanced by any cut (2.2, 3.0).

## Per-patch costs, before and after (2026-11-25, later)

The same benchmark (now with an untimed first cycle: the compiled kernels load at their first
call, which the runs above counted in their first step, ~30 ms per step in the deposit), on
`devel` and with three changes: mpiarray builds a `CopyPlan` with array operations over the
copies (116 → 43 ms for the 4422 copies of 392 patches), `deposit`, `interpolate` and the
owners group the markers by patch with one sort instead of a scan of all markers per patch,
and `marker_counts` histograms the finest level once and sums it into the coarser ones.
Serial, median of 3 runs, ms per step:

| mesh | | tag | regrid | migrate | deposit | solver | solve | halos | gather | total |
|---|---|---|---|---|---|---|---|---|---|---|
| cluster | before | 107.1 | 12.4 | 20.4 | 32.1 | 112.3 | 5.9 | 5.6 | 31.2 | 327 |
| | after | 36.6 | 15.1 | 17.8 | 27.6 | 111.4 | 6.1 | 5.6 | 26.2 | 246 (−25%) |
| cluster, `max_patch_size=16` | before | 105.4 | 94.5 | 40.5 | 103.5 | 127.1 | 12.5 | 124.0 | 74.5 | 682 |
| | after | 35.1 | 78.8 | 29.8 | 76.9 | 124.5 | 12.8 | 101.3 | 51.7 | 511 (−25%) |
| quadtree, block 8 | before | 105.6 | 285.9 | 65.9 | 206.4 | 180.2 | 22.7 | 265.8 | 130.9 | 1263 |
| | after | 35.8 | 239.1 | 35.7 | 140.3 | 182.8 | 21.9 | 203.6 | 69.1 | 928 (−27%) |
| quadtree, block 16 | before | 106.5 | 83.9 | 39.1 | 96.8 | 136.2 | 12.7 | 72.3 | 73.6 | 621 |
| | after | 36.6 | 65.9 | 29.5 | 77.2 | 138.4 | 12.4 | 56.8 | 50.9 | 468 (−25%) |
| quadtree, block 32 | before | 103.3 | 25.9 | 29.4 | 63.8 | 208.0 | 12.1 | 31.3 | 50.9 | 525 |
| | after | 35.8 | 27.7 | 26.6 | 54.0 | 200.1 | 11.7 | 20.7 | 42.3 | 419 (−20%) |

- Tagging is three times faster everywhere (one histogram instead of one per level).
- Gather, migrate and deposit lose their scan per patch: blocks of 8 gather in 69 ms instead
  of 131, migrate in 36 instead of 66.
- Halos and the regrid gain from the cheaper `CopyPlan` (blocks of 8: 266 → 204 and
  286 → 239 ms).
- What is left with blocks of 8: the regrid (239 ms, mostly the transfer's geometry per
  patch), the halos (204, the geometry of the patches that do not reuse theirs), the solver
  setup (183, mostly the composite node numbering) and the deposit (140, of which ~65 the
  schedule of `accumulate_halos`, built per hierarchy like the halos').

## Node numbering, accumulation, halo pairs and `regrid(fields=...)` (2026-11-25, later)

Against `devel` after the changes above. `CompositeNodes` finds the cells under finer
patches from one coverage mask per level instead of every patch against every patch of the
next level; the schedule of `accumulate_halos` takes the rank of a node's owner from a table
where the owner is on one rank; the halo copies between two patches of a level are kept per
pair; `in_domain` is a bounds check where the domain fills its box; and `regrid` takes
`fields`, the fields to move (the others are new and zero). Serial, median of 3 runs, ms per
step, total:

| mesh | devel (all fields moved) | this, all fields moved | this, `fields=[]` |
|---|---|---|---|
| cluster | 241 | 245 | 225 (−7%) |
| cluster, `max_patch_size=16` | 496 | 495 | 430 (−13%) |
| quadtree, block 8 | 938 | 837 (−11%) | 632 (−33%) |
| quadtree, block 16 | 474 | 455 (−4%) | 388 (−18%) |
| quadtree, block 32 | 427 | 442 (within the noise) | 401 (−6%) |

- With many patches the node numbering was quadratic: blocks of 8, solver setup 180 →
  111 ms (the numbering itself 143 → 24 ms); the rest of the setup is the assembly and the
  LU factorisation, which do not depend on the patches.
- The deposit's accumulation schedule: 141 → 112 ms with blocks of 8.
- Not moving the fields that are computed again after the regrid (the charge, deposited;
  the potential, solved for) leaves the regrid with the boxes and the new hierarchy:
  blocks of 8, 253 → 84 ms; clusters, 14 → 3 ms.
- The halos did not get faster from the pair cache (195 → 186 ms with blocks of 8, within
  the noise): what is left of their schedule is the `CopyPlan` (~65 ms), the coarse-to-fine
  interpolation of the patches whose neighbourhood changed (~35 ms) and the signatures
  (~33 ms). The next step there: keep the `CopyPlan` of the kept patches, or set up the
  interpolation of all patches at once.

## Fixed on the way

The regrid's transfer compared every new patch with every old one of its level, and copied
the nodes a new patch shares with its old neighbours although an old patch with the same box
already had them. With the bin index of the halo schedules and without the copies that a
later one overwrites, the regrid of the 386-block mesh went from 866 ms to 277 ms per step
(serial), with the same values.
