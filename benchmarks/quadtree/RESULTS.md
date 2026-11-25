# Results

2026-11-25, a cloud container with 4 cores (Intel Xeon, 2.1 GHz), one thread per rank,
`bench_quadtree.py` defaults: a 64 x 64 base, 3 levels, 200,000 markers (four fifths in a
Gaussian beam of width 0.06 moving by 0.01 per step), `per_cell=20`, 20 timed steps. Times
are the median of 3 runs of the mean ms per step; under MPI on the rank where the part is
largest. "kept" is the share of the patches of a regrid that were there before it.

## Serial, with the solve

| mesh | patches | refined cells | kept | tag | regrid | migrate | deposit | solver | solve | halos | gather | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cluster | 8 | 27,864 | 0% | 73.9 | 10.3 | 22.0 | 57.8 | 114.1 | 6.2 | 4.7 | 31.2 | 320 |
| cluster, `max_patch_size=16` | 134 | 27,864 | 2% | 73.2 | 98.6 | 43.6 | 132.7 | 133.8 | 13.1 | 100.0 | 78.2 | 673 |
| quadtree, block 8 | 386 | 24,714 | 92% | 70.6 | 277.0 | 64.9 | 231.4 | 179.7 | 23.4 | 295.5 | 131.4 | 1274 |
| quadtree, block 16 | 121 | 30,925 | 91% | 72.4 | 75.3 | 40.0 | 126.4 | 135.1 | 12.4 | 94.5 | 74.6 | 631 |
| quadtree, block 32 | 49 | 50,330 | 94% | 72.3 | 26.0 | 31.2 | 88.6 | 204.6 | 13.0 | 43.0 | 53.5 | 532 |

## 4 ranks, without the solve

| mesh | placement | imbalance | tag | regrid | migrate | deposit | halos | gather | total |
|---|---|---|---|---|---|---|---|---|---|
| cluster | balanced | 2.18 | 30.6 | 7.5 | 13.3 | 59.9 | 4.4 | 18.8 | 135 |
| cluster | morton | 2.19 | 32.0 | 7.6 | 12.4 | 57.5 | 4.8 | 23.0 | 137 |
| cluster, `max_patch_size=16` | balanced | 1.01 | 24.2 | 51.7 | 19.1 | 53.9 | 97.5 | 15.8 | 262 |
| cluster, `max_patch_size=16` | morton | 1.01 | 29.9 | 46.3 | 18.4 | 61.7 | 101.4 | 20.1 | 278 |
| quadtree, block 8 | balanced | 1.00 | 27.5 | 214.2 | 32.2 | 91.2 | 296.3 | 24.8 | 686 |
| quadtree, block 8 | morton | 1.00 | 30.7 | 189.0 | 30.9 | 95.9 | 273.5 | 25.3 | 645 |
| quadtree, block 16 | balanced | 1.02 | 24.6 | 39.4 | 17.8 | 60.5 | 96.6 | 16.6 | 256 |
| quadtree, block 16 | morton | 1.02 | 27.4 | 47.1 | 19.6 | 56.8 | 95.9 | 18.1 | 265 |
| quadtree, block 32 | balanced | 1.02 | 29.3 | 24.6 | 18.0 | 58.1 | 47.1 | 15.5 | 193 |
| quadtree, block 32 | morton | 1.02 | 25.5 | 17.3 | 14.5 | 49.7 | 36.3 | 15.2 | 158 |

Imbalance: the largest cells per rank over the mean (the base level counted).

## What they say

- **The cost follows the number of patches, not how they are placed.** Regrid, halos,
  deposit and gather grow with the patches: 8 clusters take 320 ms per step, 386 blocks of 8
  cells 1274 ms. At about the same number of patches the quadtree is as fast as clusters
  (block 16, 121 patches: 631 ms; clusters cut to 16 cells, 134 patches: 673 ms), its
  regrid a little faster since most of its patches are copied, not interpolated.
- **What the quadtree keeps is not reused yet.** 91-94% of its patches survive a regrid,
  but the hierarchy is new, so the halo schedules (`build_schedule`, most of "halos"), the
  composite node numbering and the solver are built again from scratch. Reusing them for
  kept patches is where the quadtree could win; picnic's solver already reuses the grids and
  probed operators of kept boxes (see picnic's `adaptive_rebuild_cost.py --method quadtree`).
- **Large blocks refine many cells**: blocks of 32 cover 50,330 cells for the 27,864 the
  clusters need; the solver setup grows with them (205 ms against 114 ms).
- **Morton and balanced placement are within the noise on one node** (about 10% between
  repeats). Morton keeps neighbouring patches and the patches over them on one rank, which
  saves messages between nodes; within a node the messages are cheap and the halo time is
  the schedule, built in Python. The few large clusters cannot be balanced by either (2.2).
- On four ranks tagging and gathering take about half as long as serially, but the deposit
  does not get faster (58 ms serially, 50-96 ms on the slowest rank): the placements balance
  cells, and the beam's patches hold most markers. Marker-weighted cuts of the Morton curve
  are patchwork step 8e.

## Fixed on the way

The regrid's transfer compared every new patch with every old one of its level, and copied
the nodes a new patch shares with its old neighbours although an old patch with the same box
already had them. With the bin index of the halo schedules and without the copies that a
later one overwrites, the regrid of the 386-block mesh went from 866 ms to 277 ms per step
(serial), with the same values.
