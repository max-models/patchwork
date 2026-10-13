"""A charge that moves, a mesh that follows it, and what every step costs.

A Gaussian charge crosses a grounded box. At every step its density is set on
the patches, the cells across which it changes by more than a threshold are
tagged, the hierarchy is regridded, the node-centered solver is built (numbered
and factorized) for the new mesh, and the potential is solved. A uniform mesh
as fine as the finest patches is factorized once for the whole run, but has
many more unknowns to solve for at every step.
"""

import time
from pathlib import Path

import numpy as np

from patchwork import AMRHierarchy, NodalPoisson, gradient_tags, plot_patches

BASE = 32
STEPS = 20
WIDTH = 0.002  # of the Gaussian, in squared length
THRESHOLD = 10.0  # change of the density across a cell that asks for finer cells


def density(x, y, t):
    """The charge density at time ``t``: a Gaussian of unit charge."""
    cx, cy = 0.25 + 0.5 * t, 0.3 + 0.4 * t
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / WIDTH) / (np.pi * WIDTH)


def deposit(mesh, t):
    """Set the density on every patch, as a particle code would deposit it."""
    for patch in mesh.local_patches:
        x, y = patch.local_node_coordinates(mesh, "rho")
        patch.field("rho").local[...] = density(x, y, t)


timings = {name: 0.0 for name in ("deposit", "tag", "regrid", "build", "solve")}


def timed(name, call, *args, **kwargs):
    """Call ``call`` and add its time to ``timings[name]``."""
    start = time.perf_counter()
    result = call(*args, **kwargs)
    timings[name] += time.perf_counter() - start
    return result


mesh = AMRHierarchy(base_shape=(BASE, BASE))
mesh.allocate_field("rho", centering="node")
deposit(mesh, 0.0)
while True:  # one more level per regrid, until the charge is resolved
    deeper = mesh.regrid(gradient_tags(mesh, "rho", THRESHOLD), max_patch_size=32)
    deposit(deeper, 0.0)
    if deeper.max_level == mesh.max_level:
        break
    mesh = deeper
finest = mesh.max_level
fine = BASE * 2**finest

unknowns = []
depths = set()
for step in range(1, STEPS + 1):
    t = step / STEPS
    timed("deposit", deposit, mesh, t)
    tags = timed("tag", gradient_tags, mesh, "rho", THRESHOLD)
    mesh = timed("regrid", mesh.regrid, tags, max_patch_size=32)
    timed("deposit", deposit, mesh, t)
    solver = timed("build", NodalPoisson, mesh)
    potential = timed("solve", solver.solve, "rho", potential="phi")
    unknowns.append(solver.free.size)
    depths.add(mesh.max_level)
adaptive = dict(timings)

# the uniform mesh: one factorization, a solve per step
for name in timings:
    timings[name] = 0.0
uniform = AMRHierarchy(base_shape=(fine, fine))
uniform.allocate_field("rho", centering="node")
uniform_solver = timed("build", NodalPoisson, uniform)
for step in range(1, STEPS + 1):
    timed("deposit", deposit, uniform, step / STEPS)
    reference = timed("solve", uniform_solver.solve, "rho")

# the adaptive potential on the base nodes, against the uniform one there
nodes = solver.nodes
on_base = (nodes.to_level(0) @ potential).reshape(nodes.level_shape(0))
uniform_nodes = uniform_solver.nodes
on_fine = (uniform_nodes.to_level(0) @ reference).reshape(fine + 1, fine + 1)
reference_base = on_fine[:: 2**finest, :: 2**finest]
difference = np.abs(on_base - reference_base).max() / np.abs(reference_base).max()

print(
    f"{STEPS} steps, {sorted(depths)} levels of patches on a {BASE} x {BASE} base, "
    f"{len(mesh.patches)} patches at the end"
)
print(f"{'':12s} {'adaptive':>10s} {'uniform':>10s}   (ms per step)")
for name in ("deposit", "tag", "regrid", "build", "solve"):
    per_step = adaptive[name] / STEPS * 1e3
    other = timings[name] / (1 if name == "build" else STEPS) * 1e3
    note = "  (once)" if name == "build" else ""
    print(f"{name:12s} {per_step:10.2f} {other:10.2f}{note}")
total = sum(adaptive.values()) / STEPS * 1e3
uniform_total = (timings["deposit"] + timings["solve"]) / STEPS * 1e3
print(f"{'step':12s} {total:10.2f} {uniform_total:10.2f}   (+ the uniform build once)")
print(
    f"unknowns: {min(unknowns)}-{max(unknowns)} adaptive, "
    f"{uniform_solver.free.size} uniform ({fine} x {fine} cells)"
)
print(
    f"largest difference of the potentials on the base nodes: {difference:.1e} (relative)"
)
# regridding every k steps: the solve and the deposit every step, the rest every k-th
every = {}
for k in (1, 5, 10, 20):
    fixed = (adaptive["deposit"] / 2 + adaptive["solve"]) / STEPS
    changing = (adaptive["tag"] + adaptive["regrid"] + adaptive["build"]) / STEPS
    every[k] = (fixed + changing / k) * 1e3
print(
    "estimated ms per step when regridding every k steps: "
    + ", ".join(f"k={k}: {ms:.0f}" for k, ms in every.items())
)

result = plot_patches(mesh, field="phi", show_grid=False)
if result is not None:
    figure, axes = result
    axes.set_title(f"Potential of the charge at t = 1, {len(mesh.patches)} patches")
    output = Path(__file__).with_name("regrid_moving_charge.png")
    figure.savefig(output, dpi=150)
    print(f"Saved {output}")
