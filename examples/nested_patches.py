"""Plot three nested refinement levels and their cell grids."""

from pathlib import Path

from patchwork import AMRHierarchy, plot_patches

mesh = AMRHierarchy(base_shape=(32, 32))
mesh.add_patch(level=1, box=((16, 48), (16, 48)))
mesh.add_patch(level=2, box=((32, 64), (32, 64)))

result = plot_patches(mesh)
if result is not None:
    figure, _ = result
    output = Path(__file__).with_name("nested_patches.png")
    figure.savefig(output, dpi=180)
    print(f"Saved {output}")
