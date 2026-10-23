"""The locate_points kernel: Pyccel and NumPy versions with the same arguments.

``kernel(...)`` runs the Pyccel version, or the NumPy version if Pyccel is unavailable
(see ``cunumpy.kernels.set_host_kernel_implementation`` to choose).
"""

import cunumpy as xp

from patchwork.kernels._build import KERNEL_OPTIONS

kernel = xp.kernels.Kernel.from_folder(__name__, **KERNEL_OPTIONS)
