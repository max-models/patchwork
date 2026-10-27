"""Compiled kernels of patchwork: Pyccel versions, with NumPy versions as the fallback.

Each kernel lives in a folder ``patchwork/kernels/<name>/`` with ``<name>_pyccel.py``
(compiled with Pyccel on its first call, the build cached on disk) and
``<name>_numpy.py`` (used where Pyccel is not installed or the build fails). Both take
the same arguments and give the same results. Pyccel is the optional extra
``patchwork-amr[pyccel]``.
"""

import cunumpy as xp

#: The build cache (``PATCHWORK_KERNEL_CACHE``, by default ``~/.cache/patchwork/kernels``)
#: and the options of every kernel folder: ``KERNEL_PACKAGE.from_folder(__name__)``.
KERNEL_PACKAGE = xp.kernels.KernelPackage("patchwork")
