"""The options every kernel folder passes to :meth:`cunumpy.kernels.Kernel.from_folder`.

The ``__init__.py`` of a kernel folder ``patchwork/kernels/<name>/`` declares its
kernel with these::

    kernel = xp.kernels.Kernel.from_folder(__name__, **KERNEL_OPTIONS)

which finds ``<name>_pyccel.py`` and ``<name>_numpy.py`` in the folder. Nothing is
compiled here: the Pyccel version is compiled on its first call and the build kept in
:data:`KERNEL_CACHE` (under ``PATCHWORK_KERNEL_CACHE``, by default
``~/.cache/patchwork/kernels``), so later processes import it directly. Without Pyccel
the NumPy version runs. Importing this module needs neither Pyccel nor CuPy.
"""

from __future__ import annotations

from typing import Any

import cunumpy as xp

#: The on-disk build cache of patchwork's kernels.
KERNEL_CACHE = xp.kernels.PyccelCache("patchwork")


def _compile(module: Any) -> Any:
    return KERNEL_CACHE.compile(module)


#: ``<name>_pyccel.py`` compiled with Pyccel on first use; the kernels are host
#: kernels only (no CUDA version), called with NumPy arrays.
KERNEL_OPTIONS: dict[str, Any] = {
    "host_suffix": "_pyccel",
    "compile_host": _compile,
    "dispatch": "arrays",
    "missing_cuda": "fallback",
}
