---
title: Installation
description: Create a Python environment and install patchwork.
---

Create and activate a Python environment:

```bash
python -m venv env
source env/bin/activate
pip install --upgrade pip
```

Install the release from PyPI, where the package is called `patchwork-amr` (`patchwork` there
is an unrelated project); the import is `import patchwork`:

```bash
pip install patchwork-amr
```

Or install the package and its requirements from a clone with pip:

```bash
pip install -e .
```

Run the code with:

```bash
patchwork
```

## Optional extras

| Extra  | Installs                                                |
| ------ | ------------------------------------------------------- |
| `mpi`  | mpi4py, for runs under `mpiexec`                         |
| `petsc` | petsc4py, for `NodalPoisson(backend="petsc")`: the solve distributed over the ranks |
| `test` | `pytest` and `coverage`                                 |
| `docs` | the notebook runner and griffe for the API reference    |
| `dev`  | formatters and linters, plus the `test` and `docs` extras |

```bash
pip install -e ".[dev]"
```

## Optional plotting tools

Install Matplotlib with Patchwork when you want to use `plot_patches` or run the
examples that save figures. SciPy, used by the composite Poisson solver, is installed
as a core dependency:

```bash
pip install -e ".[plot]"
```

For multi-rank runs, install the optional MPI bindings too:

```bash
pip install -e ".[mpi,plot]"
```

`mpiarray` uses its serial communicator when MPI is not installed or no MPI
launcher is active. The AMR examples therefore run as ordinary Python scripts.
