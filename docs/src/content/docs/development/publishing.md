---
title: Publishing
description: Publish the package to PyPI and the documentation to GitHub Pages.
---

## Documentation

The `Docs` workflow builds the site on every push and pull request to `main` and `devel`. It
executes the notebooks, generates the API reference and uploads the site. Pushes to `devel`
also deploy it to GitHub Pages at https://max-models.github.io/patchwork/.

Enable GitHub Pages once in the repository settings, with "GitHub Actions" as the source.

## Releases and PyPI

Before merging a release to `main`, update the version in `pyproject.toml`,
`src/patchwork/__init__.py` and `CITATION.cff` (including its release date), and add the release notes
to `CHANGELOG.md`. On every push to `main`, the `Release` workflow:

1. tags the release `vX.Y.Z` and creates the GitHub release,
2. builds the package with `uv build` and publishes it to PyPI with trusted publishing (OIDC),
   if that version is not already on PyPI.

### One-time configuration

1. Create an account on [PyPI](https://pypi.org/) and create or claim your project name.
2. In the PyPI project settings go to "Publishing" → "Add a new publisher" and fill in:
   - **PyPI project name**: `patchwork` (or your project name)
   - **Owner**: your GitHub username or organization
   - **Repository name**: `patchwork`
   - **Workflow name**: `release.yml`
   - **Environment name**: `pypi`
3. Optionally, in the GitHub repository settings create an environment named `pypi` whose
   deployment branches are limited to `main`.

### Development against the devel branches

Pull requests to `devel` test against the devel branches of the max-models packages patchwork
depends on (`maybempi` and `mpiarray`), pinned in `uv.lock` through `[tool.uv.sources]`. Pull
requests to `main` and the release use PyPI only (CI sets `UV_NO_SOURCES`), and on them the
`Test Pytest` workflow fails if the built wheel has a direct reference (`name @ git+...`), which
PyPI rejects. Refresh the pinned devel commits with `uv lock --upgrade-package maybempi`, and
reproduce a run against PyPI with `UV_NO_SOURCES=true uv sync --extra test` (it rewrites
`uv.lock`, so `git restore uv.lock` afterwards).

### Coverage

The tests upload coverage to [Codecov](https://codecov.io/). Public repositories need no
configuration; for a private one add a `CODECOV_TOKEN` repository secret.
