# template-python


<!-- README.md is generated from README.qmd: edit the .qmd and run `make readme`. -->

[![Tests](https://github.com/max-models/template-python/actions/workflows/test_pytest.yml/badge.svg)](https://github.com/max-models/template-python/actions/workflows/test_pytest.yml)
[![Static
analysis](https://github.com/max-models/template-python/actions/workflows/static_analysis.yml/badge.svg)](https://github.com/max-models/template-python/actions/workflows/static_analysis.yml)
[![Docs](https://github.com/max-models/template-python/actions/workflows/docs.yml/badge.svg)](https://max-models.github.io/template-python/)
[![codecov](https://codecov.io/gh/max-models/template-python/branch/main/graph/badge.svg)](https://codecov.io/gh/max-models/template-python)
[![PyPI](https://img.shields.io/pypi/v/template-python.png)](https://pypi.org/project/template-python/)
[![Python](https://img.shields.io/pypi/pyversions/template-python.png)](https://pypi.org/project/template-python/)

Template repository for Python projects: a `src/` package with a console
entry point, `pytest` tests, GitHub Actions for tests, static analysis,
tutorials, documentation and PyPI publishing, and an
[Astro](https://astro.build/) +
[Starlight](https://starlight.astro.build/) documentation site.

Documentation: <https://max-models.github.io/template-python/>

## Use the template

Create a repository from this template, then rename the package:

``` bash
bash setup_project.sh my-app
rm setup_project.sh
```

This replaces `template-python` with `my-app` everywhere, moves
`src/app` to `src/my_app`, and points the entry point, tests and docs at
it.

## Install

With [uv](https://docs.astral.sh/uv/):

``` bash
make install    # uv sync --extra dev, plus the pre-commit hooks
uv run template-python
```

Or create and activate a Python environment (3.10 or newer):

``` bash
python -m venv env
source env/bin/activate
pip install --upgrade pip
```

Install the code and requirements with pip:

``` bash
pip install -e .
```

Run the code with:

``` bash
template-python
```

The `test`, `docs` and `dev` extras install the test runner, the
documentation tooling and the linters:

``` bash
pip install -e ".[dev]"
```

## Development

Formatting and linting use [ruff](https://docs.astral.sh/ruff/), type
checking [pyright](https://microsoft.github.io/pyright/), run by
[pre-commit](https://pre-commit.com/) and in CI:

``` bash
make lint     # ruff check, ruff format --check, pyright
make test     # pytest with coverage
```

Commit messages follow [Conventional
Commits](https://www.conventionalcommits.org/); see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Build docs

The documentation in `docs/` is an Astro + Starlight site: hand-written
pages, the notebooks in `tutorials/` executed and published as pages,
and the API reference generated from the docstrings with
[starlight-pydocs](https://ewels.github.io/starlight-pydocs/). It needs
Node 22 or newer.

``` bash
make docs-install     # npm packages and the Python docs extra
make docs-notebooks   # execute tutorials/*.ipynb and convert them to pages
make docs-dev         # live preview at http://localhost:4321/template-python/
make docs-build       # the static site in docs/dist
```

## Build the README

`README.md` is rendered from `README.qmd` with
[Quarto](https://quarto.org/):

``` bash
make readme
```

## Releases

Before merging a release to `main`, update the version in
`pyproject.toml`, `src/app/__init__.py` and `CITATION.cff` (including
its release date), and add the release notes to `CHANGELOG.md`. The push
to `main` creates a GitHub release with a `vX.Y.Z` tag and publishes the
package to PyPI with trusted publishing (OIDC). The one-time PyPI and
GitHub configuration is described in the [publishing
guide](https://max-models.github.io/template-python/development/publishing/).
