---
title: Building the docs
description: How this documentation site is built from Markdown pages, notebooks and docstrings.
---

The site in `docs/` is built with [Astro](https://astro.build/) and
[Starlight](https://starlight.astro.build/). It has three kinds of pages:

- **Hand-written pages** in `docs/src/content/docs/`, as Markdown or MDX.
- **Tutorials**, converted from the Jupyter notebooks in `tutorials/` by `starlight-notebooks`.
  Each notebook is executed and its outputs (text, images, tables) are written into an MDX page.
- **API reference**, generated from the docstrings by
  [starlight-pydocs](https://ewels.github.io/starlight-pydocs/), one page per module.

The configuration (`defineDocs` in `docs/astro.config.mjs`), the look, the header, the notebook
tool and the build workflow come from the shared
[starlight-theme](https://github.com/max-models/starlight-theme) of the max-models projects;
change them there, and update the version in `docs/package.json` and
`.github/workflows/docs.yml` to pick up a new release.

## Requirements

- Node 22 or newer.
- The package installed with the `docs` extra: `pip install -e ".[docs]"`.

## Commands

```bash
make docs-install     # npm packages and the Python docs extra
make docs-notebooks   # execute tutorials/*.ipynb and convert them to pages
make docs-dev         # live preview at http://localhost:4321/patchwork/
make docs-build       # the static site in docs/dist, as in CI
make docs-preview     # serve docs/dist
make docs-clean       # remove generated pages and build output
```

`make docs-notebooks` only reruns a notebook whose cells changed since the last run; pass
`FORCE=1` to rerun all of them.

## Writing tutorials

Put a notebook in `tutorials/`. Its first `# Heading` becomes the page title and the first
paragraph its description. Cell tags control the output:

| Tag                | Effect                                   |
| ------------------ | ---------------------------------------- |
| `remove-cell`      | the cell is left out of the page         |
| `hide-input`       | the code is hidden, the outputs are kept |
| `fold-input`       | the code is in a closed "Show code" box  |
| `remove-output`    | the outputs are left out                 |
| `raises-exception` | an error output is shown, not a failure  |

Math in `$...$` and `$$...$$` is rendered with KaTeX.
