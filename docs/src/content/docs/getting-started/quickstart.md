---
title: Quickstart
description: Run the command line entry point of patchwork.
---

First, ensure that `patchwork` is [installed](/patchwork/getting-started/installation/).

## Basic usage

After installation, you can run the application with:

```bash
patchwork
```

which calls `patchwork.main.main` and prints a greeting.

## Use the package from Python

```python
from patchwork.main import main

main()
```

See the [API reference](/patchwork/api/patchwork/) for every module and function.
