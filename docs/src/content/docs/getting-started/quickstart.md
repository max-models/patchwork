---
title: Quickstart
description: Run the command line entry point of template-python.
---

First, ensure that `template-python` is [installed](/template-python/getting-started/installation/).

## Basic usage

After installation, you can run the application with:

```bash
template-python
```

which calls `app.main.main` and prints a greeting.

## Use the package from Python

```python
from app.main import main

main()
```

See the [API reference](/template-python/api/app/) for every module and function.
