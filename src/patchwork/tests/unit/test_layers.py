"""The three layers import downwards only: mesh, then spaces, then solvers."""

import ast
from pathlib import Path

import pytest

import patchwork

ROOT = Path(patchwork.__file__).parent

#: The layers a layer must not import when its modules load (lazy imports inside
#: functions and imports for type checking are allowed).
FORBIDDEN = {
    "mesh": ("patchwork.spaces", "patchwork.solvers"),
    "spaces": ("patchwork.solvers",),
}


def _load_time_imports(path: Path) -> list[str]:
    """The modules a file imports at load time: top level, outside ``TYPE_CHECKING``."""
    names = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
        elif isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
    return names


@pytest.mark.parametrize(
    "path",
    [path for layer in FORBIDDEN for path in sorted((ROOT / layer).glob("*.py"))],
    ids=lambda path: f"{path.parent.name}/{path.name}",
)
def test_layer_imports_only_downwards(path: Path) -> None:
    forbidden = FORBIDDEN[path.parent.name]
    upward = [name for name in _load_time_imports(path) if name.startswith(forbidden)]
    assert not upward, f"{path.name} imports {upward} when it loads"


def test_top_level_exports_every_layer() -> None:
    """The public names of the layers are also patchwork's."""
    import patchwork.mesh
    import patchwork.solvers
    import patchwork.spaces

    for layer in (patchwork.mesh, patchwork.spaces, patchwork.solvers):
        missing = [name for name in layer.__all__ if name not in patchwork.__all__]
        assert not missing, f"{layer.__name__}: {missing} not in patchwork.__all__"
