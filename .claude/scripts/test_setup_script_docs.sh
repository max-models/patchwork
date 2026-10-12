#!/usr/bin/env bash
# Copy the repository, run setup_project.sh with a new name and build the docs site of the copy,
# to check that the rename reaches the Astro config and the API pages.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
rsync -a --exclude .git --exclude node_modules --exclude dist --exclude .astro --exclude .venv "$REPO/" "$WORK/"
cd "$WORK"
bash setup_project.sh my-test-app
grep -q "const pythonPackage = 'my_test_app'" docs/astro.config.mjs
grep -q "/api/my_test_app/" docs/src/components/Header.astro
grep -q "/api/my_test_app/index.html" .github/workflows/docs.yml
grep -q "src/my_test_app/__init__.py" release-please-config.json
grep -q "source = \[ \"my_test_app\" \]" pyproject.toml
grep -q "from my_test_app.main import main" docs/src/content/docs/getting-started/quickstart.md
cp -cR "$REPO/docs/node_modules" docs/node_modules 2>/dev/null || cp -R "$REPO/docs/node_modules" docs/node_modules
cd docs && DOCS_PYTHON="$PYTHON" npm run build 2>&1 | tail -25
test -f dist/api/my_test_app/main/index.html
echo "setup script + docs build: OK"
