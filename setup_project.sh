#!/usr/bin/env bash
# Turn the template into a project of your own:
#
#     bash setup_project.sh my-app
#
# replaces "template-python" with "my-app" and "template_python" with "my_app" in every tracked
# file, moves the package src/app to src/my_app, points the entry point, tests, docs and the API
# reference at it, and removes the workflow that tests this script.
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <app-name>"
  exit 1
fi

APPNAME="$1"
APPNAME_UNDERSCORE="${APPNAME//-/_}"
PACKAGE="app"

if [[ ! "$APPNAME_UNDERSCORE" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]]; then
  echo "Error: '$APPNAME' is not a valid package name (letters, digits, '-' and '_' only)"
  exit 1
fi

if [[ ! -d "src/${PACKAGE}" ]]; then
  echo "Error: src/${PACKAGE} does not exist"
  exit 1
fi

# Files to edit: everything except git internals, the npm packages and build output.
files() {
  grep -rl "$1" . \
    --exclude-dir=.git --exclude-dir=node_modules --exclude-dir=dist --exclude-dir=.astro \
    --exclude-dir=.cache --exclude-dir=__pycache__ --exclude-dir=.venv --exclude-dir=env \
    --exclude=setup_project.sh || true
}

replace() {  # replace <pattern> <replacement> in every file that contains <pattern>
  files "$1" | while IFS= read -r file; do
    LC_ALL=C sed -i.bak "s#$1#$2#g" "$file"
    rm -f "${file}.bak"
  done
}

# 1. The project name: pyproject.toml, README, docs, workflows, release-please, ...
replace "template-python" "${APPNAME}"
replace "template_python" "${APPNAME_UNDERSCORE}"

# 2. Move the package
mv "src/${PACKAGE}" "src/${APPNAME_UNDERSCORE}"

# 3. Imports, the entry point, the API reference and the docs links
replace "from ${PACKAGE}\\.main import main" "from ${APPNAME_UNDERSCORE}.main import main"
replace "${PACKAGE}\\.main:main" "${APPNAME_UNDERSCORE}.main:main"
replace "${PACKAGE}\\.main\\.main" "${APPNAME_UNDERSCORE}.main.main"
replace "/api/${PACKAGE}/" "/api/${APPNAME_UNDERSCORE}/"
replace "src/${PACKAGE}/" "src/${APPNAME_UNDERSCORE}/"
replace "source = \[ \"${PACKAGE}\" \]" "source = [ \"${APPNAME_UNDERSCORE}\" ]"
replace "^const pythonPackage = '${PACKAGE}';" "const pythonPackage = '${APPNAME_UNDERSCORE}';"

# 4. Remove the workflow that tests this script
rm -f ".github/workflows/test_setup_script.yml"

echo "Done:"
echo "  Replaced 'template-python' → '${APPNAME}'"
echo "  Replaced 'template_python' → '${APPNAME_UNDERSCORE}'"
echo "  Moved src/${PACKAGE} → src/${APPNAME_UNDERSCORE}"
echo "  Updated the imports, entry point, docs and API reference"
echo "  Removed .github/workflows/test_setup_script.yml"
echo
echo "Next: update the GitHub links in pyproject.toml, docs/astro.config.mjs and"
echo "docs/src/components/NotebookHeader.astro, then remove this script with: rm setup_project.sh"
