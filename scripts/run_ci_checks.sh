#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${MERXEN_CI_VENV:-${repo_root}/.ci-venv}"

cd "${repo_root}"

if ! command -v uv >/dev/null 2>&1; then
    python -m pip install uv
fi

if [[ ! -x "${venv_dir}/bin/python" ]]; then
    uv venv --python 3.12 "${venv_dir}"
fi
uv pip install --python "${venv_dir}/bin/python" -r requirements.lock
uv pip install --python "${venv_dir}/bin/python" -e . --no-deps

"${venv_dir}/bin/ruff" check .
"${venv_dir}/bin/ruff" format --check .
"${venv_dir}/bin/mypy" src/
"${venv_dir}/bin/pytest" -m "not slow"
