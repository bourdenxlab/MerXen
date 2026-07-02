#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${MERXEN_CI_VENV:-${repo_root}/.ci-venv}"
install_mode="${MERXEN_CI_INSTALL_MODE:-auto}"
run_tests="${MERXEN_CI_RUN_TESTS:-auto}"

cd "${repo_root}"

if ! command -v uv >/dev/null 2>&1; then
    python -m pip install uv
fi

if [[ ! -x "${venv_dir}/bin/python" ]]; then
    uv venv --python 3.12 "${venv_dir}"
fi

if [[ "${install_mode}" == "auto" ]]; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
        install_mode="local"
    else
        install_mode="locked"
    fi
fi

case "${install_mode}" in
    locked)
        uv pip install --python "${venv_dir}/bin/python" -r requirements.lock
        uv pip install --python "${venv_dir}/bin/python" -e . --no-deps
        ;;
    local)
        uv pip install --python "${venv_dir}/bin/python" -e ".[dev]"
        ;;
    none)
        ;;
    *)
        echo "Unknown MERXEN_CI_INSTALL_MODE=${install_mode}" >&2
        echo "Expected one of: auto, locked, local, none" >&2
        exit 2
        ;;
esac

if [[ "${run_tests}" == "auto" ]]; then
    if [[ "${install_mode}" == "local" ]]; then
        run_tests="false"
    else
        run_tests="true"
    fi
fi

"${venv_dir}/bin/ruff" check .
"${venv_dir}/bin/ruff" format --check .
"${venv_dir}/bin/mypy" src/

case "${run_tests}" in
    true | 1 | yes)
        if [[ -n "${MERXEN_CI_PYTEST_ARGS:-}" ]]; then
            # shellcheck disable=SC2206
            pytest_args=(${MERXEN_CI_PYTEST_ARGS})
        else
            pytest_args=(-m "not slow")
        fi
        "${venv_dir}/bin/pytest" "${pytest_args[@]}"
        ;;
    false | 0 | no)
        echo "Skipping pytest because MERXEN_CI_RUN_TESTS=${run_tests}."
        ;;
    *)
        echo "Unknown MERXEN_CI_RUN_TESTS=${run_tests}" >&2
        echo "Expected one of: auto, true, false" >&2
        exit 2
        ;;
esac
