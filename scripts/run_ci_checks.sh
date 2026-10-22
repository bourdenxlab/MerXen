#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${MERXEN_CI_VENV:-${repo_root}/.ci-venv}"
install_mode="${MERXEN_CI_INSTALL_MODE:-auto}"
run_tests="${MERXEN_CI_RUN_TESTS:-auto}"
test_tier="${MERXEN_CI_TEST_TIER:-fast}"

# Test tiers. 'fast', the pre-push default, leaves out the tests marked
# nextflow, store_lock and integration (pyproject.toml), which take seconds
# each. 'full' is what CI runs: everything except the slow tests, which need
# GPUs, site images or extra environments.
case "${test_tier}" in
    fast)
        tier_markers="not slow and not nextflow and not store_lock and not integration"
        ;;
    full)
        tier_markers="not slow"
        ;;
    *)
        echo "Unknown MERXEN_CI_TEST_TIER=${test_tier}" >&2
        echo "Expected one of: fast, full" >&2
        exit 2
        ;;
esac

# Git exports repository-local variables while running hooks. Nested Git
# operations (for example uv installing a Git dependency) must discover their
# own repositories instead of inheriting the outer MerXen worktree.
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE

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
        # The lock is the full pinned tree (uv pip compile), so it is installed with --no-deps:
        # re-resolving would follow cell_type_mapper's unpinned abc_atlas_access git URL to the
        # upstream head, which conflicts with the lock's pinned commit whenever upstream moves.
        uv pip install --python "${venv_dir}/bin/python" --no-deps -r requirements/requirements.lock
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

# The metro map source is validated when nf-metro is available. The pytest
# suite guards the map/pipeline correspondence without it, so a venv built in
# 'none' install mode simply skips this.
if [[ -x "${venv_dir}/bin/nf-metro" ]]; then
    "${venv_dir}/bin/nf-metro" validate assets/metro_map.mmd
fi

# Nextflow lint, when Nextflow is installed. -project-dir makes the linter
# load the Groovy classes in workflows/lib; without it, run from the repo
# root, every lib class main.nf uses reads as "not defined". Warnings pass,
# errors fail (tests/test_workflows/test_nextflow_lint.py checks the same).
if command -v nextflow >/dev/null 2>&1; then
    NXF_DISABLE_CHECK_LATEST=true nextflow lint -project-dir workflows -o concise workflows/
fi

case "${run_tests}" in
    true | 1 | yes)
        # Tests run on parallel pytest-xdist workers (0 runs them in this
        # process). Each worker gets a few BLAS, OpenMP and numba threads, or
        # every worker would start one per core. Values already set win.
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
        export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
        export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
        export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-2}"
        pytest_args=(-n "${MERXEN_CI_PYTEST_WORKERS:-4}" --dist load)
        if [[ -n "${MERXEN_CI_PYTEST_ARGS:-}" ]]; then
            # Split the way a shell would, so a quoted marker expression such
            # as -m "not slow and not nextflow" stays one argument. Python
            # re-quotes each argument; nothing in the value is expanded.
            quoted_args="$("${venv_dir}/bin/python" -c \
                'import shlex, sys; print(shlex.join(shlex.split(sys.argv[1])))' \
                "${MERXEN_CI_PYTEST_ARGS}")"
            eval "pytest_args+=(${quoted_args})"
        else
            pytest_args+=(-m "${tier_markers}")
            echo "pytest: ${test_tier} tier (-m \"${tier_markers}\")."
            if [[ "${test_tier}" == "fast" ]]; then
                echo "CI also runs the nextflow, store_lock and integration tests;" \
                    "MERXEN_CI_TEST_TIER=full runs them here."
            fi
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
