# Development workflow

Everything below assumes you've finished the [Getting started](getting-started.md)
setup and have the `merxen` conda environment active.

All project standards (layout, dependencies, naming, type hints, docstrings,
git, commit messages) are defined in [Agents.md](../Agents.md). This page
documents the day-to-day mechanics of working in the repo.

## Running tests

Pytest is configured in [pyproject.toml:69-72](../pyproject.toml#L69-L72).

```bash
pytest                          # all tests except those marked slow
pytest -m "not slow"            # explicit equivalent
pytest --run-slow               # include slow integration tests
pytest tests/test_qc/           # a specific subpackage
pytest -k "gene_comparison"     # by keyword
```

Tests live under [tests/](../tests/) and mirror the source layout:

| Source subpackage | Test directory |
|-------------------|----------------|
| `src/merxen/io/` | `tests/test_io/` |
| `src/merxen/segmentation/` | `tests/test_segmentation/` |
| `src/merxen/enrichment/` | `tests/test_enrichment/` |
| `src/merxen/qc/` | `tests/test_qc/` |
| `src/merxen/visualization/` | `tests/test_visualization/` |
| `src/merxen/alignment/` | `tests/test_alignment/` |

Shared fixtures live in [tests/conftest.py](../tests/conftest.py). Mark
anything that needs a large dataset or a real Cellpose model with
`@pytest.mark.slow`.

## Linting, formatting, typing

```bash
ruff check . --fix       # lint + auto-fix
ruff format .            # format in place
mypy src/                # type-check the package
```

Ruff configuration (line length, rule set, isort) is in
[pyproject.toml:49-67](../pyproject.toml#L49-L67). Mypy configuration is in
[pyproject.toml:74-78](../pyproject.toml#L74-L78).

## Pre-commit hooks

Install both hook types after cloning:

```bash
pre-commit install
pre-commit install --hook-type pre-push
```

From [.pre-commit-config.yaml](../.pre-commit-config.yaml):

- **On commit** — trailing-whitespace, EOF fixer, check-yaml, large-file
  guard (500 KB), ruff lint + format.
- **On push** — the lockfile-backed local CI checks: lint, format, type check,
  and tests.

Both are local guardrails. They can be bypassed with `--no-verify`, but
CI is the authoritative gate — don't bypass hooks unless you have a reason
and intend to fix it before the PR is reviewed.

## Continuous integration

[.github/workflows/ci.yml](../.github/workflows/ci.yml) runs on every push
to `main` and every PR:

1. Install from `requirements.lock` with `uv`, then `pip install -e . --no-deps`.
2. `ruff check .`
3. `ruff format --check .`
4. `mypy src/`
5. `pytest -m "not slow"`

Run the same gate locally before pushing:

```bash
scripts/run_ci_checks.sh
```

The script creates or reuses `.ci-venv`, installs from
[requirements.lock](../requirements.lock) with `uv`, installs the package with
`--no-deps`, then runs the same lint, format, type-check, and test commands as
GitHub Actions on Linux. On macOS, the script defaults to a local editable
`.[dev]` install instead of the lockfile because the lockfile can include
Linux CUDA wheels from PyTorch that cannot be installed on Apple Silicon.

Set `MERXEN_CI_VENV=/path/to/venv` to keep the reproducible environment
somewhere else. Set `MERXEN_CI_INSTALL_MODE` to control dependency installation:

| Mode | Behavior |
|------|----------|
| `auto` | Default. Uses `locked` on Linux and `local` on macOS. |
| `locked` | Install exactly from `requirements.lock`, then `pip install -e . --no-deps`. Use this for Linux/server parity. |
| `local` | Install `pip install -e ".[dev]"` through `uv` without the lockfile. Use this for macOS edit/test/push workflows when CUDA wheels are unavailable. |
| `none` | Do not install dependencies; reuse the existing `.ci-venv`. |

`MERXEN_CI_RUN_TESTS` controls the pytest step. Its default is `auto`, which
runs pytest for locked installs and skips pytest for local macOS installs. This
keeps local Mac pushes from being blocked by CUDA lockfile packages or
platform-specific scientific wheel crashes while preserving Linux/server parity.
Set `MERXEN_CI_RUN_TESTS=true` to force pytest, or use
`MERXEN_CI_PYTEST_ARGS` to run a smaller target.

For example, to push from a Mac while skipping lockfile CUDA packages:

```bash
MERXEN_CI_INSTALL_MODE=local git push
```

To force Linux/server-style locked checks locally:

```bash
MERXEN_CI_INSTALL_MODE=locked scripts/run_ci_checks.sh
```

To run a targeted pytest subset through the hook environment:

```bash
MERXEN_CI_INSTALL_MODE=local \
MERXEN_CI_RUN_TESTS=true \
MERXEN_CI_PYTEST_ARGS="tests/test_cortical_depth -q" \
scripts/run_ci_checks.sh
```

Branch protection on `main` should require this workflow to pass.

## Version bumps

MerXen uses semantic versions in [pyproject.toml](../pyproject.toml) and
[src/merxen/__init__.py](../src/merxen/__init__.py). Keep both files in sync
with `bump-my-version`:

```bash
uv run bump-my-version bump patch  # bug fixes and small changes
uv run bump-my-version bump minor  # backwards-compatible features
uv run bump-my-version bump major  # breaking changes
```

For a release commit and tag from a clean working tree:

```bash
uv run bump-my-version bump patch --commit --tag
git push origin HEAD --tags
```

## Dependency management

- **Add a dependency:** edit [pyproject.toml](../pyproject.toml), then
  regenerate the lockfile:

  ```bash
  uv pip compile pyproject.toml --extra dev -o requirements.lock
  ```

- **Never `pip install <pkg>` directly.** That leaves you out of sync with
  the lockfile and CI.
- **Conda env (`environment.yml`)** is deliberately thin — Python 3.12,
  `spatialdata` from git, and `-e ".[dev]"`. Everything else comes through
  `pyproject.toml`.
- **Alignment env (`environment.alignment.yml`)** mirrors the base env for
  Nextflow `ALIGN`. Spateo/Dynamo are bootstrapped inside that env at runtime
  because their older AnnData metadata has to be followed by a modern AnnData
  restore step.

For reproducible installs (CI, onboarding):

```bash
uv pip install -r requirements.lock
pip install -e . --no-deps
```

## Git workflow

- Branch from `main`, merge via PR, never push directly to `main`.
- Commit titles follow the conventional prefix scheme from
  [Agents.md](../Agents.md#commit-messages): `[feature]`, `[bugfix]`,
  `[refactor]`, `[style]`, `[test]`, `[docs]`, `[chore]`, `[minor]`.
- Delete feature branches after merge.

## Adding a new pipeline stage

See [Python API → Adding a new stage](python-api.md#adding-a-new-stage) for
the concrete checklist.

## Writing docs

- Docs live here in `docs/` as plain markdown.
- File references should use relative markdown links
  (`[main.nf](../workflows/main.nf)`) so they render in any editor or
  GitHub preview.
- Function references should include the file path and line number
  (`[qc/metrics.py:111](../src/merxen/qc/metrics.py#L111)`) so readers can
  jump straight into the code.
- Update [docs/index.md](index.md) when you add a new page.

## Debugging the pipeline

- **Inspect a failed Nextflow task** — every task keeps its work directory
  under `./work/<hash-prefix>/<hash-rest>/`. It contains `.command.sh`,
  `.command.out`, `.command.err`, and all staged inputs.
- **Rerun one stage in isolation** — grab the JSON config Nextflow wrote
  (`build_config.json`, `segment_config.json`, ...) from the work
  directory and run `merxen <subcommand> --config <file>` directly. Add
  `--force-rerun` if you need to bypass cached outputs.
- **Check memory limits** — watch `log_status` output or the peak RSS
  column in `${outdir}/nextflow/trace.tsv`.
- **Cellpose is silent on GPU errors** — it falls back to CPU. Set
  `--cellpose_gpu false` explicitly when diagnosing GPU issues.

## Project standards summary

For the full standards, see [Agents.md](../Agents.md). The short version:

- One package per repo, `src/merxen/` layout.
- `pyproject.toml` is the single source of truth for dependencies.
- PEP 8 naming, type hints on all public functions, Google-style
  docstrings.
- Ruff for linting and formatting.
- Pre-commit hooks for linting, pre-push hooks for lockfile-backed local CI.
- CI runs lint + format + mypy + pytest on every PR.
- No production logic in notebooks. No secrets in git. No data files
  bigger than 500 KB.
