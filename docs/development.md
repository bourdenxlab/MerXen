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
| `src/merxen/annotation/` | `tests/test_annotation/` |
| `src/merxen/clustering/` | `tests/test_clustering/` |

Shared fixtures live in [tests/conftest.py](../tests/conftest.py). Mark
anything that needs a large dataset or a real Cellpose model with
`@pytest.mark.slow`.

Slow tests that need local data read its location from environment variables
and skip when they are unset (the annotation ones are also in
[.env.example](../.env.example)):

| Variable | Test |
|----------|------|
| `MERXEN_MENDER_CONTAINER` | MENDER container smoke test |
| `MERXEN_WHB_TAXONOMY_DIR`, `MERXEN_SEAAD_TERM_CSV`, `MERXEN_WMB_TAXONOMY_DIR` | the committed annotation vocab tables match the real Allen taxonomy CSVs (`build_vocab_tables.py --check`) |

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
to `main` and to the integration branch `feature/robust-celltype-annotation`
([plan §2.2](plans/robust-celltype-annotation-plan.md)), and on every PR:

1. Install from `requirements/requirements.lock` with `uv`, then `pip install -e . --no-deps`.
2. `ruff check .`
3. `ruff format --check .`
4. `mypy src/`
5. `nf-metro validate assets/metro_map.mmd`
6. `pytest -m "not slow"`

To lint the Nextflow code, point the linter at the pipeline's project
directory so it loads the Groovy classes in `workflows/lib/`:

```bash
nextflow lint -project-dir workflows workflows/
```

Without `-project-dir`, run from the repository root, the linter looks for
`lib/` in the current directory and reports every `workflows/lib` class that
`main.nf` uses (`AnnotationSettings`, `AnnotationPreflight`) as "not defined".
Those are not real errors. [tests/test_workflows/test_nextflow_lint.py](../tests/test_workflows/test_nextflow_lint.py)
runs the command above and fails on any error (warnings pass); it is skipped
when `nextflow` is not installed. `scripts/run_ci_checks.sh` runs it too when
Nextflow is on the `PATH`.

Run the same gate locally before pushing:

```bash
scripts/run_ci_checks.sh
```

The script creates or reuses `.ci-venv`, installs from
[requirements/requirements.lock](../requirements/requirements.lock) with `uv`, installs the package with
`--no-deps`, then runs the same lint, format, type-check, and test commands as
GitHub Actions on Linux. On macOS, the script defaults to a local editable
`.[dev]` install instead of the lockfile because the lockfile can include
Linux CUDA wheels from PyTorch that cannot be installed on Apple Silicon.

Set `MERXEN_CI_VENV=/path/to/venv` to keep the reproducible environment
somewhere else. Set `MERXEN_CI_INSTALL_MODE` to control dependency installation:

| Mode | Behavior |
|------|----------|
| `auto` | Default. Uses `locked` on Linux and `local` on macOS. |
| `locked` | Install exactly from `requirements/requirements.lock`, then `pip install -e . --no-deps`. Use this for Linux/server parity. |
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
  uv pip compile pyproject.toml --extra dev -o requirements/requirements.lock
  python scripts/update_env_lock_hash.py
  ```

  The second command refreshes the `# requirements.lock sha256:` header in
  `envs/environment.yml`. Nextflow caches conda envs as
  `work/conda/env-<hash of the env file's text>`, so without the header a
  lockfile change would leave the cached env on the old dependency versions.
  `tests/test_workflows/test_env_lock_sync.py` fails while the header is stale.

  **A new header invalidates `-resume` under the `conda` profile.** Any change
  to `envs/environment.yml`, the checksum included, gives the base env a new
  `work/conda/env-*` path, and Nextflow includes each task's conda env in the
  task hash. The first `-resume` under `-profile …,conda` after the change
  therefore re-runs every task that uses the base env and everything
  downstream of it: Cellpose, ProSeg, clustering, MapMyCells and the later
  stages. The new env resolves the loose `pyproject.toml` ranges on the day it
  is built, not the lock, so the re-run outputs can differ by more than the
  intended dependency change. The checksum covers the lock's raw bytes, so a
  regeneration that only rewrites `# via` comments has the same effect. Before
  merging a lockfile change, snapshot the published outputs you need to keep
  and plan a full rerun.

  When the dedicated registration stack changes, regenerate its separate lock:

  ```bash
  uv pip compile pyproject.toml --extra dev --group alignment-runtime \
    --python-platform linux -o requirements/requirements.alignment.lock
  ```

- **Never `pip install <pkg>` directly.** That leaves you out of sync with
  the lockfile and CI.
- **Known gap:** `scripts/annotation/build_vocab_tables.py` and its tests use
  PyYAML, which `pyproject.toml` does not declare yet; it comes with dask and
  pre-commit. Declare `pyyaml` in the `dev` extra (with `scikit-learn`, plan
  §3.8) in the first change that regenerates the lock anyway, since a new lock
  invalidates `-resume` (above). Until then
  `test_pyyaml_is_available_for_the_generator` fails if PyYAML disappears.
- **Conda env (`envs/environment.yml`)** is deliberately thin — Python 3.12, pip,
  and `-e ".[dev]"`. All Python dependencies come through `pyproject.toml`.
  It cannot install the lockfile, because pip rejects it (`ResolutionImpossible`:
  `cell_type_mapper` declares `abc_atlas_access` as an unpinned git URL while the
  lock pins a commit); its lockfile checksum header only forces Nextflow to
  rebuild the env when the lock changes.
- **Base image (`containers/Dockerfile`)** installs `requirements/requirements.lock`
  with `uv`, then MerXen with `--no-deps`, like CI.
- **Alignment env (`envs/environment.alignment.yml`)** installs
  `requirements/requirements.alignment.lock` plus Java/libvips for Nextflow `ALIGN`.
  VALIS 1.2 is installed exactly with `--no-deps` after the locked
  NumPy-2-compatible runtime. The explicit legacy backend still bootstraps its
  pinned Spateo/Dynamo packages at runtime.
- **Clustering GPU env (`envs/environment.clustering-gpu.yml`)** contains RAPIDS and
  its Dask pin. It receives H5AD inputs only; SpatialData reads and writes stay
  in the base environment.

For reproducible installs (CI, onboarding):

```bash
uv pip install -r requirements/requirements.lock
pip install -e . --no-deps
```

## Git workflow

- Branch from `main`, merge via PR, never push directly to `main`.
- Commit titles follow the conventional prefix scheme from
  [Agents.md](../Agents.md#commit-messages): `[feature]`, `[bugfix]`,
  `[refactor]`, `[style]`, `[test]`, `[docs]`, `[chore]`, `[minor]`.
- Delete feature branches after merge.

## Robust cell-type annotation work

The effort planned in
[docs/plans/robust-celltype-annotation-plan.md](plans/robust-celltype-annotation-plan.md)
has its own branch model and guards.

The pre-registered acceptance thresholds and the measured M3 baselines are in
[docs/acceptance/annotation-v1-preregistration.md](acceptance/annotation-v1-preregistration.md).
Baselines may only tighten a threshold; loosening one needs the user's
written approval in the gate PR.

### Integration branch

Work lands on the long-lived integration branch
`feature/robust-celltype-annotation` (plan §2.2). Each milestone is a branch
`feature/rca-m<N>-<slug>` cut from its head and merged back by PR; merge the
integration head into the milestone branch before its PR. The integration
branch is kept current by merging `main` into it, never by rebasing it, and
only the acceptance-gate PRs merge it into `main`. Data-integrity fixes go to
`main` first. Until a species flips, no milestone may change legacy results.

### Hook points shared with `feature/gaston`

`workflows/main.nf`, `workflows/nextflow.config`, `workflows/conf/dwight.config`,
`src/merxen/config.py` and `src/merxen/io/samplesheet.py` are also edited by
`feature/gaston`, so annotation changes touch them only at the hook points of
plan §2.4 (H1–H9), plus two in `main.nf` added by later milestones: H10
(`--annotation_prepare_only` builds the reference bundles and stops; M2) and
H11 (the `ANNOTATION_REPORTING` block at the end of the main workflow, map_first
runs only, after MENDER; M7):

- each hook carries one marker, `// rca-hook:H<n>` in Nextflow and
  `# rca-hook:H<n>` in Python, exactly once per file;
- every other line a hook changes carries `rca-site:H<n>`;
- [tests/test_workflows/test_rca_hooks.py](../tests/test_workflows/test_rca_hooks.py)
  pins the markers, the number of site lines per file (`EXPECTED_SITES`) and
  the code next to each marker, so a rebase or merge cannot silently drop a
  hook. When a hook legitimately grows, add the site marker and update
  `EXPECTED_SITES` in the same commit.

New behaviour goes into new files (`workflows/lib/Annotation*.groovy`,
`workflows/conf/annotation.config`, `src/merxen/annotation/`). Plan §2.4 lists
the GASTON conflict hotspots and gives the rebase guide for the GASTON author.

### Legacy clustering pin

[tests/test_workflows/test_legacy_script_pin.py](../tests/test_workflows/test_legacy_script_pin.py)
pins the sha256 of the legacy `CLUSTERING_SQUIDPY_PREPARE` / `COMPUTE` /
`FINALIZE` script text, their `main.nf` wiring and the param defaults they
read. Nextflow's `-resume` key covers the script text, so a change there
re-runs every legacy clustering task and may change legacy results. To change
it on purpose, put the change and the new digests in their own commit whose
message gives the reason and the expected effect on legacy runs.
`python tests/test_workflows/test_legacy_script_pin.py` prints the current
digests.

### Annotation vocabulary tables

The vocab and floor tables under `src/merxen/assets/annotation/` are generated;
never edit them by hand. Edit `overrides.yaml`, then regenerate from the Allen
taxonomy CSVs and commit the inputs and outputs together:

```bash
python scripts/annotation/build_vocab_tables.py \
    --whb-taxonomy-dir <abc>/metadata/WHB-taxonomy/20240330 \
    --seaad-term-csv <abc>/metadata/SEA-AD-Multiregion-taxonomy/20260711/cluster_annotation_term.csv \
    --wmb-taxonomy-dir <abc>/metadata/WMB-taxonomy/20231215
```

`--check` writes nothing and exits 1 when a committed file is stale. The
generator fails when a taxonomy term has no curated mapping, so a new Allen
release cannot change the vocabulary silently. The `NOTICE` records each
input by its ABC path, size and sha256, so a local copy may have any name.
`tests/test_annotation/test_build_vocab_tables.py` rebuilds a synthetic
taxonomy from the committed tables; its slow test checks them against the
real CSVs (variables above).

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
