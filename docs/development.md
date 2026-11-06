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

Three more markers name tests that take seconds each. The pre-push hook
leaves them out by default and CI runs them (see [Test tiers](#test-tiers)):

| Marker | Use it for |
|--------|------------|
| `nextflow` | a test that launches Nextflow |
| `store_lock` | a test where two processes contend for a store's write lock; tests that use the `store_lock_contention` fixture get it automatically |
| `integration` | a test that runs a stage end to end on a real on-disk SpatialData store |

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

Install the hooks after cloning (the hook types pre-commit, pre-merge-commit,
commit-msg and pre-push come from `default_install_hook_types`):

```bash
pre-commit install
```

From [.pre-commit-config.yaml](../.pre-commit-config.yaml):

- **On commit** — trailing-whitespace, EOF fixer, check-yaml, large-file
  guard (500 KB), ruff lint + format, and the privacy check of the staged
  lines and file names.
- **On merge and on the commit message** — the privacy check.
- **On push** — the privacy check of the branch name and every pushed commit,
  then the lockfile-backed local CI checks: lint, format, type check, and
  the fast test tier (`MERXEN_CI_TEST_TIER=changed git push` runs only the
  tests the changes can reach).

The privacy hooks keep host paths and sample names out of the repository
([Local configuration and privacy](local-configuration.md)).

Both are local guardrails. They can be bypassed with `--no-verify`, but
CI is the authoritative gate — don't bypass hooks unless you have a reason
and intend to fix it before the PR is reviewed.

## Continuous integration

[.github/workflows/ci.yml](../.github/workflows/ci.yml) runs on every push
to `main` and to the integration branch `feature/robust-celltype-annotation`
([plan §2.2](plans/robust-celltype-annotation-plan.md)), every PR (opened,
pushed, reopened or labelled), nightly on `main` and on demand
(`workflow_dispatch`). Its jobs:

- **lint**: `ruff check .`, `ruff format --check .`,
  `nf-metro validate assets/metro_map.mmd` and `mypy src/`. mypy starts from
  the cache saved by the latest push to the base branch (`main` or the
  integration branch).
- **plan**: runs [scripts/ci/select_tests.py](../scripts/ci/select_tests.py)
  (stdlib only, no install) and decides between the full suite and the tests
  the PR's changes can reach ([Test selection](#test-selection)), and how many
  shards to use: 7 for the full suite, 1 to 7 for a selection.
- **test shard i/N**: one job per shard, each set up like the former single
  test job: installs from `requirements/requirements.lock` with `uv`, then
  `pip install -e . --no-deps` (the uv cache keeps the wheels built from Git
  sources between runs), checks the locked Cellpose runtime, installs
  Nextflow, and runs `pytest -n 4 --dist load -m "not slow"` (four
  pytest-xdist workers, one BLAS/OpenMP/numba thread each) on its share of
  the tests. The job summary lists the shard's 20 slowest tests.
- **test**: the single pass/fail signal. It fails unless the plan and every
  shard succeeded, and checks the shard reports
  ([scripts/ci/check_shards.py](../scripts/ci/check_shards.py)): every shard
  collected the same tests, and together they ran each one exactly once.

More xdist workers per runner do not help: the 4-vCPU runner is already busy,
and the longest test (the manual-registration end-to-end workflow test, about
280 s on CI) sets the floor of whichever shard runs it. Shards spread the rest
over several runners instead.

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
when `nextflow` is not installed, so CI's test shards, which install Nextflow,
runs it. `scripts/run_ci_checks.sh` runs the lint too when Nextflow is on the
`PATH`.

A new push to a PR, or the `full-ci` label, cancels that PR's run in
progress; other labels start a run of their own that cancels nothing. Shards
time out after 30 minutes. The separate
[Privacy](../.github/workflows/privacy.yml) workflow checks the branch name,
PR text and commit messages.

### Sharding

[scripts/ci/pytest_shard.py](../scripts/ci/pytest_shard.py) is a pytest
plugin (`PYTHONPATH=scripts/ci pytest -p pytest_shard --shard-count=N
--shard-index=I --shard-durations=FILE --shard-report=OUT.json`; write each
path option with `=`, or pytest takes the path for a test path when it picks
the rootdir and loses the settings in `pyproject.toml`). It
partitions the collected tests (after `-m`) by recorded duration: whole test
files, longest first, onto the least loaded shard; a file heavier than half a
shard's share is split into single tests; inside a shard the tests of 10 s or
more start first. Tests without a recorded duration weigh their file's (or
the suite's) median, so new tests are always assigned. Every pytest-xdist
worker computes the same partition.

The durations come from the latest full run on `main`: the `test` job of a
green push, nightly or manual run merges the shards' measured durations and
saves them to the Actions cache (and as the `test-durations` artifact). The
plan job restores the newest one, or falls back to the committed
[scripts/ci/test_durations.json](../scripts/ci/test_durations.json), and hands
the same file to every shard. Its keys are digests of each test's file and
id (some test ids are the privacy checker's own inputs, so the file does not
spell them out). Stale durations only unbalance the shards; they never drop a
test. Refresh the committed file now and then, from the latest
green full run on `main`:

```bash
gh run list --workflow CI --branch main --event push --status success --limit 1
gh run download <run id> -n test-durations -D /tmp/test-durations
cp /tmp/test-durations/test-durations.json scripts/ci/test_durations.json
git commit -m "[chore]: refresh the recorded test durations" scripts/ci/test_durations.json
```

### Test selection

On an ordinary PR the shards run only the tests the PR's changes
(`git diff base...head`) can reach, plus a set of cheap repository-wide
guards. `select_tests.py` builds a static dependency graph of `src/merxen`,
`tests`, `scripts` and `workflows` and selects a test file when a changed
file is in its closure:

- the `merxen` modules it imports, directly or transitively, including
  imports in functions and under `TYPE_CHECKING`, the parent packages' `__init__`
  files, monkeypatch targets and `import_module` strings, and the imports of
  every `conftest.py` and test package `__init__.py` above it and of the
  helpers it imports;
- test helpers and other test modules it imports, also from another test
  package (`from test_manual_registration._mr_helpers import ...`; pytest puts
  `tests/` on `sys.path`), from directories it adds to `sys.path`, and from
  code it runs as a string (`python -c` programs);
- the repository files it names: string literals, path expressions such as
  `REPO_ROOT / "docs" / "images" / name` (a computed tail means the whole
  directory), globs (also over a package's own directory,
  `Path(pkg.__file__).parent.rglob("*.py")`, so a new module there selects
  the test), error messages that mention a path, a file's base name, and a
  script's stem (`load_script("compare_czi_readers")`); a script it names
  counts with its own imports;
- for a test that launches Nextflow (itself or through a helper or fixture),
  the workflow files it runs, their `include` closure, and the
  `merxen <command>` modules, `python -m merxen.*` modules and scripts those
  call.

On top of that, a change under `workflows/` selects all of
`tests/test_workflows` and every test that launches Nextflow, a change under
`scripts/` selects `tests/test_scripts`, and a data file under
`tests/<dir>/` selects every test in that directory. A changed Python file
under `tests/` also selects every test whose closure has a file that names
its module stem (a safety net for imports the graph cannot resolve). Docs,
notebooks, assets,
licences and the root Markdown files select only the tests that name them.

The guards always run: `tests/test_privacy`, the process script pins, the
platform golden configs, the platform sites, the metro map, analysis-mode
parity and the environment lock sync (about 25 s on four workers).

The full suite runs instead when:

- the event is a push to `main`, the nightly schedule or `workflow_dispatch`;
- the PR carries the `full-ci` label (adding it starts a new, full run; the
  plan reads the PR's current labels, so re-running a run after adding the
  label is full too);
- the PR merges a long-lived integration branch into `main`
  (`INTEGRATION_BRANCHES` in `select_tests.py`, the one place that lists them);
- a changed file is a `conftest.py`, `pyproject.toml`, `requirements/`,
  `envs/`, `.github/`, `.pre-commit-config.yaml`, `scripts/run_ci_checks.sh`,
  `scripts/ci/`, or one of the modules nearly every test imports
  (`merxen/__init__.py`, `config.py`, `io/spatialdata_io.py`,
  `io/spatialdata_schema.py`);
- a module or test helper is deleted, a non-Python file under `src/` changes,
  or any changed path has no rule;
- the selection is estimated at more than 60 % of the suite's test time.

Static analysis cannot see a path or module name computed at run time from
data, code reached only through a subprocess other than Nextflow, or package
data read by `src` code. Importing one `merxen.cli.run_*` module (and every
`merxen` subprocess) runs `merxen/cli/__init__.py`, which imports every
command; the selector does not follow that, so a change that breaks only at
import time fails the tests of the changed module but may not run every CLI
test. The selector errs towards selecting more, every
push to `main` and the nightly run test everything, and `full-ci` is there
when a PR needs certainty. All the policy constants (guards, full-suite
patterns, integration branches, label, shard sizes, cap) are at the top of
`select_tests.py`. To see what a branch would run:

```bash
python scripts/ci/select_tests.py --base origin/main --worktree --format json
```

Run the same checks locally before pushing (the fast test tier unless you
ask for another; see [Test tiers](#test-tiers)):

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
`MERXEN_CI_PYTEST_ARGS` to run a smaller target. It replaces the tier's marker
selection and is split with shell quoting rules, so
`MERXEN_CI_PYTEST_ARGS='-m "not slow and not nextflow" tests/test_io'` passes
the marker expression as one argument.

Pytest runs on `MERXEN_CI_PYTEST_WORKERS` parallel pytest-xdist workers
(default 4; `0` runs every test in one process). Each worker is limited to 2
BLAS, OpenMP and numba threads unless `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` or `NUMBA_NUM_THREADS` is already set.

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

Branch protection on `main` should require the `lint` and `test` checks to
pass (`test` aggregates the plan and the shards).

### Test tiers

`MERXEN_CI_TEST_TIER` picks the tests the script runs:

| Tier | Selection | Used by |
|------|-----------|---------|
| `fast` | `-m "not slow and not nextflow and not store_lock and not integration"` | the pre-push hook (default) |
| `changed` | the `fast` markers, on the tests the branch's changes can reach ([Test selection](#test-selection)), or on every test when the selector says FULL | opt-in: `MERXEN_CI_TEST_TIER=changed git push` |
| `full` | `-m "not slow"` | CI (sharded, on the full suite or the PR's selection) |

The `changed` tier diffs against the merge base with `MERXEN_CI_BASE_REF`
(default: the remote branch local `main` tracks, else `origin/main`, else
`upstream/main`; a stale one only selects more) up to the commit being pushed
(`PRE_COMMIT_TO_REF`, else `HEAD`), plus uncommitted and untracked files.
Without a merge base it falls back to the `fast` tier.

The `changed` and `fast` tiers miss Nextflow, cross-process lock and
end-to-end store regressions until CI. Run the full tier locally when a change
touches the workflows, store locking or a stage's store writes, or before
asking for a review:

```bash
MERXEN_CI_TEST_TIER=full git push
MERXEN_CI_TEST_TIER=full scripts/run_ci_checks.sh
```

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

- **The isolated CZI runtime** (`envs/environment.czi.yml`, lock
  `requirements/requirements.czi.lock`) holds `czifile` for reading Zeiss CZI
  files. It cannot live in the base env: squidpy caps `imagecodecs<2026` while
  current `czifile` needs `imagecodecs>=2026.8`. MerXen itself is not installed
  in it; `merxen.io.czi` re-runs itself under its interpreter. Regenerate the
  lock and its hash with:

  ```bash
  uv pip compile --group czi-runtime --python-version 3.12 --python-platform linux \
    -o requirements/requirements.czi.lock
  python scripts/update_env_lock_hash.py --lock requirements/requirements.czi.lock \
    --env envs/environment.czi.yml
  ```

  Create it locally once, then export its python (or set it in `.env`):

  ```bash
  uv venv --python 3.12 ~/merxen-czi-runtime
  uv pip install --python ~/merxen-czi-runtime/bin/python --no-deps -r requirements/requirements.czi.lock
  export MERXEN_CZI_PYTHON=~/merxen-czi-runtime/bin/python
  ```

  `scripts/run_ci_checks.sh` and GitHub CI build the same venv. The tiny CZI test
  fixtures under `tests/test_data/czi/` are regenerated, on purpose only, with
  `scripts/manual_registration/make_czi_fixtures.py` in an ephemeral environment
  (the LGPL writer is never a dependency):

  ```bash
  uv run --no-project --python 3.12 --with pylibCZIrw==6.1.0 --with numpy==2.2.6 \
      python scripts/manual_registration/make_czi_fixtures.py --out tests/test_data/czi
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

## Platform registry

Values that differ between platforms live in two registries that must agree:
[src/merxen/platforms.py](../src/merxen/platforms.py) for the Python package and
[workflows/lib/PlatformSpec.groovy](../workflows/lib/PlatformSpec.groovy) for
the Nextflow workflow (which adds the samplesheet fields of each platform). Both
raise for a platform they do not know. Look values up there rather than writing
a new `MERSCOPE`/`XENIUM` ternary or else-arm.

Tests that guard them:

- `tests/test_platforms.py` pins every registry value.
- `tests/test_workflows/test_platform_registry_groovy.py` (Nextflow tier) checks
  that the Groovy registry matches the Python one and that every converted
  workflow site raises for an unknown platform.
- `tests/test_platform_sites.py` pins, by line text, the code that still names
  original shape layers, nucleus layers, native images or a platform in an
  equality test. A new site fails it: use the registry, or add the line to the
  pin with the reason.
- `tests/test_workflows/test_platform_config_golden.py` and
  `test_process_script_pins.py` pin the rendered task configs and every process
  `script:` section, so a refactor that changes Nextflow task hashes fails. Only
  regenerate them (`MERXEN_UPDATE_GOLDEN=1`) for an announced change.

Adding a platform means an entry in both registries, the platform's arms in the
dispatch code that already raises for an unknown platform (builders,
enrichment, segmentation), and its analysis-mode alias in all three mode maps
(`tests/test_workflows/test_analysis_mode_parity.py`).

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
