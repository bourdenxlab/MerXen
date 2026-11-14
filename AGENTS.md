# AGENTS.md — Project Standards

This file defines the rules and conventions for this repository. All contributors and AI assistants must follow these standards when writing, reviewing, or modifying code. `CLAUDE.md` imports this file; tool-neutral rules belong here, not there.

## Project overview

This project is used for pre-processing and then analysing human or mouse spatial transcriptomic datasets. The datasets can be generated in pairs to enable direct comparison between two spatial transcriptomic platforms: 10x Genomics' Xenium and Vizgen's MERSCOPE. For each pair of adjacent brain sections, one is processed with Xenium and the other with MERSCOPE using the same custom gene panel. This project takes the data output from each platform and performs the following, matching all steps of analysis after the initial pre-processing steps to coerce the data into the same formats between platforms:

- Performs cell segmentation from scratch using Cellpose-SAM image based segmentation, followed by ProSeg transcript based segmentation with the cellpose assignment as a prior.
- Aligns each pair of adjacent tissue sections so that they share a common coordinate system
- Performs comparative analysis of each paired dataset

This workflow is achieved by using Nextflow to run different modules of pre-processing, segmentation, analysis etc. in a way that can be monitored per sample and scaled to many sample pairs with logging etc.

## Commands

```bash
# Run tests
pytest

# Run tests excluding slow markers
pytest -m "not slow"

# Pre-push checks: 'fast' (the default) runs the whole fast tier; 'changed'
# runs only the fast-tier tests your changes since main's remote
# (MERXEN_CI_BASE_REF) can reach; 'full' runs everything CI runs
scripts/run_ci_checks.sh
MERXEN_CI_TEST_TIER=changed scripts/run_ci_checks.sh
MERXEN_CI_TEST_TIER=full scripts/run_ci_checks.sh

# Which tests CI would run for this branch (FULL or a list; docs/development.md,
# Test selection). CI shards them over 7 runners (scripts/ci/pytest_shard.py);
# the PR label full-ci forces the full suite
python scripts/ci/select_tests.py --base origin/main --worktree --format paths

# Lint (auto-fix) and format
ruff check . --fix
ruff format .

# Type check
mypy src/

# Regenerate lockfile after changing dependencies
uv pip compile pyproject.toml --extra dev -o requirements/requirements.lock
# ...then refresh the lock checksum header in envs/environment.yml
python scripts/update_env_lock_hash.py
```

- Run `ruff check` and `ruff format` before every commit.
- Run `pytest` before pushing. The pre-push hook enforces this.

## Project Structure

```
my_project/
├── src/my_package/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   ├── models/
│   └── utils/
├── tests/
│   ├── conftest.py
│   ├── test_data/
│   └── test_models/
├── notebooks/           # Exploratory only — no production logic here
├── docs/
├── pyproject.toml       # Single source of truth for dependencies and tool config
├── envs/                # Conda envs: python version, system-level deps, pip install
│   └── environment.yml
├── containers/          # Dockerfiles for any images the project builds
├── requirements/        # Pinned dependency trees — generated, never hand-edited
│   └── requirements.lock
├── .pre-commit-config.yaml
├── .env.example         # Template for environment variables (never commit .env)
├── .gitignore
├── AGENTS.md            # Project standards (tool-neutral)
├── CLAUDE.md            # Imports AGENTS.md; Claude-specific notes only
└── README.md
```

Rules:
- One package per repo. The installable package lives in `src/my_package/`.
- Tests mirror the package structure under `tests/`.
- Notebooks are for exploration and communication. If logic in a notebook is needed in production, refactor it into `src/`.
- Data files and model weights do not go in the repo. Use `.gitignore` to exclude them.
- Conda environment files live in `envs/` and image definitions in `containers/`, so the root stays limited to files that tooling requires there. Lockfiles stay at the root because `uv`, CI, and `pip install -r` all assume that location.


## Dependency Management

### Where dependencies live

**`pyproject.toml`** is the single source of truth for all Python dependencies. Declare them loosely — specify minimum versions and upper bounds only where you know there are incompatibilities.

```toml
[project]
dependencies = [
    "numpy>=1.24,<3",
    "pandas>=2.0",
    "scikit-learn>=1.3",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "ruff>=0.4",
    "mypy>=1.0",
    "pre-commit>=3.0",
]
```

### envs/environment.yml

Keep this thin. Use it only for what conda handles better than pip: the Python version and system-level compiled dependencies. Defer everything else to pyproject.toml.

```yaml
name: my_project
channels:
  - conda-forge
dependencies:
  - python=3.11
  # Only add conda packages here if they need compiled system libraries
  # that pip can't install cleanly (GDAL, CUDA, etc.)
  - pip
  - pip:
      # Relative to this file's directory — conda runs pip from there.
      - -e "../[dev]"
```

Do not duplicate dependencies across environment.yml and pyproject.toml. If it can be installed via pip, it belongs in pyproject.toml.

### Lockfile

Generate a lockfile to pin the full resolved dependency tree:

```bash
uv pip compile pyproject.toml --extra dev -o requirements/requirements.lock
```

Commit `requirements/requirements.lock` to version control. Regenerate it when you add or update dependencies. For reproducible installs (CI, onboarding):

```bash
uv pip install -r requirements/requirements.lock
pip install -e . --no-deps
```

Never hand-edit the lockfile. Never install packages by running `pip install <package>` directly — add them to `pyproject.toml` and regenerate the lockfile.


## Environment & Configuration

- Store all configuration, secrets and machine-specific values outside the code. Locations go in
  the gitignored `.env` (template `.env.example`, placeholders only) or, for Nextflow, a gitignored
  site config (`workflows/conf/*.site.config`, templates `*.site.config.example`).
- Read `.env` through the project loader `merxen.local_config` (process environment first, then
  `.env`). A missing value raises an error that names the variable; never add a built-in host
  default.
- Never commit file paths of a machine, secrets, credentials, or the names of samples, donors,
  sections, slides, runs or instruments. Use the neutral aliases of the gitignored `.aliases.tsv`
  (template `.aliases.example.tsv`). This applies to file names, commit messages, branch names and
  pull-request text as well.

### Privacy: no host paths, no sample names

- Nothing committed may contain an absolute path of a machine (storage roots, mounts, home
  directories, another host's directories) or the name of a sample: donor, section, slide, run or
  instrument ids taken from a samplesheet, and colleagues' user names. This covers code, configs,
  tests, docs, notebooks, file names, commit messages, branch names, PR titles and PR descriptions.
- Locations live in gitignored files: `.env` (Python, scripts, tests, notebooks; template
  `.env.example`), and the Nextflow site configs `workflows/conf/site.config` and
  `workflows/conf/dwight.site.config` (templates `*.example`). Python reads `.env` through
  `merxen.local_config`; a missing value is an error, never a built-in default.
- Samples are named by aliases (HD1-HD4 paired human donors, HD1_IF / HD3_IF IF samples, HM1-HM4
  new-panel human MERSCOPE donors, MS1 / MS2 mouse sections; docs/local-configuration.md). The
  gitignored `.aliases.tsv` maps them to the real names. Code that reads a real name from data
  translates it with `merxen.privacy.to_alias`; paths to existing data are built with
  `merxen.privacy.to_real`.
- Docs write locations as `.env` variables (`$MERXEN_EVIDENCE_ROOT/...`). Inside a verbatim
  quotation a name or path is replaced by its alias or variable in square brackets.
- The hooks enforce this (`pre-commit install`). They check real names only when they find the
  alias table (docs/local-configuration.md, Discovery); "real-name check skipped" in their output
  means it was not found. Before `gh pr create` or `gh pr edit`, pipe the title and body through
  `python3 scripts/privacy/check_private_text.py --stdin`.
- Not sample names: public references (WHB, WMB, SEA-AD, MERFISH-638850 and its sections, the ABC
  atlas, public 10x and Vizgen datasets, and the public Xenium hippocampus deposits: KI hippocampus,
  Zenodo 14879552 and 14879626, and JHU/LIBD dentate gyrus lifespan, GEO GSE317381, with their
  section, slide, run, instrument and donor ids) and the `.vzg2` file format. Personal data inside
  a deposit, such as a contact email in its metadata, stays private.


## Naming Conventions

Follow PEP 8 strictly:

| Element              | Convention       | Example                |
|----------------------|------------------|------------------------|
| Functions/variables  | snake_case       | `fetch_user_data()`    |
| Classes              | PascalCase       | `UserProfile`          |
| Constants            | UPPER_SNAKE_CASE | `MAX_RETRIES = 3`      |
| Private members      | _leading_underscore | `_internal_cache`   |
| Modules/packages     | short snake_case | `data_loader.py`       |

Additional rules:
- Be descriptive. `user_id` not `uid`, `calculate_total` not `calc_tot`. Code is read far more than it is written.
- Prefix internal-use attributes and methods with a single underscore. Avoid double underscore name mangling unless you specifically intend it.
- Boolean variables and functions should read as yes/no questions: `is_valid`, `has_permission`, `should_retry`.


## Type Hints

Add type hints to all public functions and methods.

```python
def load_dataset(path: str, *, sample_frac: float = 1.0) -> pd.DataFrame:
    ...

def train_model(X: np.ndarray, y: np.ndarray) -> sklearn.base.BaseEstimator:
    ...

def get_user(user_id: int) -> User | None:
    ...
```

Rules:
- Use built-in generics (`list[str]`, `dict[str, int]`, `tuple[int, ...]`) — no need to import from `typing` for these on Python 3.10+.
- Use `X | None` instead of `Optional[X]`.
- For structured data passed between functions, use `dataclass`, `TypedDict`, or Pydantic `BaseModel` — not plain dicts.
- Run mypy in CI. If adopting on an existing codebase, start with `--ignore-missing-imports` and tighten over time.

```python
from dataclasses import dataclass

@dataclass
class TrainResult:
    model: sklearn.base.BaseEstimator
    metrics: dict[str, float]
    feature_names: list[str]
```


## Documentation

### Docstrings

Write docstrings for all public functions, classes, and modules. Use Google-style format.

```python
def split_dataset(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into train and test sets with stratification.

    Args:
        df: Input dataframe with features and target.
        target_col: Name of the target column for stratified splitting.
        test_size: Fraction of data to reserve for testing.

    Returns:
        A tuple of (train_df, test_df).

    Raises:
        ValueError: If target_col is not in df.
    """
```

### README

The README must cover:
1. What the project does (one paragraph).
2. How to set up the environment (`conda env create -f envs/environment.yml`).
3. How to install pre-commit hooks (both commit and push types).
4. How to run tests (`pytest`).
5. How to run the main workflow / entry points.
6. Any required environment variables (point to `.env.example`).

If a new team member cannot get from clone to running tests within 10 minutes, the README needs work.

### Comments

Comment *why*, not *what*. The code shows what is happening. Comments explain tricky decisions, edge cases, known limitations, or constraints that aren't obvious from reading the code.

```python
# Retry with exponential backoff because the upstream API
# rate-limits to 10 req/s and returns 429 without Retry-After.
for attempt in range(MAX_RETRIES):
    ...
```


## Git Workflow

### Branches

- Commit routine work directly to `main`.
- Serious, risky or breaking changes, and changes that span unrelated areas, go on a short-lived branch and merge via pull request. When unsure which applies, ask before committing.
- When using branches, branch from `main`, keep them short-lived, merge via pull request, and delete them after merge.
- Exception: a long-lived integration branch is allowed only when a plan in `docs/plans/` names it; it follows the same PR, CI and commit-prefix rules and is deleted when the effort closes. Milestone branches may be cut from it and merged back into it by PR, and are deleted after merge.
- Do not resume the former integration branches (see docs/development.md, Integration branch).
- Never edit, check out, reset, pull or remove a worktree or code export a pipeline run executes from; see docs/development.md, Worktrees and running pipelines.

### Commit messages

Use imperative mood. Reference issue numbers where applicable.

```
[feature]: add stratified sampling to data loader (#42)
[bugfix]: handle NaN values in feature pipeline
[docs]: update README with environment setup instructions
[refactor]: extract preprocessing into separate module
[test]: add coverage for edge cases in validation
[chore]: refresh CI dependency pins
[release]: v1.3.0
```

A change users would notice also adds its entry under `## [Unreleased]` in
`CHANGELOG.md`, in the same commit, or in the same PR when the change goes on a branch
([versioning and releases](docs/versioning.md)).
Only release PRs change the version.

Commit title format is required:
- `[feature]:` changes intended behavior of the code.
- `[minor]:` minor non-breaking behavior change (for example, changing a default optional argument).
- `[refactor]:` refactoring without changing functionality.
- `[style]:` code style/formatting only.
- `[bugfix]:` bug fix.
- `[test]:` testing-related changes.
- `[docs]:` documentation, comments, or docstrings.
- `[chore]:` maintenance/configuration changes.
- `[release]:` release PRs only: the version bump and the release's changelog section (docs/versioning.md).

A commit message, branch name or PR text never names a sample or a host path (the commit-msg and pre-push hooks check).


## Linting, Formatting & Hooks

### Ruff

Use ruff for both linting and formatting. Configure it in pyproject.toml:

```toml
[tool.ruff]
line-length = 88
src = ["src"]

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "F",    # pyflakes
    "I",    # isort
    "N",    # pep8-naming
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "SIM",  # flake8-simplify
    "ANN",  # flake8-annotations (public functions)
]
ignore = ["ANN101"]  # don't require type annotation for self

[tool.ruff.lint.isort]
known-first-party = ["my_package"]
```

### Pre-commit and pre-push hooks

Two stages of automated checks:

**On every commit** — fast checks only (linting, formatting, file hygiene, and the privacy check of the staged lines and the commit message). These run in seconds and keep the codebase clean without slowing you down.

**On every push** — the privacy check of the pushed commits and branch name, then `scripts/run_ci_checks.sh`: ruff, mypy, the Nextflow lint, the changelog entry check on pushes to `main`, and the fast test tier (docs/development.md, Test tiers; on macOS the tests are skipped unless `MERXEN_CI_RUN_TESTS=true`). Catches broken code before it reaches the remote, without blocking rapid work-in-progress commits locally.

The hooks themselves are defined in `.pre-commit-config.yaml`; its ruff `rev` must match the ruff pinned in `pyproject.toml` and the lockfile.

After cloning, every contributor must install the hooks. The hook types (pre-commit,
pre-merge-commit, commit-msg, pre-push) come from `default_install_hook_types` in
`.pre-commit-config.yaml`:

```bash
pre-commit install
```

Note: these are local guardrails, and for work committed directly to `main` they are the only gate before the push: the PR-only Changelog and Privacy jobs never run on a direct push, and `ci.yml` runs only after it. Never bypass them with `--no-verify`. On a push to `main` the pre-push hook also runs the Changelog job's entry check (`scripts/run_ci_checks.sh`; `MERXEN_SKIP_CHANGELOG=1` stands in for the `skip-changelog` label). For branches, CI (see below) is the gate before merge.


## CI (GitHub Actions)

Every pull request must pass CI before merge. Branch protection on `main` may require status checks, but must not require pull requests: routine work is committed to `main` directly (Git Workflow).

The workflows live in `.github/workflows/`: `ci.yml` (lint, types and the tests, sharded over several runners, with test selection on PRs and a nightly full run), `changelog.yml` and `privacy.yml` (PR-only) and `release.yml`. docs/development.md, Test selection, explains which tests a PR runs.

Notes:
- CI uses `requirements/requirements.lock` for deterministic installs — not a fresh resolve.
- CI uses pip/uv directly, not conda. Conda is a local development convenience; CI needs speed and reproducibility. If you need system-level libraries in CI, install them with `apt-get` in a prior step.
- Branch protection, if used: Settings → Branches → Add rule for `main` → check "Require status checks to pass before merging" and select the CI job. Leave "Require a pull request before merging" off.


## Testing

Configure pytest in pyproject.toml:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
markers = ["slow: marks tests as slow (deselect with '-m \"not slow\"')"]
```

Rules:
- Every public function should have at least one test.
- Test names describe the scenario: `test_load_dataset_raises_on_missing_file`, not `test_load_1`.
- Use fixtures in `conftest.py` for shared test data, temporary directories, and mock objects.
- For data science code: test data transformations on small, known inputs with expected outputs. Test that models can fit and predict without error (smoke tests). Test edge cases like empty DataFrames, NaN values, and single-row inputs.
- Keep tests fast. If a test needs a large dataset or trained model, mark it with `@pytest.mark.slow` and exclude from default runs.


## Things to Avoid

- Do not add production logic to notebooks. Explore there, then refactor into `src/`.
- Do not use `from my_package import *`. Use explicit imports.
- Do not suppress exceptions broadly (`except Exception: pass`). Catch specific exceptions.
- Do not commit `.env` files, data files, model weights, or anything over 500KB.
- Do not leave `print()` statements in production code. Use `logging`.


## Rule Severity Reference

**Must** — non-negotiable. Violations block merge or push.
- Standard project layout (`src/`, `tests/`, `docs/`)
- `pyproject.toml` as single source of truth for dependencies
- PEP 8 naming
- Type hints on all public functions
- Docstrings on all public functions (Google-style)
- README covers install, run, test, deploy
- Routine work committed to `main`; serious, risky, breaking or wide changes on a short-lived branch merged via PR
- Meaningful commit messages with conventional prefixes
- Ruff for linting and formatting
- Pre-commit hooks for linting, pre-push hooks for testing
- CI runs tests, lint, and type checking on every PR

**Should** — strong default. Deviate only with a stated reason.
- Descriptive names, no abbreviations
- Prefix private members with `_`
- mypy in CI
- `TypedDict` / `dataclass` / Pydantic for structured data
- Comment *why*, not *what*
- Config via environment variables (never hardcoded)
- Pin dependencies with a lockfile
