# Agents.md — Project Standards

This file defines the rules and conventions for this repository. All contributors and AI assistants must follow these standards when writing, reviewing, or modifying code.

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
├── environment.yml      # Conda env: python version, system-level deps, pip install
├── requirements.lock    # Pinned dependency tree — generated, never hand-edited
├── .pre-commit-config.yaml
├── .env.example         # Template for environment variables (never commit .env)
├── .gitignore
├── CLAUDE.md
└── README.md
```

Rules:
- One package per repo. The installable package lives in `src/my_package/`.
- Tests mirror the package structure under `tests/`.
- Notebooks are for exploration and communication. If logic in a notebook is needed in production, refactor it into `src/`.
- Data files and model weights do not go in the repo. Use `.gitignore` to exclude them.


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

### environment.yml

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
      - -e ".[dev]"
```

Do not duplicate dependencies across environment.yml and pyproject.toml. If it can be installed via pip, it belongs in pyproject.toml.

### Lockfile

Generate a lockfile to pin the full resolved dependency tree:

```bash
uv pip compile pyproject.toml -o requirements.lock
```

Commit `requirements.lock` to version control. Regenerate it when you add or update dependencies. For reproducible installs (CI, onboarding):

```bash
uv pip install -r requirements.lock
pip install -e . --no-deps
```

Never hand-edit the lockfile. Never install packages by running `pip install <package>` directly — add them to `pyproject.toml` and regenerate the lockfile.


## Environment & Configuration

- Store all configuration, secrets, and environment-specific values in environment variables.
- Use `.env` files locally with `python-decouple` or `pydantic-settings` to load them.
- Never hardcode secrets, API keys, file paths to local machines, or database credentials.
- Commit a `.env.example` with placeholder values so new contributors know what's needed.
- Add `.env` to `.gitignore`.

```python
from decouple import config

DATABASE_URL = config("DATABASE_URL")
API_KEY = config("API_KEY")
DEBUG = config("DEBUG", default=False, cast=bool)
```


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
2. How to set up the environment (`conda env create -f environment.yml`).
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

- Branch from `main` for every piece of work. Use short-lived feature branches — aim to merge within 1–2 days.
- Merge via pull request. Do not push directly to `main`.
- Delete branches after merge.

### Commit messages

Use imperative mood. Reference issue numbers where applicable.

```
[feature]: add stratified sampling to data loader (#42)
[bugfix]: handle NaN values in feature pipeline
[docs]: update README with environment setup instructions
[refactor]: extract preprocessing into separate module
[test]: add coverage for edge cases in validation
[chore]: refresh CI dependency pins
```

Commit title format is required:
- `[feature]:` changes intended behavior of the code.
- `[minor]:` minor non-breaking behavior change (for example, changing a default optional argument).
- `[refactor]:` refactoring without changing functionality.
- `[style]:` code style/formatting only.
- `[bugfix]:` bug fix.
- `[test]:` testing-related changes.
- `[docs]:` documentation, comments, or docstrings.
- `[chore]:` maintenance/configuration changes.


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

**On every commit** — fast checks only (linting, formatting, file hygiene). These run in seconds and keep the codebase clean without slowing you down.

**On every push** — the full test suite. Catches broken code before it reaches the remote, without blocking rapid work-in-progress commits locally.

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=500']

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        stages: [pre-push]
        always_run: true
        pass_filenames: false
```

After cloning, every contributor must install both hook types:

```bash
pre-commit install
pre-commit install --hook-type pre-push
```

Note: these are local guardrails. Contributors can bypass them with `--no-verify`. CI (see below) is the authoritative gate that cannot be bypassed.


## CI (GitHub Actions)

Every pull request must pass CI before merge. Configure branch protection rules on `main` to enforce this.

```yaml
# .github/workflows/ci.yml
name: CI
on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install uv
        run: pip install uv
      - name: Install dependencies
        run: |
          uv pip install --system -r requirements.lock
          uv pip install --system -e . --no-deps
      - name: Lint
        run: ruff check .
      - name: Format check
        run: ruff format --check .
      - name: Type check
        run: mypy src/
      - name: Tests
        run: pytest
```

Notes:
- CI uses `requirements.lock` for deterministic installs — not a fresh resolve.
- CI uses pip/uv directly, not conda. Conda is a local development convenience; CI needs speed and reproducibility. If you need system-level libraries in CI, install them with `apt-get` in a prior step.
- Set up branch protection: Settings → Branches → Add rule for `main` → check "Require status checks to pass before merging" and select the CI job.


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

**Must** — non-negotiable. Violations block merge.
- Standard project layout (`src/`, `tests/`, `docs/`)
- `pyproject.toml` as single source of truth for dependencies
- PEP 8 naming
- Type hints on all public functions
- Docstrings on all public functions (Google-style)
- README covers install, run, test, deploy
- Feature branches merged via PR
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
