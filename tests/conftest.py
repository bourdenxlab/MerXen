"""Shared test fixtures for MerXen.

Tests are hermetic by default: they see what GitHub CI sees (no ``.env``, no
alias table, no Nextflow site configs), here and in every subprocess they
start. The real locations and the alias table are captured once, before the
isolation, and only tests that request :func:`real_path` or
:func:`real_aliases` see them; those skip with a stated reason when a value is
unset or its path is absent (docs/local-configuration.md).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest

from merxen import local_config, privacy
from merxen.local_config import LocalConfigError

# Hermetic settings: the private files are absent on purpose.
_ISOLATED = {
    "MERXEN_ENV_FILE": "none",
    "MERXEN_ALIAS_FILE": "none",
    "MERXEN_SITE_CONFIG": "/dev/null",
    "MERXEN_DWIGHT_SITE_CONFIG": "/dev/null",
}


def _capture() -> tuple[dict[str, str], Path | None, str | None, str]:
    """Real locations, alias table and .env, taken before the isolation."""
    try:
        locations = local_config.snapshot()
        alias_file = privacy.alias_file()
        env_file = local_config.env_file()
    except LocalConfigError as error:
        return {}, None, None, str(error)
    return locations, alias_file, str(env_file) if env_file else None, ""


_REAL_LOCATIONS, _REAL_ALIAS_FILE, _REAL_ENV_FILE, _CAPTURE_ERROR = _capture()

# Session-wide isolation (also covers module- and session-scoped fixtures and
# the subprocesses they start); the autouse fixture repeats it per test.
for _name in local_config.KNOWN:
    os.environ.pop(_name, None)
os.environ.update(_ISOLATED)
local_config.clear_cache()
privacy.clear_cache()


@pytest.fixture(autouse=True)
def _no_private_files(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Hide every private location and the alias table from each test."""
    for name in list(os.environ):
        if name in local_config.KNOWN:
            monkeypatch.delenv(name)
    for name, value in _ISOLATED.items():
        monkeypatch.setenv(name, value)
    local_config.clear_cache()
    privacy.clear_cache()
    yield
    local_config.clear_cache()
    privacy.clear_cache()


@pytest.fixture
def real_path(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], Path]:
    """Factory: ``real_path("MERXEN_EVIDENCE_ROOT")`` -> Path, or skip.

    Skips when the variable is unset (``.env``), when the value lies under the
    refused prefix (``MERXEN_REFUSED_PATH_PREFIX``: no test reads the failing
    disk) or when the path does not exist on this host. The variable is also
    set for the test, because the code under test reads it through
    ``merxen.local_config``.
    """

    def factory(name: str) -> Path:
        if _CAPTURE_ERROR:
            pytest.skip(f"private configuration unreadable: {_CAPTURE_ERROR}")
        value = _REAL_LOCATIONS.get(name)
        if not value:
            pytest.skip(f"{name} is not set (.env): this test reads real data")
        prefix = (_REAL_LOCATIONS.get("MERXEN_REFUSED_PATH_PREFIX") or "").rstrip("/")
        if prefix and (value == prefix or value.startswith(prefix + "/")):
            pytest.skip(f"{name} lies under MERXEN_REFUSED_PATH_PREFIX: not read")
        path = Path(value)
        if not path.exists():
            pytest.skip(f"{name} does not exist on this host")
        monkeypatch.setenv(name, value)
        local_config.clear_cache()
        return path

    return factory


@pytest.fixture
def real_aliases(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Re-enable the real alias table for one test (skip without one)."""
    if _REAL_ALIAS_FILE is None:
        pytest.skip("no alias table: needs real names")
    monkeypatch.setenv(privacy.ALIAS_FILE_VAR, str(_REAL_ALIAS_FILE))
    privacy.clear_cache()
    yield _REAL_ALIAS_FILE
    privacy.clear_cache()


@pytest.fixture
def real_env_file() -> Path:
    """The real ``.env`` captured before the isolation (skip without one)."""
    if _REAL_ENV_FILE is None:
        pytest.skip("no .env: needs the private configuration")
    return Path(_REAL_ENV_FILE)


@pytest.fixture
def tiny_mask() -> np.ndarray:
    """A 32x32 labeled mask with 3 cell regions."""
    mask = np.zeros((32, 32), dtype=np.int32)
    mask[2:8, 2:8] = 1
    mask[12:20, 12:20] = 2
    mask[24:30, 5:15] = 3
    return mask


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Temporary directory for test outputs."""
    return tmp_path / "output"
