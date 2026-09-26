"""Keep the Nextflow conda env and the base image in step with the lockfile.

Nextflow reuses ``work/conda/env-<hash>`` for as long as the text of
``envs/environment.yml`` is unchanged. The file installs MerXen with a thin
editable line, so without a lockfile checksum in its text a dependency bump
(for example cell_type_mapper 1.5.5 -> 1.7.2) never reaches the cached env.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / "envs" / "environment.yml"
LOCK_PATH = REPO_ROOT / "requirements" / "requirements.lock"
DOCKERFILE_PATH = REPO_ROOT / "containers" / "Dockerfile"
SCRIPT_PATH = REPO_ROOT / "scripts" / "update_env_lock_hash.py"

HEADER_PATTERN = re.compile(
    r"^# requirements\.lock sha256: (?P<digest>[0-9a-f]{64})$", re.MULTILINE
)


@pytest.fixture(scope="module")
def lock_hash_script() -> ModuleType:
    """Import ``scripts/update_env_lock_hash.py`` as a module."""
    spec = importlib.util.spec_from_file_location("update_env_lock_hash", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_environment_yml_header_matches_lockfile_sha256() -> None:
    env_text = ENV_PATH.read_text()
    matches = HEADER_PATTERN.findall(env_text)
    expected = hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest()

    assert len(matches) == 1, "envs/environment.yml needs exactly one lock header"
    assert matches[0] == expected, (
        "envs/environment.yml records a stale requirements.lock sha256, so "
        "Nextflow would keep reusing an old work/conda env. Run "
        "`python scripts/update_env_lock_hash.py`."
    )


def test_environment_yml_keeps_thin_editable_install() -> None:
    # pip raises ResolutionImpossible on the lock (cell_type_mapper's unpinned
    # abc_atlas_access git URL against the lock's pinned commit), so the conda
    # env must not install it with -r.
    env_text = ENV_PATH.read_text()

    assert '- -e "../[dev]"' in env_text
    assert "-r ../requirements/requirements.lock" not in env_text


def test_dockerfile_installs_lockfile_with_uv_then_package_without_deps() -> None:
    run_lines = [
        line.strip()
        for line in DOCKERFILE_PATH.read_text().splitlines()
        if line.strip().startswith("RUN ")
    ]
    lock_installs = [
        index
        for index, line in enumerate(run_lines)
        if "uv pip install" in line and "-r requirements/requirements.lock" in line
    ]
    package_installs = [
        index
        for index, line in enumerate(run_lines)
        if re.search(r"\bpip install\b.*\s\.$", line)
    ]

    assert len(lock_installs) == 1
    assert len(package_installs) == 1
    package_line = run_lines[package_installs[0]]
    assert "uv pip install" in package_line
    assert "--no-deps" in package_line, "the package must not re-resolve its deps"
    assert lock_installs[0] < package_installs[0]


def test_update_env_text_replaces_existing_header(
    lock_hash_script: ModuleType,
) -> None:
    old = f"# requirements.lock sha256: {'0' * 64}\nname: merxen\n"

    updated = lock_hash_script.update_env_text(old, "a" * 64)

    assert updated == f"# requirements.lock sha256: {'a' * 64}\nname: merxen\n"
    assert lock_hash_script.read_header_digest(updated) == "a" * 64


def test_update_env_text_prepends_missing_header(
    lock_hash_script: ModuleType,
) -> None:
    updated = lock_hash_script.update_env_text("name: merxen\n", "b" * 64)

    assert updated == f"# requirements.lock sha256: {'b' * 64}\nname: merxen\n"
    assert lock_hash_script.read_header_digest("name: merxen\n") is None


def test_main_check_fails_on_stale_header_and_rewrite_fixes_it(
    lock_hash_script: ModuleType, tmp_path: Path
) -> None:
    lock_path = tmp_path / "requirements.lock"
    lock_path.write_text("numpy==2.0.0\n")
    env_path = tmp_path / "environment.yml"
    env_path.write_text(f"# requirements.lock sha256: {'0' * 64}\nname: merxen\n")
    arguments = ["--lock", str(lock_path), "--env", str(env_path)]

    assert lock_hash_script.main([*arguments, "--check"]) == 1
    assert lock_hash_script.main(arguments) == 0
    assert lock_hash_script.main([*arguments, "--check"]) == 0
    assert lock_hash_script.read_header_digest(env_path.read_text()) == (
        hashlib.sha256(b"numpy==2.0.0\n").hexdigest()
    )
