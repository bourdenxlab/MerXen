"""Shared test fixtures for MerXen.

Tests are hermetic by default: they see what GitHub CI sees (no ``.env``, no
alias table, no Nextflow site configs), here and in every subprocess they
start. The real locations and the alias table are captured once, before the
isolation, and only tests that request :func:`real_path` or
:func:`real_aliases` see them; those skip with a stated reason when a value is
unset or its path is absent (docs/local-configuration.md).
"""

from __future__ import annotations

import fcntl
import json
import os
import select
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, TypeVar

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


# The environment as it was before the isolation, handed to pytest-xdist
# workers (they start from the controller's isolated environment). No code
# under test reads this variable.
_STASH_VAR = "_MERXEN_TEST_PRIVATE_STASH"


def _original_environment() -> dict[str, str]:
    stash = os.environ.get(_STASH_VAR)
    if stash is not None:
        return dict(json.loads(stash))
    names = [*local_config.KNOWN, *_ISOLATED]
    return {name: os.environ[name] for name in names if name in os.environ}


def _capture(
    original: dict[str, str],
) -> tuple[dict[str, str], Path | None, str | None, str]:
    """Real locations, alias table and .env, taken before the isolation."""
    for name in [*local_config.KNOWN, *_ISOLATED]:
        os.environ.pop(name, None)
    os.environ.update(original)
    local_config.clear_cache()
    privacy.clear_cache()
    try:
        locations = local_config.snapshot()
        alias_file = privacy.alias_file()
        env_file = local_config.env_file()
    except LocalConfigError as error:
        return {}, None, None, str(error)
    return locations, alias_file, str(env_file) if env_file else None, ""


_ORIGINAL = _original_environment()
_REAL_LOCATIONS, _REAL_ALIAS_FILE, _REAL_ENV_FILE, _CAPTURE_ERROR = _capture(_ORIGINAL)

# Session-wide isolation (also covers module- and session-scoped fixtures and
# the subprocesses they start); the autouse fixture repeats it per test.
for _name in local_config.KNOWN:
    os.environ.pop(_name, None)
os.environ.update(_ISOLATED)
os.environ[_STASH_VAR] = json.dumps(_ORIGINAL)
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


_WriterResult = TypeVar("_WriterResult")

# A holder or writer may import SpatialData from scratch, so every wait that
# covers a start-up or a whole writer run is generous.
_STORE_LOCK_TIMEOUT_SECONDS = 180.0
_STORE_LOCK_POLL_SECONDS = 0.05

# Holds the store write lock through the path in argv[1] until stdin closes or
# a line arrives, in its own process (as a second Nextflow task would).
_STORE_LOCK_HOLDER = """
import sys
from merxen.io.spatialdata_io import spatialdata_write_lock

with spatialdata_write_lock(sys.argv[1]):
    print("held", flush=True)
    sys.stdin.readline()
"""


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


class StoreLockContention:
    """Run a store writer while another process holds that store's write lock.

    The store is staged twice, as Nextflow stages it into two task work
    directories: a holder process takes the lock through one symlink and the
    writer under test is given the other. The writer must wait for the holder
    (it is seen blocked on the same lock file, and has not read the store),
    then read and persist only while it holds the lock.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self._monkeypatch = monkeypatch
        self._work_root = tmp_path / "store_lock_work"

    def stage(self, store: Path, task: str) -> Path:
        """Symlink ``store`` into a task work directory the way Nextflow does."""
        work_dir = self._work_root / task
        work_dir.mkdir(parents=True)
        staged = work_dir / store.name
        staged.symlink_to(store, target_is_directory=True)
        return staged

    def run(
        self,
        run_writer: Callable[[Path], _WriterResult],
        store: Path,
        *,
        reads: Sequence[tuple[Any, str]],
        persists: Sequence[tuple[Any, str]],
    ) -> _WriterResult:
        """Run ``run_writer`` on a second staging of ``store`` under contention.

        Args:
            run_writer: Calls the writer with the path it should use for the store.
            store: Real store directory both stagings point at.
            reads: ``(owner, attribute)`` of the functions the writer reads the
                store with; each is wrapped to record that it was called.
            persists: ``(owner, attribute)`` of the functions the writer persists
                with; each is wrapped to check that the lock is held at the time.

        Returns:
            What ``run_writer`` returned.
        """
        from merxen.io.spatialdata_io import spatialdata_write_lock_path

        if not Path("/proc/locks").is_file():
            pytest.skip("needs /proc/locks to see a writer waiting on the lock")
        holder_path = self.stage(store, "holder")
        writer_path = self.stage(store, "writer")
        reads_seen: list[str] = []
        persists_seen: list[tuple[str, bool]] = []
        for owner, name in reads:
            self._wrap(owner, name, lambda label: reads_seen.append(label))
        for owner, name in persists:
            self._wrap(
                owner,
                name,
                lambda label: persists_seen.append((label, _lock_is_held(holder_path))),
            )

        holder = subprocess.Popen(
            [sys.executable, "-c", _STORE_LOCK_HOLDER, str(holder_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, sys.path))},
        )
        outcome: dict[str, Any] = {}
        writer = threading.Thread(
            target=_record_outcome,
            args=(run_writer, writer_path, outcome),
            daemon=True,
        )
        try:
            _wait_for_holder(holder)
            lock_inode = spatialdata_write_lock_path(holder_path).stat().st_ino
            writer.start()
            _wait_until_blocked_on(lock_inode, writer, outcome)
            assert reads_seen == [], "the writer read the store before the lock"
            assert persists_seen == [], "the writer persisted before the lock"
        finally:
            _release_holder(holder)
        writer.join(timeout=_STORE_LOCK_TIMEOUT_SECONDS)
        assert not writer.is_alive(), "the writer did not finish after the release"
        if "error" in outcome:
            raise outcome["error"]
        assert reads_seen, "the writer never read the store"
        assert persists_seen, "the writer never persisted to the store"
        unlocked = [label for label, held in persists_seen if not held]
        assert not unlocked, f"persisted without holding the store lock: {unlocked}"
        return outcome["result"]

    def _wrap(self, owner: Any, name: str, record: Callable[[str], None]) -> None:
        original = getattr(owner, name)
        label = f"{getattr(owner, '__name__', type(owner).__name__)}.{name}"

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            record(label)
            return original(*args, **kwargs)

        self._monkeypatch.setattr(owner, name, wrapper)


@pytest.fixture
def store_lock_contention(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> StoreLockContention:
    """Check that a writer serializes with another process on the store lock."""
    return StoreLockContention(monkeypatch, tmp_path)


def _record_outcome(
    run_writer: Callable[[Path], Any], writer_path: Path, outcome: dict[str, Any]
) -> None:
    try:
        outcome["result"] = run_writer(writer_path)
    except BaseException as exc:  # noqa: BLE001 - re-raised in the test thread
        outcome["error"] = exc


def _lock_is_held(store_path: Path) -> bool:
    """Probe, through a new open file, whether anyone holds the store lock."""
    from merxen.io.spatialdata_io import spatialdata_write_lock_path

    with spatialdata_write_lock_path(store_path).open("a", encoding="utf-8") as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(probe, fcntl.LOCK_UN)
        return False


def _wait_for_holder(holder: subprocess.Popen[str]) -> None:
    assert holder.stdout is not None
    ready, _, _ = select.select([holder.stdout], [], [], _STORE_LOCK_TIMEOUT_SECONDS)
    assert ready, "the lock holder did not start"
    assert holder.stdout.readline().strip() == "held", "the lock holder failed"


def _release_holder(holder: subprocess.Popen[str]) -> None:
    if holder.stdin is not None and not holder.stdin.closed:
        try:
            holder.stdin.write("release\n")
            holder.stdin.close()
        except BrokenPipeError:
            pass
    try:
        holder.wait(timeout=_STORE_LOCK_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        holder.kill()
        holder.wait()
    if holder.stdout is not None:
        holder.stdout.close()
    assert holder.returncode == 0, "the lock holder failed"


def _wait_until_blocked_on(
    lock_inode: int, writer: threading.Thread, outcome: dict[str, Any]
) -> None:
    """Wait until this process has a blocked ``flock`` request on the lock file."""
    deadline = time.monotonic() + _STORE_LOCK_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _has_blocked_flock(lock_inode):
            return
        if not writer.is_alive():
            if "error" in outcome:
                raise outcome["error"]
            pytest.fail("the writer finished without waiting for the store lock")
        writer.join(timeout=_STORE_LOCK_POLL_SECONDS)
    pytest.fail("the writer never waited for the store lock")


def _has_blocked_flock(lock_inode: int) -> bool:
    # /proc/locks lists a waiter as "N: -> FLOCK ADVISORY WRITE <pid> <dev>:<ino> ...".
    # The lock file is unique to this test, so only the writer can wait on it.
    for line in Path("/proc/locks").read_text().splitlines():
        fields = line.split()
        if len(fields) < 7 or fields[1] != "->" or fields[2] != "FLOCK":
            continue
        if fields[6].rsplit(":", 1)[-1] == str(lock_inode):
            return True
    return False
