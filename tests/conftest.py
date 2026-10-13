"""Shared test fixtures for MerXen."""

from __future__ import annotations

import fcntl
import os
import select
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pytest

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
