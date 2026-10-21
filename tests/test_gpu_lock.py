"""Tests for the process-level GPU lock.

The lock must serialise Python holders with each other and with the shell
``flock`` the Nextflow GPU tasks take on the same file, and it must never need
a GPU: no test here imports torch.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import shutil
import stat
import subprocess
import sys
import threading
import types
from multiprocessing.synchronize import Event
from pathlib import Path

import pytest

from merxen import gpu_lock
from merxen.gpu_lock import GPU_LOCK_ENV_VAR, gpu_lock_path_from_env, gpu_process_lock

# Spawned children import this module from scratch, so waits that cover a child
# start-up are generous; only the "still blocked" window is short, and it starts
# after the contender reports that it is about to ask for the lock.
_STARTUP_TIMEOUT_SECONDS = 120.0
_BLOCKED_WINDOW_SECONDS = 1.0

_needs_flock = pytest.mark.skipif(
    shutil.which("flock") is None, reason="the flock command is unavailable"
)


def _hold_lock(path: str, held: Event, release: Event) -> None:
    with gpu_process_lock(path):
        held.set()
        release.wait(timeout=_STARTUP_TIMEOUT_SECONDS)


def _take_lock(path: str, ready: Event, acquired: Event) -> None:
    ready.set()
    with gpu_process_lock(path):
        acquired.set()


def test_two_processes_are_serialised(tmp_path: Path) -> None:
    """G7: a process asking for the lock blocks while another holds it."""
    lock = tmp_path / "gpu.lock"
    context = multiprocessing.get_context("spawn")
    held, release = context.Event(), context.Event()
    ready, acquired = context.Event(), context.Event()
    holder = context.Process(
        target=_hold_lock, args=(str(lock), held, release), daemon=True
    )
    contender = context.Process(
        target=_take_lock, args=(str(lock), ready, acquired), daemon=True
    )
    holder.start()
    try:
        assert held.wait(_STARTUP_TIMEOUT_SECONDS)
        contender.start()
        assert ready.wait(_STARTUP_TIMEOUT_SECONDS)
        assert not acquired.wait(_BLOCKED_WINDOW_SECONDS)
        release.set()
        assert acquired.wait(_STARTUP_TIMEOUT_SECONDS)
    finally:
        release.set()
        for process in (holder, contender):
            process.join(timeout=_STARTUP_TIMEOUT_SECONDS)
            if process.is_alive():
                process.terminate()
    assert holder.exitcode == 0
    assert contender.exitcode == 0


@_needs_flock
def test_waits_for_a_shell_flock_holder(tmp_path: Path) -> None:
    """G7: the Nextflow tasks hold the lock with ``exec 9<file; flock 9``."""
    lock = tmp_path / "gpu.lock"
    lock.touch()
    holder = subprocess.Popen(
        ["bash", "-c", f'exec 9<"{lock}"; flock 9; echo held; sleep 3'],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "held"
        acquired = threading.Event()

        def contend() -> None:
            with gpu_process_lock(lock):
                acquired.set()

        thread = threading.Thread(target=contend, daemon=True)
        thread.start()
        assert not acquired.wait(_BLOCKED_WINDOW_SECONDS)
        holder.kill()
        holder.wait()
        assert acquired.wait(30.0)
        thread.join(timeout=30.0)
    finally:
        holder.kill()
        holder.wait()


@_needs_flock
def test_a_shell_flock_is_refused_while_python_holds_the_lock(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    with gpu_process_lock(lock):
        shell = subprocess.run(
            ["flock", "-n", str(lock), "true"], capture_output=True, check=False
        )
        assert shell.returncode != 0
    free = subprocess.run(
        ["flock", "-n", str(lock), "true"], capture_output=True, check=False
    )
    assert free.returncode == 0


def test_no_path_is_a_no_op() -> None:
    with gpu_process_lock(None) as held:
        assert held is None


def test_a_missing_lock_file_and_folder_are_created(tmp_path: Path) -> None:
    lock = tmp_path / "locks" / "gpu.lock"
    old_umask = os.umask(0o022)
    try:
        with gpu_process_lock(lock) as held:
            assert held == lock
            assert lock.is_file()
    finally:
        os.umask(old_umask)
    assert stat.S_IMODE(lock.stat().st_mode) == 0o644


def test_an_existing_lock_file_is_not_truncated(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    lock.write_text("owner")
    with gpu_process_lock(str(lock)):
        pass
    assert lock.read_text() == "owner"


def test_the_lock_is_released_when_the_block_raises(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    with pytest.raises(ZeroDivisionError), gpu_process_lock(lock):
        raise ZeroDivisionError
    # A second acquisition would hang if the first had leaked.
    with gpu_process_lock(lock):
        pass


def test_a_nested_acquisition_raises_instead_of_deadlocking(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    with gpu_process_lock(lock):
        assert lock.is_file()
        with pytest.raises(RuntimeError, match="already holds"):
            gpu_process_lock(lock).__enter__()
    with gpu_process_lock(lock):
        pass


def test_wait_acquire_and_release_are_logged_with_seconds(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    lock = tmp_path / "gpu.lock"
    with (
        caplog.at_level(logging.INFO, logger="merxen.gpu_lock"),
        gpu_process_lock(lock),
    ):
        pass
    messages = [record.getMessage() for record in caplog.records]
    assert messages[0].startswith("Waiting for MerXen GPU process lock")
    assert messages[1].startswith("Acquired MerXen GPU process lock")
    assert "waited " in messages[1] and " s)" in messages[1]
    assert messages[2].startswith("Released MerXen GPU process lock")
    assert "held " in messages[2]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every file")
def test_an_unreadable_lock_file_is_reported(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    lock.touch()
    lock.chmod(0)
    try:
        with (
            pytest.raises(PermissionError, match="not readable"),
            gpu_process_lock(lock),
        ):
            pass  # pragma: no cover - the open raises
    finally:
        lock.chmod(0o644)


def test_the_cuda_cache_is_cleared_after_release_when_torch_is_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "gpu.lock"
    events: list[str] = []
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setattr(
        "merxen.memory.clear_cuda_cache", lambda: events.append("clear")
    )

    with gpu_process_lock(lock):
        events.append("held")
    events.append("after")

    assert events == ["held", "clear", "after"]


def test_a_failing_cache_clear_does_not_hide_the_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken() -> None:
        raise RuntimeError("driver gone")

    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setattr("merxen.memory.clear_cuda_cache", broken)
    with gpu_process_lock(tmp_path / "gpu.lock"):
        pass


def test_releasing_the_lock_never_imports_torch(tmp_path: Path) -> None:
    code = (
        "import sys\n"
        "from merxen.gpu_lock import gpu_process_lock\n"
        f"with gpu_process_lock({str(tmp_path / 'gpu.lock')!r}):\n"
        "    pass\n"
        "assert 'torch' not in sys.modules and 'triton' not in sys.modules\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_the_lock_path_comes_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(GPU_LOCK_ENV_VAR, raising=False)
    assert gpu_lock_path_from_env() is None
    monkeypatch.setenv(GPU_LOCK_ENV_VAR, "  ")
    assert gpu_lock_path_from_env() is None
    monkeypatch.setenv(GPU_LOCK_ENV_VAR, str(tmp_path / "gpu.lock"))
    assert gpu_lock_path_from_env() == tmp_path / "gpu.lock"
    assert gpu_lock.GPU_LOCK_ENV_VAR == "MERXEN_GPU_LOCK_FILE"
