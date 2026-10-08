"""Tests for the store-level SpatialData writer lock.

Nextflow stages a SpatialData store into each task's work directory as a
symlink, and results directories are often reached through a symlinked parent.
Every path that reaches one store must therefore take the same lock, or
concurrent writers to that store are not serialized.
"""

from __future__ import annotations

import fcntl
import multiprocessing
from multiprocessing.synchronize import Event
from pathlib import Path

import pytest

from merxen.io.spatialdata_io import (
    spatialdata_write_lock,
    spatialdata_write_lock_path,
)

# Spawned children import this module (and SpatialData) from scratch, so every
# wait that covers a child start-up is generous; only the "still blocked"
# check uses a short window, and it starts after the contender reports ready.
_STARTUP_TIMEOUT_SECONDS = 120.0
_BLOCKED_WINDOW_SECONDS = 1.0


def _hold_lock(store_path: str, held: Event, release: Event) -> None:
    with spatialdata_write_lock(store_path):
        held.set()
        release.wait(timeout=_STARTUP_TIMEOUT_SECONDS)


def _take_lock(store_path: str, ready: Event, acquired: Event) -> None:
    ready.set()
    with spatialdata_write_lock(store_path):
        acquired.set()


def _assert_writers_serialize(first_path: Path, second_path: Path) -> None:
    """Assert a writer through ``second_path`` waits for one through ``first_path``.

    The holder and the contender are separate processes, as two Nextflow tasks
    are. While the holder has the lock, a non-blocking probe through the second
    path must also be refused.
    """
    context = multiprocessing.get_context("spawn")
    held = context.Event()
    release = context.Event()
    ready = context.Event()
    acquired = context.Event()
    holder = context.Process(
        target=_hold_lock, args=(str(first_path), held, release), daemon=True
    )
    contender = context.Process(
        target=_take_lock, args=(str(second_path), ready, acquired), daemon=True
    )
    holder.start()
    try:
        assert held.wait(timeout=_STARTUP_TIMEOUT_SECONDS)

        probe_path = spatialdata_write_lock_path(second_path)
        with (
            probe_path.open("a", encoding="utf-8") as probe,
            pytest.raises(BlockingIOError),
        ):
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)

        contender.start()
        assert ready.wait(timeout=_STARTUP_TIMEOUT_SECONDS)
        assert not acquired.wait(timeout=_BLOCKED_WINDOW_SECONDS)

        release.set()
        assert acquired.wait(timeout=_STARTUP_TIMEOUT_SECONDS)
    finally:
        release.set()
        for process in (holder, contender):
            if process.ident is not None:
                process.join(timeout=_STARTUP_TIMEOUT_SECONDS)
                if process.is_alive():
                    process.kill()
                    process.join()
    assert holder.exitcode == 0
    assert contender.exitcode == 0


def _stage_like_nextflow(store: Path, work_dir: Path) -> Path:
    """Symlink ``store`` into a task work directory the way Nextflow stages it."""
    work_dir.mkdir(parents=True)
    staged = work_dir / store.name
    staged.symlink_to(store, target_is_directory=True)
    return staged


def test_writers_through_two_staged_symlinks_serialize(tmp_path: Path) -> None:
    store = tmp_path / "results" / "P1" / "merscope" / "latest" / "latest.zarr"
    store.mkdir(parents=True)
    first = _stage_like_nextflow(store, tmp_path / "work" / "aa" / "1111")
    second = _stage_like_nextflow(store, tmp_path / "work" / "bb" / "2222")

    _assert_writers_serialize(first, second)


def test_writers_through_symlinked_parent_serialize(tmp_path: Path) -> None:
    real_results = tmp_path / "storage" / "results"
    store = real_results / "P1" / "merscope" / "latest" / "latest.zarr"
    store.mkdir(parents=True)
    linked_results = tmp_path / "checkout" / "results"
    linked_results.parent.mkdir()
    linked_results.symlink_to(real_results, target_is_directory=True)
    through_link = linked_results / "P1" / "merscope" / "latest" / "latest.zarr"

    _assert_writers_serialize(through_link, store)


def test_lock_path_sits_beside_the_real_store(tmp_path: Path) -> None:
    store = tmp_path / "results" / "latest.zarr"
    store.mkdir(parents=True)
    staged = _stage_like_nextflow(store, tmp_path / "work" / "aa")
    # Some modules add a second hop: latest_input.zarr -> latest.zarr -> store.
    chained = staged.parent / "latest_input.zarr"
    chained.symlink_to(staged.name, target_is_directory=True)

    expected = store.resolve().parent / "latest.zarr.merxen-write.lock"
    assert spatialdata_write_lock_path(store) == expected
    assert spatialdata_write_lock_path(staged) == expected
    assert spatialdata_write_lock_path(chained) == expected
    assert spatialdata_write_lock_path(f"{staged}/") == expected

    with spatialdata_write_lock(chained) as acquired_lock:
        assert acquired_lock == expected
    assert expected.exists()
    assert not (staged.parent / "latest.zarr.merxen-write.lock").exists()
    assert not (staged.parent / "latest_input.zarr.merxen-write.lock").exists()


def test_relative_store_path_gets_the_absolute_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = tmp_path / "results" / "latest.zarr"
    store.mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "results")

    lock_path = spatialdata_write_lock_path("latest.zarr")

    assert lock_path.is_absolute()
    assert lock_path == spatialdata_write_lock_path(store)
    assert lock_path == spatialdata_write_lock_path("../results/./latest.zarr")


def test_store_not_yet_written_gets_the_lock_it_will_keep(tmp_path: Path) -> None:
    real_results = tmp_path / "storage" / "results"
    real_results.mkdir(parents=True)
    linked_results = tmp_path / "results"
    linked_results.symlink_to(real_results, target_is_directory=True)
    future_store = real_results / "P1" / "latest.zarr"
    # A task may also be handed a symlink to a store that is not written yet.
    dangling = tmp_path / "work" / "latest.zarr"
    dangling.parent.mkdir()
    dangling.symlink_to(future_store, target_is_directory=True)

    before = spatialdata_write_lock_path(linked_results / "P1" / "latest.zarr")
    assert before == spatialdata_write_lock_path(future_store)
    assert before == spatialdata_write_lock_path(dangling)

    future_store.mkdir(parents=True)

    assert spatialdata_write_lock_path(linked_results / "P1" / "latest.zarr") == (
        before
    )
    assert spatialdata_write_lock_path(dangling) == before
    assert before == future_store.resolve().parent / "latest.zarr.merxen-write.lock"
