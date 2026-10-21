"""Process-level GPU lock shared with the Nextflow GPU tasks.

The Cellpose, alignment and clustering tasks serialise their use of the one
workstation GPU with a shell ``flock`` on a fixed lock file held for the whole
task (``exec 9<file; flock 9`` in their ``beforeScript``). Python code that
needs the GPU for only part of its run, such as the bead-removal correction in
the SpatialData build, takes the same kernel lock around that part with
:func:`gpu_process_lock`, so it queues behind and in front of those tasks
without holding the GPU for the rest of its run.

The lock file is opened read-only and locked with ``flock(2)``, exactly like
the shell side, so the two interoperate.
"""

from __future__ import annotations

import errno
import fcntl
import gc
import logging
import os
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

GPU_LOCK_ENV_VAR = "MERXEN_GPU_LOCK_FILE"

_HELD = threading.local()


def gpu_lock_path_from_env() -> Path | None:
    """Return the lock file named by ``MERXEN_GPU_LOCK_FILE``.

    Returns:
        The path, or ``None`` when the variable is unset or empty (no lock is
        configured, so GPU use is not serialised).
    """
    value = os.environ.get(GPU_LOCK_ENV_VAR, "").strip()
    return Path(value) if value else None


def _create_lock_file(path: Path) -> None:
    """Create the lock file when it is missing, never truncating it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return
    os.close(descriptor)


def _held_paths() -> set[str]:
    held: set[str] | None = getattr(_HELD, "paths", None)
    if held is None:
        held = set()
        _HELD.paths = held
    return held


def _release_cuda_cache() -> None:
    """Hand cached GPU memory back before the lock is released.

    The cache is only cleared when torch is already loaded, so releasing the
    lock never imports it. ``gc.collect`` runs first so that tensors that only
    a reference cycle keeps alive are freed and ``empty_cache`` can return them.
    """
    if "torch" not in sys.modules:
        return
    from merxen.memory import clear_cuda_cache

    # A failed cleanup must not replace the block's own outcome.
    try:
        gc.collect()
        clear_cuda_cache()
    except Exception:
        logger.warning("Could not clear the CUDA cache after the GPU lock.")


@contextmanager
def gpu_process_lock(path: Path | str | None) -> Iterator[Path | None]:
    """Hold the exclusive host GPU lock for the duration of the block.

    The lock file is created when it does not exist (mode 0o644, like the
    shell side), opened read-only and locked with ``flock(LOCK_EX)``. The call
    blocks until every other holder, shell or Python, has released it. It is
    released in a ``finally``; when torch is loaded the CUDA cache is cleared
    first, so the next holder does not start while this process still keeps its
    reserved GPU memory.

    The lock is not re-entrant: a second open of the file conflicts with the
    first even within one process, so a nested acquisition would wait forever
    and raises instead.

    Args:
        path: Lock file, or ``None`` for no lock (the block runs unserialised).

    Yields:
        The lock path while the lock is held, or ``None`` without a lock.

    Raises:
        RuntimeError: This thread already holds the lock.
        PermissionError: The lock file is not readable.
    """
    if path is None:
        yield None
        return
    lock_path = Path(path)
    key = str(lock_path.absolute())
    held = _held_paths()
    if key in held:
        raise RuntimeError(
            f"This thread already holds the MerXen GPU process lock {lock_path}; "
            "taking it again would deadlock."
        )
    _create_lock_file(lock_path)
    try:
        descriptor = os.open(lock_path, os.O_RDONLY)
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EPERM):
            raise PermissionError(
                f"MerXen GPU lock is not readable: {lock_path}"
            ) from exc
        raise
    try:
        logger.info("Waiting for MerXen GPU process lock: %s", lock_path)
        waiting_since = time.monotonic()
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        acquired_at = time.monotonic()
        held.add(key)
        logger.info(
            "Acquired MerXen GPU process lock: %s (waited %.1f s)",
            lock_path,
            acquired_at - waiting_since,
        )
        try:
            yield lock_path
        finally:
            held.discard(key)
            # Before the unlock: the next holder must find the memory free.
            _release_cuda_cache()
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            logger.info(
                "Released MerXen GPU process lock: %s (held %.1f s)",
                lock_path,
                time.monotonic() - acquired_at,
            )
    finally:
        os.close(descriptor)
