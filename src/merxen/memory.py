"""Memory monitoring, garbage collection, and CUDA cache management."""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import signal
import threading
from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path
from types import TracebackType
from typing import Final

import psutil

logger = logging.getLogger(__name__)


def memory_snapshot_gb() -> dict[str, float]:
    """Query process RSS and system memory stats.

    Returns:
        Dict with keys rss_gb, used_gb, available_gb.
    """
    vm = psutil.virtual_memory()
    rss = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    return {
        "rss_gb": rss,
        "used_gb": vm.used / (1024**3),
        "available_gb": vm.available / (1024**3),
    }


def log_status(msg: str) -> None:
    """Log a message with current memory metrics.

    Args:
        msg: The message to log.
    """
    mem = memory_snapshot_gb()
    logger.info(
        "%s | RSS=%.1f GB | System used=%.1f GB | System avail=%.1f GB",
        msg,
        mem["rss_gb"],
        mem["used_gb"],
        mem["available_gb"],
    )


def enforce_memory_limit(
    stage: str = "",
    max_gb: float = 600.0,
    warn_gb: float = 560.0,
) -> None:
    """Raise MemoryError if system RAM exceeds the configured limit.

    Args:
        stage: Description of the current pipeline stage (for error messages).
        max_gb: Hard limit in GB. Raises MemoryError if exceeded.
        warn_gb: Soft limit in GB. Logs a warning if exceeded.

    Raises:
        MemoryError: If system RAM usage exceeds max_gb.
    """
    used_gb = psutil.virtual_memory().used / (1024**3)
    if used_gb > max_gb:
        raise MemoryError(
            f"System RAM usage exceeded limit at '{stage}': "
            f"{used_gb:.1f} GB > {max_gb:.1f} GB"
        )
    if used_gb > warn_gb:
        logger.warning(
            "High RAM usage during '%s': %.1f GB > warn %.1f GB",
            stage,
            used_gb,
            warn_gb,
        )


def force_release(note: str = "") -> None:
    """Explicit garbage collection and malloc_trim for memory cleanup.

    Args:
        note: Optional description logged after cleanup.
    """
    gc.collect()
    with suppress(Exception):
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    if note:
        log_status(f"Memory cleanup complete: {note}")


def clear_cuda_cache() -> None:
    """Clear CUDA cache if torch is available and a GPU is present."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            with suppress(Exception):
                torch.cuda.ipc_collect()
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# Process-tree memory watchdog
# --------------------------------------------------------------------------- #

#: Exit status of a task stopped by its watchdog. It differs from 137 (a
#: SIGKILL from the kernel or the scheduler), so the workflow retries an
#: external kill with more memory but not a task that hit its own cap.
RSS_LIMIT_EXIT_CODE: Final = 99

_GIB: Final = float(1 << 30)


def _anonymous_rss_bytes(process: psutil.Process) -> int:
    """Anonymous resident memory of one process (``RssAnon``, else USS).

    File-backed pages (a memory-mapped mask touched by every polygonisation
    worker) are shared page cache, so they are not counted.
    """
    try:
        with open(f"/proc/{process.pid}/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return int(process.memory_full_info().uss)


def process_tree_anonymous_rss_bytes(pid: int | None = None) -> int:
    """Anonymous resident memory of a process and all its descendants.

    Args:
        pid: Root process; this process when ``None``.

    Returns:
        Bytes summed over the tree; processes that exit while being read are
        skipped.
    """
    root = psutil.Process(pid)
    total = 0
    for process in (root, *root.children(recursive=True)):
        try:
            total += _anonymous_rss_bytes(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            continue
    return total


def stop_process_tree(
    used_gb: float, limit_gb: float, cleanup: Sequence[Path], *, name: str = "task"
) -> None:
    """Kill the process tree of a task over its memory cap and exit.

    The exit skips ``finally`` blocks, so ``cleanup`` is removed here, after
    the child processes are gone.

    Args:
        used_gb: Measured anonymous RSS in GiB.
        limit_gb: The cap in GiB.
        cleanup: Paths to remove before exiting.
        name: Short task name for the log line.
    """
    from merxen.path_utils import remove_path

    logger.error(
        "[%s] Stopped by the memory watchdog: %.1f GB anonymous RSS > %.1f GB cap. "
        "Raise the cap and the task memory.",
        name,
        used_gb,
        limit_gb,
    )
    for handler in logging.getLogger().handlers:
        handler.flush()
    for child in psutil.Process().children(recursive=True):
        try:
            child.send_signal(signal.SIGKILL)
        except psutil.Error:
            continue
    for path in cleanup:
        try:
            remove_path(path)
        except OSError:
            continue
    os._exit(RSS_LIMIT_EXIT_CODE)


class RssWatchdog:
    """Stop a task when its process tree exceeds an anonymous-memory cap.

    A background thread polls the anonymous RSS of this process and its
    descendants (``RssAnon`` from ``/proc``, else USS; never ``rss``, which
    counts shared page cache once per process). The local executor only
    schedules memory and does not cap it, so the watchdog is the cap that
    keeps a runaway task from the kernel OOM killer, which may pick another
    session's process. With no cap it only records the peak. (Moved from the
    ATERA builder, which keeps a subclass with its own stop message.)

    Use it as a context manager, or call :meth:`start` and :meth:`stop`.

    Attributes:
        cleanup: Paths the default action removes before it exits.
    """

    def __init__(
        self,
        limit_gb: float | None,
        *,
        interval_s: float = 2.0,
        on_exceed: Callable[[float, float], None] | None = None,
        measure: Callable[[], int] = process_tree_anonymous_rss_bytes,
        name: str = "task",
    ) -> None:
        """Prepare a watchdog; nothing runs until :meth:`start`.

        Args:
            limit_gb: Cap in GiB; ``None`` only records the peak.
            interval_s: Seconds between two measurements.
            on_exceed: Called with ``(used_gb, limit_gb)`` when the cap is
                exceeded; the default logs, kills the child processes,
                removes :attr:`cleanup` and exits with
                :data:`RSS_LIMIT_EXIT_CODE`.
            measure: Returns the bytes to compare with the cap.
            name: Short task name used in the thread name and log lines.
        """
        self.limit_gb = limit_gb
        self.interval_s = interval_s
        self.cleanup: list[Path] = []
        self.name = name
        self._on_exceed = on_exceed or (
            lambda used, limit: stop_process_tree(used, limit, self.cleanup, name=name)
        )
        self._measure = measure
        self._failed = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_gb = 0.0

    def check(self) -> float:
        """Measure once, update the peak and act on the cap.

        Returns:
            The measured anonymous RSS in GiB.
        """
        used_gb = self._measure() / _GIB
        self.peak_gb = max(self.peak_gb, used_gb)
        if self.limit_gb is not None and used_gb > self.limit_gb:
            self._on_exceed(used_gb, self.limit_gb)
        return used_gb

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            try:
                self.check()
            except Exception:
                # A failed reading must not end the thread and lift the cap.
                if not self._failed:
                    logger.exception("[%s] Memory watchdog reading failed", self.name)
                    self._failed = True

    def start(self) -> RssWatchdog:
        """Start polling in a daemon thread.

        Returns:
            This watchdog.
        """
        if self._thread is None:
            self.check()
            self._thread = threading.Thread(
                target=self._run, name=f"{self.name.lower()}-rss-watchdog", daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        """Stop polling (idempotent) after a last measurement."""
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join()
            self._thread = None
            self.check()

    def __enter__(self) -> RssWatchdog:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()
