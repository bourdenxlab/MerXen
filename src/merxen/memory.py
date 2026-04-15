"""Memory monitoring, garbage collection, and CUDA cache management."""

from __future__ import annotations

import ctypes
import gc
import logging
import os
from contextlib import suppress

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
