"""Process-tree memory sampling for PREP steps and panel simulations (plan §8.7).

cell_type_mapper parallelises over worker processes, so the peak RSS that GNU
``time`` reports (the largest single process, as the evidence logs measured
it) understates what a step holds at once. ``ProcessTreeSampler`` samples a
process and all its descendants in a background thread and records the peak
of their summed RSS and of their summed PSS (proportional set size: shared
pages split between the processes that map them, so copy-on-write pages of
forked workers are not counted twice). The PSS peak is what a memory reserve
must cover; the summed RSS peak is an upper bound.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Final

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_S: Final = 2.0
_GIB: Final = 1024**3


@dataclass(frozen=True)
class TreeMemoryPeak:
    """Peak memory of a process tree.

    Attributes:
        peak_rss_gb: Peak of the summed RSS (GiB).
        peak_pss_gb: Peak of the summed PSS (GiB), ``None`` when unreadable.
        peak_processes: Processes in the tree at the RSS peak.
        n_samples: Samples taken.
        interval_s: Sampling interval.
    """

    peak_rss_gb: float
    peak_pss_gb: float | None
    peak_processes: int
    n_samples: int
    interval_s: float

    def to_json(self) -> dict[str, Any]:
        """Return the peak as a JSON object."""
        return {
            "peak_tree_rss_gb": round(self.peak_rss_gb, 3),
            "peak_tree_pss_gb": None
            if self.peak_pss_gb is None
            else round(self.peak_pss_gb, 3),
            "peak_tree_processes": self.peak_processes,
            "tree_samples": self.n_samples,
            "tree_sample_interval_s": self.interval_s,
        }


class ProcessTreeSampler:
    """Sample the summed RSS / PSS of a process and its descendants.

    Use as a context manager around the work (``pid`` defaults to the current
    process, i.e. the work and every subprocess it starts)::

        with ProcessTreeSampler(pid) as sampler:
            ...
        peak = sampler.peak()

    Args:
        pid: Root process id; ``None`` for the current process.
        interval_s: Seconds between samples.
    """

    def __init__(
        self, pid: int | None = None, *, interval_s: float = DEFAULT_INTERVAL_S
    ) -> None:
        import os

        self.pid = os.getpid() if pid is None else int(pid)
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss = 0
        self._peak_pss: int | None = None
        self._peak_processes = 0
        self._samples = 0

    def sample(self) -> tuple[int, int | None, int]:
        """Take one sample: summed RSS, summed PSS (or ``None``), processes."""
        import psutil

        try:
            root = psutil.Process(self.pid)
            processes = [root, *root.children(recursive=True)]
        except psutil.Error:
            return 0, None, 0
        rss = 0
        pss: int | None = 0
        counted = 0
        for process in processes:
            try:
                info = process.memory_full_info()
            except (psutil.AccessDenied, AttributeError):
                try:
                    rss += int(process.memory_info().rss)
                    counted += 1
                except psutil.Error:
                    pass
                pss = None
                continue
            except psutil.Error:
                continue
            rss += int(info.rss)
            counted += 1
            if pss is not None:
                value = getattr(info, "pss", None)
                pss = None if value is None else pss + int(value)
        return rss, pss, counted

    def _record(self) -> None:
        rss, pss, counted = self.sample()
        self._samples += 1
        if rss > self._peak_rss:
            self._peak_rss = rss
            self._peak_processes = counted
        if pss is not None and (self._peak_pss is None or pss > self._peak_pss):
            self._peak_pss = pss

    def _run(self) -> None:
        while not self._stop.is_set():
            self._record()
            self._stop.wait(self.interval_s)

    def start(self) -> ProcessTreeSampler:
        """Start sampling in a daemon thread."""
        self._thread = threading.Thread(
            target=self._run, name=f"tree-memory-{self.pid}", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> TreeMemoryPeak:
        """Stop sampling and return the peak."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10 * self.interval_s + 5)
        return self.peak()

    def peak(self) -> TreeMemoryPeak:
        """Return the peak so far."""
        return TreeMemoryPeak(
            peak_rss_gb=self._peak_rss / _GIB,
            peak_pss_gb=None if self._peak_pss is None else self._peak_pss / _GIB,
            peak_processes=self._peak_processes,
            n_samples=self._samples,
            interval_s=self.interval_s,
        )

    def __enter__(self) -> ProcessTreeSampler:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()
