"""Chunk-major driver: bead-corrected max projection of a whole mosaic.

The per-plane (stain-major) loop of the builder cannot host the bead model: the
model needs the planes 0, 1 and 2 and the projected planes of a window at the
same time. The corrected stains are therefore projected chunk by chunk:

* One task is one core of ``core_px`` x ``core_px`` px (default 4096, the zarr
  store chunk) read with a ``halo_px`` halo (default 256 px, larger than the
  support of every operation of the model and of the fill).
* Only the bead-corrected stains go through the driver; every other stain keeps
  the builder's own plain max projection.
* The window correction is a :class:`WindowCorrector` chosen by the resolved
  device. On ``"cpu"`` tasks run in a :class:`concurrent.futures.ProcessPoolExecutor`
  with the ``"spawn"`` start method and one numeric thread per worker, and
  workers return the uint16 core arrays. On ``"cuda"`` one consumer in the parent
  process runs the windows one after the other on the GPU while a small thread
  pool reads the next windows and one thread writes the finished ones; the host
  GPU lock is held once, for the whole window loop.
* Only the parent writes the store, in task order, so no two processes write
  the same store chunk.
* The result does not depend on the number of workers or reader threads, or on
  the order in which tasks finish (tested). Window order, store layout, per-chunk
  statistics and QC records are the same for both devices.

Scripts that call :func:`run_bead_removal` must keep the call under
``if __name__ == "__main__":``: the ``"spawn"`` start method re-imports the
calling module in every worker.
"""

from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import json
import logging
import math
import multiprocessing as mp
import os
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from merxen.gpu_lock import gpu_lock_path_from_env, gpu_process_lock
from merxen.io.bead_removal.apply import SPECTRAL_STAIN, correct_window
from merxen.io.bead_removal.backend import (
    DEVICE_REMEDY,
    backend_record,
    load_gpu_backend,
)
from merxen.io.bead_removal.calibrate import BeadCalibration
from merxen.io.bead_removal.fill import FillParams, fill_params_sha256, support_px
from merxen.io.bead_removal.model import beads_table
from merxen.io.bead_removal.params import (
    DEFAULT_PARAMS,
    ENVELOPE_BLOCK_PX,
    NZ_MAX,
    PsfSubtractParams,
    params_sha256,
)
from merxen.io.bead_removal.qc import OVERVIEW_FACTOR, chunk_qc_probe
from merxen.io.bead_removal.reader import MosaicPlaneReader

logger = logging.getLogger(__name__)

DEFAULT_CORE_PX = 4096
DEFAULT_HALO_PX = 256
# Worker memory added when the fill is on (measured: +1.7 GB peak per worker).
FILL_EXTRA_RSS_GB = 1.8
# Fraction of the available memory the workers may use, and the memory kept for
# the parent process.
MEMORY_USE_FRACTION = 0.8
PARENT_RESERVE_GB = 8.0
# GPU consumer: threads that read the next windows, the number of windows that
# may be read or waiting ahead of the one on the GPU (about 1.2 GB of float32
# planes each for a 4096 px core), and the finished chunks waiting for the
# writer thread.
GPU_READER_THREADS = 3
GPU_QUEUE_WINDOWS = 2
GPU_WRITE_QUEUE = 2
_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_PLANE_FILE_RE = re.compile(r"^mosaic_.+_z(?P<z>\d+)\.tif$")

# (y0, y1, x0, x1) in absolute mosaic px, half-open.
Core = tuple[int, int, int, int]


@dataclass(frozen=True)
class ChunkTask:
    """Everything a worker needs to process one chunk.

    The task is small and picklable; the arrays are read in the worker.

    Attributes:
        index: Position of the chunk in the (row-major) task list, also its
            deterministic write order.
        core: (y0, y1, x0, x1) of the core in absolute mosaic px.
        halo_px: Halo read around the core.
        images_dir: Folder with the mosaic planes (a string: it crosses the
            process boundary).
        stains: Stains of the output channels.
        z_layers: Planes that are projected.
        calibration_dir: Folder of the saved calibration, which each worker loads
            once (None: the caller supplies the object).
        params: Algorithm parameters.
        fill: Fill configuration, or None (off).
    """

    index: int
    core: Core
    halo_px: int
    images_dir: str
    stains: tuple[str, ...]
    z_layers: tuple[int, ...]
    calibration_dir: str | None
    params: PsfSubtractParams = DEFAULT_PARAMS
    fill: FillParams | None = None


@dataclass
class ChunkResult:
    """Output of one chunk.

    Attributes:
        index: Task index.
        core: Core of the chunk (absolute mosaic px).
        projections: Stain -> uint16 [core_h, core_w] projected core.
        stats: Counts, timings and per-plane change statistics (JSON-able).
        beads: float64 [n, 8] bead table (see ``model.beads_table``) restricted
            to the beads whose centre lies in the core.
        qc: Arrays of the QC probe that are collected over the whole mosaic
            (``"change"``: stain -> 1/16 block maximum of raw - corrected); the
            sink consumes them. The probe's small statistics are in
            ``stats["ring_score"]``.
    """

    index: int
    core: Core
    projections: dict[str, np.ndarray]
    stats: dict[str, Any]
    beads: np.ndarray
    qc: dict[str, Any] = field(default_factory=dict)


@dataclass
class BeadRemovalRunResult:
    """Summary of a driver run (also written to ``work_dir/run_summary.json``).

    ``read_s``, ``correct_s`` and ``project_s`` are sums over the chunks of the
    time spent in each step.

    Attributes:
        n_chunks: Number of chunks.
        n_workers: Worker processes that were used.
        wall_s: Wall time of the run.
        worker_cpu_s: Summed CPU time of the chunks.
        worker_wall_s: Summed wall time of the chunks.
        read_s: Summed read time.
        correct_s: Summed correction time.
        project_s: Summed projection (rounding) time.
        parent_write_s: Time the parent spent writing the store.
        n_beads: Beads whose centre lies in a core.
        peak_tree_rss_gb: Peak summed RSS of the parent and its workers.
        peak_tree_anon_gb: Peak summed anonymous memory of the same.
        peak_worker_rss_gb: Peak RSS of one worker.
        peak_worker_anon_gb: Peak anonymous memory of one worker.
        work_dir: Folder with the beads table, chunk statistics and summary.
        device: Backend device that ran the correction.
        backend: Backend record (device and update order; for the GPU also the
            library versions and the GPU name).
        gpu: GPU run record, ``None`` on the CPU: ``lock_wait_s`` (time spent
            waiting for the host GPU lock, 0 without a lock), ``locked`` (a lock
            file was configured), ``held_s`` (time the lock was held), ``gpu_s``
            (summed correction seconds), ``gpu_s_per_window`` (median, max),
            ``reader_threads`` and ``queue_windows``.
        chunk_stats: Per-chunk statistics (not part of ``to_dict``).
    """

    n_chunks: int
    n_workers: int
    wall_s: float
    worker_cpu_s: float
    worker_wall_s: float
    read_s: float
    correct_s: float
    project_s: float
    parent_write_s: float
    n_beads: int
    peak_tree_rss_gb: float
    peak_tree_anon_gb: float
    peak_worker_rss_gb: float
    peak_worker_anon_gb: float
    work_dir: str
    device: str = "cpu"
    backend: dict[str, Any] = field(default_factory=dict)
    gpu: dict[str, Any] | None = None
    chunk_stats: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-able summary without the per-chunk list.

        The ``gpu`` entry is present only for a GPU run.
        """
        summary = dataclasses.asdict(self)
        summary.pop("chunk_stats")
        if summary["gpu"] is None:
            summary.pop("gpu")
        return summary


# ---------------------------------------------------------------------------
# planning
def _edges(n: int, core_px: int, shift_px: int) -> list[tuple[int, int]]:
    """Core intervals along one axis: cuts at shift + k x core_px in [0, n)."""
    cuts = [0]
    shift = shift_px % core_px
    cut = shift if shift > 0 else core_px
    while cut < n:
        cuts.append(cut)
        cut += core_px
    cuts.append(n)
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


def plan_chunks(
    region: Core, *, core_px: int = DEFAULT_CORE_PX, shift_px: int = 0
) -> list[Core]:
    """List the cores (absolute mosaic px) that tile ``region``, row-major.

    Args:
        region: (y0, y1, x0, x1) of the area to process (absolute mosaic px).
        core_px: Core edge length.
        shift_px: Offset of the grid from the region origin (0 = cores aligned to
            the region; the chunk-invariance test cuts the same region
            differently).

    Returns:
        Cores, y outer and x inner; edge cores are smaller.
    """
    y0, y1, x0, x1 = region
    ys = _edges(y1 - y0, core_px, shift_px)
    xs = _edges(x1 - x0, core_px, shift_px)
    return [(y0 + a, y0 + b, x0 + c, x0 + d) for (a, b) in ys for (c, d) in xs]


def window_of(core: Core, halo_px: int) -> Core:
    """Return the read window of a core: core plus halo on the 16 px grid.

    The model's lower-envelope blocks sit on the absolute 16 px grid, and a
    window that starts and ends on it needs no internal padding (which would
    turn real pixels at the far edge into "invalid" ones). The origin is snapped
    down and the end up. Negative origins and ends beyond the mosaic are fine:
    the reader returns zeros there, which is the invalid value.

    Args:
        core: (y0, y1, x0, x1) of the core.
        halo_px: Halo around the core.

    Returns:
        (y0, y1, x0, x1) of the window.
    """
    g = ENVELOPE_BLOCK_PX
    y0, y1, x0, x1 = core
    wy0 = ((y0 - halo_px) // g) * g
    wx0 = ((x0 - halo_px) // g) * g
    wy1 = -((-(y1 + halo_px)) // g) * g
    wx1 = -((-(x1 + halo_px)) // g) * g
    return (wy0, wy1, wx0, wx1)


def planes_needed(
    calibration: BeadCalibration, z_layers: Sequence[int]
) -> dict[str, list[int]]:
    """List the planes each corrected stain has to supply to ``correct_window``.

    Args:
        calibration: The dataset calibration.
        z_layers: Planes that are projected.

    Returns:
        Stain -> planes: the reference stain needs 0-2 and ``z_layers`` (and
        planes 2..zmax of the rescue's z-profile test when the calibration has
        one), the others 0-1 and ``z_layers`` (DAPI is read at 0-1 for the
        spectral test in any case).
    """
    zs = {int(z) for z in z_layers}
    need = {s: sorted({0, 1} | zs) for s in calibration.corrected_stains}
    need[calibration.reference_stain] = sorted(
        {0, 1, 2} | zs | set(calibration.thresholds.rescue_planes)
    )
    need[SPECTRAL_STAIN] = sorted(set(need.get(SPECTRAL_STAIN, [])) | {0, 1})
    return need


def count_plane_stacks(calibration: BeadCalibration, z_layers: Sequence[int]) -> int:
    """Count the float32 planes of a window that a worker holds at once.

    Args:
        calibration: The dataset calibration.
        z_layers: Planes that are projected.

    Returns:
        The number of (stain, plane) pairs of ``planes_needed`` (14 for the
        default z 1-6 with PolyT and DAPI; more with 7 planes or the rescue's
        extra reference planes on a short z range).
    """
    return sum(len(planes) for planes in planes_needed(calibration, z_layers).values())


def estimate_worker_rss_gb(
    window_px: tuple[int, int],
    n_plane_stacks: int = 14,
    overhead_gb: float = 3.5,
) -> float:
    """Estimate the peak RSS of one worker.

    Measured: a 4608 x 4608 window with PolyT 0-6 and DAPI 0-6 (14 float32
    planes) peaked at 9.9 GB, about half of it reclaimable memmap pages. The
    estimate is ``planes x 4 B x pixels x 2 + overhead``; measure the real value
    on one dense chunk before a large run.

    Args:
        window_px: Window (height, width) in px.
        n_plane_stacks: Float32 planes held at once.
        overhead_gb: Fixed overhead of the process.

    Returns:
        Estimated peak RSS in GB.
    """
    return n_plane_stacks * 4 * window_px[0] * window_px[1] * 2 / 1e9 + overhead_gb


# ---------------------------------------------------------------------------
# resources seen by this process (affinity and cgroup limits, not the host)
_CGROUP_ROOT = Path("/sys/fs/cgroup")
_PROC_CGROUP = Path("/proc/self/cgroup")
# cgroup v1 reports "no limit" as a huge number close to 2**63.
_CGROUP_V1_UNLIMITED = 2**60


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _read_int(path: Path) -> int | None:
    text = _read_text(path)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:  # "max"
        return None


def _cgroup_dirs(root: Path, proc_cgroup: Path, controller: str) -> list[Path]:
    """List the cgroup directories of this process, from its own up to the root."""
    text = _read_text(proc_cgroup)
    if not text:
        return []
    v2_rel: str | None = None
    v1_rel: str | None = None
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, rel = parts
        if controllers == "":
            v2_rel = rel
        elif controller in controllers.split(","):
            v1_rel = rel
    candidates: list[tuple[Path, str]] = []
    if v2_rel is not None:
        candidates.append((root, v2_rel))
    if v1_rel is not None:
        candidates.append((root / controller, v1_rel))
    dirs: list[Path] = []
    for base, rel in candidates:
        parts_rel = [p for p in rel.split("/") if p]
        for depth in range(len(parts_rel), -1, -1):
            dirs.append(base.joinpath(*parts_rel[:depth]))
    return dirs


def cgroup_memory_headroom_gb(
    root: Path = _CGROUP_ROOT, proc_cgroup: Path = _PROC_CGROUP
) -> float | None:
    """Return the memory (GB) this process's cgroup can still use.

    This is the tightest ``limit - usage`` up the hierarchy. cgroup v2
    (``memory.max`` / ``memory.current``) and v1 (``memory.limit_in_bytes`` /
    ``memory.usage_in_bytes``) are supported. Usage excludes the inactive file
    cache: it is reclaimable, and the memory-mapped planes the workers read
    would otherwise look like used memory.

    Args:
        root: cgroup file system mount (a parameter so that tests can build a
            fake tree).
        proc_cgroup: The process's ``/proc/<pid>/cgroup`` file.

    Returns:
        Headroom in GB, or None when no finite limit applies.
    """
    best: float | None = None
    for directory in _cgroup_dirs(root, proc_cgroup, "memory"):
        limit = _read_int(directory / "memory.max")
        usage = _read_int(directory / "memory.current")
        stat_names: tuple[str, ...] = ("inactive_file",)
        if limit is None:  # v1 names
            limit = _read_int(directory / "memory.limit_in_bytes")
            usage = _read_int(directory / "memory.usage_in_bytes")
            stat_names = ("total_inactive_file", "inactive_file")
            if limit is not None and limit >= _CGROUP_V1_UNLIMITED:
                limit = None
        if limit is None or usage is None:
            continue
        inactive = 0
        stat = _read_text(directory / "memory.stat")
        if stat:
            for line in stat.splitlines():
                key, _, val = line.partition(" ")
                if key in stat_names and val.strip().isdigit():
                    inactive = int(val)
                    break
        free = (limit - max(usage - inactive, 0)) / 1e9
        best = free if best is None else min(best, free)
    return best


def cgroup_cpu_limit(
    root: Path = _CGROUP_ROOT, proc_cgroup: Path = _PROC_CGROUP
) -> int | None:
    """Return the CPU limit of the cgroup in cores (rounded down, at least 1).

    Args:
        root: cgroup file system mount.
        proc_cgroup: The process's ``/proc/<pid>/cgroup`` file.

    Returns:
        Cores from ``cpu.max`` or ``cpu.cfs_quota_us``, or None for no limit.
    """
    best: float | None = None
    for directory in _cgroup_dirs(root, proc_cgroup, "cpu"):
        quota = period = None
        text = _read_text(directory / "cpu.max")
        if text:
            fields = text.split()
            if len(fields) == 2 and fields[0] != "max":
                quota, period = int(fields[0]), int(fields[1])
        else:
            q = _read_int(directory / "cpu.cfs_quota_us")
            p = _read_int(directory / "cpu.cfs_period_us")
            if q is not None and p and q > 0:
                quota, period = q, p
        if quota and period:
            cores = quota / period
            best = cores if best is None else min(best, cores)
    return None if best is None else max(1, math.floor(best))


def available_cpu_count() -> int:
    """Return the CPUs this process may use.

    The lowest of the affinity mask, the cgroup CPU quota and ``DASK_NUM_WORKERS``
    (when set) applies: ``os.cpu_count()`` would report the whole host inside a
    container or a scheduler allocation.

    Returns:
        Usable CPUs, at least 1.
    """
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):  # not Linux
        n = os.cpu_count() or 1
    quota = cgroup_cpu_limit()
    if quota is not None:
        n = min(n, quota)
    dask = os.environ.get("DASK_NUM_WORKERS", "").strip()
    if dask.isdigit() and int(dask) > 0:
        n = min(n, int(dask))
    return max(1, n)


def read_mem_available_gb() -> float:
    """Return the memory (GB) this process tree can use.

    This is the lower of the cgroup headroom and the host's ``MemAvailable``.

    Returns:
        GB, or infinity when neither can be read.
    """
    host = float("inf")
    text = _read_text(Path("/proc/meminfo"))
    if text:
        for line in text.splitlines():
            if line.startswith("MemAvailable:"):
                host = int(line.split()[1]) / 1e6
                break
    headroom = cgroup_memory_headroom_gb()
    return host if headroom is None else min(host, headroom)


def resolve_worker_count(
    requested: int,
    *,
    memory_available_gb: float | None,
    rss_per_worker_gb: float,
    n_tasks: int | None = None,
    cpus: int | None = None,
) -> int:
    """Choose the number of workers.

    The count is min(requested, usable CPUs, memory-limited count, number of
    tasks), at least 1. The memory-limited count is
    ``floor((0.8 x available - 8 GB) / per-worker estimate)``: 20 % head-room and
    8 GB for the parent process.

    Args:
        requested: Requested number of workers.
        memory_available_gb: Memory the process tree may use (None: do not limit
            by memory).
        rss_per_worker_gb: Estimated peak RSS of one worker.
        n_tasks: Number of tasks (more workers would idle).
        cpus: Usable CPUs (default ``available_cpu_count()``: affinity, cgroup
            quota and ``DASK_NUM_WORKERS``).

    Returns:
        The worker count.
    """
    n = max(1, int(requested))
    n = min(n, cpus if cpus is not None else available_cpu_count())
    if (
        memory_available_gb is not None
        and math.isfinite(memory_available_gb)
        and rss_per_worker_gb > 0
    ):
        usable = MEMORY_USE_FRACTION * memory_available_gb - PARENT_RESERVE_GB
        fit = int(usable // rss_per_worker_gb)
        if fit < 1:
            logger.warning(
                "[BEADS] %.1f GB available is less than one worker (%.1f GB) "
                "needs; running one worker anyway",
                memory_available_gb,
                rss_per_worker_gb,
            )
        n = min(n, max(1, fit))
    if n_tasks is not None:
        n = min(n, max(1, n_tasks))
    return n


# ---------------------------------------------------------------------------
# backend hook: the window correction chosen by the resolved device
@dataclass(frozen=True)
class WindowCorrector:
    """A window correction callable and the device it runs on.

    Calling it is ``fn(stacks, origin_yx=, calibration=, z_layers=, params=,
    fill=, core=)`` with ``core`` the (y0, y1, x0, x1) of the output inside the
    window. It returns ``(projections, model, stats)``: ``projections`` maps
    each corrected stain to a float32 [core_h, core_w] array, the max over
    ``z_layers`` of the corrected planes on the core.

    Attributes:
        device: Device name (``"cpu"`` or ``"cuda"``).
        fn: The correction function.
    """

    device: str
    fn: Callable[..., tuple[dict[str, np.ndarray], Any, dict[str, Any]]]

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[dict[str, np.ndarray], Any, dict[str, Any]]:
        """Correct one window through ``fn``."""
        return self.fn(*args, **kwargs)


def _cpu_window_fn(
    stacks: Mapping[str, Any],
    *,
    origin_yx: tuple[int, int],
    calibration: BeadCalibration,
    z_layers: Sequence[int],
    params: PsfSubtractParams,
    fill: FillParams | None,
    core: Core,
) -> tuple[dict[str, np.ndarray], Any, dict[str, Any]]:
    """Run ``correct_window`` and take the max over ``z_layers`` on the core."""
    out, model, stats = correct_window(
        stacks,
        origin_yx=origin_yx,
        calibration=calibration,
        z_layers=list(z_layers),
        params=params,
        fill=fill,
        core=core,
    )
    y0, y1, x0, x1 = core
    return {s: out[s][:, y0:y1, x0:x1].max(axis=0) for s in out}, model, stats


def _cuda_window_fn(
    stacks: Mapping[str, Any],
    *,
    origin_yx: tuple[int, int],
    calibration: BeadCalibration,
    z_layers: Sequence[int],
    params: PsfSubtractParams,
    fill: FillParams | None,
    core: Core,
) -> tuple[dict[str, np.ndarray], Any, dict[str, Any]]:
    """Run the GPU window correction and crop its max projection to the core.

    The GPU function projects the whole window on the device (one transfer of
    one plane per stain instead of one per z layer); the CPU function takes the
    core only, so the core is cut out here.
    """
    if fill is not None and fill.enabled:
        raise ValueError(
            "the bead-removal fill has no GPU implementation; the device resolves "
            "to cpu when the fill is on"
        )
    gpu = load_gpu_backend()
    projections, model, stats = gpu.correct_window(
        stacks,
        origin_yx=origin_yx,
        calibration=calibration,
        z_layers=list(z_layers),
        params=params,
        core=core,
        output="maxproj",
    )
    y0, y1, x0, x1 = core
    return {s: a[y0:y1, x0:x1] for s, a in projections.items()}, model, stats


def _make_cuda_corrector() -> WindowCorrector:
    """Load the GPU backend and fix its numerics (no TF32, deterministic).

    Raises:
        BeadRemovalBackendError: The GPU backend cannot be imported.
    """
    gpu = load_gpu_backend()
    gpu.ops.configure(True)
    return WindowCorrector("cuda", _cuda_window_fn)


CPU_CORRECTOR = WindowCorrector("cpu", _cpu_window_fn)
_CORRECTORS: dict[str, WindowCorrector] = {"cpu": CPU_CORRECTOR}
_VALID_DEVICES = ("cpu", "cuda")


def get_window_corrector(device: str) -> WindowCorrector:
    """Return the window corrector of a resolved device.

    The CUDA corrector is created on first use: it imports the GPU backend
    (torch and triton), which the CPU path never does.

    Args:
        device: Resolved device, ``"cpu"`` or ``"cuda"``.

    Returns:
        The corrector.

    Raises:
        ValueError: The device is unknown.
        BeadRemovalBackendError: ``"cuda"`` and the GPU backend cannot be
            imported.
    """
    if device not in _VALID_DEVICES:
        raise ValueError(
            f"unknown bead-removal device {device!r}; expected one of "
            f"{list(_VALID_DEVICES)}"
        )
    if device not in _CORRECTORS:
        _CORRECTORS[device] = _make_cuda_corrector()
    return _CORRECTORS[device]


def describe_backend(device: str) -> dict[str, Any]:
    """Return the backend record of the run summary and the provenance.

    Args:
        device: Resolved device.

    Returns:
        The device and the order in which overlapping beads are updated; for
        the GPU also the torch, triton and CUDA versions and the GPU name.
    """
    return backend_record(device)


# ---------------------------------------------------------------------------
# worker side
_WORKER: dict[str, Any] = {}


def _set_single_thread_env() -> dict[str, str | None]:
    old = {k: os.environ.get(k) for k in _THREAD_ENV}
    for k in _THREAD_ENV:
        os.environ[k] = "1"
    return old


def _restore_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _proc_mem_gb(pid: int | str = "self") -> tuple[float, float]:
    """Return (VmRSS, RssAnon) of a process in GB, (0, 0) when unavailable."""
    rss = anon = 0.0
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) / 1e6
            elif line.startswith("RssAnon:"):
                anon = int(line.split()[1]) / 1e6
    except (OSError, ValueError):
        pass
    return rss, anon


def _init_worker(calibration_dir: str) -> None:
    """Pool initializer: single-threaded numerics, calibration loaded once."""
    _set_single_thread_env()
    _WORKER["calibration"] = BeadCalibration.load(calibration_dir)
    _WORKER["calibration_dir"] = calibration_dir
    _WORKER["readers"] = {}


def _calibration_for(task: ChunkTask) -> BeadCalibration:
    if task.calibration_dir is None:
        raise ValueError(
            "the task has no calibration_dir and no calibration object was supplied"
        )
    # In-process use or an initializer-less pool: load on first use.
    if _WORKER.get("calibration_dir") != task.calibration_dir:
        _WORKER["calibration"] = BeadCalibration.load(task.calibration_dir)
        _WORKER["calibration_dir"] = task.calibration_dir
        _WORKER["readers"] = {}
    calibration: BeadCalibration = _WORKER["calibration"]
    return calibration


def _reader_for(
    images_dir: str, stains: Sequence[str], planes: Sequence[int]
) -> MosaicPlaneReader:
    """Return the cached reader for (folder, stains, planes) of this worker."""
    key = (images_dir, tuple(stains), tuple(planes))
    readers = _WORKER.setdefault("readers", {})
    if key not in readers:
        readers[key] = MosaicPlaneReader(images_dir, stains, planes)
    reader: MosaicPlaneReader = readers[key]
    return reader


class _PeakSampler(threading.Thread):
    """Sample this process's RSS and RssAnon while a chunk runs.

    The kernel keeps no high-water mark for anonymous memory.
    """

    def __init__(self, period_s: float = 2.0) -> None:
        super().__init__(daemon=True)
        self.period_s = period_s
        self.peak_rss = 0.0
        self.peak_anon = 0.0
        self._stop_evt = threading.Event()

    def run(self) -> None:
        """Sample until ``finish`` is called."""
        while not self._stop_evt.is_set():
            rss, anon = _proc_mem_gb()
            self.peak_rss = max(self.peak_rss, rss)
            self.peak_anon = max(self.peak_anon, anon)
            self._stop_evt.wait(self.period_s)

    def finish(self) -> None:
        """Stop sampling and take a last sample."""
        self._stop_evt.set()
        self.join()
        rss, anon = _proc_mem_gb()
        self.peak_rss = max(self.peak_rss, rss)
        self.peak_anon = max(self.peak_anon, anon)


def _to_uint16(arr: np.ndarray) -> np.ndarray:
    """Round a float32 projection to uint16.

    Values are integer counts minus a non-negative float and never above raw.
    """
    rounded: np.ndarray = np.clip(np.rint(arr), 0, 65535).astype(np.uint16)
    return rounded


def _summarise_per_stain(
    per_stain: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for stain, d in per_stain.items():
        out[stain] = {
            "rho": float(d["rho"]),
            "frac_gt10": [float(v) for v in d["frac_gt10"]],
            "frac_gt100": [float(v) for v in d["frac_gt100"]],
            "removed_sum": [float(v) for v in d["removed_sum"]],
            "removed_p99": [float(v) for v in d["removed_p99"]],
            "max_brightening": float(d["max_brightening"]),
        }
    return out


def read_window_stacks(
    reader: MosaicPlaneReader,
    need: Mapping[str, Sequence[int]],
    window: Core,
) -> dict[str, dict[int, np.ndarray]]:
    """Read the planes a window needs as float32 ``{stain: {z: plane}}``.

    Args:
        reader: Plane reader.
        need: Stain -> planes to read.
        window: (y0, y1, x0, x1) of the window in absolute mosaic px; pixels
            outside the mosaic are zero.

    Returns:
        The planes.
    """
    wy0, wy1, wx0, wx1 = window
    return {
        stain: {int(z): reader.read(stain, int(z), wy0, wy1, wx0, wx1) for z in zs}
        for stain, zs in need.items()
    }


_MODEL_STAT_KEYS = (
    "status",
    "n_cand",
    "n_bead",
    "n_rescued",
    "n_pruned",
    "n_blob",
    "n_stretch_beads",
    "n_ring_evidence",
    "n_beads_final",
    "gamma_med",
    "ell_median",
    "ell_sd",
    "n_saturated_px",
    "time_detect_s",
    "time_z0_s",
    "time_stretch_s",
    "time_s",
    "time_fit_s",
    "time_total_s",
    "k_fit",
)


def process_chunk(
    task: ChunkTask,
    *,
    corrector: WindowCorrector | None = None,
    calibration: BeadCalibration | None = None,
    stacks: Mapping[str, Mapping[int, np.ndarray]] | None = None,
    t_read_s: float = 0.0,
) -> ChunkResult:
    """Project one chunk.

    This is a top-level function so that it can be pickled to a worker. It reads
    the window (unless the caller already did), runs ``correct_window`` through
    ``corrector``, takes the max over ``z_layers`` on the core and rounds to
    uint16. Every stain of the task is bead-corrected.

    Args:
        task: The chunk to process.
        corrector: Window correction to use (default: the numpy reference). On
            the CUDA corrector the chunk statistics gain ``gpu_s``.
        calibration: Calibration object (default: loaded once per process from
            ``task.calibration_dir``).
        stacks: Window planes that the caller already read, ``{stain: {z:
            float32 plane}}``; None reads them here.
        t_read_s: Read time to record when ``stacks`` was read by the caller.

    Returns:
        The projected cores, statistics and the beads whose centre lies in the
        core.
    """
    corrector = corrector if corrector is not None else CPU_CORRECTOR
    t_start = time.time()
    cpu0 = time.process_time()
    sampler = _PeakSampler()
    sampler.start()
    try:
        cal = calibration if calibration is not None else _calibration_for(task)
        zs = list(task.z_layers)
        cy0, cy1, cx0, cx1 = task.core
        wy0, wy1, wx0, wx1 = window_of(task.core, task.halo_px)
        need = planes_needed(cal, zs)
        stats: dict[str, Any] = {
            "index": task.index,
            "core": list(task.core),
            "window": [wy0, wy1, wx0, wx1],
            "pid": os.getpid(),
            "device": corrector.device,
        }
        t_read = t_read_s
        if stacks is None:
            reader = _reader_for(
                task.images_dir,
                sorted(need),
                sorted({z for planes in need.values() for z in planes}),
            )
            t0 = time.time()
            stacks = read_window_stacks(reader, need, (wy0, wy1, wx0, wx1))
            t_read += time.time() - t0
        fill = task.fill if task.fill is not None and task.fill.enabled else None
        t0 = time.time()
        proj, model, window_stats = corrector(
            stacks,
            origin_yx=(wy0, wx0),
            calibration=cal,
            z_layers=zs,
            params=task.params,
            fill=fill,
            core=(cy0 - wy0, cy1 - wy0, cx0 - wx0, cx1 - wx0),
        )
        t_correct = time.time() - t0
        t0 = time.time()
        projections = {s: _to_uint16(a) for s, a in proj.items()}
        del proj
        t_project = time.time() - t0
        beads: np.ndarray = np.zeros((0, 8))
        table = beads_table(model, (wy0, wx0))
        if len(table):
            inside = (
                (table[:, 0] >= cy0)
                & (table[:, 0] < cy1)
                & (table[:, 1] >= cx0)
                & (table[:, 1] < cx1)
            )
            beads = table[inside]
        # The QC probe needs the raw planes (raw projection) next to the
        # corrected one, so the stacks are released only after it.
        t0 = time.time()
        probe = chunk_qc_probe(
            stacks=stacks,
            window_origin=(wy0, wx0),
            core=task.core,
            z_layers=zs,
            projections=projections,
            beads=beads,
            calibration=cal,
            index=task.index,
        )
        del stacks
        stats["ring_score"] = probe["ring"]
        stats["t_qc_s"] = time.time() - t0
        qc_arrays: dict[str, Any] = {"change": probe["change"]}
        stats["model"] = {
            k: window_stats[k] for k in _MODEL_STAT_KEYS if k in window_stats
        }
        stats["per_stain"] = _summarise_per_stain(window_stats.get("per_stain", {}))
        if "fill" in window_stats:
            stats["fill"] = window_stats["fill"]
        stats["n_beads_in_core"] = int(len(beads))
        stats.update(t_read_s=t_read, t_correct_s=t_correct, t_project_s=t_project)
        if corrector.device == "cuda":
            # The correction call returns host arrays, so its wall time is the
            # time the GPU (and the transfers around it) were busy with this window.
            stats["gpu_s"] = t_correct
    finally:
        # Drop the memmaps after every chunk: the file pages touched by earlier
        # chunks would otherwise stay mapped and grow the worker's RSS to tens of
        # GB (reclaimable page cache, but it hides the real memory use and
        # inflates cgroup accounting). Re-opening a plane is cheap.
        for reader_ in _WORKER.get("readers", {}).values():
            reader_.close()
        sampler.finish()
    stats["wall_s"] = time.time() - t_start
    stats["cpu_s"] = time.process_time() - cpu0
    stats["peak_rss_gb"] = sampler.peak_rss
    stats["peak_anon_gb"] = sampler.peak_anon
    return ChunkResult(
        index=task.index,
        core=task.core,
        projections=projections,
        stats=stats,
        beads=beads,
        qc=qc_arrays,
    )


# ---------------------------------------------------------------------------
# parent side
class _TreeSampler(threading.Thread):
    """Sample the summed RSS of the parent and its live children (Linux only)."""

    def __init__(self, period_s: float = 2.0) -> None:
        super().__init__(daemon=True)
        self.period_s = period_s
        self.peak_rss = 0.0
        self.peak_anon = 0.0
        self._stop_evt = threading.Event()

    def sample(self) -> None:
        """Take one sample of the process tree."""
        rss, anon = _proc_mem_gb()
        for child in mp.active_children():
            if child.pid is None:
                continue
            child_rss, child_anon = _proc_mem_gb(child.pid)
            rss += child_rss
            anon += child_anon
        self.peak_rss = max(self.peak_rss, rss)
        self.peak_anon = max(self.peak_anon, anon)

    def run(self) -> None:
        """Sample until ``finish`` is called."""
        while not self._stop_evt.is_set():
            self.sample()
            self._stop_evt.wait(self.period_s)

    def finish(self) -> None:
        """Stop sampling."""
        self._stop_evt.set()
        self.join()


def _fmt_eta(seconds: float) -> str:
    hours, rem = divmod(int(max(seconds, 0)), 3600)
    return f"{hours:d}h{rem // 60:02d}m"


class _ChunkSink:
    """Write finished chunks into the store (in task order) and collect records.

    Besides the projections the sink assembles the 1/16 change overview of the
    whole region (largest raw - corrected per block) from the chunks' QC probes.
    """

    def __init__(
        self,
        store: Any,
        channels: Mapping[str, int],
        origin_yx: tuple[int, int],
        n_total: int,
        t_run: float,
        progress_every: int,
        region_shape: tuple[int, int] = (0, 0),
    ) -> None:
        self.store = store
        factor = OVERVIEW_FACTOR
        self.change: dict[str, np.ndarray] = {
            stain: np.zeros(
                (-(-region_shape[0] // factor), -(-region_shape[1] // factor)),
                np.float32,
            )
            for stain in channels
        }
        self.channels = dict(channels)
        self.ry, self.rx = origin_yx
        self.n_total = n_total
        self.t_run = t_run
        self.progress_every = max(1, progress_every)
        self.t_write = 0.0
        self.stats: list[dict[str, Any]] = []
        self.beads: list[np.ndarray] = []

    def write(self, result: ChunkResult) -> None:
        """Store one chunk's projections and keep its stats and beads."""
        t0 = time.time()
        y0, y1, x0, x1 = result.core
        for stain, arr in result.projections.items():
            self.store[
                self.channels[stain],
                y0 - self.ry : y1 - self.ry,
                x0 - self.rx : x1 - self.rx,
            ] = arr
        dt = time.time() - t0
        self.t_write += dt
        result.stats["t_write_s"] = dt
        factor = OVERVIEW_FACTOR
        for stain, blocks in result.qc.get("change", {}).items():
            by, bx = (y0 - self.ry) // factor, (x0 - self.rx) // factor
            target = self.change[stain]
            view = target[by : by + blocks.shape[0], bx : bx + blocks.shape[1]]
            view[...] = blocks[: view.shape[0], : view.shape[1]]
        # The arrays are not needed again; free them with the result.
        result.qc = {}
        self.stats.append(result.stats)
        self.beads.append(result.beads)
        done = len(self.stats)
        if done % self.progress_every == 0 or done == self.n_total:
            elapsed = time.time() - self.t_run
            eta = elapsed * (self.n_total - done) / max(done, 1)
            logger.info(
                "[BEADS] %d/%d chunks done, elapsed %s, ETA %s, last chunk %.0f s",
                done,
                self.n_total,
                _fmt_eta(elapsed),
                _fmt_eta(eta),
                result.stats["wall_s"],
            )


def _check_inputs(
    images_dir: str | Path,
    stains: Sequence[str],
    z_layers: Sequence[int],
    calibration: BeadCalibration,
    params: PsfSubtractParams,
) -> tuple[list[int], tuple[int, int]]:
    """Validate the run inputs and return (z layers, mosaic shape).

    Raises:
        ValueError: Invalid ``z_layers`` or stains, a parameter-hash mismatch, a
            calibration without the noise level of a plane that is fitted, or a
            non-uint16 mosaic.
        FileNotFoundError: A plane the ``z_layers`` need does not exist.
    """
    zs = [int(z) for z in z_layers]
    if not zs:
        raise ValueError("z_layers is empty")
    if min(zs) < 1:
        raise ValueError(
            "bead removal needs z_layers to start at plane 1 or later (plane 0 is "
            "the bead plane and is read for detection only); got "
            f"{sorted(zs)}. Use for example 1-6."
        )
    if max(zs) > NZ_MAX - 1:
        raise ValueError(
            f"bead removal supports planes up to {NZ_MAX - 1}; got {max(zs)}"
        )
    if calibration.params_sha256 != params_sha256(params):
        raise ValueError(
            "the calibration was made with different bead-removal parameters "
            "(params_sha256 mismatch); recalibrate or pass the parameters it was "
            "made with"
        )
    # The model fits the reference planes 0-2 and every plane of z_layers, and
    # each needs the calibration's noise level; a plane it lacks would fail in a
    # worker (a KeyError) after the pool has started.
    noise_planes = sorted({0, 1, 2} | set(zs))
    uncovered = [k for k in noise_planes if k not in calibration.noise]
    if uncovered:
        raise ValueError(
            f"the calibration has no noise level for plane(s) {uncovered} (it "
            f"covers planes {sorted(calibration.noise)}); recalibrate with these "
            f"z_layers {zs}"
        )
    # The stains must be exactly the corrected ones: anything else is projected
    # by the builder's plain loop, and a corrected stain left out would be
    # silently uncorrected.
    corrected = list(calibration.corrected_stains)
    for stain in [*corrected, calibration.reference_stain, SPECTRAL_STAIN]:
        if stain not in stains:
            raise ValueError(
                f"stain {stain!r} is needed by the bead calibration but is not "
                f"among the output stains {list(stains)}"
            )
    for stain in stains:
        if stain not in corrected:
            raise ValueError(
                f"stain {stain!r} is not bead-corrected ({corrected}); project it "
                "with the builder's plain max projection instead of the "
                "bead-removal driver"
            )
    folder = Path(images_dir)
    missing = sorted(
        f"mosaic_{s}_z{z}.tif"
        for s, planes in planes_needed(calibration, zs).items()
        for z in planes
        if not (folder / f"mosaic_{s}_z{z}.tif").exists()
    )
    if missing:
        have = sorted(
            {
                int(m.group("z"))
                for f in folder.iterdir()
                if (m := _PLANE_FILE_RE.match(f.name))
            }
        )
        raise FileNotFoundError(
            f"{len(missing)} mosaic plane file(s) needed by z_layers {zs} are "
            f"missing (e.g. {missing[0]}); planes found in the folder: {have}"
        )
    import tifffile

    reference = folder / f"mosaic_{calibration.reference_stain}_z{zs[0]}.tif"
    with tifffile.TiffFile(reference) as tf:
        page = tf.pages[0]
        if np.dtype(page.dtype) != np.uint16:
            raise ValueError(
                f"mosaic planes are {page.dtype}; the driver writes uint16 projections"
            )
        shape = (int(page.shape[0]), int(page.shape[1]))
    return zs, shape


def fill_support_px(fill: FillParams) -> int:
    """Return the distance (px) over which a fill result depends on other pixels.

    Args:
        fill: Fill parameters.

    Returns:
        Total support of the fill (fine, cover, reach and coarse levels).
    """
    return int(support_px(fill)["total"])


def check_fill_support(fill: FillParams | None, halo_px: int) -> None:
    """Refuse a fill whose support does not fit in the halo.

    Chunk invariance would be lost otherwise.

    Args:
        fill: Fill parameters, or None.
        halo_px: Halo around each core.

    Raises:
        ValueError: ``support_px(fill) > halo_px - 2``.
    """
    if fill is None or not fill.enabled:
        return
    support = fill_support_px(fill)
    if support > halo_px - 2:
        raise ValueError(
            f"the fill's support ({support} px) exceeds halo_px - 2 = "
            f"{halo_px - 2}: results would depend on the chunk grid; lower the "
            "fill's reach or raise halo_px"
        )


def _run_pool(
    tasks: list[ChunkTask],
    sink: _ChunkSink,
    *,
    n_workers: int,
    calibration_dir: str,
    max_tasks_per_child: int | None,
) -> None:
    """Run the tasks in a process pool and write finished chunks in task order."""
    results: dict[int, ChunkResult] = {}
    next_pos = 0
    executor = cf.ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(calibration_dir,),
        max_tasks_per_child=max_tasks_per_child,
    )
    try:
        # All tasks are submitted at once. A result that finishes ahead of a slow
        # earlier chunk waits in ``results`` (about 64 MB each) until the prefix is
        # complete: a few hundred MB to about 1 GB with five workers and one chunk
        # three times slower than the rest, inside the PARENT_RESERVE_GB share.
        futures = {executor.submit(process_chunk, t): t for t in tasks}
        for future in cf.as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.error(
                    "[BEADS] chunk %d (core %s) failed: %r", task.index, task.core, exc
                )
                for pending in futures:
                    pending.cancel()
                # A worker killed by the kernel (out of memory) breaks the pool;
                # any other error is the chunk's own and no hint applies.
                hint = (
                    " (fewer workers if this was out of memory)"
                    if isinstance(exc, BrokenProcessPool)
                    else ""
                )
                raise RuntimeError(
                    f"bead removal failed on chunk {task.index} (core "
                    f"{task.core}): {exc!r}{hint}"
                ) from exc
            results[result.index] = result
            # Deterministic write order: write the finished prefix of the list.
            while next_pos < len(tasks) and tasks[next_pos].index in results:
                sink.write(results.pop(tasks[next_pos].index))
                next_pos += 1
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _read_window_job(
    task: ChunkTask, need: Mapping[str, Sequence[int]]
) -> tuple[dict[str, dict[int, np.ndarray]], float]:
    """Read the planes of one task's window (a reader thread).

    The reader and its memory maps live only for this call: the arrays it
    returns are copies, so the file pages are not kept resident.

    Returns:
        The float32 planes and the read time in seconds.
    """
    planes = sorted({z for zs in need.values() for z in zs})
    reader = MosaicPlaneReader(task.images_dir, sorted(need), planes)
    try:
        t0 = time.time()
        stacks = read_window_stacks(reader, need, window_of(task.core, task.halo_px))
        return stacks, time.time() - t0
    finally:
        reader.close()


def _run_gpu_consumer(
    tasks: list[ChunkTask],
    sink: _ChunkSink,
    *,
    corrector: WindowCorrector,
    calibration: BeadCalibration,
    lock_path: Path | None,
) -> dict[str, Any]:
    """Correct the windows on the GPU, one after the other, in task order.

    The host GPU lock is taken once, here, for the whole window loop (and not
    before: the calibration and the earlier part of the build do not hold the
    GPU). Reader threads read up to ``GPU_QUEUE_WINDOWS`` windows ahead of the
    one on the GPU, starting before the lock is awaited so that the GPU does not
    idle during the first read, and one writer thread stores the finished
    chunks, so reading and writing overlap the GPU work. The consumer takes the
    windows in task order and the writer stores them in that order, so the
    result does not depend on the number of reader threads.

    Args:
        tasks: The chunks, in task order.
        sink: Receives the finished chunks (called from the writer thread only).
        corrector: The CUDA window corrector.
        calibration: The dataset calibration.
        lock_path: Host GPU lock file, or ``None`` to run without a lock.

    Returns:
        The GPU run record (see ``BeadRemovalRunResult.gpu``).

    Raises:
        RuntimeError: A chunk failed to read, correct or write (the message
            names it); the lock is released on the way out.
    """
    need = planes_needed(calibration, tasks[0].z_layers) if tasks else {}
    read_ahead = max(1, int(GPU_QUEUE_WINDOWS))
    threads = max(1, int(GPU_READER_THREADS))
    t_wait = time.monotonic()
    readers = cf.ThreadPoolExecutor(max_workers=threads, thread_name_prefix="bead-read")
    writer = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="bead-write")
    reading: deque[tuple[ChunkTask, cf.Future[Any]]] = deque()
    writing: deque[tuple[ChunkTask, cf.Future[None]]] = deque()
    submitted = 0
    gpu_seconds: list[float] = []
    lock_wait_s = held_s = 0.0

    def _read_ahead() -> None:
        nonlocal submitted
        while submitted < len(tasks) and len(reading) < read_ahead:
            ahead = tasks[submitted]
            reading.append((ahead, readers.submit(_read_window_job, ahead, need)))
            submitted += 1

    def _fail(task: ChunkTask, step: str, exc: BaseException) -> RuntimeError:
        logger.error(
            "[BEADS] chunk %d (core %s) failed while %s: %r",
            task.index,
            task.core,
            step,
            exc,
            exc_info=exc,
        )
        # The traceback keeps the failing window's frames, and with them its GPU
        # tensors, alive until the process ends: drop it (it is logged above) so
        # the lock release can hand the GPU memory back before the next holder.
        exc.__traceback__ = None
        hint = (
            f" {DEVICE_REMEDY} if the GPU ran out of memory or a kernel failed."
            if step == "correcting it"
            else ""
        )
        return RuntimeError(
            f"bead removal failed on chunk {task.index} (core {task.core}) "
            f"while {step}: {exc!r}.{hint}"
        )

    def _drain_writes(limit: int) -> None:
        while len(writing) > limit:
            done_task, done = writing.popleft()
            try:
                done.result()
            except Exception as exc:
                raise _fail(done_task, "writing the store", exc) from exc

    try:
        # The first windows are read while the lock is awaited, so the GPU does
        # not sit idle during the first read once the lock is granted. The
        # host memory bound is the same as in the loop (GPU_QUEUE_WINDOWS).
        _read_ahead()
        with gpu_process_lock(lock_path):
            t_held = time.monotonic()
            lock_wait_s = t_held - t_wait
            while reading:
                task, pending = reading.popleft()
                try:
                    stacks, t_read = pending.result()
                except Exception as exc:
                    raise _fail(task, "reading its window", exc) from exc
                del pending
                # The next windows are read while this one is on the GPU.
                _read_ahead()
                try:
                    result = process_chunk(
                        task,
                        corrector=corrector,
                        calibration=calibration,
                        stacks=stacks,
                        t_read_s=t_read,
                    )
                except Exception as exc:
                    raise _fail(task, "correcting it", exc) from exc
                finally:
                    del stacks
                gpu_seconds.append(float(result.stats["gpu_s"]))
                writing.append((task, writer.submit(sink.write, result)))
                del result
                _drain_writes(GPU_WRITE_QUEUE)
            _drain_writes(0)
            held_s = time.monotonic() - t_held
    finally:
        for _, pending_read in reading:
            pending_read.cancel()
        readers.shutdown(wait=True, cancel_futures=True)
        writer.shutdown(wait=True, cancel_futures=True)
    return _gpu_run_record(
        locked=lock_path is not None,
        lock_wait_s=lock_wait_s,
        held_s=held_s,
        gpu_seconds=gpu_seconds,
    )


def _gpu_run_record(
    *,
    locked: bool,
    lock_wait_s: float,
    held_s: float,
    gpu_seconds: Sequence[float],
) -> dict[str, Any]:
    """Build the ``gpu`` entry of the run summary."""
    per_window = np.asarray(gpu_seconds, float)
    return {
        "lock_wait_s": float(lock_wait_s),
        "locked": bool(locked),
        "held_s": float(held_s),
        "gpu_s": float(per_window.sum()),
        "gpu_s_per_window": {
            "median": float(np.median(per_window)) if len(per_window) else 0.0,
            "max": float(per_window.max()) if len(per_window) else 0.0,
        },
        "reader_threads": max(1, int(GPU_READER_THREADS)),
        "queue_windows": max(1, int(GPU_QUEUE_WINDOWS)),
    }


def run_bead_removal(
    *,
    images_dir: str | Path,
    stains: Sequence[str],
    z_layers: Sequence[int],
    store: Any,
    calibration: BeadCalibration,
    calibration_dir: str | Path | None,
    work_dir: str | Path,
    params: PsfSubtractParams = DEFAULT_PARAMS,
    fill: FillParams | None = None,
    store_channels: Mapping[str, int] | None = None,
    region: Core | None = None,
    core_px: int = DEFAULT_CORE_PX,
    halo_px: int = DEFAULT_HALO_PX,
    shift_px: int = 0,
    max_workers: int = 5,
    memory_available_gb: float | None = None,
    rss_per_worker_gb: float | None = None,
    max_tasks_per_child: int | None = None,
    progress_every: int = 10,
    device: str = "cpu",
) -> BeadRemovalRunResult:
    """Write the bead-corrected max projection of ``stains`` into ``store``.

    Args:
        images_dir: Folder with the ``mosaic_<stain>_z<N>.tif`` planes.
        stains: The stains to project. They must be exactly the corrected stains
            of the calibration (any other stain is projected by the builder's
            plain loop, not here).
        z_layers: Planes to project; all >= 1 and <= 7.
        store: zarr array [channels, H', W'] of dtype uint16 (H' x W' = region
            size; the full mosaic by default).
        calibration: The dataset calibration.
        calibration_dir: Folder with the saved calibration, which the workers load
            themselves.
        work_dir: Folder for the beads table, the per-chunk statistics and the run
            summary.
        params: Algorithm parameters (the calibration must have been made with the
            same ones).
        fill: Optional fill configuration; the fill runs only when
            ``fill.enabled``.
        store_channels: Stain -> channel index in ``store`` (default: the position
            in ``stains``, i.e. a store with one channel per stain).
        region: (y0, y1, x0, x1) of the mosaic area to process (default: the whole
            mosaic); store index 0 is the region origin.
        core_px: Core edge, which is the store chunk.
        halo_px: Halo around each core (>= 256 for the validated behaviour).
        shift_px: Offset of the core grid from the region origin (chunk-invariance
            test).
        max_workers: Requested number of worker processes, reduced to the usable
            CPUs and the memory available to this process.
        memory_available_gb: Memory the process tree may use (default
            ``read_mem_available_gb()``: cgroup headroom and MemAvailable). The
            workers get 80 % of it less 8 GB for the parent.
        rss_per_worker_gb: Peak RSS of one worker (default
            ``estimate_worker_rss_gb``, plus ``FILL_EXTRA_RSS_GB`` with the fill;
            measure it on a dense chunk).
        max_tasks_per_child: Recycle a worker after this many chunks (frees
            fragmented heap; default never).
        progress_every: Log progress with ETA every this many finished chunks.
        device: Resolved backend, ``"cpu"`` (process pool) or ``"cuda"`` (one GPU
            consumer in this process under the host GPU lock named by
            ``MERXEN_GPU_LOCK_FILE``; no fill; ``max_workers``, the memory
            arguments and ``max_tasks_per_child`` do not apply).

    Returns:
        Run summary with wall time, CPU time, I/O share, peak memory and the
        per-chunk statistics.

    Raises:
        ValueError: Invalid ``z_layers``, stains, device or dtype, a parameter-hash
            mismatch, a fill that does not fit the halo (or any fill on the GPU),
            or a store shape that does not match the region.
        BeadRemovalBackendError: ``device`` is ``"cuda"`` and the GPU backend
            cannot be imported.
        FileNotFoundError: A mosaic plane that ``z_layers`` needs does not exist.
        RuntimeError: A chunk failed (the message names it); the store then holds
            only the chunks written so far.
    """
    t_run = time.time()
    # Refuses an unknown device before any work (and loads the GPU backend).
    corrector = get_window_corrector(device)
    if calibration_dir is None:
        raise ValueError(
            "calibration_dir is required (the workers load the saved calibration)"
        )
    fill_on = fill is not None and fill.enabled
    if fill_on and device == "cuda":
        raise ValueError(
            "the bead-removal fill has no GPU implementation; run it with device cpu"
        )
    check_fill_support(fill, halo_px)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    zs, mosaic_shape = _check_inputs(images_dir, stains, z_layers, calibration, params)
    region = region if region is not None else (0, mosaic_shape[0], 0, mosaic_shape[1])
    channels = (
        dict(store_channels)
        if store_channels is not None
        else {s: i for i, s in enumerate(stains)}
    )
    _check_store(store, stains, channels, region, exact_channels=store_channels is None)
    cores = plan_chunks(region, core_px=core_px, shift_px=shift_px)
    cal_dir = str(calibration_dir)
    tasks = [
        ChunkTask(
            index=i,
            core=core,
            halo_px=halo_px,
            images_dir=str(images_dir),
            stains=tuple(stains),
            z_layers=tuple(zs),
            calibration_dir=cal_dir,
            params=params,
            fill=fill,
        )
        for i, core in enumerate(cores)
    ]

    use_gpu = device == "cuda"
    if use_gpu:
        # One consumer in this process; the memory of the readers is bounded by
        # the read-ahead, not by a worker count.
        n_workers = 1
        estimate = memory = 0.0
        logger.info(
            "[BEADS] %d chunks, core %d px, halo %d px, device cuda, %d reader "
            "thread(s), %d window(s) read ahead, z_layers %s, stains %s, fill off",
            len(tasks),
            core_px,
            halo_px,
            GPU_READER_THREADS,
            GPU_QUEUE_WINDOWS,
            zs,
            list(stains),
        )
    else:
        window = core_px + 2 * halo_px + ENVELOPE_BLOCK_PX
        estimate = (
            rss_per_worker_gb
            if rss_per_worker_gb is not None
            else estimate_worker_rss_gb(
                (window, window),
                n_plane_stacks=count_plane_stacks(calibration, zs),
            )
            + (FILL_EXTRA_RSS_GB if fill_on else 0.0)
        )
        memory = (
            memory_available_gb
            if memory_available_gb is not None
            else read_mem_available_gb()
        )
        n_workers = resolve_worker_count(
            max_workers,
            memory_available_gb=memory,
            rss_per_worker_gb=estimate,
            n_tasks=len(tasks),
        )
        logger.info(
            "[BEADS] %d chunks, core %d px, halo %d px, device %s, %d worker(s) "
            "(requested %d, %.1f GB per worker, %.0f GB available, %d usable "
            "CPUs), z_layers %s, stains %s, fill %s",
            len(tasks),
            core_px,
            halo_px,
            device,
            n_workers,
            max_workers,
            estimate,
            memory,
            available_cpu_count(),
            zs,
            list(stains),
            f"on ({fill_params_sha256(fill)[:8]})"
            if fill is not None and fill.enabled
            else "off",
        )

    sink = _ChunkSink(
        store,
        channels,
        (region[0], region[2]),
        len(tasks),
        t_run,
        progress_every,
        region_shape=(region[1] - region[0], region[3] - region[2]),
    )
    # The thread variables are for the pool's children; a GPU run has none.
    old_env = {} if use_gpu else _set_single_thread_env()
    tree = _TreeSampler()
    tree.start()
    gpu_record: dict[str, Any] | None = None
    try:
        if tasks and use_gpu:
            gpu_record = _run_gpu_consumer(
                tasks,
                sink,
                corrector=corrector,
                calibration=calibration,
                lock_path=gpu_lock_path_from_env(),
            )
        elif tasks:
            _run_pool(
                tasks,
                sink,
                n_workers=n_workers,
                calibration_dir=cal_dir,
                max_tasks_per_child=max_tasks_per_child,
            )
        elif use_gpu:
            gpu_record = _gpu_run_record(
                locked=False, lock_wait_s=0.0, held_s=0.0, gpu_seconds=[]
            )
    finally:
        tree.sample()
        tree.finish()
        _restore_env(old_env)

    bead_arrays = [b for b in sink.beads if len(b)]
    n_beads = sum(len(b) for b in bead_arrays)
    np.save(
        work / "beads.npy",
        np.concatenate(bead_arrays) if bead_arrays else np.zeros((0, 8)),
    )
    np.savez(work / "change_overview.npz", **sink.change)  # type: ignore[arg-type]
    all_stats = sink.stats
    (work / "chunk_stats.jsonl").write_text(
        "\n".join(json.dumps(s) for s in all_stats) + "\n"
    )
    summary = BeadRemovalRunResult(
        n_chunks=len(tasks),
        n_workers=n_workers,
        wall_s=time.time() - t_run,
        worker_cpu_s=sum(s["cpu_s"] for s in all_stats),
        worker_wall_s=sum(s["wall_s"] for s in all_stats),
        read_s=sum(s["t_read_s"] for s in all_stats),
        correct_s=sum(s["t_correct_s"] for s in all_stats),
        project_s=sum(s["t_project_s"] for s in all_stats),
        parent_write_s=sink.t_write,
        n_beads=int(n_beads),
        peak_tree_rss_gb=tree.peak_rss,
        peak_tree_anon_gb=tree.peak_anon,
        peak_worker_rss_gb=max((s["peak_rss_gb"] for s in all_stats), default=0.0),
        peak_worker_anon_gb=max((s["peak_anon_gb"] for s in all_stats), default=0.0),
        work_dir=str(work),
        device=device,
        backend=describe_backend(device),
        gpu=gpu_record,
        chunk_stats=all_stats,
    )
    (work / "run_summary.json").write_text(json.dumps(summary.to_dict(), indent=1))
    logger.info(
        "[BEADS] done: %d chunks in %s (%s), worker CPU %.2f h, I/O share %.1f %%, "
        "peak tree RSS %.0f GB (anon %.0f GB)",
        len(tasks),
        _fmt_eta(summary.wall_s),
        device,
        summary.worker_cpu_s / 3600,
        100 * summary.read_s / max(summary.worker_wall_s, 1e-9),
        summary.peak_tree_rss_gb,
        summary.peak_tree_anon_gb,
    )
    return summary


def _check_store(
    store: Any,
    stains: Sequence[str],
    channels: Mapping[str, int],
    region: Core,
    *,
    exact_channels: bool,
) -> None:
    """Check that the store can take the region and has a channel per stain.

    Args:
        store: The output array.
        stains: Stains that are written.
        channels: Stain -> channel index.
        region: (y0, y1, x0, x1) that is processed.
        exact_channels: The store has one channel per stain (no other channels).

    Raises:
        ValueError: A stain has no channel or the shape does not fit.
    """
    missing = [s for s in stains if s not in channels]
    if missing:
        raise ValueError(f"store_channels has no channel for stains {missing}")
    height, width = region[1] - region[0], region[3] - region[2]
    n_channels = max(channels[s] for s in stains) + 1
    shape = tuple(store.shape)
    channels_ok = shape[0] == n_channels if exact_channels else shape[0] >= n_channels
    if len(shape) != 3 or not channels_ok or shape[1:] != (height, width):
        raise ValueError(
            f"store shape {shape} does not match {n_channels} channel(s) x region "
            f"{height} x {width}"
        )
