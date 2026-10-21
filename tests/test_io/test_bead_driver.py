"""Tests of the chunk-major bead-removal driver on synthetic mosaics.

Fast: chunk planning, worker count (affinity, cgroup, DASK_NUM_WORKERS), the
cgroup readers on fake trees, refusals, the fill support check, the backend
hook, the store sink and the GPU consumer (reader threads, writer thread and
host GPU lock, driven by a numpy stand-in for the GPU correction so that no GPU
or torch is needed). Slow: runs through the process pool (1 vs 2 workers
bit-identical, fill on 1 vs 2 workers, a 7-plane mosaic, the core-grid shift).
The real CUDA path is tested in ``test_bead_driver_gpu.py``.
"""

from __future__ import annotations

import dataclasses
import fcntl
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from merxen.io.bead_removal import driver
from merxen.io.bead_removal.calibrate import BeadCalibration
from merxen.io.bead_removal.detect import empty_sources
from merxen.io.bead_removal.fill import FillParams
from merxen.io.bead_removal.model import BeadModel
from merxen.io.bead_removal.params import DEFAULT_PARAMS

from . import bead_synth as bs
from .bead_build_fixtures import (
    open_projection_store,
    raw_max,
    real_calibration,
    write_mosaic,
)

ZS = [1, 2, 3]
SHAPE = (2048, 2560)
STAINS = ["PolyT", "DAPI"]

MosaicFixture = tuple[Path, BeadCalibration, Path, dict[str, Any]]


@pytest.fixture(scope="module")
def mosaic(tmp_path_factory: pytest.TempPathFactory) -> MosaicFixture:
    folder = tmp_path_factory.mktemp("mosaic")
    stack = bs.make_stack(n_planes=4, size=SHAPE, n_beads=90, seed=7, n_nuclei=14)
    write_mosaic(folder, stack, 4)
    calibration = real_calibration()
    cal_dir = folder / "cal"
    calibration.save(cal_dir)
    return folder, calibration, cal_dir, stack


def _run(
    fixture: MosaicFixture, out: Path, *, shape: tuple[int, int] = SHAPE, **kw: Any
) -> tuple[np.ndarray, driver.BeadRemovalRunResult]:
    folder, calibration, cal_dir, _ = fixture
    store = open_projection_store(
        out / "p.zarr", n_channels=len(STAINS), shape=shape, chunk_px=1024
    )
    args: dict[str, Any] = {
        "images_dir": folder,
        "stains": STAINS,
        "z_layers": ZS,
        "store": store,
        "calibration": calibration,
        "calibration_dir": cal_dir,
        "work_dir": out / "work",
        "core_px": 1024,
        "max_workers": 1,
        "rss_per_worker_gb": 1.0,
        "memory_available_gb": 1000.0,
    }
    args.update(kw)
    result = driver.run_bead_removal(**args)
    return np.asarray(store[:]), result


# ---------------------------------------------------------------------------
# planning and resources
def test_chunk_plan_and_windows() -> None:
    cores = driver.plan_chunks((0, 2048, 0, 2560), core_px=1024)
    assert len(cores) == 6
    assert cores[0] == (0, 1024, 0, 1024)
    assert cores[-1] == (1024, 2048, 2048, 2560)
    covered = np.zeros((2048, 2560), int)
    for y0, y1, x0, x1 in cores:
        covered[y0:y1, x0:x1] += 1
    assert (covered == 1).all()
    shifted = driver.plan_chunks((100, 2148, 0, 2560), core_px=1024, shift_px=512)
    assert shifted[0] == (100, 612, 0, 512)
    assert shifted[-1][1] == 2148
    y0, y1, x0, x1 = driver.window_of((1024, 2048, 2048, 2560), 256)
    assert all(v % 16 == 0 for v in (y0, y1, x0, x1))
    assert (y0, y1, x0, x1) == (768, 2304, 1792, 2816)
    # An unaligned core: the window origin is snapped down onto the 16 px grid.
    wy0, _, wx0, _ = driver.window_of((37, 900, 21, 900), 256)
    assert wy0 % 16 == 0 and wy0 <= 37 - 256
    assert wx0 % 16 == 0 and wx0 <= 21 - 256


def test_planes_needed() -> None:
    cal = real_calibration()
    need = driver.planes_needed(cal, [2, 4])
    assert need["PolyT"] == [0, 1, 2, 4]
    assert need["DAPI"] == [0, 1, 2, 4]
    assert driver.planes_needed(cal, [3])["DAPI"] == [0, 1, 3]
    assert driver.planes_needed(cal, [3])["PolyT"] == [0, 1, 2, 3]


def test_worker_count_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    # Memory-limited count: floor((0.8 x available - 8 GB) / per-worker estimate).
    count = driver.resolve_worker_count
    assert count(16, memory_available_gb=100, rss_per_worker_gb=11, cpus=64) == 6
    assert count(5, memory_available_gb=1000, rss_per_worker_gb=11, cpus=64) == 5
    assert count(5, memory_available_gb=1, rss_per_worker_gb=11, cpus=64) == 1
    assert (
        count(8, memory_available_gb=None, rss_per_worker_gb=11, n_tasks=3, cpus=64)
        == 3
    )
    assert count(8, memory_available_gb=float("inf"), rss_per_worker_gb=11, cpus=3) == 3
    # CPUs: the affinity mask (not the host), the cgroup quota, DASK_NUM_WORKERS.
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(4)))
    monkeypatch.setattr(driver, "cgroup_cpu_limit", lambda *a, **k: None)
    monkeypatch.delenv("DASK_NUM_WORKERS", raising=False)
    assert driver.available_cpu_count() == 4
    assert count(16, memory_available_gb=1000, rss_per_worker_gb=11) == 4
    monkeypatch.setenv("DASK_NUM_WORKERS", "2")
    assert driver.available_cpu_count() == 2
    monkeypatch.setenv("DASK_NUM_WORKERS", "junk")
    assert driver.available_cpu_count() == 4
    monkeypatch.setattr(driver, "cgroup_cpu_limit", lambda *a, **k: 3)
    assert driver.available_cpu_count() == 3


def test_worker_count_warns_when_memory_fits_no_worker(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger=driver.logger.name):
        n = driver.resolve_worker_count(
            4, memory_available_gb=10, rss_per_worker_gb=11, cpus=8
        )
    assert n == 1
    assert "less than one worker" in caplog.text


def test_worker_memory_estimate() -> None:
    # 14 float32 planes held twice, plus a fixed overhead.
    expected = 14 * 4 * 1000 * 1000 * 2 / 1e9 + 3.5
    assert driver.estimate_worker_rss_gb((1000, 1000)) == pytest.approx(expected)
    # Measured: the fill adds 1.7 GB per worker at its peak.
    assert driver.FILL_EXTRA_RSS_GB == 1.8


def _fake_cgroup_v2(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "cg"
    (root / "job" / "task").mkdir(parents=True)
    (root / "job" / "memory.max").write_text("80000000000\n")
    (root / "job" / "memory.current").write_text("20000000000\n")
    (root / "job" / "memory.stat").write_text(
        "anon 1\ninactive_file 4000000000\nactive_file 5\n"
    )
    (root / "job" / "cpu.max").write_text("400000 100000\n")
    (root / "job" / "task" / "memory.max").write_text("max\n")
    (root / "job" / "task" / "memory.current").write_text("1000\n")
    (root / "job" / "task" / "cpu.max").write_text("max 100000\n")
    proc = tmp_path / "proc_cgroup"
    proc.write_text("0::/job/task\n")
    return root, proc


def test_cgroup_v2_limits(tmp_path: Path) -> None:
    root, proc = _fake_cgroup_v2(tmp_path)
    # Limit 80 GB, usage 20 GB of which 4 GB is reclaimable file cache.
    assert driver.cgroup_memory_headroom_gb(root, proc) == pytest.approx(64.0)
    assert driver.cgroup_cpu_limit(root, proc) == 4
    # A tighter limit lower in the hierarchy wins.
    (root / "job" / "task" / "memory.max").write_text("10000000000\n")
    (root / "job" / "task" / "memory.current").write_text("2000000000\n")
    assert driver.cgroup_memory_headroom_gb(root, proc) == pytest.approx(8.0)
    # No limits anywhere.
    (root / "job" / "memory.max").write_text("max\n")
    (root / "job" / "task" / "memory.max").write_text("max\n")
    (root / "job" / "cpu.max").write_text("max 100000\n")
    assert driver.cgroup_memory_headroom_gb(root, proc) is None
    assert driver.cgroup_cpu_limit(root, proc) is None
    nothing = tmp_path / "nothing"
    assert driver.cgroup_memory_headroom_gb(nothing, nothing) is None


def test_cgroup_v1_limits(tmp_path: Path) -> None:
    root = tmp_path / "cg"
    (root / "memory" / "job").mkdir(parents=True)
    (root / "cpu" / "job").mkdir(parents=True)
    job = root / "memory" / "job"
    (job / "memory.limit_in_bytes").write_text("50000000000\n")
    (job / "memory.usage_in_bytes").write_text("10000000000\n")
    (job / "memory.stat").write_text("cache 1\ntotal_inactive_file 2000000000\n")
    # "Unlimited" at the top of the hierarchy.
    (root / "memory" / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    (root / "memory" / "memory.usage_in_bytes").write_text("1\n")
    (root / "cpu" / "job" / "cpu.cfs_quota_us").write_text("200000\n")
    (root / "cpu" / "job" / "cpu.cfs_period_us").write_text("100000\n")
    proc = tmp_path / "proc_cgroup"
    proc.write_text("4:cpu,cpuacct:/job\n3:memory:/job\n")
    assert driver.cgroup_memory_headroom_gb(root, proc) == pytest.approx(42.0)
    assert driver.cgroup_cpu_limit(root, proc) == 2


def test_memory_available_is_cgroup_aware(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(driver, "cgroup_memory_headroom_gb", lambda *a, **k: 3.5)
    assert driver.read_mem_available_gb() <= 3.5
    monkeypatch.setattr(driver, "cgroup_memory_headroom_gb", lambda *a, **k: None)
    assert driver.read_mem_available_gb() > 0


# ---------------------------------------------------------------------------
# refusals
def test_refusals(mosaic: MosaicFixture, tmp_path: Path) -> None:
    folder, calibration, cal_dir, _ = mosaic
    store = open_projection_store(
        tmp_path / "p.zarr", n_channels=2, shape=SHAPE, chunk_px=1024
    )
    base: dict[str, Any] = {
        "images_dir": folder,
        "stains": STAINS,
        "store": store,
        "calibration": calibration,
        "calibration_dir": cal_dir,
        "work_dir": tmp_path / "w",
    }
    run = driver.run_bead_removal
    with pytest.raises(ValueError, match="plane 1 or later"):
        run(z_layers=[0, 1, 2], **base)
    with pytest.raises(ValueError, match="empty"):
        run(z_layers=[], **base)
    with pytest.raises(ValueError, match="up to 7"):
        run(z_layers=[1, 8], **base)
    with pytest.raises(ValueError, match="not among the output stains"):
        run(z_layers=ZS, **{**base, "stains": ["PolyT"]})
    with pytest.raises(ValueError, match="not bead-corrected"):
        run(z_layers=ZS, **{**base, "stains": ["PolyT", "DAPI", "Anti-X"]})
    one_channel = open_projection_store(
        tmp_path / "q.zarr", n_channels=1, shape=SHAPE, chunk_px=1024
    )
    with pytest.raises(ValueError, match="store shape"):
        run(z_layers=ZS, **{**base, "store": one_channel})
    with pytest.raises(ValueError, match="unknown bead-removal device"):
        run(z_layers=ZS, device="tpu", **base)
    with pytest.raises(ValueError, match="calibration_dir is required"):
        run(z_layers=ZS, **{**base, "calibration_dir": None})
    # A calibration made with other parameters is refused before any work.
    other = dataclasses.replace(DEFAULT_PARAMS, stretch_kmin=3)
    with pytest.raises(ValueError, match="params_sha256"):
        run(z_layers=ZS, params=other, **base)
    # The fixture mosaic has planes 0-3 only.
    with pytest.raises(
        FileNotFoundError, match=r"planes found in the folder: \[0, 1, 2, 3\]"
    ):
        run(z_layers=[1, 2, 3, 4], **base)


def test_store_channels_must_cover_the_stains(
    mosaic: MosaicFixture, tmp_path: Path
) -> None:
    folder, calibration, cal_dir, _ = mosaic
    store = open_projection_store(
        tmp_path / "p.zarr", n_channels=3, shape=SHAPE, chunk_px=1024
    )
    common: dict[str, Any] = {
        "images_dir": folder,
        "stains": STAINS,
        "z_layers": ZS,
        "store": store,
        "calibration": calibration,
        "calibration_dir": cal_dir,
        "work_dir": tmp_path / "w",
    }
    with pytest.raises(ValueError, match="no channel for stains"):
        driver.run_bead_removal(store_channels={"PolyT": 0}, **common)
    # A store with more channels than stains needs the mapping.
    with pytest.raises(ValueError, match="store shape"):
        driver.run_bead_removal(**common)
    # A mapped channel beyond the store.
    with pytest.raises(ValueError, match="store shape"):
        driver.run_bead_removal(store_channels={"PolyT": 0, "DAPI": 3}, **common)


def test_fill_support_must_fit_the_halo(mosaic: MosaicFixture, tmp_path: Path) -> None:
    fill = FillParams(enabled=True)
    support = driver.fill_support_px(fill)
    assert 0 < support <= driver.DEFAULT_HALO_PX - 2, "the shipped fill must fit"
    driver.check_fill_support(fill, driver.DEFAULT_HALO_PX)
    driver.check_fill_support(None, 16)
    driver.check_fill_support(FillParams(enabled=False), 16)
    with pytest.raises(ValueError, match=r"exceeds halo_px - 2"):
        driver.check_fill_support(fill, support + 1)
    driver.check_fill_support(fill, support + 2)  # exactly halo - 2 is allowed
    folder, calibration, cal_dir, _ = mosaic
    store = open_projection_store(
        tmp_path / "p.zarr", n_channels=2, shape=SHAPE, chunk_px=1024
    )
    with pytest.raises(ValueError, match=r"exceeds halo_px - 2"):
        driver.run_bead_removal(
            images_dir=folder,
            stains=STAINS,
            z_layers=ZS,
            store=store,
            calibration=calibration,
            calibration_dir=cal_dir,
            work_dir=tmp_path / "w",
            core_px=1024,
            halo_px=64,
            fill=fill,
        )


# ---------------------------------------------------------------------------
# backend hook and sink
def test_process_chunk_uses_the_given_corrector(mosaic: MosaicFixture) -> None:
    folder, calibration, cal_dir, _ = mosaic
    task = driver.ChunkTask(
        index=0,
        core=(0, 512, 0, 512),
        halo_px=256,
        images_dir=str(folder),
        stains=tuple(STAINS),
        z_layers=(1, 2),
        calibration_dir=str(cal_dir),
    )
    seen: dict[str, Any] = {}

    def fn(
        stacks: Any,
        *,
        origin_yx: tuple[int, int],
        calibration: BeadCalibration,
        z_layers: list[int],
        params: Any,
        fill: Any,
        core: tuple[int, int, int, int],
    ) -> tuple[dict[str, np.ndarray], BeadModel, dict[str, Any]]:
        seen.update(
            origin=origin_yx, core=core, planes=sorted(stacks["PolyT"]), z=z_layers
        )
        height, width = core[1] - core[0], core[3] - core[2]
        empty = BeadModel(empty_sources(), {}, {}, {}, {"status": "no beads"})
        out = {s: np.full((height, width), 7.4, np.float32) for s in STAINS}
        return out, empty, {"status": "no beads"}

    result = driver.process_chunk(
        task, corrector=driver.WindowCorrector("fake", fn), calibration=calibration
    )
    # The window is core + halo; the reader zero-pads outside the mosaic.
    assert seen["origin"] == (-256, -256)
    assert seen["planes"] == [0, 1, 2] and seen["z"] == [1, 2]
    assert seen["core"] == (256, 768, 256, 768)
    assert result.stats["device"] == "fake"
    assert all(
        a.dtype == np.uint16 and (a == 7).all() for a in result.projections.values()
    )
    assert driver.get_window_corrector("cpu") is driver.CPU_CORRECTOR
    with pytest.raises(ValueError, match="unknown bead-removal device"):
        driver.get_window_corrector("tpu")


def test_sink_writes_into_the_mapped_channels_in_task_order() -> None:
    store = np.zeros((3, 20, 30), np.uint16)
    sink = driver._ChunkSink(
        store, {"PolyT": 2, "DAPI": 0}, (100, 200), 2, 0.0, progress_every=1
    )
    for core, value in (((100, 110, 200, 215), 5), ((110, 120, 215, 230), 9)):
        stats = {"wall_s": 0.0}
        projections = {
            s: np.full((10, 15), value, np.uint16) for s in ("PolyT", "DAPI")
        }
        sink.write(driver.ChunkResult(0, core, projections, stats, np.zeros((0, 8))))
    assert (store[1] == 0).all(), "a channel that is not mapped stays untouched"
    assert (store[2, :10, :15] == 5).all() and (store[2, 10:, 15:] == 9).all()
    assert (store[0, :10, :15] == 5).all() and (store[0, 10:, 15:] == 9).all()
    assert len(sink.stats) == 2 and sink.t_write >= 0


def test_a_failing_worker_names_its_chunk(
    mosaic: MosaicFixture, tmp_path: Path
) -> None:
    """A pool that cannot start (the calibration cannot be loaded) is reported."""
    folder, calibration, _, _ = mosaic
    empty = tmp_path / "no_calibration"
    empty.mkdir()
    store = open_projection_store(
        tmp_path / "p.zarr", n_channels=2, shape=SHAPE, chunk_px=1024
    )
    with pytest.raises(RuntimeError, match=r"bead removal failed on chunk"):
        driver.run_bead_removal(
            images_dir=folder,
            stains=STAINS,
            z_layers=ZS,
            store=store,
            calibration=calibration,
            calibration_dir=empty,
            work_dir=tmp_path / "w",
            core_px=1024,
            max_workers=1,
            rss_per_worker_gb=1.0,
            memory_available_gb=1000.0,
        )


# ---------------------------------------------------------------------------
# the GPU consumer, driven by a numpy stand-in for the GPU correction
class _StandIn:
    """A "cuda" window corrector that returns the raw max projection.

    It records its calls and the state of the host GPU lock at each call, and
    can fail on a chosen chunk, so the consumer's reading, ordering, writing and
    locking are tested without a GPU.
    """

    def __init__(
        self, lock: Path | None = None, fail_on_call: int | None = None
    ) -> None:
        self.lock = lock
        self.fail_on_call = fail_on_call
        self.cores: list[tuple[int, int, int, int]] = []
        self.lock_held: list[bool] = []
        self.threads: set[str] = set()

    def __call__(
        self,
        stacks: Any,
        *,
        origin_yx: tuple[int, int],
        calibration: BeadCalibration,
        z_layers: list[int],
        params: Any,
        fill: Any,
        core: tuple[int, int, int, int],
    ) -> tuple[dict[str, np.ndarray], BeadModel, dict[str, Any]]:
        if self.fail_on_call is not None and len(self.cores) == self.fail_on_call:
            raise ValueError("stand-in failure")
        self.cores.append(core)
        self.threads.add(threading.current_thread().name)
        if self.lock is not None:
            self.lock_held.append(_lock_is_held(self.lock))
        y0, y1, x0, x1 = core
        out = {
            s: np.max([stacks[s][z][y0:y1, x0:x1] for z in z_layers], axis=0).astype(
                np.float32
            )
            for s in STAINS
        }
        return out, BeadModel(empty_sources(), {}, {}, {}, {"status": "ok"}), {}

    def corrector(self) -> driver.WindowCorrector:
        return driver.WindowCorrector("cuda", self)


def _lock_is_held(path: Path) -> bool:
    """Tell whether another open of the lock file is blocked by a holder."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


def _run_stand_in(
    mosaic: MosaicFixture,
    out: Path,
    stand_in: _StandIn,
    monkeypatch: pytest.MonkeyPatch,
    **kw: Any,
) -> tuple[np.ndarray, driver.BeadRemovalRunResult]:
    monkeypatch.setitem(driver._CORRECTORS, "cuda", stand_in.corrector())
    monkeypatch.setattr(
        driver,
        "describe_backend",
        lambda device: {"device": device, "update_order": "colour"},
    )
    return _run(mosaic, out, device="cuda", **kw)


def _consumer_threads() -> list[str]:
    return [
        t.name
        for t in threading.enumerate()
        if t.name.startswith(("bead-read", "bead-write"))
    ]


@pytest.mark.parametrize("reader_threads", [1, 3])
def test_gpu_consumer_writes_the_projection_whatever_the_reader_threads(
    mosaic: MosaicFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader_threads: int,
) -> None:
    """Window read, core crop and store order: the stand-in gives the raw max."""
    _, _, _, stack = mosaic
    lock = tmp_path / "gpu.lock"
    monkeypatch.setenv("MERXEN_GPU_LOCK_FILE", str(lock))
    monkeypatch.setattr(driver, "GPU_READER_THREADS", reader_threads)
    stand_in = _StandIn(lock)
    out, run = _run_stand_in(mosaic, tmp_path / "w", stand_in, monkeypatch)
    for i, stain in enumerate(STAINS):
        assert np.array_equal(out[i], raw_max(stack, stain, ZS)), stain
    # Identical window order and store layout: the cores come in task order, as
    # (y0, y1, x0, x1) inside their windows.
    expected = []
    for core in driver.plan_chunks((0, SHAPE[0], 0, SHAPE[1]), core_px=1024):
        wy0, _, wx0, _ = driver.window_of(core, driver.DEFAULT_HALO_PX)
        expected.append((core[0] - wy0, core[1] - wy0, core[2] - wx0, core[3] - wx0))
    assert stand_in.cores == expected
    assert run.device == "cuda" and run.n_workers == 1 and run.n_chunks == 6
    assert [s["index"] for s in run.chunk_stats] == list(range(6))
    assert all(s["device"] == "cuda" and s["gpu_s"] >= 0 for s in run.chunk_stats)
    assert all(stand_in.lock_held) and len(stand_in.lock_held) == 6
    assert not _consumer_threads()
    assert not _lock_is_held(lock)
    gpu = run.gpu
    assert gpu is not None
    assert gpu["locked"] is True and gpu["lock_wait_s"] >= 0
    assert gpu["reader_threads"] == reader_threads and gpu["queue_windows"] == 2
    assert gpu["gpu_s"] == pytest.approx(sum(s["gpu_s"] for s in run.chunk_stats))
    assert gpu["gpu_s_per_window"]["max"] >= gpu["gpu_s_per_window"]["median"] >= 0
    assert gpu["held_s"] >= 0
    summary = json.loads((tmp_path / "w" / "work" / "run_summary.json").read_text())
    assert summary["gpu"]["reader_threads"] == reader_threads
    assert summary["backend"] == {"device": "cuda", "update_order": "colour"}
    assert (tmp_path / "w" / "work" / "beads.npy").exists()
    assert sorted(p.name for p in (tmp_path / "w" / "work").iterdir()) == [
        "beads.npy",
        "change_overview.npz",
        "chunk_stats.jsonl",
        "run_summary.json",
    ]


def test_the_cpu_summary_has_no_gpu_entry(mosaic: MosaicFixture) -> None:
    result = driver.BeadRemovalRunResult(
        n_chunks=1,
        n_workers=1,
        wall_s=1.0,
        worker_cpu_s=1.0,
        worker_wall_s=1.0,
        read_s=0.0,
        correct_s=0.0,
        project_s=0.0,
        parent_write_s=0.0,
        n_beads=0,
        peak_tree_rss_gb=0.0,
        peak_tree_anon_gb=0.0,
        peak_worker_rss_gb=0.0,
        peak_worker_anon_gb=0.0,
        work_dir="w",
    )
    assert "gpu" not in result.to_dict() and result.gpu is None
    with_gpu = dataclasses.replace(result, gpu={"lock_wait_s": 0.0})
    assert with_gpu.to_dict()["gpu"] == {"lock_wait_s": 0.0}


def test_gpu_consumer_takes_the_lock_once_for_the_whole_loop(
    mosaic: MosaicFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "locks" / "gpu.lock"
    monkeypatch.setenv("MERXEN_GPU_LOCK_FILE", str(lock))
    events: list[str] = []
    real_lock = driver.gpu_process_lock

    class _Spy:
        def __init__(self, path: Any) -> None:
            self._inner = real_lock(path)

        def __enter__(self) -> Any:
            events.append("enter")
            return self._inner.__enter__()

        def __exit__(self, *exc: Any) -> Any:
            events.append("exit")
            return self._inner.__exit__(*exc)

    monkeypatch.setattr(driver, "gpu_process_lock", _Spy)
    stand_in = _StandIn(lock)
    _run_stand_in(mosaic, tmp_path / "w", stand_in, monkeypatch)
    assert events == ["enter", "exit"]
    assert len(stand_in.cores) == 6 and all(stand_in.lock_held)


def test_gpu_consumer_waits_for_the_host_lock_and_records_the_wait(
    mosaic: MosaicFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "gpu.lock"
    lock.write_text("")
    monkeypatch.setenv("MERXEN_GPU_LOCK_FILE", str(lock))
    holder = os.open(lock, os.O_RDONLY)
    fcntl.flock(holder, fcntl.LOCK_EX)
    stand_in = _StandIn()

    def release_later() -> None:
        time.sleep(0.6)
        fcntl.flock(holder, fcntl.LOCK_UN)

    thread = threading.Thread(target=release_later)
    thread.start()
    try:
        _, run = _run_stand_in(mosaic, tmp_path / "w", stand_in, monkeypatch)
    finally:
        thread.join()
        os.close(holder)
    assert run.gpu is not None and run.gpu["lock_wait_s"] >= 0.5
    # Nothing was read or corrected while the lock was not held.
    assert len(stand_in.cores) == 6


def test_gpu_consumer_runs_unlocked_without_a_lock_file(
    mosaic: MosaicFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MERXEN_GPU_LOCK_FILE", raising=False)
    _, run = _run_stand_in(mosaic, tmp_path / "w", _StandIn(), monkeypatch)
    assert run.gpu is not None
    assert run.gpu["locked"] is False and run.gpu["lock_wait_s"] < 1.0


def test_gpu_consumer_releases_the_lock_when_a_window_fails(
    mosaic: MosaicFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "gpu.lock"
    monkeypatch.setenv("MERXEN_GPU_LOCK_FILE", str(lock))
    stand_in = _StandIn(lock, fail_on_call=2)
    with pytest.raises(RuntimeError, match=r"failed on chunk 2 .*while correcting it"):
        _run_stand_in(mosaic, tmp_path / "w", stand_in, monkeypatch)
    assert len(stand_in.cores) == 2
    assert not _lock_is_held(lock), "the lock must be released after an error"
    assert not _consumer_threads(), "reader and writer threads must be joined"
    # The chunks written before the failure are in the store.
    store = np.asarray(zarr.open_array(str(tmp_path / "w" / "p.zarr"), mode="r")[:])
    assert store[0, :1024, :1024].any() and not store[0, 1024:, 1024:].any()


def test_gpu_consumer_names_a_chunk_that_cannot_be_read(
    mosaic: MosaicFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "gpu.lock"
    monkeypatch.setenv("MERXEN_GPU_LOCK_FILE", str(lock))
    real = driver.read_window_stacks
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("disk went away")
        return real(*args, **kwargs)

    monkeypatch.setattr(driver, "read_window_stacks", flaky)
    stand_in = _StandIn(lock)
    with pytest.raises(RuntimeError, match=r"chunk 2 .*while reading its window"):
        _run_stand_in(mosaic, tmp_path / "w", stand_in, monkeypatch)
    assert not _lock_is_held(lock) and not _consumer_threads()
    assert len(stand_in.cores) == 2


def test_gpu_consumer_names_a_chunk_that_cannot_be_written(
    mosaic: MosaicFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "gpu.lock"
    monkeypatch.setenv("MERXEN_GPU_LOCK_FILE", str(lock))
    real = driver._ChunkSink.write
    calls = {"n": 0}

    def failing(self: Any, result: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("store is full")
        real(self, result)

    monkeypatch.setattr(driver._ChunkSink, "write", failing)
    with pytest.raises(RuntimeError, match=r"chunk \d .*while writing the store"):
        _run_stand_in(mosaic, tmp_path / "w", _StandIn(lock), monkeypatch)
    assert not _lock_is_held(lock) and not _consumer_threads()


def test_the_fill_is_refused_on_the_gpu(
    mosaic: MosaicFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stand_in = _StandIn()
    with pytest.raises(ValueError, match="no GPU implementation"):
        _run_stand_in(
            mosaic, tmp_path / "w", stand_in, monkeypatch, fill=FillParams(enabled=True)
        )
    assert not stand_in.cores
    # The corrector itself refuses too, before it loads the GPU backend.
    with pytest.raises(ValueError, match="no GPU implementation"):
        driver._cuda_window_fn(
            {},
            origin_yx=(0, 0),
            calibration=real_calibration(),
            z_layers=[1],
            params=DEFAULT_PARAMS,
            fill=FillParams(enabled=True),
            core=(0, 1, 0, 1),
        )


def test_get_window_corrector_loads_the_gpu_backend_only_for_cuda() -> None:
    import subprocess
    import sys

    code = (
        "import sys\n"
        "from merxen.io.bead_removal import driver\n"
        "assert driver.get_window_corrector('cpu') is driver.CPU_CORRECTOR\n"
        "try:\n"
        "    driver.get_window_corrector('tpu')\n"
        "except ValueError:\n"
        "    pass\n"
        "assert 'torch' not in sys.modules and 'triton' not in sys.modules\n"
        "print('ok')\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == "ok"


# ---------------------------------------------------------------------------
# slow: the real pool
@pytest.mark.slow
def test_one_and_two_workers_are_bit_identical(
    mosaic: MosaicFixture, tmp_path: Path
) -> None:
    _, _, _, stack = mosaic
    one, run1 = _run(mosaic, tmp_path / "w1", max_workers=1)
    two, run2 = _run(mosaic, tmp_path / "w2", max_workers=2)
    assert np.array_equal(one, two), "1 and 2 workers must give identical stores"
    assert run1.n_workers == 1 and run2.n_workers == 2 and run1.device == "cpu"
    raw = {s: raw_max(stack, s, ZS) for s in STAINS}
    assert (one[0] <= raw["PolyT"]).all() and (one[1] <= raw["DAPI"]).all()
    assert (one[0] < raw["PolyT"]).mean() > 0.05, "beads must have been removed"
    assert run1.n_beads > 20 and run1.n_chunks == 6 and run1.read_s >= 0
    work = tmp_path / "w1" / "work"
    summary = json.loads((work / "run_summary.json").read_text())
    assert summary["n_chunks"] == 6
    assert summary["backend"] == {"device": "cpu", "update_order": "brightness"}
    assert (work / "beads.npy").exists()
    lines = (work / "chunk_stats.jsonl").read_text().splitlines()
    assert len(lines) == 6
    assert len(np.load(work / "beads.npy")) == run1.n_beads
    assert sorted(p.name for p in work.iterdir()) == [
        "beads.npy",
        "change_overview.npz",
        "chunk_stats.jsonl",
        "run_summary.json",
    ]


@pytest.mark.slow
def test_core_grid_shift_changes_little(mosaic: MosaicFixture, tmp_path: Path) -> None:
    a, _ = _run(mosaic, tmp_path / "a", max_workers=2)
    b, _ = _run(mosaic, tmp_path / "b", max_workers=2, shift_px=512)
    diff = np.abs(a[0].astype(int) - b[0].astype(int))
    assert (diff > 10).mean() < 0.02, f"grid dependence {(diff > 10).mean():.4f}"


@pytest.mark.slow
def test_seven_plane_mosaic(tmp_path: Path) -> None:
    """Planes 0-6 (a 7-plane dataset) with z_layers 1-6; plane 7 is refused."""
    shape = (1536, 1536)
    folder = tmp_path / "m7"
    stack = bs.make_stack(n_planes=7, size=shape, n_beads=45, seed=11, n_nuclei=8)
    write_mosaic(folder, stack, 7)
    calibration = real_calibration()
    calibration.save(tmp_path / "cal")
    z6 = [1, 2, 3, 4, 5, 6]
    store = open_projection_store(
        tmp_path / "p.zarr", n_channels=2, shape=shape, chunk_px=1024
    )
    common: dict[str, Any] = {
        "images_dir": folder,
        "stains": STAINS,
        "store": store,
        "calibration": calibration,
        "calibration_dir": tmp_path / "cal",
        "core_px": 1024,
    }
    result = driver.run_bead_removal(
        z_layers=z6,
        work_dir=tmp_path / "work",
        max_workers=2,
        rss_per_worker_gb=1.0,
        memory_available_gb=1000.0,
        **common,
    )
    out = np.asarray(store[:])
    assert result.n_chunks == 4 and result.n_beads > 10
    for i, stain in enumerate(STAINS):
        assert (out[i] <= raw_max(stack, stain, z6)).all(), f"{stain} brightened"
    assert (out[0] < raw_max(stack, "PolyT", z6)).mean() > 0.02
    with pytest.raises(
        FileNotFoundError, match=r"planes found in the folder: \[0, 1, 2, 3, 4, 5, 6\]"
    ):
        driver.run_bead_removal(z_layers=[1, 7], work_dir=tmp_path / "work2", **common)


@pytest.mark.slow
def test_fill_on_one_and_two_workers_are_identical(
    mosaic: MosaicFixture, tmp_path: Path
) -> None:
    """The fill depends on more of the window; not on the worker count."""
    fill = FillParams(enabled=True)
    one, run1 = _run(mosaic, tmp_path / "f1", max_workers=1, fill=fill)
    two, run2 = _run(mosaic, tmp_path / "f2", max_workers=2, fill=fill)
    assert np.array_equal(one, two), "fill on: 1 and 2 workers must be identical"
    off, _ = _run(mosaic, tmp_path / "f0", max_workers=2)
    assert (one[0] <= off[0]).all(), "the fill never adds light"
    assert (one[1] == off[1]).all(), "the fill acts on PolyT only"
    assert run1.n_workers == 1 and run2.n_workers == 2
    assert "PolyT" in run1.chunk_stats[0]["fill"]
