"""Tests of the CUDA path of the bead-removal driver and builder (``gpu`` marker).

They run the real GPU consumer on small synthetic mosaics:

* the driver's CUDA store agrees with its CPU store within the GPU acceptance
  bars (colour update order against the brightness order of the numpy
  algorithm), never brightens, and records the same chunk layout and statistics;
* the host GPU lock is taken once for the whole window loop and released when a
  window fails;
* the builder (B7) runs the whole MERSCOPE build with ``device="cuda"`` and
  writes the backend, lock wait and per-window GPU seconds into the preprocessing
  record and the QC folder.

The tests are opt-in (``flock <gpu lock file> pytest -m gpu``) and skip themselves
without CUDA. They never use the host lock file named in the environment (the
outer ``flock`` holds it): they point ``MERXEN_GPU_LOCK_FILE`` at a temporary
file. The stand-in tests of the consumer (no GPU needed) are in
``test_bead_driver.py``.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from merxen.config import MerscopeBeadRemovalConfig, MerscopeBuildConfig
from merxen.io.bead_removal import driver
from merxen.io.bead_removal.provenance import IMAGE_PREPROCESSING_ATTR
from merxen.io.builders import merscope as merscope_module
from merxen.io.builders.merscope import write_merscope_spatialdata

from . import bead_synth as bs
from .bead_build_fixtures import (
    open_projection_store,
    raw_max,
    real_calibration,
    write_mosaic,
)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

ZS = [1, 2, 3]
STAINS = ["PolyT", "DAPI"]
SHAPE = (1536, 2048)
# Gate A bars for the colour order against the CPU brightness order (O4).
MAX_FRAC_GT10 = 0.005
RATIO_TOLERANCE = 0.006


@pytest.fixture(scope="module")
def cuda() -> None:
    """Skip the test unless torch can use a CUDA device (imports torch)."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")


@pytest.fixture
def lock_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private lock file as the host lock of the run."""
    path = tmp_path / "gpu.lock"
    monkeypatch.setenv("MERXEN_GPU_LOCK_FILE", str(path))
    return path


def _lock_is_held(path: Path) -> bool:
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


@pytest.fixture(scope="module")
def mosaic(
    cuda: None, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[Path, Any, Path, dict[str, Any]]]:
    folder = tmp_path_factory.mktemp("gpu_mosaic")
    stack = bs.make_stack(n_planes=4, size=SHAPE, n_beads=70, seed=5, n_nuclei=10)
    write_mosaic(folder / "images", stack, 4)
    calibration = real_calibration()
    cal_dir = folder / "cal"
    calibration.save(cal_dir)
    yield folder / "images", calibration, cal_dir, stack


def _run(
    mosaic: tuple[Path, Any, Path, dict[str, Any]], out: Path, **kw: Any
) -> tuple[np.ndarray, driver.BeadRemovalRunResult]:
    images, calibration, cal_dir, _ = mosaic
    store = open_projection_store(
        out / "p.zarr", n_channels=2, shape=SHAPE, chunk_px=1024
    )
    args: dict[str, Any] = {
        "images_dir": images,
        "stains": STAINS,
        "z_layers": ZS,
        "store": store,
        "calibration": calibration,
        "calibration_dir": cal_dir,
        "work_dir": out / "work",
        "core_px": 1024,
        "max_workers": 2,
        "rss_per_worker_gb": 1.0,
        "memory_available_gb": 1000.0,
    }
    args.update(kw)
    result = driver.run_bead_removal(**args)
    return np.asarray(store[:]), result


# Keys of the chunk statistics that depend on the device or the run, not on the
# image: timings, memory, process ids, the GPU seconds.
_RUN_DEPENDENT = {
    "pid",
    "device",
    "gpu_s",
    "wall_s",
    "cpu_s",
    "peak_rss_gb",
    "peak_anon_gb",
}


def _static_keys(stats: dict[str, Any]) -> set[str]:
    return {k for k in stats if k not in _RUN_DEPENDENT and not k.startswith("t_")}


def test_cuda_driver_agrees_with_the_cpu_driver(
    mosaic: tuple[Path, Any, Path, dict[str, Any]],
    tmp_path: Path,
    lock_file: Path,
) -> None:
    stack = mosaic[3]
    cpu, run_cpu = _run(mosaic, tmp_path / "cpu", device="cpu")
    gpu, run_gpu = _run(mosaic, tmp_path / "gpu", device="cuda")
    assert run_cpu.device == "cpu" and run_gpu.device == "cuda"
    assert run_gpu.backend["device"] == "cuda"
    assert run_gpu.backend["update_order"] == "colour"
    assert run_gpu.backend["torch"] and run_gpu.backend["triton"]
    for i, stain in enumerate(STAINS):
        raw = raw_max(stack, stain, ZS).astype(np.float64)
        assert (gpu[i] <= raw).all(), f"{stain}: the GPU store is above raw"
        diff = np.abs(gpu[i].astype(np.float64) - cpu[i])
        assert (diff > 10).mean() <= MAX_FRAC_GT10, (
            f"{stain}: {(diff > 10).mean():.3%} of the pixels differ by > 10 counts"
        )
    raw_p = raw_max(stack, "PolyT", ZS).astype(np.float64)
    assert (raw_p - cpu[0]).sum() > 0 and (cpu[0] < raw_p).mean() > 0.01
    ratio = (raw_p - gpu[0]).sum() / (raw_p - cpu[0]).sum()
    assert abs(ratio - 1.0) <= RATIO_TOLERANCE, f"removed-signal ratio {ratio:.4f}"
    # The same chunks in the same order with the same statistics.
    assert run_gpu.n_chunks == run_cpu.n_chunks == 4
    assert [s["core"] for s in run_gpu.chunk_stats] == [
        s["core"] for s in run_cpu.chunk_stats
    ]
    assert [s["index"] for s in run_gpu.chunk_stats] == list(range(4))
    for g, c in zip(run_gpu.chunk_stats, run_cpu.chunk_stats, strict=True):
        assert _static_keys(g) == _static_keys(c)
        assert set(g["model"]) == set(c["model"])
        assert set(g["per_stain"]) == set(c["per_stain"])
        assert g["device"] == "cuda" and g["gpu_s"] > 0
        assert "gpu_s" not in c
    assert abs(run_gpu.n_beads - run_cpu.n_beads) <= max(3, 0.05 * run_cpu.n_beads)
    work = tmp_path / "gpu" / "work"
    assert sorted(p.name for p in work.iterdir()) == sorted(
        p.name for p in (tmp_path / "cpu" / "work").iterdir()
    )
    summary = json.loads((work / "run_summary.json").read_text())
    assert summary["gpu"]["locked"] is True
    assert summary["gpu"]["reader_threads"] == driver.GPU_READER_THREADS
    assert summary["gpu"]["gpu_s"] == pytest.approx(
        sum(s["gpu_s"] for s in run_gpu.chunk_stats)
    )
    assert "gpu" not in json.loads(
        (tmp_path / "cpu" / "work" / "run_summary.json").read_text()
    )
    assert not _lock_is_held(lock_file)


def test_cuda_driver_is_bit_identical_across_runs(
    mosaic: tuple[Path, Any, Path, dict[str, Any]], tmp_path: Path, lock_file: Path
) -> None:
    first, _ = _run(mosaic, tmp_path / "a", device="cuda")
    second, _ = _run(mosaic, tmp_path / "b", device="cuda")
    assert first.any() and np.array_equal(first, second)


def test_the_lock_is_taken_once_and_released_when_a_window_fails(
    mosaic: tuple[Path, Any, Path, dict[str, Any]],
    tmp_path: Path,
    lock_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    taken: list[Path | None] = []
    real_lock = driver.gpu_process_lock

    def counting(path: Path | str | None) -> Any:
        taken.append(None if path is None else Path(path))
        return real_lock(path)

    monkeypatch.setattr(driver, "gpu_process_lock", counting)
    held_in_windows: list[bool] = []
    real_chunk = driver.process_chunk
    calls = {"n": 0}

    def failing_second(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        held_in_windows.append(_lock_is_held(lock_file))
        if calls["n"] == 2:
            raise RuntimeError("stand-in GPU failure")
        return real_chunk(*args, **kwargs)

    monkeypatch.setattr(driver, "process_chunk", failing_second)
    with pytest.raises(RuntimeError, match=r"chunk 1 .*while correcting it"):
        _run(mosaic, tmp_path / "w", device="cuda")
    assert taken == [lock_file], "one acquisition for the whole window loop"
    assert held_in_windows == [True, True]
    assert not _lock_is_held(lock_file), "the lock must be released after an error"
    # A later run can take the lock again and finishes.
    monkeypatch.setattr(driver, "process_chunk", real_chunk)
    stores, run = _run(mosaic, tmp_path / "again", device="cuda")
    assert stores.any() and run.gpu is not None and run.gpu["lock_wait_s"] < 5


def test_the_second_lock_holder_waits_for_the_first(
    mosaic: tuple[Path, Any, Path, dict[str, Any]],
    tmp_path: Path,
    lock_file: Path,
) -> None:
    import threading
    import time

    lock_file.write_text("")
    holder = os.open(lock_file, os.O_RDONLY)
    fcntl.flock(holder, fcntl.LOCK_EX)

    def release() -> None:
        time.sleep(1.0)
        fcntl.flock(holder, fcntl.LOCK_UN)

    thread = threading.Thread(target=release)
    thread.start()
    try:
        _, run = _run(mosaic, tmp_path / "w", device="cuda")
    finally:
        thread.join()
        os.close(holder)
    assert run.gpu is not None and run.gpu["lock_wait_s"] >= 0.9


def test_b7_builder_runs_the_gpu_path_end_to_end(
    cuda: None,
    tmp_path: Path,
    lock_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B7: tiny raw folder, device cuda, whole build including the QC folder."""
    from .test_bead_qc import _calibration, _patch_build, _raw_folder

    written = _patch_build(monkeypatch)
    attrs: list[dict[str, Any]] = []
    plain_write = merscope_module.write_spatialdata_zarr  # creates the folder

    def _write(sdata: Any, path: Path, overwrite: bool | None = None) -> None:
        attrs.append(dict(sdata.attrs))
        plain_write(sdata, path, overwrite=overwrite)

    monkeypatch.setattr("merxen.io.builders.merscope.write_spatialdata_zarr", _write)
    calibration_backends: list[str] = []

    def _calibrate(reader: Any, **kwargs: Any) -> Any:
        calibration_backends.append(kwargs["backend"])
        return _calibration()

    monkeypatch.setattr(
        "merxen.io.bead_removal.calibrate.calibrate_bead_model", _calibrate
    )
    raw = _raw_folder(tmp_path)

    write_merscope_spatialdata(
        input_path=raw,
        output_path=tmp_path / "source.zarr",
        build_config=MerscopeBuildConfig(
            z_layers=ZS,
            bead_removal=MerscopeBeadRemovalConfig(
                enabled=True, max_workers=1, device="cuda"
            ),
        ),
    )

    assert calibration_backends == ["cuda"], "the forecast is made for the GPU"
    record = attrs[0][IMAGE_PREPROCESSING_ATTR]["bead_removal"]
    assert record["backend"]["device"] == "cuda"
    assert record["backend"]["update_order"] == "colour"
    assert record["backend"]["torch"] and record["backend"]["triton"]
    assert record["run"]["gpu"]["locked"] is True
    assert record["run"]["n_chunks"] == 1 and record["run"]["n_beads"] > 0
    qc_dir = written[0] / "bead_removal_qc"
    summary = json.loads((qc_dir / "summary.json").read_text())
    assert summary["errors"] == []
    assert summary["device"] == "cuda"
    assert summary["backend"]["update_order"] == "colour"
    assert summary["gpu"]["lock_wait_s"] >= 0 and summary["gpu"]["gpu_s"] > 0
    assert summary["gpu"]["gpu_s_per_window"]["max"] > 0
    header, *rows = (qc_dir / "chunks.csv").read_text().splitlines()
    columns = header.split(",")
    assert "gpu_s" in columns
    assert all(float(r.split(",")[columns.index("gpu_s")]) > 0 for r in rows)
    assert not _lock_is_held(lock_file)
    assert list(tmp_path.glob("merscope_projection_*")) == []
