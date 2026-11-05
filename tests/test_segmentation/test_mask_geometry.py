"""Tests for mask polygon extraction."""

from __future__ import annotations

import logging
import os
import signal
from collections.abc import Iterator
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from pathlib import Path
from types import FrameType

import numpy as np
import pytest
from shapely.geometry.base import BaseGeometry

import merxen.segmentation.mask_geometry as mask_geometry
from merxen.segmentation.mask_geometry import (
    masks_to_labeled_polygons,
    masks_to_polygons,
)


def test_masks_to_polygons_extracts_expected_count() -> None:
    """Two disconnected labels should produce two polygons."""
    masks = np.zeros((10, 10), dtype=np.int32)
    masks[1:4, 1:4] = 1
    masks[6:9, 6:9] = 2

    polys = masks_to_polygons(masks, n_jobs=1)
    assert len(polys) == 2
    assert all(poly.is_valid and not poly.is_empty for poly in polys)


def test_masks_to_polygons_applies_scale_factor() -> None:
    """Scaling polygons by 2x should increase area by approximately 4x."""
    masks = np.zeros((8, 8), dtype=np.int32)
    masks[2:6, 2:6] = 1

    base = masks_to_polygons(masks, factor_rescale=0.0, n_jobs=1)[0]
    scaled = masks_to_polygons(masks, factor_rescale=2.0, n_jobs=1)[0]

    assert base.area > 0
    ratio = scaled.area / base.area
    assert np.isclose(ratio, 4.0, rtol=1e-3)


def test_masks_to_labeled_polygons_preserves_sparse_label_ids() -> None:
    """Polygon extraction should retain non-contiguous mask identifiers."""
    masks = np.zeros((10, 10), dtype=np.int32)
    masks[1:4, 1:4] = 2
    masks[6:9, 6:9] = 9

    labeled = masks_to_labeled_polygons(masks, n_jobs=1)

    assert [label_id for label_id, _ in labeled] == [2, 9]


def test_masks_to_labeled_polygons_preserves_disconnected_label_components() -> None:
    """One mask ID with two components should retain both pieces."""
    masks = np.zeros((12, 12), dtype=np.int32)
    masks[1:4, 1:4] = 5
    masks[8:11, 8:11] = 5

    [(label_id, geometry)] = masks_to_labeled_polygons(masks, n_jobs=1)

    assert label_id == 5
    assert geometry.geom_type == "MultiPolygon"
    assert len(geometry.geoms) == 2


def _grid_mask(n_side: int = 9, block: int = 4) -> np.ndarray:
    """Return a mask of ``n_side**2`` square labels separated by background."""
    size = n_side * block
    rows, cols = np.indices((size, size))
    labels = (rows // block) * n_side + (cols // block) + 1
    inside = (rows % block != 0) & (cols % block != 0)
    return np.where(inside, labels, 0).astype(np.int32)


class _FailingExecutor:
    """Process pool stand-in that cannot start and records its start method."""

    start_methods: list[str] = []

    def __init__(self, *args: object, mp_context: object, **kwargs: object) -> None:
        _FailingExecutor.start_methods.append(mp_context.get_start_method())  # type: ignore[attr-defined]
        raise OSError("simulated pool start failure")


@pytest.fixture
def failing_executor(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make every pool fail to start; return the start methods requested."""
    _FailingExecutor.start_methods = []
    monkeypatch.setattr(mask_geometry, "ProcessPoolExecutor", _FailingExecutor)
    return _FailingExecutor.start_methods


def _fail_in_workers(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    """Make polygon extraction fail in forked workers but not in this process.

    Args:
        monkeypatch: Fixture used to patch the extraction function, which
            forked workers inherit.
        failure: ``"raise"`` to raise ``ValueError`` in a worker, ``"exit"``
            to end the worker process abruptly, as an out-of-memory kill does.
    """
    parent_pid = os.getpid()
    extract = mask_geometry._polygon_from_bbox_with_mask

    def _extract(
        seg_masks: np.ndarray, label_and_slice: tuple[int, tuple[slice, slice]]
    ) -> BaseGeometry | None:
        if os.getpid() != parent_pid and label_and_slice[0] == 40:
            if failure == "exit":
                os._exit(1)
            raise ValueError("simulated worker failure")
        return extract(seg_masks, label_and_slice)

    monkeypatch.setattr(mask_geometry, "_polygon_from_bbox_with_mask", _extract)


@contextmanager
def _time_limit(seconds: int) -> Iterator[None]:
    """Fail instead of hanging when the block runs longer than ``seconds``."""

    def _timeout(signum: int, frame: FrameType | None) -> None:
        raise TimeoutError(f"still running after {seconds} s")

    previous = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _wkb_pairs(labeled: list[tuple[int, BaseGeometry]]) -> list[tuple[int, bytes]]:
    return [(label_id, polygon.wkb) for label_id, polygon in labeled]


def test_masks_to_labeled_polygons_pool_matches_serial() -> None:
    """The default fork pool returns the same labels and polygons as serial."""
    masks = _grid_mask()

    serial = masks_to_labeled_polygons(masks, n_jobs=1)
    pooled = masks_to_labeled_polygons(masks, n_jobs=2)

    assert len(serial) == 81
    assert _wkb_pairs(pooled) == _wkb_pairs(serial)


def test_masks_to_labeled_polygons_falls_back_to_serial_by_default(
    failing_executor: list[str], caplog: pytest.LogCaptureFixture
) -> None:
    """A pool failure falls back to the serial loop and says so in the log."""
    masks = _grid_mask()
    serial = masks_to_labeled_polygons(masks, n_jobs=1)

    with caplog.at_level(logging.WARNING, logger=mask_geometry.__name__):
        pooled = masks_to_labeled_polygons(masks, n_jobs=2)

    assert _wkb_pairs(pooled) == _wkb_pairs(serial)
    assert failing_executor == ["fork"]
    assert "Polygonisation pool failed" in caplog.text
    assert "retrying 81 labels serially" in caplog.text


def test_masks_to_labeled_polygons_raises_on_pool_failure_without_fallback(
    failing_executor: list[str],
) -> None:
    """``serial_fallback=False`` turns a pool failure into an error."""
    with pytest.raises(RuntimeError, match="Polygonisation pool") as excinfo:
        masks_to_labeled_polygons(_grid_mask(), n_jobs=2, serial_fallback=False)

    assert isinstance(excinfo.value.__cause__, OSError)


@pytest.mark.parametrize(
    ("failure", "cause"),
    [("raise", ValueError), ("exit", BrokenProcessPool)],
)
def test_masks_to_labeled_polygons_raises_on_worker_failure_without_fallback(
    monkeypatch: pytest.MonkeyPatch, failure: str, cause: type[BaseException]
) -> None:
    """An error in a worker, or a worker dying, raises instead of hanging."""
    _fail_in_workers(monkeypatch, failure)

    with (
        _time_limit(60),
        pytest.raises(RuntimeError, match="Polygonisation pool") as excinfo,
    ):
        masks_to_labeled_polygons(_grid_mask(), n_jobs=2, serial_fallback=False)

    assert isinstance(excinfo.value.__cause__, cause)


@pytest.mark.parametrize("failure", ["raise", "exit"])
def test_masks_to_labeled_polygons_falls_back_on_worker_failure_by_default(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """A failed or dead worker falls back to the serial loop by default."""
    masks = _grid_mask()
    serial = masks_to_labeled_polygons(masks, n_jobs=1)
    _fail_in_workers(monkeypatch, failure)

    with _time_limit(60):
        pooled = masks_to_labeled_polygons(masks, n_jobs=2)

    assert _wkb_pairs(pooled) == _wkb_pairs(serial)


def test_masks_to_labeled_polygons_without_fallback_matches_serial() -> None:
    """A healthy pool is unaffected by ``serial_fallback=False``."""
    masks = _grid_mask()

    pooled = masks_to_labeled_polygons(masks, n_jobs=2, serial_fallback=False)

    assert _wkb_pairs(pooled) == _wkb_pairs(masks_to_labeled_polygons(masks, n_jobs=1))


@pytest.mark.parametrize(
    ("kwargs", "expected_method"),
    [({}, "fork"), ({"mp_context": "forkserver"}, "forkserver")],
)
def test_masks_to_labeled_polygons_passes_the_start_method(
    failing_executor: list[str],
    kwargs: dict[str, str],
    expected_method: str,
) -> None:
    """The pool is created from the requested start method, ``fork`` by default."""
    with pytest.raises(RuntimeError):
        masks_to_labeled_polygons(
            _grid_mask(), n_jobs=2, serial_fallback=False, **kwargs
        )

    assert failing_executor == [expected_method]


def test_masks_to_labeled_polygons_spawn_pool_matches_serial() -> None:
    """Workers started with ``spawn`` receive the mask and return the same output."""
    masks = _grid_mask()

    pooled = masks_to_labeled_polygons(
        masks, n_jobs=2, serial_fallback=False, mp_context="spawn"
    )

    assert _wkb_pairs(pooled) == _wkb_pairs(masks_to_labeled_polygons(masks, n_jobs=1))


def _saved_memmap(tmp_path: Path, masks: np.ndarray) -> np.memmap:
    """Save ``masks`` as ``.npy`` and map it back read-only."""
    path = tmp_path / "masks.npy"
    np.save(path, masks)
    loaded = np.load(path, mmap_mode="r")
    assert isinstance(loaded, np.memmap)
    return loaded


@pytest.mark.parametrize("method", ["forkserver", "spawn"])
def test_memmapped_mask_reaches_clean_workers_as_its_file(
    tmp_path: Path, method: mask_geometry.MpStartMethod
) -> None:
    """A mask mapped from a file is mapped again, not copied, in clean workers."""
    masks = _grid_mask()
    memmapped = _saved_memmap(tmp_path, masks)

    sent = mask_geometry._pool_mask_argument(memmapped, method)
    mask_geometry._pool_init(sent)

    assert isinstance(sent, mask_geometry._MemmapSpec)
    assert isinstance(mask_geometry._POOL_SEG_MASKS, np.memmap)
    np.testing.assert_array_equal(mask_geometry._POOL_SEG_MASKS, masks)
    mask_geometry._POOL_SEG_MASKS = None


def test_fortran_ordered_memmapped_mask_keeps_its_layout(tmp_path: Path) -> None:
    """A Fortran-ordered file is mapped again with the same element order."""
    masks = np.asfortranarray(_grid_mask()[:, :-3])
    sent = mask_geometry._pool_mask_argument(_saved_memmap(tmp_path, masks), "spawn")

    mask_geometry._pool_init(sent)

    assert isinstance(sent, mask_geometry._MemmapSpec)
    assert sent.order == "F"
    np.testing.assert_array_equal(mask_geometry._POOL_SEG_MASKS, masks)
    mask_geometry._POOL_SEG_MASKS = None


def test_mask_is_sent_as_an_array_when_its_file_cannot_stand_in(
    tmp_path: Path,
) -> None:
    """Fork pools, in-memory masks, memmap views and copy-on-write maps send data."""
    masks = _grid_mask()
    memmapped = _saved_memmap(tmp_path, masks)
    copy_on_write = np.load(tmp_path / "masks.npy", mmap_mode="c")

    assert mask_geometry._pool_mask_argument(memmapped, "fork") is memmapped
    assert mask_geometry._pool_mask_argument(masks, "spawn") is masks
    view = memmapped[4:]
    assert mask_geometry._pool_mask_argument(view, "spawn") is view
    assert mask_geometry._pool_mask_argument(copy_on_write, "spawn") is copy_on_write


def test_masks_to_labeled_polygons_spawn_pool_reads_a_memmapped_mask(
    tmp_path: Path,
) -> None:
    """Spawned workers polygonise a memmapped mask like the serial loop."""
    masks = _grid_mask()

    pooled = masks_to_labeled_polygons(
        _saved_memmap(tmp_path, masks),
        n_jobs=2,
        serial_fallback=False,
        mp_context="spawn",
    )

    assert _wkb_pairs(pooled) == _wkb_pairs(masks_to_labeled_polygons(masks, n_jobs=1))


def test_masks_to_labeled_polygons_rejects_an_unknown_start_method() -> None:
    """A misspelt start method fails even when the serial fallback is on."""
    with pytest.raises(ValueError, match="mp_context"):
        masks_to_labeled_polygons(_grid_mask(), n_jobs=2, mp_context="threads")  # type: ignore[arg-type]


def test_masks_to_labeled_polygons_small_masks_do_not_start_a_pool(
    failing_executor: list[str],
) -> None:
    """Fewer than 64 labels run serially, so no pool can fail."""
    labeled = masks_to_labeled_polygons(
        _grid_mask(n_side=7), n_jobs=2, serial_fallback=False
    )

    assert len(labeled) == 49
    assert failing_executor == []
