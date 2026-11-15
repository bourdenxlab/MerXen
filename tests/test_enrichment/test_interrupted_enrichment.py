"""ENRICH killed part-way is enriched again on resume, never skipped as done.

ENRICH writes into the persistent latest store in place, and its skip checks
look only at which elements exist. A run killed after its last element but
before its attrs write (or during the incremental nuclei write) left a store
those checks call enriched, without the branch registry, the model identities
or the alignment invalidation. The stand-in store (``dst``) keeps what the
killed run wrote, as the disk would.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import merxen.enrichment.enrich as enrich_module
from merxen.enrichment.enrich import (
    CELLPOSE_NUCLEI_SHAPE_NAME,
    MOSAIK_CELLPOSE_SHAPE_NAME,
    MOSAIK_PROSEG_SHAPE_NAME,
    ORIGINAL_TABLE_NAME,
    _read_latest_zarr_for_enrichment,
    enrich_single_latest,
)
from merxen.io.bead_removal.provenance import write_store_segmentation_sidecar
from merxen.segmentation.output_markers import (
    enrichment_marker_path,
    store_marker_path,
)

from .test_enrich_paths import (
    _bead_record,
    _stub_merscope_enrichment,
    _stub_merscope_enrichment_config,
    _write_preprocessing_record,
)


class Killed(BaseException):
    """Stands in for SIGKILL: nothing below the raise runs, no handler cleans up."""


class Recorder:
    """Counts element writes and the steps a complete enrichment ends with."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.writes: list[str] = []
        self.finishes = 0
        self.kill_finish = False
        real_write = enrich_module.write_or_replace_element

        def _write(sdata: Any, key: str, *args: Any, **kwargs: Any) -> bool:
            self.writes.append(key)
            return bool(real_write(sdata, key, *args, **kwargs))

        def _finish(sdata: Any, **kwargs: Any) -> None:
            if self.kill_finish:
                self.kill_finish = False
                raise Killed
            self.finishes += 1

        monkeypatch.setattr(enrich_module, "write_or_replace_element", _write)
        monkeypatch.setattr(enrich_module, "write_spatialdata_metadata", _finish)
        monkeypatch.setattr(
            enrich_module,
            "invalidate_alignment_materialization",
            lambda sdata, **kwargs: None,
        )


def _setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[SimpleNamespace, Any, Recorder]:
    stub = _stub_merscope_enrichment(monkeypatch, tmp_path)
    config = _stub_merscope_enrichment_config(
        tmp_path, stub, persistent_output_path=stub.target
    )
    return stub, config, Recorder(monkeypatch)


def test_a_kill_before_the_attrs_write_is_enriched_again_not_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stub, config, recorder = _setup(monkeypatch, tmp_path)
    marker = enrichment_marker_path(stub.target)
    recorder.kill_finish = True

    with pytest.raises(Killed):
        enrich_single_latest(config)
    # Every element is there: the element checks alone would skip this store.
    assert enrich_module._is_already_enriched(stub.dst, "MERSCOPE")
    assert marker.is_file()
    recorder.writes.clear()

    enrich_single_latest(config)

    assert ORIGINAL_TABLE_NAME in recorder.writes
    assert MOSAIK_CELLPOSE_SHAPE_NAME in recorder.writes
    assert recorder.finishes == 1
    assert not marker.exists()
    # Complete now: the next run skips.
    recorder.writes.clear()
    enrich_single_latest(config)
    assert recorder.writes == []


def test_a_kill_during_the_incremental_nuclei_write_is_rebuilt_in_full(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stub, config, recorder = _setup(monkeypatch, tmp_path)
    enrich_single_latest(config)
    # An enrichment from before the nuclei layer.
    del stub.dst.shapes[CELLPOSE_NUCLEI_SHAPE_NAME]
    recorder.writes.clear()
    recorder.kill_finish = True

    with pytest.raises(Killed):
        enrich_single_latest(config)
    assert recorder.writes == [CELLPOSE_NUCLEI_SHAPE_NAME]
    assert enrich_module._is_already_enriched(stub.dst, "MERSCOPE")
    recorder.writes.clear()

    enrich_single_latest(config)

    assert MOSAIK_PROSEG_SHAPE_NAME in recorder.writes
    assert CELLPOSE_NUCLEI_SHAPE_NAME in recorder.writes
    assert not enrichment_marker_path(stub.target).exists()


def test_the_marker_sits_beside_the_real_store_not_in_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reached through the staged link, the marker still lands beside the store;
    and it is not the ProSeg marker, which would send the store back to ProSeg."""
    stub, _, recorder = _setup(monkeypatch, tmp_path)
    config = _stub_merscope_enrichment_config(
        tmp_path, stub, persistent_output_path=stub.latest
    )
    recorder.kill_finish = True
    with pytest.raises(Killed):
        enrich_single_latest(config)

    marker = enrichment_marker_path(stub.latest)
    assert marker.parent == stub.target.parent
    assert marker.name == ".latest_spatialdata.zarr.enrich.merxen-incomplete"
    assert not any(stub.target.glob("*merxen-incomplete"))
    assert not store_marker_path(stub.target).exists()


def test_a_refused_store_leaves_no_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A refusal before the first write must not force a rebuild later."""
    stub, config, _ = _setup(monkeypatch, tmp_path)
    _write_preprocessing_record(stub.original, _bead_record())

    with pytest.raises(ValueError, match="predates the correction"):
        enrich_single_latest(config)

    assert not enrichment_marker_path(stub.target).exists()
    write_store_segmentation_sidecar(stub.target, _bead_record())
    enrich_single_latest(config)
    assert not enrichment_marker_path(stub.target).exists()


def test_an_unreadable_interrupted_store_loses_only_what_enrich_rebuilds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An element written for the first time when the kill landed can block the
    read; only the elements the rebuild writes again are removed."""
    store = tmp_path / "latest.zarr"
    kept = [
        f"shapes/{MOSAIK_PROSEG_SHAPE_NAME}",
        "shapes/cell_boundaries",
        "tables/table_MOSAIK_proseg_hybrid",
    ]
    removed = [
        f"shapes/{MOSAIK_CELLPOSE_SHAPE_NAME}",
        f"tables/{ORIGINAL_TABLE_NAME}",
    ]
    for rel_path in kept + removed:
        (store / rel_path).mkdir(parents=True)
    reads: list[Path] = []

    def _read(path: Path) -> object:
        reads.append(Path(path))
        if len(reads) == 1:
            raise ValueError("partial element")
        return object()

    monkeypatch.setattr(enrich_module.sd, "read_zarr", _read)

    with pytest.raises(ValueError, match="partial element"):
        _read_latest_zarr_for_enrichment(store, "MERSCOPE", "P1")
    assert all((store / rel_path).is_dir() for rel_path in kept + removed)
    reads.clear()

    _read_latest_zarr_for_enrichment(store, "MERSCOPE", "P1", interrupted=True)

    assert len(reads) == 2
    assert all((store / rel_path).is_dir() for rel_path in kept)
    assert not any((store / rel_path).exists() for rel_path in removed)
