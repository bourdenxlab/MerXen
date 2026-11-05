"""Atera builder on the synthetic bundle (plan §7.2 contract, config,
atomicity and metadata; §7.3 integration: the on-disk store re-read).

One store is built per module through ``build_spatialdata_artifact`` and shared
by the read-only tests; tests that build, break or refuse a store use their
own copies under ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dask
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import spatialdata as sd
import zarr
from pydantic import ValidationError
from spatialdata.transformations import Identity, Scale, get_transformation

from merxen import privacy
from merxen.config import AteraBuildConfig, SpatialDataBuildConfig
from merxen.io import atera_reader as ar
from merxen.io.builders import atera as atera_builder
from merxen.io.builders.pipeline import build_spatialdata_artifact
from merxen.io.builders.xenium import check_not_atera_preview
from merxen.io.spatialdata_schema import (
    MERXEN_SCHEMA_ATTR,
    validate_merxen_schema,
    validate_merxen_schema_streaming,
)
from merxen.platforms import find_um_per_px

from .atera_synth import (
    EMPTY_POLYGON_CELL,
    FAKE_IDENTIFIERS,
    AteraSynth,
    write_atera_bundle,
)

#: Small partitions and pools keep the fixture build at a few seconds.
FAST_OPTIONS: dict[str, Any] = {
    "transcript_partition_rows": 1_048_576,
    "polygon_n_jobs": 2,
    "morphology_levels": 3,
}
_HERE = Path(__file__).parent


@dataclass(frozen=True)
class Built:
    """The module's synthetic bundle and the store built from it."""

    synth: AteraSynth
    store: Path
    config: SpatialDataBuildConfig
    log_text: str


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def _config(
    roi: Path, output: Path, *, persistent: Path | None = None, **options: Any
) -> SpatialDataBuildConfig:
    return SpatialDataBuildConfig(
        dataset_name="SYN1_ATERA",
        platform="ATERA",
        input_path=roi,
        output_path=output,
        persistent_output_path=persistent,
        atera=AteraBuildConfig(**{**FAST_OPTIONS, **options}),
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Built]:
    """Build the default synthetic bundle once, collecting every log line."""
    root = tmp_path_factory.mktemp("atera_build")
    synth = write_atera_bundle(root / "roi")
    config = _config(synth.root, root / "stage" / "source.zarr")
    handler = _Collect()
    merxen_logger = logging.getLogger("merxen")
    level = merxen_logger.level
    merxen_logger.addHandler(handler)
    merxen_logger.setLevel(logging.DEBUG)
    try:
        store = build_spatialdata_artifact(config)
    finally:
        merxen_logger.removeHandler(handler)
        merxen_logger.setLevel(level)
    yield Built(synth, store, config, "\n".join(handler.lines))


def _attrs(store: Path) -> dict[str, Any]:
    return dict(zarr.open_group(store, mode="r").attrs)


# --------------------------------------------------------------------------- #
# Config and dispatch
# --------------------------------------------------------------------------- #


def test_build_config_accepts_atera_with_defaults(tmp_path: Path) -> None:
    config = SpatialDataBuildConfig(
        dataset_name="SYN1_ATERA",
        platform="ATERA",
        input_path=tmp_path,
        output_path=tmp_path / "out.zarr",
        atera={"max_rss_gb": 120, "species": "human"},
    )
    assert config.atera.min_qv == 20.0
    assert config.atera.spatial_order == "row_range"
    assert config.atera.boundary_source == "masks"
    assert config.atera.max_rss_gb == 120
    # Existing platforms keep their configs; the ATERA block is a default.
    xenium = SpatialDataBuildConfig(
        dataset_name="P_XENIUM",
        platform="XENIUM",
        input_path=tmp_path,
        output_path=tmp_path / "out.zarr",
    )
    assert xenium.atera == AteraBuildConfig()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"transcript_partition_rows": 1000}, "multiple of 1,048,576"),
        ({"cells_boundaries": False}, "cells_table needs cells_boundaries"),
        ({"spatial_order": "hilbert"}, "row_range"),
        ({"no_such_option": True}, "no_such_option"),
        ({"min_qv": -1}, "min_qv"),
    ],
)
def test_atera_build_config_rejects_bad_options(
    options: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        AteraBuildConfig(**options)


def test_dispatch_calls_the_atera_writer(
    built: Built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_write(**kwargs: Any) -> Path:
        calls.append(kwargs)
        Path(kwargs["output_path"]).mkdir(parents=True)
        return Path(kwargs["output_path"])

    monkeypatch.setattr(atera_builder, "write_atera_spatialdata", fake_write)
    output = tmp_path / "stage" / "source.zarr"
    config = _config(built.synth.root, output, min_qv=30.0)
    assert build_spatialdata_artifact(config) == output
    (call,) = calls
    assert call["input_path"] == built.synth.root
    assert call["output_path"] == output
    assert call["build_config"] == config.atera
    assert call["dataset_name"] == "SYN1_ATERA"


def test_zip_input_is_refused(tmp_path: Path) -> None:
    bundle_zip = tmp_path / "outs.zip"
    bundle_zip.write_bytes(b"PK")
    with pytest.raises(ValueError, match="extract the outs bundle"):
        build_spatialdata_artifact(_config(bundle_zip, tmp_path / "out.zarr"))


def test_missing_input_and_store_fail(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="atera_dir"):
        build_spatialdata_artifact(_config(tmp_path / "absent", tmp_path / "o.zarr"))


# --------------------------------------------------------------------------- #
# Concurrency and the memory watchdog
# --------------------------------------------------------------------------- #


def test_build_sets_dask_and_zarr_concurrency(
    built: Built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def capture(*_args: Any, **_kwargs: Any) -> None:
        seen["scheduler"] = dask.config.get("scheduler")
        seen["num_workers"] = dask.config.get("num_workers")
        seen["zarr_workers"] = zarr.config.get("threading.max_workers")
        raise RuntimeError("stop after the concurrency check")

    monkeypatch.setattr(atera_builder, "assemble_atera_sdata", capture)
    monkeypatch.setenv("DASK_NUM_WORKERS", "3")
    with pytest.raises(RuntimeError, match="stop after"):
        atera_builder.write_atera_spatialdata(
            input_path=built.synth.root,
            output_path=tmp_path / "out.zarr",
            build_config=AteraBuildConfig(),
        )
    assert seen == {"scheduler": "threads", "num_workers": 3, "zarr_workers": 3}
    assert not list(tmp_path.iterdir())


def test_resolve_worker_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASK_NUM_WORKERS", "16")
    assert atera_builder.resolve_worker_count() == 16
    assert atera_builder.resolve_worker_count(2) == 2
    monkeypatch.setenv("DASK_NUM_WORKERS", "many")
    with pytest.raises(ValueError, match="DASK_NUM_WORKERS"):
        atera_builder.resolve_worker_count()
    monkeypatch.delenv("DASK_NUM_WORKERS")
    assert atera_builder.resolve_worker_count() >= 1
    with pytest.raises(ValueError, match="positive"):
        atera_builder.resolve_worker_count(0)


def test_watchdog_records_the_peak_and_acts_on_the_cap() -> None:
    readings = iter([1, 3, 2, 5, 4, 4, 4, 4])
    exceeded: list[tuple[float, float]] = []
    watchdog = atera_builder.RssWatchdog(
        3.5 / (1 << 30),
        interval_s=0.001,
        measure=lambda: next(readings, 4),
        on_exceed=lambda used, limit: exceeded.append((used, limit)),
    )
    for _ in range(4):
        watchdog.check()
    assert watchdog.peak_gb == pytest.approx(5 / (1 << 30))
    assert len(exceeded) == 1
    with watchdog:
        pass
    assert len(exceeded) > 1


def test_watchdog_without_a_cap_only_records() -> None:
    with atera_builder.RssWatchdog(None, interval_s=0.01) as watchdog:
        np.ones(1 << 20)
    assert watchdog.peak_gb > 0


def test_process_tree_rss_counts_anonymous_memory() -> None:
    before = atera_builder.process_tree_anonymous_rss_bytes()
    block = np.ones(64 << 20, dtype=np.uint8)
    after = atera_builder.process_tree_anonymous_rss_bytes()
    assert after - before > 32 << 20
    del block


# --------------------------------------------------------------------------- #
# Store contents and the contract
# --------------------------------------------------------------------------- #


def test_store_holds_every_element_with_the_source_contract(built: Built) -> None:
    sdata = sd.read_zarr(built.store)
    assert list(sdata.images) == ["morphology_focus"]
    assert set(sdata.labels) == {"cell_labels", "nucleus_labels"}
    assert set(sdata.points) == {"transcripts"}
    assert set(sdata.shapes) == {"cell_boundaries", "nucleus_boundaries"}
    assert set(sdata.tables) == {"table"}
    schema = sdata.attrs[MERXEN_SCHEMA_ATTR]
    assert schema["platform"] == "ATERA"
    assert schema["primary_points"] == "transcripts"
    original = schema["segmentations"]["original"]
    assert original["shape"] == "cell_boundaries"
    assert original["assignment_column"] == "original_assignment"
    validate_merxen_schema(sdata, deep=False)


def test_transforms_follow_the_pixel_convention(built: Built) -> None:
    sdata = sd.read_zarr(built.store)
    scale = 1.0 / built.synth.pixel_size_um
    for element in (
        sdata.points["transcripts"],
        sdata.shapes["cell_boundaries"],
        sdata.shapes["nucleus_boundaries"],
    ):
        transform = get_transformation(element, "global")
        assert isinstance(transform, Scale)
        # z (points only) stays in microns.
        by_axis = dict(zip(transform.axes, transform.scale, strict=True))
        np.testing.assert_allclose([by_axis["x"], by_axis["y"]], [scale, scale])
        assert by_axis.get("z", 1.0) == 1.0
    for element in (sdata.images["morphology_focus"], sdata.labels["cell_labels"]):
        assert isinstance(get_transformation(element, "global"), Identity)
    image_shapes = [
        sdata.images["morphology_focus"][level]["image"].shape[1:]
        for level in sdata.images["morphology_focus"]
    ]
    label_shapes = [
        sdata.labels["cell_labels"][level]["image"].shape
        for level in sdata.labels["cell_labels"]
    ]
    assert image_shapes == label_shapes == [(97, 67), (48, 33), (24, 16)]


def test_points_match_the_vendor_rows_after_the_qv_filter(built: Built) -> None:
    sdata = sd.read_zarr(built.store)
    points = sdata.points["transcripts"].compute()
    truth = built.synth.transcripts
    truth = truth[truth["qv"].astype(np.float64) >= 20.0]
    assert len(points) == len(truth)
    assert sorted(points["transcript_id"]) == sorted(truth["transcript_id"])
    assert points["qv"].min() >= 20.0
    # Known categories are lost on a parquet round trip; attrs keep them.
    attrs = sdata.attrs["merxen_atera"]["transcripts"]
    known = points["feature_name"].cat.set_categories(attrs["feature_names"])
    assert list(known.cat.categories) == list(built.synth.gene_names)


def test_label_and_table_values(built: Built) -> None:
    sdata = sd.read_zarr(built.store)
    labels = sdata.labels["cell_labels"]["scale0"]["image"].values
    np.testing.assert_array_equal(labels, built.synth.cell_mask)
    table = sdata.tables["table"]
    shape_ids = sdata.shapes["cell_boundaries"]["instance_id"].to_numpy()
    assert sorted(table.obs["instance_id"]) == sorted(shape_ids)
    assert "highly_variable" not in table.var
    assert set(sdata.shapes["nucleus_boundaries"]["cell_instance_id"]) <= set(shape_ids)


# --------------------------------------------------------------------------- #
# Vendor metadata kept whole (plan §3.11, user decision 2026-11-05)
# --------------------------------------------------------------------------- #


def test_sidecars_are_byte_identical_with_original_names(built: Built) -> None:
    records = _attrs(built.store)["merxen_atera"]["sidecars"]
    assert records
    by_original = {record["original_name"]: record for record in records}
    for sidecar in ar.resolve_atera_bundle(built.synth.root).sidecars:
        record = by_original[sidecar.original_name]
        stored = built.store / record["path"]
        source = sidecar.source.read_bytes()
        assert stored.read_bytes() == source
        assert record["size_bytes"] == len(source)
        assert record["sha256"] == hashlib.sha256(source).hexdigest()
    names = {record["path"] for record in records}
    assert {
        "atera/experiment.spatial",
        "atera/metrics_summary.csv",
        "atera/analysis_summary.html",
        "atera/panel_config.json",
    } <= names
    panel = (built.store / "atera" / "panel_config.json").read_text()
    assert FAKE_IDENTIFIERS["customer_email"] in panel
    assert FAKE_IDENTIFIERS["designer_name"] in panel


def test_attrs_carry_the_full_vendor_header(built: Built) -> None:
    header = _attrs(built.store)["merxen_atera"]
    experiment = json.loads((built.synth.root / "experiment.spatial").read_text())
    assert header["experiment"] == experiment
    for key in ("run_name", "slide_name", "slide_id", "region_name", "instrument_sn"):
        assert header["experiment"][key] == FAKE_IDENTIFIERS[key]
    (metrics,) = header["metrics_summary"]
    assert metrics["run_name"] == FAKE_IDENTIFIERS["run_name"]
    assert metrics["reagent_plate_1"] == FAKE_IDENTIFIERS["reagent_lot"]
    (payload,) = header["panel_config"]["payloads"]
    assert payload["customer"]["email"] == FAKE_IDENTIFIERS["customer_email"]
    assert payload["customer"]["name"] == FAKE_IDENTIFIERS["customer_name"]
    assert header["bundle_files"]["transcripts"] == "transcripts.zarr.zip"
    build = header["build"]
    assert build["pixel_size_um"] == built.synth.pixel_size_um
    assert build["morphology_channels"] == list(built.synth.channel_names)
    assert build["options"]["min_qv"] == 20.0
    assert set(header["shapes"]) == {"cell_boundaries", "nucleus_boundaries"}


def test_transcript_statistics_reach_the_attrs(built: Built) -> None:
    transcripts = _attrs(built.store)["merxen_atera"]["transcripts"]
    truth = built.synth.transcripts
    kept = truth[truth["qv"].astype(np.float64) >= 20.0]
    assert transcripts["n_rows_read"] == len(truth)
    assert transcripts["n_points_written"] == len(kept)
    assert transcripts["rows_kept_per_tile"] == {
        key: int((kept["tile"] == key).sum()) for key in built.synth.grid_keys_order
    }
    q20 = np.bincount(kept["gene"], minlength=len(built.synth.gene_names))
    assert transcripts["gene_counts_q20"] == q20.tolist()


def test_points_index_matches_the_parquet_files(built: Built) -> None:
    index = json.loads((built.store / atera_builder.POINTS_INDEX_NAME).read_text())
    entry = index["points"]["transcripts"]
    directory = built.store / entry["directory"]
    files = sorted(path.name for path in directory.glob("*.parquet"))
    assert sorted(item["file"] for item in entry["files"]) == files
    total = 0
    for item in entry["files"]:
        frame = pq.read_table(directory / item["file"]).to_pandas()
        total += len(frame)
        assert item["n_rows"] == len(frame)
        if len(frame):
            assert set(frame["source_tile"].astype(str)) == {item["tile"]}
            assert item["transcript_id_range"] == [
                int(frame["transcript_id"].min()),
                int(frame["transcript_id"].max()),
            ]
            x0, x1, y0, y1 = item["bbox_um"]
            assert x0 == pytest.approx(frame["x"].min())
            assert y1 == pytest.approx(frame["y"].max())
            assert x1 >= x0 and y1 >= y0
    assert entry["fingerprint"] == {"n_files": len(files), "n_rows": total}


def test_marker_records_the_fingerprint(built: Built) -> None:
    marker = atera_builder.read_build_marker(built.store)
    assert marker is not None
    assert marker["platform"] == "ATERA"
    bundle = ar.resolve_atera_bundle(built.synth.root)
    fingerprint = atera_builder.atera_build_fingerprint(bundle, built.config.atera)
    assert marker["build_fingerprint"] == fingerprint
    header = _attrs(built.store)["merxen_atera"]
    assert header["build"]["build_fingerprint"] == fingerprint
    assert marker["peak_anonymous_rss_gb"] > 0


def test_fingerprint_ignores_runtime_options_only(built: Built) -> None:
    bundle = ar.resolve_atera_bundle(built.synth.root)
    base = AteraBuildConfig(**FAST_OPTIONS)
    fingerprint = atera_builder.atera_build_fingerprint(bundle, base)
    runtime = base.model_copy(update={"max_rss_gb": 9.0, "polygon_n_jobs": 7})
    assert atera_builder.atera_build_fingerprint(bundle, runtime) == fingerprint
    other = base.model_copy(update={"min_qv": 30.0})
    assert atera_builder.atera_build_fingerprint(bundle, other) != fingerprint


def test_logs_name_the_alias_and_no_manifest_value(built: Built) -> None:
    """Human-facing output names the dataset by alias (plan §3.11)."""
    assert "SYN1_ATERA" in built.log_text
    for value in FAKE_IDENTIFIERS.values():
        assert value not in built.log_text
    assert str(built.synth.root) not in built.log_text


# --------------------------------------------------------------------------- #
# Atomic write, completion marker and reuse (plan §3.12)
# --------------------------------------------------------------------------- #


def test_failed_write_leaves_no_store_and_a_rerun_rebuilds(
    built: Built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persistent = tmp_path / "cache" / "source.zarr"
    output = tmp_path / "stage" / "source.zarr"
    config = _config(built.synth.root, output, persistent=persistent)
    finish = atera_builder._finish_store

    def crash(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("killed mid-write")

    monkeypatch.setattr(atera_builder, "_finish_store", crash)
    with pytest.raises(RuntimeError, match="killed mid-write"):
        build_spatialdata_artifact(config)
    assert not persistent.exists()
    assert sorted(path.name for path in persistent.parent.iterdir()) == []

    monkeypatch.setattr(atera_builder, "_finish_store", finish)
    assert build_spatialdata_artifact(config) == output
    assert atera_builder.read_build_marker(persistent) is not None
    assert output.resolve() == persistent.resolve()


def test_incomplete_store_is_rebuilt_and_stale_partials_removed(
    built: Built, tmp_path: Path
) -> None:
    persistent = tmp_path / "cache" / "source.zarr"
    shutil.copytree(built.store, persistent)
    (persistent / atera_builder.BUILD_MARKER_NAME).unlink()
    (persistent / "points").rename(persistent / "points_half_written")
    stale = tmp_path / "cache" / "source.zarr.partial-0123"
    stale.mkdir()
    config = _config(built.synth.root, tmp_path / "out.zarr", persistent=persistent)
    build_spatialdata_artifact(config)
    assert atera_builder.read_build_marker(persistent) is not None
    assert (persistent / "points").is_dir()
    assert not (persistent / "points_half_written").exists()
    assert not stale.exists()


def test_complete_matching_store_is_reused_without_a_build(
    built: Built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persistent = tmp_path / "cache" / "source.zarr"
    shutil.copytree(built.store, persistent)

    def refuse(**_kwargs: Any) -> Path:
        raise AssertionError("a matching complete store must be reused")

    monkeypatch.setattr(atera_builder, "write_atera_spatialdata", refuse)
    output = tmp_path / "stage" / "source.zarr"
    config = _config(built.synth.root, output, persistent=persistent)
    assert build_spatialdata_artifact(config) == output
    # Without the raw folder, a complete store is reused as it is.
    zarr_only = _config(persistent, output, persistent=persistent)
    assert build_spatialdata_artifact(zarr_only) == output


def test_fingerprint_mismatch_refuses_unless_forced(
    built: Built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persistent = tmp_path / "cache" / "source.zarr"
    shutil.copytree(built.store, persistent)
    output = tmp_path / "stage" / "source.zarr"
    config = _config(built.synth.root, output, persistent=persistent, min_qv=30.0)
    with pytest.raises(ValueError, match="build_fingerprint mismatch"):
        build_spatialdata_artifact(config)

    calls: list[Path] = []

    def record(**kwargs: Any) -> Path:
        calls.append(kwargs["output_path"])
        return Path(kwargs["output_path"])

    monkeypatch.setattr(atera_builder, "write_atera_spatialdata", record)
    build_spatialdata_artifact(config, force_rerun=True)
    assert calls == [persistent]


def test_store_without_marker_and_without_raw_input_fails(
    built: Built, tmp_path: Path
) -> None:
    persistent = tmp_path / "cache" / "source.zarr"
    shutil.copytree(built.store, persistent)
    (persistent / atera_builder.BUILD_MARKER_NAME).unlink()
    config = _config(persistent, tmp_path / "out.zarr", persistent=persistent)
    with pytest.raises(FileNotFoundError, match="complete SpatialData store"):
        build_spatialdata_artifact(config)
    with pytest.raises(ValueError, match="force-rerun"):
        shutil.copy(
            built.store / atera_builder.BUILD_MARKER_NAME,
            persistent / atera_builder.BUILD_MARKER_NAME,
        )
        build_spatialdata_artifact(config, force_rerun=True)


def test_check_cached_build(tmp_path: Path) -> None:
    store = tmp_path / "store.zarr"
    store.mkdir()
    assert not atera_builder.check_cached_atera_build(store, "a", force_rerun=False)
    (store / atera_builder.BUILD_MARKER_NAME).write_text("{broken")
    assert not atera_builder.check_cached_atera_build(store, "a", force_rerun=False)
    (store / atera_builder.BUILD_MARKER_NAME).write_text(
        json.dumps({"build_fingerprint": "a"})
    )
    assert atera_builder.check_cached_atera_build(store, "a", force_rerun=False)
    assert atera_builder.check_cached_atera_build(store, None, force_rerun=False)
    assert not atera_builder.check_cached_atera_build(store, "b", force_rerun=True)
    with pytest.raises(ValueError, match="mismatch"):
        atera_builder.check_cached_atera_build(store, "b", force_rerun=False)


# --------------------------------------------------------------------------- #
# Options
# --------------------------------------------------------------------------- #


def test_vendor_polygon_fallback_builds_and_needs_a_shape_per_cell(
    built: Built, tmp_path: Path
) -> None:
    bundle = ar.resolve_atera_bundle(built.synth.root)
    config = AteraBuildConfig(boundary_source="vendor_polygons", **FAST_OPTIONS)
    assembly = atera_builder.assemble_atera_sdata(
        bundle, config, n_jobs=1, workdir=tmp_path
    )
    nuclei = assembly.sdata.shapes["nucleus_boundaries"]
    owners = set(nuclei["cell_instance_id"])
    # The fixture's empty vendor polygon is a nucleus: it is skipped.
    assert len(nuclei) == len(built.synth.nucleus_cell_index) - 1
    assert owners <= set(assembly.sdata.shapes["cell_boundaries"]["instance_id"])
    assert EMPTY_POLYGON_CELL + 1 in set(
        assembly.sdata.shapes["cell_boundaries"]["instance_id"]
    )
    cells = ar.read_atera_cells(bundle)
    shapes = ar.atera_shapes_from_vendor_polygons(cells, bundle.pixel_size_um)
    partial = shapes["cell_boundaries"].iloc[1:]
    with pytest.raises(ar.AteraFormatError, match="no cell polygon"):
        atera_builder._check_table_cells_have_shapes(partial, cells)


def test_elements_can_be_switched_off(built: Built, tmp_path: Path) -> None:
    bundle = ar.resolve_atera_bundle(built.synth.root)
    config = AteraBuildConfig(
        transcripts=False,
        morphology_2d=False,
        nucleus_labels=False,
        nucleus_boundaries=False,
        **FAST_OPTIONS,
    )
    assembly = atera_builder.assemble_atera_sdata(
        bundle, config, n_jobs=1, workdir=tmp_path
    )
    sdata = assembly.sdata
    assert not sdata.points and not sdata.images
    assert set(sdata.labels) == {"cell_labels"}
    assert set(sdata.shapes) == {"cell_boundaries"}
    assert assembly.aggregates is None


# --------------------------------------------------------------------------- #
# XENIUM preview refusal and privacy of committed fixtures
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("experiment", "refused"),
    [
        ({"analysis_sw_version": "atera-1.1.1.0"}, True),
        (
            {
                "analysis_sw_version": "xenium-4.0.0",
                "panel_num_targets_predesigned": 18049,
            },
            True,
        ),
        (
            {
                "analysis_sw_version": "xenium-3.0.0",
                "panel_num_targets_predesigned": 5001,
                "panel_num_targets_custom": 100,
            },
            False,
        ),
        ({"analysis_sw_version": "xenium-2.0.0"}, False),
    ],
)
def test_xenium_builder_refuses_atera_preview_bundles(
    tmp_path: Path, experiment: dict[str, Any], refused: bool
) -> None:
    (tmp_path / "experiment.xenium").write_text(json.dumps(experiment))
    if refused:
        with pytest.raises(ValueError, match="Atera run"):
            check_not_atera_preview(tmp_path)
    else:
        check_not_atera_preview(tmp_path)
    check_not_atera_preview(tmp_path / "no_experiment_here")


def test_fixture_identifiers_are_obviously_fake() -> None:
    """Committed fixtures hold no real sample, run, slide or instrument name."""
    markers = ("SYNTH", "Synthetic", "deadbeef", "example.invalid", "1999-")
    for key, value in FAKE_IDENTIFIERS.items():
        assert any(marker.lower() in value.lower() for marker in markers), key
    for name in ("atera_synth.py", "test_atera_build.py", "test_atera_reader.py"):
        text = (_HERE / name).read_text()
        assert not privacy.find_host_paths(text), name
        assert not privacy.find_generic_donor_ids(text), name


# --------------------------------------------------------------------------- #
# Integration: the on-disk store re-read (plan §7.3)
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_store_passes_the_streaming_deep_check(built: Built) -> None:
    sdata = sd.read_zarr(built.store)
    rows = _attrs(built.store)["merxen_atera"]["transcripts"]["rows_kept_per_tile"]
    summaries = validate_merxen_schema_streaming(
        sdata,
        group_column="source_tile",
        expected_group_rows={"transcripts": rows},
    )
    assert summaries["transcripts"].n_rows == sum(rows.values())
    assert summaries["transcripts"].null_group_rows == 0


@pytest.mark.integration
def test_pixel_size_resolves_through_the_shared_helper(built: Built) -> None:
    assert find_um_per_px(built.store, "ATERA") == built.synth.pixel_size_um
    experiment = _attrs(built.store)["merxen_atera"]["experiment"]
    assert experiment["pixel_size"] == built.synth.pixel_size_um


@pytest.mark.integration
def test_rewritten_store_keeps_the_sidecars_and_index(
    built: Built, tmp_path: Path
) -> None:
    """``atera/`` and the root JSON files survive a store rewrite."""
    from merxen.io.spatialdata_io import _copy_non_spatialdata_sidecars

    target = tmp_path / "rewritten.zarr"
    sdata = sd.read_zarr(built.store)
    sdata.write(target)
    _copy_non_spatialdata_sidecars(built.store, target)
    for name in (
        "atera/experiment.spatial",
        atera_builder.POINTS_INDEX_NAME,
        atera_builder.BUILD_MARKER_NAME,
    ):
        assert (target / name).read_bytes() == (built.store / name).read_bytes()
    reread = sd.read_zarr(target)
    assert reread.attrs["merxen_atera"] == sdata.attrs["merxen_atera"]
    table = reread.tables["table"].obs
    assert isinstance(table["region"].dtype, pd.CategoricalDtype)
