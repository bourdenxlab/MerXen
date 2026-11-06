"""ENRICH in native mode on the AT-6 reference latest store of an ATERA row.

The reference store (``test_io.atera_native_fixture``) has the layout
NATIVE_LATEST produces: the vendor shapes renamed to ``atera_cell_boundaries``
and ``atera_nucleus``, the vendor table as ``table_original`` and the
``original``/``original_nucleus`` registry entries. Native enrichment must keep
all of that, add only the label-check counts, and never write a Cellpose layer
or Cellpose provenance.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
from click.testing import CliRunner
from test_io.atera_native_fixture import (
    CELL_KEY,
    CELL_SHAPES_KEY,
    NUCLEUS_SHAPES_KEY,
    TABLE_KEY,
    make_native_reference,
)

from merxen.cli.run_enrichment import enrich_command
from merxen.config import EnrichmentConfig
from merxen.enrichment import enrich as enrich_module
from merxen.enrichment.assignment import run_per_shape_assignment_for_dataset
from merxen.enrichment.enrich import (
    NATIVE_ENRICHMENT_ATTR,
    _is_already_enriched,
    _is_enriched_except_nuclei,
    _partial_enrichment_artifact_paths,
    _remove_partial_enrichment_artifacts_from_zarr_path,
    enrich_single_latest,
    native_label_check_path,
)
from merxen.enrichment.label_lookup import NATIVE_COUNT_COLUMNS
from merxen.io.points_aggregates import points_parquet_dir
from merxen.io.spatialdata_schema import MERXEN_SCHEMA_ATTR
from merxen.segmentation import native as native_segmentation

PS = 0.21249222
DATASET = "ATERA_SYNTH"


@dataclass(frozen=True)
class Stores:
    source: Path
    latest: Path
    output_dir: Path


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return make_native_reference(tmp_path_factory.mktemp("native_enrich_ref"))


@pytest.fixture
def stores(reference: tuple[Path, Path], tmp_path: Path) -> Iterator[Stores]:
    """A private copy of the reference latest store; the source is shared."""
    source, latest = reference
    copy = tmp_path / "latest_spatialdata.zarr"
    shutil.copytree(latest, copy)
    yield Stores(source=source, latest=copy, output_dir=tmp_path / "enrich_out")


def _config(stores: Stores, **overrides: Any) -> EnrichmentConfig:
    fields: dict[str, Any] = {
        "dataset_name": DATASET,
        "platform": "ATERA",
        "latest_zarr_path": stores.latest,
        "mask_path": None,
        "nuclei_mask_path": None,
        "original_data_path": stores.source,
        "output_dir": stores.output_dir,
        "segmentation_source": "native",
    }
    fields.update(overrides)
    return EnrichmentConfig(**fields)


def _manifest(root: Path) -> dict[str, tuple[int, str]]:
    return {
        str(path.relative_to(root)): (
            path.stat().st_ino,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _points(store: Path) -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_parquet(path)
            for path in sorted(points_parquet_dir(store).glob("*.parquet"))
        ],
        ignore_index=True,
    )


def _labels(store: Path, key: str) -> np.ndarray:
    sdata = sd.read_zarr(store)
    return np.asarray(sd.get_pyramid_levels(sdata.labels[key], n=0))


def _pixel_labels(store: Path, key: str, points: pd.DataFrame) -> np.ndarray:
    labels = _labels(store, key)
    column = np.floor(points["x"].to_numpy(np.float64) / PS).astype(int)
    row = np.floor(points["y"].to_numpy(np.float64) / PS).astype(int)
    inside = (
        (column >= 0)
        & (row >= 0)
        & (column < labels.shape[1])
        & (row < labels.shape[0])
    )
    out = np.zeros(len(points), dtype=np.int64)
    out[inside] = labels[row[inside], column[inside]]
    return out


# --------------------------------------------------------------------------- #
# The enriched store
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_native_enrichment_adds_only_counts_and_vendor_provenance(
    stores: Stores,
) -> None:
    """No MOSAIK_*/Cellpose layer or identity; table_original only gains counts."""
    before = sd.read_zarr(stores.latest)
    table_before = before.tables[TABLE_KEY].copy()
    shapes_before = {key: before.shapes[key].copy() for key in before.shapes}
    source_manifest = _manifest(stores.source)

    written = enrich_single_latest(_config(stores))

    assert written == stores.latest
    after = sd.read_zarr(stores.latest)
    assert set(after.shapes) == {CELL_SHAPES_KEY, NUCLEUS_SHAPES_KEY}
    assert set(after.tables) == {TABLE_KEY}
    assert set(after.labels) == {"cell_labels", "nucleus_labels"}
    assert not any(str(key).startswith("MOSAIK") for key in after.shapes)

    schema = after.attrs[MERXEN_SCHEMA_ATTR]
    assert "model_identities" not in schema
    assert schema["segmentation_source"] == "native"
    provenance = schema["native_segmentation"]
    assert provenance["source"] == "vendor"
    assert provenance["analysis_sw_version"] == "atera-1.1.1.0"
    assert sum(provenance["cell_segmentation_methods"].values()) == len(
        after.shapes[CELL_SHAPES_KEY]
    )
    registry = schema["segmentations"]
    assert set(registry) == {"original", "original_nucleus"}
    assert registry["original"]["table"] == TABLE_KEY
    assert registry["original"]["shape"] == CELL_SHAPES_KEY
    assert registry["original_nucleus"]["table"] is None
    assert registry["original_nucleus"]["cell_key"] == CELL_KEY
    assert not any("model_identity" in entry for entry in registry.values())

    for key, shape in shapes_before.items():
        pd.testing.assert_frame_equal(
            pd.DataFrame(after.shapes[key].drop(columns="geometry")),
            pd.DataFrame(shape.drop(columns="geometry")),
        )
        assert after.shapes[key].geometry.equals(shape.geometry)

    table = after.tables[TABLE_KEY]
    added = [column for column in table.obs.columns if column.startswith("merxen_")]
    assert sorted(added) == sorted(NATIVE_COUNT_COLUMNS.values())
    pd.testing.assert_frame_equal(table.obs.drop(columns=added), table_before.obs)
    pd.testing.assert_frame_equal(table.var, table_before.var)
    assert (table.X != table_before.X).nnz == 0
    assert set(table.obsm) == set(table_before.obsm)
    assert table.uns["spatialdata_attrs"] == table_before.uns["spatialdata_attrs"]

    record = after.attrs[NATIVE_ENRICHMENT_ATTR]
    assert record["version"] == enrich_module.NATIVE_ENRICHMENT_VERSION
    assert record["label_check"]["passed"] is True
    assert _manifest(stores.source) == source_manifest


@pytest.mark.integration
def test_native_label_check_matches_the_vendor_exactly(stores: Stores) -> None:
    """0 mismatches; 100% overlaps_nucleus agreement; counts equal brute force."""
    enrich_single_latest(_config(stores))

    summary = json.loads(
        native_label_check_path(stores.output_dir, DATASET).read_text()
    )
    points = _points(stores.latest)
    assert summary["passed"] is True
    assert summary["n_mismatch"] == 0
    assert summary["n_rows"] == len(points)
    assert summary["n_vendor_assigned"] == int(
        points["original_assignment"].notna().sum()
    )
    assert summary["nucleus"]["overlaps_agreement"] == 1.0
    assert summary["nucleus"]["n_nucleus_outside_own_cell"] == 0
    assert summary["store_points_scale_agrees"] is True

    cells = _pixel_labels(stores.latest, "cell_labels", points)
    table = sd.read_zarr(stores.latest).tables[TABLE_KEY]
    ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    expected = np.array([(cells == cell).sum() for cell in ids])
    np.testing.assert_array_equal(
        table.obs[NATIVE_COUNT_COLUMNS["cell"]].to_numpy(), expected
    )
    assigned = points["original_assignment"].fillna(0).to_numpy(np.int64)
    np.testing.assert_array_equal(
        expected, np.array([(assigned == cell).sum() for cell in ids])
    )


@pytest.mark.integration
def test_two_nucleus_cell_gets_both_nuclei_transcripts(stores: Stores) -> None:
    """Nuclear counts go to the owning cell, never to the cell with the nucleus id."""
    enrich_single_latest(_config(stores))
    sdata = sd.read_zarr(stores.latest)
    nuclei = sdata.shapes[NUCLEUS_SHAPES_KEY]
    owners = dict(
        zip(
            nuclei["instance_id"].astype(int),
            nuclei[CELL_KEY].astype(int),
            strict=True,
        )
    )
    points = _points(stores.latest)
    per_nucleus = pd.Series(
        _pixel_labels(stores.latest, "nucleus_labels", points)
    ).value_counts()
    two_nucleus_cells = [
        cell for cell in set(owners.values()) if list(owners.values()).count(cell) > 1
    ]
    assert two_nucleus_cells, "the fixture has a two-nucleus cell"

    table = sdata.tables[TABLE_KEY]
    got = dict(
        zip(
            table.obs["instance_id"].astype(int),
            table.obs[NATIVE_COUNT_COLUMNS["nucleus"]].astype(int),
            strict=True,
        )
    )
    for cell, count in got.items():
        mine = [nucleus for nucleus, owner in owners.items() if owner == cell]
        assert count == sum(int(per_nucleus.get(nucleus, 0)) for nucleus in mine)
    for cell in two_nucleus_cells:
        mine = [nucleus for nucleus, owner in owners.items() if owner == cell]
        assert sum(int(per_nucleus.get(n, 0) > 0) for n in mine) == 2
        assert got[cell] > max(int(per_nucleus.get(n, 0)) for n in mine)


@pytest.mark.integration
def test_a_shifted_pixel_size_fails_the_check(stores: Stores, tmp_path: Path) -> None:
    """A wrong transform is caught before any count is written."""
    transform = tmp_path / "shifted_transform.txt"
    scale = 1.0 / (PS * 1.05)
    np.savetxt(transform, np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]]))

    with pytest.raises(RuntimeError, match="Native label check failed"):
        enrich_single_latest(_config(stores, transform_path=transform))

    summary = json.loads(
        native_label_check_path(stores.output_dir, DATASET).read_text()
    )
    assert summary["passed"] is False
    assert summary["n_mismatch"] > 0
    assert summary["store_points_scale_agrees"] is False
    after = sd.read_zarr(stores.latest)
    assert not any(c.startswith("merxen_") for c in after.tables[TABLE_KEY].obs)
    assert NATIVE_ENRICHMENT_ATTR not in after.attrs


@pytest.mark.integration
def test_a_second_run_is_a_skip_and_rewrites_the_summary(
    stores: Stores, monkeypatch: pytest.MonkeyPatch
) -> None:
    enrich_single_latest(_config(stores))
    summary_path = native_label_check_path(stores.output_dir, DATASET)
    first = summary_path.read_text()
    summary_path.unlink()

    def _fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("the label check must not run again")

    monkeypatch.setattr(enrich_module, "native_label_check", _fail)
    enrich_single_latest(_config(stores))
    assert summary_path.read_text() == first

    with pytest.raises(AssertionError, match="must not run again"):
        enrich_single_latest(_config(stores), force_rerun=True)


@pytest.mark.integration
def test_the_label_check_can_be_switched_off(stores: Stores) -> None:
    enrich_single_latest(_config(stores, native_label_check=False))
    after = sd.read_zarr(stores.latest)
    assert after.attrs[NATIVE_ENRICHMENT_ATTR]["label_check"] is None
    assert not any(c.startswith("merxen_") for c in after.tables[TABLE_KEY].obs)
    assert not native_label_check_path(stores.output_dir, DATASET).exists()
    # Switching it on later runs it.
    enrich_single_latest(_config(stores))
    assert native_label_check_path(stores.output_dir, DATASET).exists()


# --------------------------------------------------------------------------- #
# Partial stores
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_missing_renamed_elements_are_recreated_from_the_source(
    stores: Stores, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ENRICH calls ensure_native_elements, re-reads, and completes."""
    sdata = sd.read_zarr(stores.latest)
    nuclei = sdata.shapes[NUCLEUS_SHAPES_KEY].copy()
    sdata.delete_element_from_disk(NUCLEUS_SHAPES_KEY)
    calls: list[tuple[Path, Path]] = []

    def _ensure(latest_path: Path, source_path: Path) -> list[str]:
        calls.append((Path(latest_path), Path(source_path)))
        store = sd.read_zarr(latest_path)
        store.shapes[NUCLEUS_SHAPES_KEY] = nuclei
        store.write_element(NUCLEUS_SHAPES_KEY)
        store.write_consolidated_metadata()
        return [f"shapes/{NUCLEUS_SHAPES_KEY}"]

    monkeypatch.setattr(native_segmentation, "ensure_native_elements", _ensure)
    enrich_single_latest(_config(stores))

    assert calls == [(stores.latest, stores.source)]
    after = sd.read_zarr(stores.latest)
    assert NUCLEUS_SHAPES_KEY in after.shapes
    assert after.attrs[NATIVE_ENRICHMENT_ATTR]["label_check"]["passed"] is True


@pytest.mark.integration
def test_missing_payload_elements_are_refused(stores: Stores) -> None:
    """Images, labels and points are shared payloads only NATIVE_LATEST restores."""
    shutil.rmtree(stores.latest / "labels" / "nucleus_labels")
    enrich_module._prune_zarr_consolidated_metadata(
        stores.latest / "zarr.json", {"labels/nucleus_labels"}
    )
    enrich_module._prune_zarr_consolidated_metadata(
        stores.latest / "labels" / "zarr.json", {"nucleus_labels"}
    )
    with pytest.raises(RuntimeError, match="payload elements.*nucleus_labels"):
        enrich_single_latest(_config(stores))


@pytest.mark.integration
def test_partial_cleanup_of_a_native_store_removes_only_derived_tables(
    stores: Stores,
) -> None:
    """Payloads, renamed shapes and table_original survive; the source too."""
    derived = stores.latest / "tables" / "table_atera_cell_boundaries_extra"
    derived.mkdir(parents=True)
    source_manifest = _manifest(stores.source)

    assert _partial_enrichment_artifact_paths(stores.latest, "ATERA") == {
        "tables/table_atera_cell_boundaries_extra"
    }
    _remove_partial_enrichment_artifacts_from_zarr_path(stores.latest, "ATERA")

    assert not derived.exists()
    for rel in (
        f"shapes/{CELL_SHAPES_KEY}",
        f"shapes/{NUCLEUS_SHAPES_KEY}",
        f"tables/{TABLE_KEY}",
        "images/morphology_focus",
        "labels/cell_labels",
        "labels/nucleus_labels",
        "points/transcripts",
    ):
        assert (stores.latest / rel).is_dir(), rel
    assert _manifest(stores.source) == source_manifest
    sdata = sd.read_zarr(stores.latest)
    assert set(sdata.shapes) == {CELL_SHAPES_KEY, NUCLEUS_SHAPES_KEY}


def _namespace(attrs: dict[str, Any] | None = None, **missing: str) -> Any:
    elements = {
        "shapes": {CELL_SHAPES_KEY, NUCLEUS_SHAPES_KEY},
        "tables": {TABLE_KEY},
        "images": {"morphology_focus"},
        "labels": {"cell_labels", "nucleus_labels"},
        "points": {"transcripts"},
    }
    for kind, name in missing.items():
        elements[kind].discard(name)
    return SimpleNamespace(
        attrs=attrs or {},
        **{kind: dict.fromkeys(names, object()) for kind, names in elements.items()},
    )


@pytest.mark.parametrize(
    "missing",
    [
        {},
        {"shapes": NUCLEUS_SHAPES_KEY},
        {"tables": TABLE_KEY},
        {"images": "morphology_focus"},
        {"labels": "cell_labels"},
        {"points": "transcripts"},
    ],
)
def test_native_completeness_needs_vendor_elements_not_cellpose_layers(
    missing: dict[str, str],
) -> None:
    store = _namespace(**missing)
    assert _is_already_enriched(store, "ATERA") is (not missing)
    assert _is_enriched_except_nuclei(store, "ATERA") is False


# --------------------------------------------------------------------------- #
# Per-shape assignment and the CLI
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_native_assignment_registers_table_original_without_a_clone(
    stores: Stores,
) -> None:
    enrich_single_latest(_config(stores))
    summaries = run_per_shape_assignment_for_dataset(
        DATASET,
        stores.latest,
        platform="ATERA",
        segmentation_source="native",
    )

    assert [(s["shape_key"], s["table_key"], s["mode"]) for s in summaries] == [
        (CELL_SHAPES_KEY, TABLE_KEY, "native_table_original")
    ]
    after = sd.read_zarr(stores.latest)
    assert set(after.tables) == {TABLE_KEY}
    registry = after.attrs[MERXEN_SCHEMA_ATTR]["segmentations"]
    assert registry["original"]["table"] == TABLE_KEY
    assert registry["original"]["assignment_column"] == "original_assignment"
    assert registry["original_nucleus"]["cell_key"] == CELL_KEY
    assert registry["original_nucleus"]["table"] is None


@pytest.mark.integration
def test_native_assignment_needs_a_native_platform(stores: Stores) -> None:
    with pytest.raises(ValueError, match="native segmentation"):
        run_per_shape_assignment_for_dataset(
            DATASET, stores.latest, platform="XENIUM", segmentation_source="native"
        )


@pytest.mark.integration
def test_enrich_cli_native_row_end_to_end(stores: Stores, tmp_path: Path) -> None:
    """The ENRICH task command on a native row: counts, registry, summaries."""
    config = tmp_path / "enrich_config.json"
    config.write_text(_config(stores).model_dump_json(indent=2))

    result = CliRunner().invoke(enrich_command, ["--config", str(config)])
    assert result.exit_code == 0, result.output

    summary_csv = stores.output_dir / "atera_synth_per_shape_assignment_summary.csv"
    rows = pd.read_csv(summary_csv)
    assert rows[["shape_key", "table_key", "mode"]].values.tolist() == [
        [CELL_SHAPES_KEY, TABLE_KEY, "native_table_original"]
    ]
    assert native_label_check_path(stores.output_dir, DATASET).exists()
    after = sd.read_zarr(stores.latest)
    assert set(after.tables) == {TABLE_KEY}
    assert {"MOSAIK_proseg", "MOSAIK_cellpose", "cellpose_nuclei"}.isdisjoint(
        after.shapes
    )

    # A second task run on the enriched store is a no-op apart from summaries.
    result = CliRunner().invoke(enrich_command, ["--config", str(config)])
    assert result.exit_code == 0, result.output
    assert set(sd.read_zarr(stores.latest).tables) == {TABLE_KEY}


def test_native_partial_paths_hold_only_derived_tables(tmp_path: Path) -> None:
    """ATERA (native only): no shapes, images, labels or points are rebuildable."""
    store = tmp_path / "store.zarr"
    assert _partial_enrichment_artifact_paths(store, "ATERA") == set()
    for name in ("table_original", "table_atera_cell_boundaries_x", "not_a_table"):
        (store / "tables" / name).mkdir(parents=True)
    assert _partial_enrichment_artifact_paths(store, "ATERA") == {
        "tables/table_atera_cell_boundaries_x"
    }


@pytest.mark.parametrize("platform", ["XENIUM", "MERSCOPE"])
def test_a_native_record_never_makes_xenium_or_merscope_native(platform: str) -> None:
    attrs = {MERXEN_SCHEMA_ATTR: {"segmentation_source": "native"}}
    assert enrich_module._is_native_store(platform, attrs) is False
    assert enrich_module._is_native_store(platform, {}) is False
    assert enrich_module._is_native_store("ATERA", {}) is True
